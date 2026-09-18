# GDN 算子：代码实现与公式对照

> 分析对象：`vllm_ascend/ops/gdn.py`（Ascend 910 路径）
> 关联实现：`vllm_ascend/ops/gdn_attn_builder.py`、`vllm_ascend/ops/triton/fla/*`、`csrc/moe/chunk_*`、`csrc/attention/recurrent_gated_delta_rule`
> 重点：prefill 阶段的并行计算优化原理
>
> **公式排版约定**：GitHub 会先对正文做一遍 markdown 转义处理，行内公式（`$...$`）里的 `\_`、`\,`、`\{` 等反斜杠会被吃掉（`\text{num\_spec}` 会变成 `\text{num_spec}`，进而在 MathJax 里报 “'_' allowed only in math mode”），且 GitHub 的 MathJax 未加载 ams 扩展，不认 `\operatorname`。另外 GitHub 对行内公式的**定界符位置**有硬性要求，不满足就整条公式失效（`$` 原样显示）。因此本文约定四条：
>
> 1. 公式内不写下划线，标识符改用连字符（如 `\text{num-spec}` 对应变量 `num_spec`）；函数名统一用 `\mathrm{}` 而非 `\operatorname{}`。
> 2. **开符号 `$` 的左侧必须是 ASCII 空格或行首**。紧跟汉字、全角标点（`：`、`（`、`，`、`。`）时公式不会被识别，需要在标点后补一个空格：`（ $T$ 很小）`。
> 3. **闭符号 `$` 的右侧不能紧跟 ASCII 字母、数字或下划线**（`$T=32$k` 会失效）。这类单位后缀写进公式内部即可：`$T=32\text{k}$`。
> 4. 定界符内侧不留空白：`$ x $` 不生效。
>
> 改动公式时请保持这些约定。

---

## 0. 符号表

| 符号 | 代码变量 | 含义 |
|---|---|---|
| $C$ | `chunk_size` / `_GDN_CHUNK_SIZE` | chunk 大小，固定 64 |
| $T$ | `num_actual_tokens` / `T` | 序列总 token 数 |
| $N$ | `len(cu_seqlens)-1` | 序列（请求）条数 |
| $H$ / $N_v$ | `num_v_heads` | value head 数 |
| $H_g$ / $N_k$ | `num_k_heads` | key/query head 数（GQA） |
| $D_k$ | `head_k_dim` | key/query 维度，典型 128 |
| $D_v$ | `head_v_dim` | value 维度，典型 128 |
| $\beta_t$ | `beta` | 写入门控， $b$ 经 sigmoid 得到， $\in(0,1)$ |
| $g_t$ | `g` | log 遗忘门， $\le 0$，fp32 |
| $\tilde g_t$ | `g_cumsum` | **块内**累积 log 遗忘门 |
| $S_t$ / $H_n$ | `ssm_state` / `h` | 递归状态， $[D_k, D_v]$ |
| $\Phi_n,\ P_n$ | — | 状态递推的仿射系数矩阵 |

**状态张量 `ssm_state` 物理布局**：`[N, H, D_v, D_k]`。
（证据：`gdn.py:628` chunk 路径做 `.transpose(-1,-2)` 得到 `[N,H,D_k,D_v]`；`gdn.py:609` 注释说明融合算子的 `[N, Nv, Dv, Dk]` 与 `ssm_state` 直接对齐，无需转置。）

---

## 1. 算子数学原型

### 1.1 Gated Delta Rule（recurrent 形式）

GDN 在 delta rule 的基础上引入遗忘门。单步递推：

$$
S_t = S_{t-1}\cdot \mathrm{diag}(\alpha_t)\Big(I - \beta_t k_t k_t^{\top}\Big) + \beta_t v_t k_t^{\top},
\qquad \alpha_t = e^{g_t}
$$

输出：

$$
o_t = S_t^{\top} q_t
$$

等价展开形式（delta 修正视角，更直观）：

$$
S_t = S_{t-1}\cdot\mathrm{diag}(\alpha_t) - \beta_t\Big(S_{t-1}\mathrm{diag}(\alpha_t) k_t - v_t\Big)k_t^{\top}
$$

**这一步天然串行**： $S_t$ 严格依赖 $S_{t-1}$，递推深度 $O(T)$，tensor core 几乎闲置。所以它只用于 decode（ $T$ 很小，通常 $=1$ 或 $=1+\text{num-spec}$）。

### 1.2 门控生成

`fused_gdn_gating.py:48-56`：

$$
\begin{aligned}
x_t &= a_t + \text{dt-bias} \\
\mathrm{softplus}_\beta(x) &= \begin{cases}\dfrac{1}{\beta}\log\big(1+e^{\beta x}\big), & \beta x \le \tau \\ x, & \beta x > \tau\end{cases} \qquad (\beta=1.0,\ \tau=20.0)\\
g_t &= -e^{A_{\log}} \cdot \mathrm{softplus}_\beta(x_t) \\
\beta_t &= \sigma(b_t)
\end{aligned}
$$

对应代码：

```python
x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)[None, :]
softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
```

两个工程要点：

- `softplus` 的 `threshold=20.0` 分支是标准数值稳定技巧—— $\beta x$ 大时 $e^{\beta x}$ 溢出，改用线性近似（误差可忽略）。这同时也是 **AscendC 算子需要 `threshold` 参数的原因**。
- $g_t$ 恒 $\le 0$（负号 × 非负 softplus），这个性质被后文的 `safe_exp` 依赖。
- 输出 dtype 有意区分：`g` 是 **fp32**，`beta` 跟随输入 dtype（bf16）。

调用点（`gdn.py:516`）：

```python
g, beta = DeviceOperator.fused_gdn_gating(self.A_log, a, b, self.dt_bias)
```

### 1.3 Chunkwise Parallel Form（prefill 形式）

将序列按 $C=64$ 切块，第 $n$ 块覆盖 token $[nC,\ (n+1)C)$。块内累积门与衰减矩阵：

$$
\tilde g_t = \sum_{i=nC}^{t} g_i, \qquad
D_{ij} = \exp\big(\tilde g_i - \tilde g_j\big)
$$

（注意 $i<j$ 时 $D_{ij}>1$ 会溢出，下文 §5.1 专门处理。）

**块内辅助量**（WY 表示）：

$$
\begin{aligned}
L &= \mathrm{tril}_{i>j}\Big(\mathrm{diag}(\beta)\,K K^{\top}\odot D\Big) \\
A &= \big(I - L\big)^{-1} \\
W &= A\cdot\big(\beta \odot e^{\tilde g}\odot K\big) \\
U &= A\cdot\big(\beta \odot V\big)
\end{aligned}
$$

**块间状态递推**（`chunk_delta_h.py:97-166`，逐行核对）：

$$
V'_n = \Big(U_n - W_n H_n\Big)\odot\exp\big(\tilde g_{last,n} - \tilde g^{(n)}\big)
$$

$$
\boxed{\ H_{n+1} = \exp\big(\tilde g_{last,n}\big)\,H_n \;+\; K_n^{\top} V'_n\ }
$$

**块内输出**（`chunk_o.py:70-109`）：

$$
O_n = \text{scale}\cdot\left[\Big(Q_n H_n\Big)\odot e^{\tilde g^{(n)}} \;+\; \mathrm{tril}_{i\ge j}\Big(\big(Q_nK_n^{\top}\big)\odot D\Big)\,V'_n\right]
$$

### 1.4 为什么可以并行：仿射性是关键

把 §1.3 的 state 递推展开：

$$
H_{n+1} = \exp\big(\tilde g_{last,n}\big)H_n + K_n^{\top}\Big[\big(U_n - W_n H_n\big)\odot e^{\tilde g_{last}-\tilde g}\Big]
$$

所有对 $H_n$ 的依赖都是**线性**的，因此可以写成仿射形式：

$$
H_{n+1} = \Phi_n H_n + P_n
$$

$$
\begin{aligned}
\Phi_n &= \exp\big(\tilde g_{last,n}\big) I - K_n^{\top}\mathrm{diag}\big(e^{\tilde g_{last}-\tilde g}\big) W_n \in \mathbb{R}^{D_k\times D_k}\\
P_n &= K_n^{\top}\mathrm{diag}\big(e^{\tilde g_{last}-\tilde g}\big) U_n \in \mathbb{R}^{D_k\times D_v}
\end{aligned}
$$

**这个仿射性同时支撑了三处优化**：

1. 块间解耦 → `A/W/U` 只依赖本块，可完全并行（§4.2）
2. 状态递推的串行长度从 $T$ 降到 $T/C$（§4.1）
3. 跨卡（PCP）修正可以做到**精确**而非近似（§4.3）

---

## 2. 代码结构总览

### 2.1 `forward`：三段式（`gdn.py:264-344`）

```python
def forward(self, hidden_states, output=None):
    # Part 1: 输入投影  → mixed_qkv, z, b, a
    # Part 2: 核心注意力（custom op）
    core_attn_out = torch.zeros(...)          # 注意：zeros 而非 empty
    torch.ops.vllm.qwen_gdn_attention_core(mixed_qkv, b, a, core_attn_out, self.prefix, False)
    # Part 3: 输出投影  → gated RMSNorm → out_proj
```

Part 3 的门控归一化（对应 GDN 论文里的 output gate $z$）：

$$
y = \mathrm{RMSNorm}_{\text{weighted}}(o,\ z)
$$

代码 `gdn.py:338`：`core_attn_out = self.norm(core_attn_out, z)`。

### 2.2 `core_attn_out` 为何用 `zeros`

`gdn.py:313-320` 注释引用了上游 PR #28182。原因：部分执行路径（例如全部 token 落在 spec 分支、`core_attn_out_non_spec` 返回 `None`）不会写满整个张量，用 `empty` 会残留脏数据。**这是踩坑后留下的防御性写法，不要"优化"掉。**

### 2.3 `_forward_core` 的分流决策（`gdn.py:346-686`）

```
输入 mixed_qkv, b, a
  │
  ├─ attn_metadata is None ────────────────► V1 profile run，直接 return
  │
  ├─ 1. causal conv1d（三路）
  │    ├─ spec       : gdn.py:401-420   （run_mode=1）
  │    ├─ prefill    : gdn.py:423-488   （run_mode=0，含 PCP halo 交换）
  │    └─ decode     : gdn.py:489-508   （run_mode=1）
  │
  ├─ 2. gating: g, beta = fused_gdn_gating(...)
  │    └─ 按 spec_token_indx / non_spec_token_indx 拆分
  │
  ├─ 3. 核心计算（三路互斥）
  │    ├─ 2.1 spec decode : npu_recurrent_gated_delta_rule
  │    ├─ 2.2 mixed decode: npu_recurrent_gated_delta_rule（先切出 decode 段）
  │    └─ 2.3 prefill     : chunk_gated_delta_rule  ← 本文重点
  │
  └─ 4. 合并输出: index_copy_ / cat
```

| 路径 | 触发条件 | 算法 | 代码位置 |
|---|---|---|---|
| speculative | `spec_sequence_masks is not None` | recurrent（带 `num_accepted_tokens` 变长） | `:540-559` |
| mixed decode | `split_non_spec == True` | recurrent（只有 1 token/seq） | `:564-583` |
| prefill | `attn_metadata.num_prefills > 0` | **chunkwise parallel** | `:588-650` |
| pure decode | `num_decodes > 0` 且无 prefill | recurrent | `:651-667` |

`split_non_spec` 的定义（`gdn.py:534-536`）：

```python
split_non_spec = (spec_sequence_masks is None
                  and attn_metadata.num_prefills > 0
                  and attn_metadata.num_decodes > 0)
```

即"同一个非 spec batch 里既有 prefill 又有 decode"。此时先把前 `num_decode_tokens` 个 token 切出来走 recurrent，prefill 段再走 chunk，最后 `torch.cat` 拼回（`:646-650`）。decode 段只有 1 个 token，串行代价可忽略。

---

## 3. 逐阶段代码 ↔ 公式对照

### 3.1 输入投影与 q/k/v 拆分

```python
# gdn.py:512-513
query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(mixed_qkv_non_spec)
```

`mixed_qkv` 是打包形式 `[(2·key_dim + value_dim)/tp]`，拆分后 reshape 成 `[1, T, H, D]`。GQA 交错布局时走融合拆分内核 `fused_qkvzba_split_reshape_cat`（`:302-309`），一次完成 qkvzba 四路拆分 + reshape，省掉 4 次独立 view。

| 公式量 | 代码 | 形状 |
|---|---|---|
| $q$ | `query_non_spec` | `[1, T, H_g, D_k]` |
| $k$ | `key_non_spec` | `[1, T, H_g, D_k]` |
| $v$ | `value_non_spec` | `[1, T, H, D_v]` |
| $z$ | `z` | `[T, H, D_v]` |

**L2 归一化**：prefill 路径通过 `use_qk_l2norm_in_kernel=True` 传入，在 `chunk.py:238-240` 内部完成：

$$
\hat q_t = \frac{q_t}{\lVert q_t\rVert_2},\qquad \hat k_t = \frac{k_t}{\lVert k_t\rVert_2}
$$

对应 `l2norm_fwd(q)` / `l2norm_fwd(k)`。融合算子路径则在外层显式调用（`gdn.py:211-212`）。

### 3.2 Causal Conv1d

`npu_causal_conv1d_custom` 完成（`gdn.py:406`、`:452`、`:474`、`:494`）。数学形式：

$$
x_t = \sum_{w=0}^{W-1} u_{t-w}\cdot c_w + b,\qquad W = \text{conv-kernel-size}
$$

权重预打包（把 `[D,1,W]` 转成内核要的 `[W,D]`）见 `gdn.py:62-80`：

```python
packed_weight = (source_weight.view(source_weight.size(0), source_weight.size(2))
                                .transpose(0, 1)          # [W, D]
                                .to(device=..., dtype=...).contiguous())
```

`initialize_packed_conv_weight`（`:83-114`）用 `@wraps` 劫持 `quant_method.process_weights_after_loading`，保证每次权重加载后自动重打包。`_get_base_conv1d`（`:50-58`）处理 LoRA 场景：LoRA 会把 `conv1d` 换成 wrapper 但保留 `base_layer`，打包参数必须挂在 base 上，否则 layerwise reload 时源权重与派生权重会脱节。

### 3.3 Prefill 各阶段的公式与代码对照

驱动代码：`vllm_ascend/ops/triton/fla/chunk.py:30-218`（`chunk_gated_delta_rule_fwd`）。

#### 阶段 ① 块内累积门

```python
# chunk.py:58-63
g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens, block_indices=block_indices_cumsum)
```

$$
\tilde g_t = \sum_{i=nC}^{t} g_i \quad \text{（仅在块内累加，不跨块）}
$$

**这是整个并行化的前提**：cumsum 只在块内做，块间衰减交由 $H$ 递推中的 $\exp(\tilde g_{last})$ 因子承担。

实现细节（`cumsum.py:91-95`）：`OPTIM_BLOCK_SIZE = 2**18 // (H * chunk_size)`，一次处理多个 chunk。工作集常量 `_GDN_CUMSUM_WORKING_SET = 2**18`。

#### 阶段 ② 构造 $L$ 的原始形式

```python
# chunk.py:65-72
A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g_cumsum=g, cu_seqlens=cu_seqlens,
                             chunk_indices=chunk_indices_chunk64, output_dtype=torch.float32)
```

内核（`chunk_scaled_dot_kkt.py:70-87`）：

```python
b_A = tl.zeros([BT, BT], dtype=tl.float32)
for i_k in range(tl.cdiv(K, BK)):
    b_k = tl.load(p_k, ...)
    b_A += tl.dot(b_k, tl.trans(b_k))        # KKᵀ
if USE_G:
    b_g_diff = b_g[:, None] - b_g[None, :]
    b_A *= safe_exp(b_g_diff)                # ⊙ D
b_A *= b_beta[:, None]                       # ⊙ diag(β)
b_A = tl.where(o_t_fp32[:, None] > o_t_fp32[None, :], b_A, 0)   # 严格下三角
```

$$
M_{ij} = \begin{cases}
\beta_i\,(k_i\cdot k_j)\,\exp(\tilde g_i - \tilde g_j), & i > j\\
0, & i \le j
\end{cases}
$$

#### 阶段 ③ 求 $(I-L)^{-1}$

```python
# chunk.py:73-79
A = solve_tril(A=A, cu_seqlens=cu_seqlens,
               chunk_indices_large_block=chunk_indices_large_block,
               chunk_indices_bt=chunk_indices_chunk64, output_dtype=k.dtype)
```

`solve_tril` 内部先取负再补对角线单位元（`solve_tril.py:101-119`）：

```python
is_lower = (rows > cols).to(b_A.dtype)
b_A = -b_A * is_lower                       # 变成 -L
...
on_diagonal = rows == cols
b_A = tl.where(on_diagonal, b_A + 1.0, b_A) # 变成 I - L
```

$$
(I - L)^{-1} = \Big(I - \mathrm{tril}_{i>j}\big(\mathrm{diag}(\beta)KK^{\top}\odot D\big)\Big)^{-1}
$$

即 §1.3 中的 $A$。

#### 阶段 ④ 重算 $W$ / $U$

```python
# chunk.py:80-88
w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g_cumsum=g,
                         cu_seqlens=cu_seqlens, chunk_indices=chunk_indices_chunk64)
```

内核（`wy_fast.py:66-95`）：

```python
b_g = tl.exp(tl.load(ptr_g, ...)).to(tl.float32)     # e^{g̃}
b_vb = b_v * b_beta[:, None]
b_u = tl.dot(b_A, b_vb, allow_tf32=False)            # U = A(βV)
b_kb = b_k * b_beta[:, None] * b_g[:, None]
b_w = tl.dot(b_A, b_kb)                              # W = A(β⊙e^{g̃}⊙K)
```

$$
W = A\Big(\beta\odot e^{\tilde g}\odot K\Big),\qquad U = A\Big(\beta\odot V\Big)
$$

#### 阶段 ⑤ 状态扫描

```python
# chunk.py:115-129
h, v_new, final_state = torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
    k_ascendc, w_ascendc, u_ascendc, g=g_ascendc, gk=None,
    initial_state=initial_state_kern, output_final_state=True,
    chunk_size=64, save_new_value=True,
    cu_seqlens=cu_seqlens_kern, chunk_indices=chunk_indices_chunk64_host,
    use_exp2=False, transpose_state_layout=False)
```

Triton 参考实现 `chunk_delta_h.py:97-166` 的循环体逐行对应：

| 公式 | 代码 |
|---|---|
| $V' \leftarrow U - W H_n$ | `b_v_new1 = b_v1 - tl.dot(b_w, b_h1_bv1)` |
| $V' \leftarrow V'\odot e^{\tilde g_{last}-\tilde g}$ | `b_g = safe_exp(b_g_last - b_g); b_v_new1 = b_v_new1 * b_g[:, None]` |
| $H \leftarrow H\cdot e^{\tilde g_{last}}$ | `b_h1_bv1 = b_h1_bv1 * b_g_last` |
| $H \leftarrow H + K^{\top}V'$ | `b_h1_bv1 += tl.dot(b_k, b_v_new1)` |

$$
V'_n = \big(U_n - W_nH_n\big)\odot\exp\big(\tilde g_{last,n}-\tilde g^{(n)}\big),\qquad
H_{n+1} = e^{\tilde g_{last,n}}H_n + K_n^{\top}V'_n
$$

其中 `b_g_last = tl.load(g + ... + last_idx)` 取的是块内**最后一个** token 的 $\tilde g$，`b_g` 是该 token 自身的 $\tilde g_t$。

#### 阶段 ⑥ 计算输出

```python
# chunk.py:197-209
o_ascendc = torch.ops._C_ascend.chunk_fwd_o(q_ascendc, k_ascendc, v_new, h, scale,
                                           g=g_ascendc, g_gamma=None,
                                           cu_seqlens=cu_seqlens_host,
                                           chunk_indices=chunk_indices_chunk64_host,
                                           chunk_size=64, transpose_state_layout=False)
```

内核（`chunk_o.py:70-109`）逐行对应：

| 公式 | 代码 |
|---|---|
| $QH_n$ | `b_o += tl.dot(b_q, b_h)` |
| $QK^{\top}$ | `b_A += tl.dot(b_q, b_k)` |
| $\odot e^{\tilde g}$ | `b_o = b_o * tl.exp(b_g)[:, None]` |
| $\odot D$ | `b_A = b_A * safe_exp(b_g[:, None] - b_g[None, :])` |
| 因果掩码 $i \ge j$ | `m_A = o_i[:, None] >= o_i[None, :]; b_A = tl.where(m_A, b_A, 0)` |
| $\times$ scale | `b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale` |

$$
O_n = \text{scale}\cdot\left[\big(Q_nH_n\big)\odot e^{\tilde g^{(n)}} + \mathrm{tril}_{i\ge j}\Big(\big(Q_nK_n^{\top}\big)\odot D\Big)V'_n\right]
$$

注意掩码是 **inclusive**（`>=`），因为对角线元素 $q_i\cdot k_i$ 也是合法的（ $D_{ii}=\exp(0)=1$）。这与阶段 ② 的严格下三角（`>`）不同——阶段 ② 构造的是 $L$，对角必须为空。

### 3.4 Decode / Spec 路径

```python
# gdn.py:657-667（pure decode）
core_attn_out_non_spec = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
    query=query_non_spec.squeeze(0),    # [T, H, Dk]
    key=key_non_spec.squeeze(0),        # [T, H, Dk]
    value=value_non_spec.squeeze(0),    # [T, H, Dv]
    g=g_non_spec.squeeze(0),            # [T, H]
    beta=beta_non_spec.squeeze(0),      # [T, H]
    state=ssm_state,                    # [N, H, Dv, Dk]
    scale=key_non_spec.shape[-1] ** -0.5,
    actual_seq_lengths=actual_seq_lengths,
    ssm_state_indices=non_spec_state_indices_tensor,
).unsqueeze(0)
```

这里是 §1.1 的原样实现，串行推进。spec 路径（`:548-559`）多一个 `num_accepted_tokens` 参数以支持投机解码的变长接受。

`actual_seq_lengths` 由 `_build_actual_seq_lengths`（`gdn_attn_builder.py:160-174`）从 `query_start_loc` 差分得到。

---

## 4. Prefill 并行化设计原理

### 4.1 并行维度分解

| 维度 | 规模 | 是否独立 | 覆盖阶段 |
|---|---|---|---|
| **chunk 轴** | $NT = T/C$ | 阶段 ①~④、⑥ 独立；⑤ 串行 | 全部 |
| **head 轴** | $H$（或 $H_g$） | 完全独立 | 全部 |
| **序列轴** | $N$ | 完全独立 | 全部 |
| **V 维** | $D_v$ | 可切分（state 对 V 逐通道独立） | ⑤ ⑥ |
| **K 维** | $D_k$ | 仅累加方向，非并行轴 | ①②⑥ |
| **跨卡（PCP）** | `world_size` | 需状态修正 | ⑤ |

**核心收益**：串行深度从 $O(T)$ 降到 $O(T/C)$（ $C=64$，降 64 倍），剩余计算全部变成可在 tensor core 上展开的矩阵乘。

### 4.2 各阶段的并行策略与代码证据

#### 阶段 ① 块内 cumsum：大块摊薄启动开销

```python
# cumsum.py:91-95
OPTIM_BLOCK_SIZE = triton.next_power_of_2((2**18) // (H * chunk_size))
if cu_seqlens is not None and block_indices is None:
    block_indices = prepare_chunk_indices(cu_seqlens, chunk_size=OPTIM_BLOCK_SIZE)
num_blocks = len(block_indices) if cu_seqlens is not None else triton.cdiv(T, OPTIM_BLOCK_SIZE)
grid = (num_blocks, B)
```

`OPTIM_BLOCK_SIZE` 让一次处理多个 chunk，使单次工作集约 $2^{18}$ 个元素（与 `_GDN_CUMSUM_WORKING_SET` 对应），把 kernel 启动次数摊薄。

**"局部 cumsum"的实现关键**在 `cumsum.py:65-67`：

```python
b_s = tl.reshape(b_s, (N_CHUNKS, CHUNK_SIZE, H))
b_s = tl.trans(b_s, (1, 0, 2))                      # → (CHUNK_SIZE, N_CHUNKS, H)
b_o = tl.cumsum(b_s, axis=0, reverse=REVERSE)       # axis=0 是块内 token 轴
b_o = tl.trans(b_o, (1, 0, 2))
b_o = tl.reshape(b_o, (BLOCK_T, H))
```

`axis=0` 经过转置后对应**块内** token 轴，累加不会跨过 chunk 边界。若误在序列轴上 cumsum，§1.3 的块间解耦假设就不成立了。

#### 阶段 ② 持久化内核 + 任务均分

```python
# chunk_scaled_dot_kkt.py:131-154
num_core = get_aicore_num()
bh_step = B * H
task_num = NT * bh_step
A = DeviceOperator.chunk_scaled_dot_kkt_fwd(
    num_core=num_core, bh_step=bh_step, task_num=task_num, k=k,
    beta=torch.permute(beta, (2, 0, 1)).contiguous(),      # → [H, B, T]
    g_cumsum=torch.permute(g_cumsum, (2, 0, 1)).contiguous(),
    A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    T=T, B=B, H=H, Hg=Hg, K=K, BT=BT, BK=128)
```

注意 grid 是 `(num_core,)` 而非 `(NT, B*H)`：

```python
# chunk_scaled_dot_kkt.py:26,48-52
@triton.jit(do_not_specialize=["T", "B", "bh_step", "task_num", "num_core"])
def chunk_scaled_dot_kkt_fwd_kernel(...):
    core_id = tl.program_id(0)
    for task_id in tl.range(core_id, task_num, num_core):
        i_t_i = task_id // bh_step
        i_bh  = task_id %  bh_step
```

这是**持久化 kernel（persistent kernel）**：固定开 `num_core` 个 program，每个核用 grid-stride 循环领取任务。

- **收益 1**：消除调度开销。 $NT\cdot B\cdot H$ 在长序列下可达数千，直接开数千 program 会让调度成为瓶颈；固定为核数后开销变常数。
- **收益 2**：`multibuffer=True` + `num_stages=3`（`device_op.py:703-704`），UB↔GM 搬运与计算重叠。

配套优化：`beta`/`g_cumsum` 显式 `permute(2,0,1)` 成 `[H,B,T]`（head-major），同 head 相邻 chunk 内存连续；`BK=128` 让 $D_k=128$ 时一次 dot 走完，消除 K 维循环。

#### 阶段 ③ 三级分块求逆

`LARGE_BLOCK_T = 608 * 2 = 1216`（`solve_tril.py:359`），配合：

```python
NTASKS: tl.constexpr = 2
N_BLOCKS: tl.constexpr = LARGE_BLOCK_T // 16 // NTASKS   # = 38
```

三个设计意图：

1. **grid 缩小 19 倍**：按 `BT=64` 算 grid 是 `(T/64, B*H)`，按 1216 算是 `(T/1216, B*H)`。 $T=32\text{k}$ 时 grid 从 512 降到 27。
2. **块内批量化**：`b_A` 是 `(38, 16, 16)` 三维张量，38 个块同时驻留 UB。
3. **前向代入向量化**（`solve_tril.py:105-116`）：

```python
local_ori_A = tl.trans(b_A, (1, 0, 2))                    # (16, N_BLOCKS, 16)
local_ori_A = tl.reshape(local_ori_A, (16, 16 * N_BLOCKS))
for i in range(1, 16):
    nblks_vec16 = -extract_slice(local_ori_A, (i, 0), (1, 16*N_BLOCKS),
                                 (EXTRACT_SLICE_STRIDE_1, 1))
    b_a = tl.reshape(nblks_vec16, (N_BLOCKS, 16))
    dot_tmp = tl.trans(b_a[:, :, None] * b_A, (1, 0, 2))
    dot_product = tl.sum(dot_tmp, 0)
    b_a = b_a + dot_product                                # 批量 b_a += b_a @ b_A
    b_A = insert_slice(ful=b_A, sub=b_a[:, None, :],
                       offsets=[0, i, 0], sizes=[N_BLOCKS, 1, 16], strides=[1, 1, 1])
```

$16\times16$ 块内求逆本身是前向代入（串行 16 步），但**38 个块的同一行被一次 strided load 取出来，在 `(N_BLOCKS, 16)` 形状上做批量矩阵运算**，把 38 次独立标量代入折叠成 16 次批量运算。

`is_lower` 用外积构造掩码再乘，而不是生成循环——注释明确写了原因：

```python
# solve_tril.py:97-102
# Convert mask into matrix multiplication to avoid for loops ub oom
tmp = tl.arange(0, 16).to(tl.float32)
rows = tmp[:, None]; cols = tmp[None, :]
is_lower = (rows > cols).to(b_A.dtype)
b_A = -b_A * is_lower
```

**第二 / 三级：分块合并**。用分块三角求逆公式：

$$
\begin{pmatrix} A_{11} & 0 \\ A_{21} & A_{22}\end{pmatrix}^{-1}
= \begin{pmatrix} A_{11}^{-1} & 0 \\ -A_{22}^{-1}A_{21}A_{11}^{-1} & A_{22}^{-1}\end{pmatrix}
$$

- 16→32（`solve_tril.py:139-198`，`merge_16x16_to_32x32_inverse_kernel`）
- 32→64（`solve_tril.py:203-328`，`merge_16x16_to_64x64_inverse_kernel`，双层展开 + `insert_slice` 拼装）

在 $BT=64$ 粒度上 grid 恢复为 `(NT, B*H)`，`num_warps=4, num_stages=3`。

整体并行结构：**超粗块（1216）内批量求 16×16 → 逐级合并到 64×64**，把 $O(64^3)$ 的串行深度压成 $\log$ 级。

#### 阶段 ④ 换取 compute/memory 比

```python
# wy_fast.py:42-44, 122
i_t_o = tl.program_id(0)
for i_bh in range(H):          # 一个 program 处理一个 chunk 的全部 head
    ...
recompute_w_u_fwd_kernel[(NT, B)](...)
```

grid 只有 `(NT, B)`，缺一个 head 维度。目的是把 `A / g / beta` 的加载摊到 $H$ 个 head 上复用，刷高算术强度。代价见 §6 待优化点 #1。

#### 阶段 ⑤ 唯一的串行段，但被两轴并行

并行轴 1：**序列 × head**。Triton 参考实现用 `i_nh = tl.program_id(1)`（`chunk_delta_h.py:54`）展开为 $N\times H$。

并行轴 2：**V 维二分**。`chunk_delta_h.py:74-79`：

```python
b_h1_bv1 = tl.zeros([128, 64], dtype=tl.float32)
b_h1_bv2 = tl.zeros([128, 64], dtype=tl.float32)
v_start1 = 0
v_start2 = 64
```

状态矩阵 $[D_k{=}128,\ D_v{=}128]$ 沿 V 切成两个 $[128,64]$ 子块（`b_h1_bv1` / `b_h1_bv2`），交给不同核并行推进。**因为 state 更新对 V 的每个通道独立，这个切分零代价。**

主机侧 tiling：`blockDim = ctx_.aicCoreNum`（`chunk_gated_delta_rule_fwd_h_tiling_processor.h:90`），占满所有 AI Core。

一次扫描同时产出两样东西：`h`（每 chunk 起始状态，供阶段 ⑥）和 `v_new`（`SAVE_NEW_VALUE=True`），避免额外一遍 matmul。

#### 阶段 ⑥ 回到全并行

```python
# chunk_o.py:138-139
def grid(meta):
    return (triton.cdiv(V, meta["BV"]), N * H)
```

grid 为 `(V/BV, N*H)`， $BV=128$ 时第一维退化为 1。此阶段无任何跨 chunk 依赖。

#### 融合算子路径

`gdn.py:180-235`（`_chunk_gated_delta_rule_fused`）用 `torch_npu.npu_chunk_gated_delta_rule` 一次覆盖阶段 ②~⑥。

**收益是 HBM 往返**。中间张量尺度（ $T=32\text{k}$, $H=16$, $d=128$, bf16）：

| 张量 | 形状 | 大小 |
|---|---|---|
| `k/w/u/q/v` | `[B,T,H,128]` | ~128 MB / 个 |
| `A` | `[B,T,H,64]` fp32 | ~256 MB |
| `Ad`/`Ai` | `[B,T,H,16/64]` | 32~256 MB |

五步之间反复读写这些量，HBM 带宽成为天花板。融合后中间结果留在片上，只剩 `q/k/v/g/beta` 读 + `o/final_state` 写。

**三条启用条件**（`gdn.py:607`）：

```python
use_fused_chunk = AscendGatedDeltaNetAttention._probe_fused_chunk() and get_pcp_group().world_size == 1
```

1. `_probe_fused_chunk()`（`:132-178`）：接口存在 **且** 最小 smoke call 真能跑通。注释明确说明 **A5 平台的实现在官方 CANN 包里尚未就绪，调了会报错**，因此必须实测而非仅 `hasattr`。结果缓存在类变量 `_fused_chunk_available`，只第一层付一次探测成本。
2. **PCP 下禁用**：融合算子没有暴露 `h_update`，跨卡修正拿不到 $\Phi_i$，必须回退 Triton。
3. **layout / dtype 抹平**（`:210-234`）：

| 项目 | 融合算子要求 | Triton 路径要求 |
|---|---|---|
| q/k 布局 | TND `[T, N_k, D_k]` | `[B,H,T,D_k]` |
| `initial_state` | bf16，**不需要转置** | 需 `[D_k,D_v]`（要 `transpose(-1,-2)`） |
| g 的 cumsum | 不算，传原始 g | 内部算 |

**顺带的收益**：`ssm_state` 物理布局是 `[N,H,D_v,D_k]`，而 chunk kernel 要 `[D_k,D_v]`。Triton 路径必须进 `transpose(-1,-2)`（`gdn.py:628`）、出 `transpose(-1,-2)`（`:643`）。以 $N=32,H=16,128\times128$, bf16 估，单个状态张量 16.7 MB，一来一回 33 MB，乘 30+ 层即 GB 级纯搬运。融合算子的 `[N,Nv,Dv,Dk]` 与 `ssm_state` 对齐，这笔开销归零。

### 4.3 PCP：跨卡状态的精确修正

长序列切到 `world_size` 张卡上并行。朴素做法是串行传递状态，等于白切。`chunk.py:138-195` 的解法：

**步骤 1**：每张卡用本地 `initial_state` 独立跑一遍 `fwd_h`，得到（含误差的）`final_state` = $F_i$。

**步骤 2**：同时用 `chunk_gated_delta_rule_fwd_hupdate`（`chunk_delta_hupdate.py`）算出转移矩阵 `h_update` = $\Phi_i$。

**步骤 3**：`all_gather`（`chunk.py:155-160`）：

```python
all_final_state = get_pcp_group().all_gather(final_state.unsqueeze(0), 0)
final_h_update = h_update[:, final_chunk_indices, :, :, :]
all_final_h_update = get_pcp_group().all_gather(final_h_update, 0)
```

**步骤 4**：串行修正循环（`chunk.py:162-171`）：

```python
updated_state = final_state.new_empty(get_pcp_group().world_size, *final_state.shape)
updated_state[0, ...] = all_final_state[0]
for i in range(1, get_pcp_group().world_size):
    # correct_i = all_final_state[i] + Φ_i · (correct_{i-1} - s0) = Φ_i · correct_{i-1} + p_i
    updated_final_state = all_final_state[i] + torch.matmul(
        all_final_h_update[i, ...], updated_state[i - 1, ...] - initial_state)
    updated_state[i, ...] = updated_final_state
final_state = updated_state[-1, ...]
```

$$
S_i = F_i + \Phi_i\big(S_{i-1} - S_0\big)
$$

**为什么精确**：由 §1.4， $H_{n+1} = \Phi_n H_n + P_n$ 对 $H_n$ 是仿射的。以错误初值 $S_0$ 算出的 $F_i$ 与真实值只差一个线性传递项，上式**严格成立而非近似**。

**步骤 5**：`rank > 0` 用修正后状态重跑本地 `fwd_h`（`chunk.py:178-195`）：

```python
if get_pcp_group().rank_in_group > 0:
    rerun_initial_state = initial_state.clone()
    prefill_slice = slice(actual_num_decodes, final_state.shape[0])
    rerun_initial_state[prefill_slice] = updated_h_state[prefill_slice]
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(..., initial_state=rerun_initial_state, ...)
```

**代价对比**：

| 方案 | 串行深度 | 额外开销 |
|---|---|---|
| 朴素串行传递 | $O(T)$ | 无 |
| PCP 修正 | $O(1)$ 并行 + $O(\text{world-size})$ 小串行 | 1 次 `all_gather` + 1 次重跑 |

`chunk_gated_delta_rule_fwd_hupdate` 仅在 `world_size > 1` 时调用（`chunk.py:138`），单卡零开销。

**同一逻辑在 conv1d 上**（`gdn.py:429-470`）：`extract_last_width` 取每段末尾 `width-1` 个 token，`all_gather` 后把上游尾巴写入本卡 conv state，解决 causal 窗口跨卡边界：

```python
last_width_prefill_x = extract_last_width(mixed_qkv_non_spec_T,
                                          non_spec_query_start_loc[prefill_seq_offset:], state_len)
all_last_width_prefill_x = get_pcp_group().all_gather(last_width_prefill_x.unsqueeze(0).contiguous(), 0)
if pcp_rank > 0 and prefill_cache_indices.shape[0] > 0:
    self_kv_cache[0][prefill_cache_indices, :state_len, :] = \
        all_last_width_prefill_x[pcp_rank - 1, ...].transpose(-1, -2)
```

注意 `pcp_rank > 0` 时 builder 会强制把 `initial_state_mode` 置 True（`gdn_attn_builder.py:423-426`），否则空段压缩会把人为填充的 state 清掉：

```python
if pcp_rank > 0 and attn_metadata.num_prefills > 0:
    prefill_seq_offset = max(0, prefill_num_rows - attn_metadata.num_prefills)
    initial_state_mode = initial_state_mode.clone()
    initial_state_mode[prefill_seq_offset:] = True
```

### 4.4 元数据预计算：把 D2H 同步从 30+ 次压到 1 次

`gdn_attn_builder.py:113-125` 定义了预计算容器：

```python
@dataclass
class GDNChunkedPrefillMetadata:
    cu_seqlens_host: tuple[int, ...]
    chunk_indices_chunk64_host: tuple[int, ...]
    chunk_indices_chunk64: torch.Tensor
    chunk_offsets_chunk64: torch.Tensor
    update_chunk_offsets_chunk64: torch.Tensor
    final_chunk_indices_chunk64: torch.Tensor
    chunk_indices_large_block: torch.Tensor
    block_indices_cumsum: torch.Tensor
    num_decodes: int = 0
    cu_seqlens_kern: tuple[int, ...] | None = None
    keep_meta: torch.Tensor | None = None
```

关键在**同时缓存 host 端 tuple**。`.tolist()` 会触发 device→host 同步；若放在每层算子里做，30 层就是 30 次同步，每次都要等 NPU 流水线排空。这里在 metadata build 阶段一次算好，全层复用。

预计算内容（`gdn_attn_builder.py:222-230`）：

```python
chunk_indices_chunk64 = prepare_chunk_indices(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
chunk_offsets_chunk64 = prepare_chunk_offsets(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
update_chunk_offsets_chunk64 = prepare_update_chunk_offsets(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
final_chunk_indices_chunk64 = prepare_final_chunk_indices(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
chunk_indices_large_block = prepare_chunk_indices(cu_seqlens_cpu, _GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE)
block_indices_cumsum = prepare_chunk_indices(cu_seqlens_cpu, cumsum_chunk_size)
```

`cumsum_chunk_size` 由 UB 反推（`:219-220`）：

```python
cumsum_chunks = max(1, _GDN_CUMSUM_WORKING_SET // (gdn_num_heads * _GDN_CHUNK_SIZE))
cumsum_chunk_size = 1 if cumsum_chunks <= 1 else 1 << (cumsum_chunks - 1).bit_length()
```

使 `chunk_local_cumsum` 单次工作集稳定卡在 `2**18` 个元素。

**跨文件常量耦合**需要警惕（`gdn_attn_builder.py:43-46`）：

```python
_GDN_CHUNK_SIZE = 64
# Keep this aligned with solve_tril.LARGE_BLOCK_T in ops/triton/fla/solve_tril.py.
_GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE = 608 * 2
_GDN_CUMSUM_WORKING_SET = 2**18
```

这三个常量必须与 `solve_tril.py:359` 的 `LARGE_BLOCK_T` 和 `cumsum.py:92` 的 `OPTIM_BLOCK_SIZE` 保持一致，改一处必须改另一处。**代码里只有注释提醒，没有断言保护。**

---

## 5. 数值与工程细节

### 5.1 `safe_exp`：不只是优化，是正确性保障

```python
# fla/utils.py:72-74
@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))
```

在阶段 ②（`chunk_scaled_dot_kkt.py:82`）和阶段 ⑥（`chunk_o.py:96`）都用到。

**为什么必需**： $D_{ij}=\exp(\tilde g_i - \tilde g_j)$ 在 $i<j$（上三角）时指数为正，直接 `exp` 会溢出成 `inf`。虽然紧接着有 `tl.where(rows > cols, b_A, 0)` 清零，**但溢出发生在清零之前**——`inf × 0 = nan`。`safe_exp` 把非法区域先压成 $-\infty$（`exp(-inf)=0`），从源头消除 `nan`。

### 5.2 空段压缩

`_compact_empty_segments`（`gdn_attn_builder.py:177-201`）剔除长度为 0 的序列段：

```python
cu = torch.tensor(cu_seqlens_host, dtype=torch.int64)
keep = (cu[1:] - cu[:-1]) > 0
if bool(keep.all()):
    return cu_seqlens_host, initial_state, None
cu_kern = torch.cat([cu[:1], cu[1:][keep]]).tolist()
keep = keep.to(device) if device is not None else keep
st_kern = initial_state[keep] if initial_state is not None else None
return cu_kern, st_kern, keep
```

**目的**：让 AscendC 内核的索引严格对齐，避免空循环。**投机解码的 padding 场景下空段非常普遍**（graph padding 请求、无 draft token 的占位），不做这一步会白跑一大截。

压缩后需要把结果 scatter 回原布局（`chunk.py:130-136`）：

```python
if keep_meta is not None:
    _fs_full = initial_state.clone()
    _fs_full[keep_meta] = final_state
    final_state = _fs_full
```

即空段的 `final_state` 保持其 `initial_state`。

配套的请求侧裁剪：`_remove_spec_graph_padding_queries`（`gdn_attn_builder.py:55-84`）把 FIA 图的 padding 请求 query 长度压成 0；`_treat_single_token_prefills_with_state_as_decodes`（`:87-110`）把已有状态的单 token prompt chunk 归入 decode 路径。

### 5.3 dtype 与 layout 约定

| 张量 | dtype | layout | 说明 |
|---|---|---|---|
| `g` | fp32 | `[1, T, H]` | 恒 $\le 0$ |
| `beta` | bf16 | `[1, T, H]` | sigmoid 输出 |
| `g_cumsum` | fp32 | `[B, T, H]` → 内核内 `[B,H,T]` | 块内 cumsum |
| `A` | fp32 | `[B, T, H, 64]` | 求逆输入 |
| `Ai` | `output_dtype` (bf16) | `[B, T, H, 64]` | 求逆输出 |
| `ssm_state` | fp32 或 bf16 | `[N, H, D_v, D_k]` | 递归路径保留 fp32 |
| `h` (AscendC) | — | `[B, H, NT, D_k, D_v]` | `transpose_state_layout=False` |

**`initial_state` 的清零时机很关键**（`gdn.py:609-613`）：

```python
# The fused op's state layout [N, Nv, Dv, Dk] matches ssm_state
# directly, so no transpose is needed. Advanced indexing already
# returns a copy, safe to clear in place.
initial_state = ssm_state[prefill_state_indices]
clear_ssm_states(initial_state, prefill_has_initial_state)
```

`ssm_state[prefill_state_indices]` 是高级索引，返回**副本**，所以可以安全地就地清零。若换成 `index_select` 语义或 view 就会污染真实状态。

### 5.4 上游接口兼容与图模式约束

**`@torch.compiler.disable`**（`chunk.py:258`）加在 `chunk_gated_delta_rule` 上，把整条 Triton 流水线挡在 Dynamo 追踪之外。

**`_forward_core` 整体被包进 custom op** `torch.ops.vllm.qwen_gdn_attention_core`。原因见 `gdn.py:363-367` 的注释：

```python
# Layerwise KV pool hooks must stay inside the custom op body: the
# forward() caller region is traced by Dynamo in fullgraph mode, and
# these side effects (thread locks, connector waits) would break the
# graph. Waiting here still orders the deferred mamba state copy and
# the layer load before conv/attention kernels touch mamba state.
wait_for_kv_layer_from_connector(self.prefix)
record_attention_compute_start()
```

**`_warmup_prefill_kernels` / `_warmup_prefill_kernels_v0202` 被重写成 no-op**（`gdn.py:255-259`）。上游的 warmup 针对 CUDA Triton 缓存预热，在 Ascend 上无收益且多几次 kernel 启动。

---

## 6. 待优化点

> 以下均为**分析发现的优化空间**，非缺陷。每条给出定位、根因、影响面与验证建议。

### #1 阶段 ④ `recompute_w_u_fwd_kernel` grid 欠并行

| 项 | 内容 |
|---|---|
| **位置** | `vllm_ascend/ops/triton/fla/wy_fast.py:42-44, 122` |
| **现象** | grid 只有 `(NT, B)`，缺少 head 维度 |
| **根因** | `for i_bh in range(H)` 把一个 chunk 的全部 head 放在同一 program 内串行处理，以复用 `A/g/beta` 的加载 |
| **影响** | 短 prefill（ $T \le 2048$）时 $NT \approx 32$，对 40 核的 910B 明显欠并行；长序列（ $T=32\text{k}$ 时 $NT=512$）无此问题 |
| **验证** | 用 $T \in \lbrace 512, 2048, 8192, 32768 \rbrace$ 扫一遍该 kernel 的单独耗时占比，确认短序列下是否真是瓶颈 |
| **可能改法** | 改造成与阶段 ① 相同的持久化 + task 均分（`NT*B*H` 个任务摊到 `num_core`）。代价是丢掉 head 间数据复用，需实测权衡 |

### #2 `LARGE_BLOCK_T = 1216` 硬编码 UB 拟合

| 项 | 内容 |
|---|---|
| **位置** | `solve_tril.py:359`、`gdn_attn_builder.py:45`、`triton_utils.py:112-122` |
| **现象** | `N_BLOCKS = 38` 使 `b_A`（38×16×16×4B ≈ 39 KB）加 `local_ori_A` 等刚好卡进 UB |
| **根因** | 常量按某一代硬件的 UB 容量手调得出，但 910B / A3 / A5 的 UB 大小并不一致 |
| **影响** | 换平台可能 UB 溢出（编译失败/性能劣化）或 UB 未用满（浪费并行度） |
| **验证** | 在目标平台读 `get_ub_size_bytes()`，反算 1216 是否仍合适 |
| **可能改法** | 由 `get_ub_size_bytes()` 动态推导 `LARGE_BLOCK_T`；同时给 `_GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE` 加一个启动期断言，防止两处常量失配 |

### #3 PCP 修正的 host 侧 Python 循环

| 项 | 内容 |
|---|---|
| **位置** | `chunk.py:164-169` |
| **现象** | `for i in range(1, world_size)` 是 host 侧循环，每轮一次 matmul + 一次 buffer 写 |
| **根因** | 修正链条天然串行，直接照写 |
| **影响** | `world_size=8` 时 7 次串行下发，会切断图模式下的算子重叠；单卡/PCP=2 时影响可忽略 |
| **验证** | 对比 PCP=1/2/4/8 时的 prefill 端到端耗时，观察是否线性劣化 |
| **可能改法** | 用一个 device 侧 scan kernel 替掉；但收益取决于实际 PCP 规模，优先级不高 |

### #4 `_probe_fused_chunk` 懒加载引入首 token 抖动

| 项 | 内容 |
|---|---|
| **位置** | `gdn.py:132-178`，调用点 `:607` |
| **现象** | smoke call 在第一次 prefill 时才执行，含 `torch.npu.synchronize()` |
| **根因** | 探测结果缓存在类变量上，天然是懒加载 |
| **影响** | 一次性开销，但落在首 token 延迟的关键路径上 |
| **验证** | profile 首个 prefill 请求的耗时，看探测占比 |
| **可能改法** | 挪到模型加载阶段（如 `process_weights_after_loading` 的 hook 时机）预探测；注意此时 AI Core 可用的前提需要确认 |

### #5 A5 平台融合算子缺失

| 项 | 内容 |
|---|---|
| **位置** | `gdn.py:148-150` 的 TODO |
| **现象** | `# TODO(2026/8/6): The A5-specific implementation is not available in the official release.` 探测必然失败，A5 只能走 Triton 五段式 |
| **影响** | §4.2 中融合算子省掉的中间张量 HBM 往返（百 MB 量级 × 30+ 层）在 A5 上一点都省不掉。若 A5 是主要部署目标，影响不小 |
| **验证** | 在 A5 上量化 Triton 路径 vs（A3 上的）融合路径的 prefill 吞吐差 |
| **可能改法** | 等新 CANN 包；期间可考虑在 A5 上用 AscendC 自实现阶段 ②③ 的融合 |

### #6 跨文件魔数缺乏断言保护

| 项 | 内容 |
|---|---|
| **位置** | `gdn_attn_builder.py:43-46` ↔ `solve_tril.py:359` ↔ `cumsum.py:92` |
| **现象** | `_GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE = 608*2`、`_GDN_CUMSUM_WORKING_SET = 2**18` 与另两处常量必须一致，但只有注释提醒 |
| **影响** | 任一处被改动而忘记同步，会静默产生错误分块（索引越界或结果错误），排查成本高 |
| **验证** | — |
| **可能改法** | 抽到一个共享常量模块，或在 `_build_non_spec_chunked_prefill_metadata` 入口加 `assert` |

### #7 `chunk_o` 的 BV 分块在 $D_v=128$ 时退化

| 项 | 内容 |
|---|---|
| **位置** | `chunk_o.py:138-139, 158-159` |
| **现象** | `grid = (cdiv(V, BV), N*H)`，`BV=128` 且 $D_v=128$ 时第一维恒为 1 |
| **根因** | BV 写死 128 |
| **影响** | $D_v=128$ 时该维度不提供额外并行度（仅 $N\times H$，小 batch 短序列下偏低）。属观察项，非确定问题 |
| **验证** | 小 batch（ $N=1$）场景下测该 kernel 的 AI Core 利用率 |
| **可能改法** | $D_v=128$ 时把 BV 降到 64 换 2 倍并行度，需实测是否被 UB 容量收益抵消 |

---

## 7. 关键常量与文件索引

### 常量

| 常量 | 值 | 位置 |
|---|---|---|
| chunk 大小 | 64 | `chunk.py:49`、`gdn_attn_builder.py:43` |
| solve_tril 超粗块 | 1216 | `solve_tril.py:359`、`gdn_attn_builder.py:45` |
| cumsum 工作集 | $2^{18}$ | `gdn_attn_builder.py:46`、`cumsum.py:92` |
| softplus threshold | 20.0 | `fused_gdn_gating.py:65` |
| `BLK_HEADS` / `BLK_BATCHES` | 8 / 64 | `fused_gdn_gating.py:72,76` |
| `_PACKED_CONV_WEIGHT_NAME` | `ascend_conv1d_weight` | `gdn.py:47` |
| q/k l2norm eps | — | `fla/l2norm.py` |

### 文件清单

| 文件 | 职责 |
|---|---|
| `vllm_ascend/ops/gdn.py` | 主实现：投影、conv1d、gating、三路分流、输出合并 |
| `vllm_ascend/ops/gdn_attn_builder.py` | 元数据构建、chunk 索引预计算、空段压缩 |
| `vllm_ascend/ops/triton/fla/chunk.py` | prefill 五阶段驱动 + PCP 修正 |
| `vllm_ascend/ops/triton/fla/chunk_scaled_dot_kkt.py` | 阶段 ② 构造 $L$ |
| `vllm_ascend/ops/triton/fla/solve_tril.py` | 阶段 ③ 求 $(I-L)^{-1}$ |
| `vllm_ascend/ops/triton/fla/wy_fast.py` | 阶段 ④ 重算 $W/U$ |
| `vllm_ascend/ops/triton/fla/chunk_delta_h.py` | 阶段 ⑤ 状态扫描（Triton 参考） |
| `vllm_ascend/ops/triton/fla/chunk_delta_hupdate.py` | 阶段 ⑤ 转移矩阵 $\Phi$（PCP 专用） |
| `vllm_ascend/ops/triton/fla/chunk_o.py` | 阶段 ⑥ 输出 |
| `vllm_ascend/ops/triton/fla/cumsum.py` | 阶段 ① 块内 cumsum |
| `vllm_ascend/ops/triton/fla/utils.py` | `safe_exp`、`clear_ssm_states`、chunk 索引工具 |
| `vllm_ascend/ops/triton/fused_gdn_gating.py` | $g/\beta$ 生成 |
| `vllm_ascend/device/device_op.py` | 设备算子分发（300 系 / 310P 双实现） |
| `csrc/moe/chunk_gated_delta_rule_fwd_h/` | 阶段 ⑤ AscendC 内核 |
| `csrc/moe/chunk_fwd_o/` | 阶段 ⑥ AscendC 内核 |
| `csrc/attention/recurrent_gated_delta_rule/` | decode 路径 AscendC 内核 |

### 相关测试

| 测试 | 覆盖 |
|---|---|
| `tests/ut/ops/test_gdn_mixed_batch.py` | mixed prefill+decode 的 token 边界与 qkv 复用 |
| `tests/ut/ops/test_gdn_layerwise_kv.py` | layerwise KV pool 交互 |
| `tests/e2e/pull_request/one_card/aclgraph/test_aclgraph_accuracy.py` | 图模式精度一致性 |

---

## 8. 一句话总结

GDN 的 prefill 优化本质是把 §1.1 的 $O(T)$ 串行递推改写为 §1.3 的 chunkwise 形式，得到三层并行：

1. **chunk 内**：阶段 ①~④、⑥ 在 `(chunk, head)` 轴上完全独立（§4.2）
2. **chunk 间**：阶段 ⑤ 保留串行，但长度只有 $T/64$，且在 `(seq, head, V 块)` 三轴上并行
3. **跨卡**：长序列经 PCP 切分，用仿射修正把串行段从 $O(T)$ 降到 $O(\text{world-size})$

工程上再用四种手段把理论并行度兑现为吞吐：持久化内核消调度开销（阶段 ②）、超粗块消启动开销（阶段 ③）、host 端元数据预计算消 D2H 同步（§4.4）、融合算子消中间张量 HBM 往返（§4.2）。
