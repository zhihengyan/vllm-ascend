# ChunkGatedDeltaRule 融合算子（CANN）分析与优化路径

> 对象：`ops-transformer/attention/chunk_gated_delta_rule`（aclnn `aclnnChunkGatedDeltaRule` / torch_npu `npu_chunk_gated_delta_rule`）
> 对照：`gdn_impletiion.md` §6 的五条路径 P1–P5
> 结论口径：所有百分比都是**静态估算**，落地前以本文 §5 的实测为准。
>

---

## 0. 一句话结论

融合算子把 **算子数（约 10 → 1）、外部布局转置、`h` 物化** 三件事一次性消掉了，但**总访存流量并没有真正下降**（复算约 41u vs 多算子路径 37.45u，同量级）；它把成本换到了另一处——**阶段 2（状态扫描）是一条只用到 `Nv` 个核的串行链**，其单核算术负载约为阶段 1 的 3.8 倍，折进关键路径后占每轮 cube 时间的约 75%。

因此 P1–P5 不能再直接当作融合算子的优化清单：**P1/P2 已在算子内部解决，P3/P4 已"实现"，但换来了新瓶颈**，真正该做的是 F1–F5（见 §4）。

---

## 1. 算子形态速览

### 1.1 三阶段流水 + 双架构

入口 `op_kernel/chunk_gated_delta_rule_arch22.h`（A2/A3）与 `..._arch35.h`（Ascend 950）。两者共用同一套算法骨架，差异只在 matmul 实现与向量融合（见 §1.3）。

```
Process()                                     // arch22.h:201-233 / arch35.h:153-186
  SyncAll
  for bid in B:                               // 逐 batch
    for pos in seq, step = maxGroupLength:    // 逐 chunk group
      RunStage1(cg); SyncAll
      RunStage2(cg, curState, stateOut); SyncAll
      RunStage3(cg); SyncAll
```

与文档 §6 的阶段对应关系：

| 原文档阶段 | 融合算子实现 | 位置 |
|---|---|---|
| ① 块内 cumsum | stage1 AIV `GCumExpCompute`（arch35 用 `CumSumExpVF` 把 cumsum+exp+gamma 合成一次 VF） | `stage1:423-448` / `:423-448` |
| ② 构造 $L$ | stage1 AIC `key @ key` → `kkWsGm_`；AIV `KKBetaCompute`（乘 beta）+ 衰减掩码 | `stage1:288, 450` |
| ③ 求逆 $(I+L)^{-1}$ | AIV `InverseCompute`（**32×32 对角块**，arch35 用 `InverseAIVVF`）+ AIC `AttnInverseMMCompute`（两次 32×32 matmul 做块合并） | `stage1:469, 710` |
| ④ 重算 $W/U$ | AIC `attn @ gBK` → `kCumDecay`；`attn @ vBeta` → `vInner` | `stage1:309-322` |
| ⑤ 状态扫描 | stage2：**跨 chunk 串行推 state，`h` 完全不物化**，inter-chunk 注意力直接 atomic 累加进 `out` | `stage2` 全文 |
| ⑥ 输出 | stage3：块内 `masked_qkt @ v_inner`，同样 atomic 累加进 `out` | `stage3` 全文 |

三阶段各自的并行任务空间（这是后面 F1 的关键）：

| 阶段 | 任务划分 | 实际并行度 |
|---|---|---|
| stage1 | `nv × numChunk` 均分到 `aiCoreNum`；每核再用 `subBlockIdx` 把 chunk 劈成两半（32 行）；核内 `paraNum = 4` 个任务连发做 latency hiding | ≈ `2 × aiCoreNum` 子核 |
| **stage2** | **`nvPerCore = ceil(Nv / aiCoreNum)`，只按 `Nv` 分** | **`min(Nv, aiCoreNum)`** |
| stage3 | `Nv × numChunk` 均分 | ≈ `aiCoreNum` |

### 1.2 关键 tiling 与 workspace

`op_host/chunk_gated_delta_rule_tiling.cpp`：

| 量 | 值 / 公式 | 位置 |
|---|---|---|
| chunk 大小 $c$ | 64（硬编码） | `:119` |
| 单核最大 chunk 数 $p$ | `P_NUM = 2`（硬编码） | `:62` |
| `maxGroupLength` | $p \times \text{aiCoreNum} \times c$（40 核 → 5120） | `:122` |
| `stageOneParaNum` | 4（硬编码） | `:60` |
| 掩码份数 | `MASK_NUM = 4`（2 张掩码 × `TASK_RATIO`） | `:61` |
| 求逆对角块 | `INVERSE_SHAPE = 32`、`INVERSE_COUNT = 5` | `stage1:28-29` |
| UB 预算 | `UB_REST_BYTES = 140KB`（硬编码） | `stage1:27` |

workspace 的三段（`DoOpTiling` `:125-155`，`GetWorkspaceSize` `:255-261`）：

- **`interWorkspaceSz` —— 按 group 尺寸而非 $T$ 分配**，这是融合算子相对多算子路径最重要的结构差异：
  `gCumExp`(fp32) + `kCumDecay` + `vInner` + `qPrime` + `attnInter` + `kg` + `qkt` + `state` + `mask`，
  即 $\mathcal{O}(\text{maxGroupLength})$，随 group 复用，而不是 $\mathcal{O}(T)$。
- `stageWorkspaceSz` = $2 \cdot c \cdot (2c + 3d_k + d_v) \cdot \text{paraNum} \cdot \text{aiCoreNum}$
  （ $d_k=d_v=128$、40 核 → 约 13 MB），是 stage1 的核内 scratch。
- 固定 16 MB 系统 workspace。

`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`：1 AIC : 2 AIV 的混合核。

### 1.3 架构差异（决定 F3）

| | arch22（A2/A3） | arch35（Ascend 950） |
|---|---|---|
| matmul | `MatmulImpl`（库 matmul，自带流水） | `CGDRMatmulBasic` **手写 basic 通路**（GM→L1→L0A/L0B→Mmad→fixpipe） |
| 操作数复用 | 无 | `BSource::SameAsA`（`kk` 的 A/B 共用一次 L1 搬运）、`BSource::Reuse`（stage2 两次 matmul 共用 state 的 L1 副本） |
| 向量 | 多轮 `PipeBarrier<PIPE_V>` 分段 | `VF_CALL` 融合（`CumSumExpVF` / `InverseAIVVF` / `ComputeMaskedQkt*VF` / `Scale*State*VF`） |
| state 精度 | 仅 bf16 | bf16 / fp32 双路（fp32 路额外保一份 `curStateBf16`、`vInnerBf16`） |

### 1.4 布局：P1/P2 为什么在算子内部就没了

- **入侧**：算子只接受 TND（`query/key: (T,Nk,Dk)`、`value/out: (T,Nv,Dv)`、`beta/g: (T,Nv)`、`state: (B,Nv,Dv,Dk)`），内部用 `DataCopyPad` 的 `srcGap`（`stage1:684-697`）和 `Nd2NzParams.srcDValue`（`matmul_basic.h:131-144`）**带行步长直读**，不需要外部 `transpose`。
- **出侧**：stage2/stage3 把 matmul 的 C 矩阵按 `dstStride = Nv*Dv` 用 fixpipe 直接写进 TND 的 `out`（`matmul_basic.h:174-191`，调用点 `stage2:264`、`stage3:119-122`），出侧也没有转置。
- **状态池**：`state` 布局 `(B,Nv,Dv,Dk)` 与 vLLM 的 `ssm_state` 直接对齐，`gdn.py:609-612` 的融合路径注释已说明——**零拷贝**。

---

## 2. P1–P5 对照

### 2.1 总表

| 原路径 | 可消除流量（原文） | 在融合算子中的状态 | 一句话结论 |
|---|---|---|---|
| P1 进侧布局对齐 | 8.03u · 21.4% | ✅ **已消解** | 算子直接吃 TND，转置被 stride 搬运吸收 |
| P2 出侧边界布局 | 8.00u · 21.4% | ✅ **已消解** | fixpipe `dstStride` 直接写 TND，`transpose_state_layout` 开关不再需要 |
| P3 ②③④ 中间量往返 | 4.25u · 11.3% | ⚠️ **粒度已统一，流量没省** | 中间量从"算子间传"变成"核内 GM scratch 传"，复算约 14u |
| P4 ⑤⑥ 融合 + 状态池 | 22.0u · 58.7% | ⚠️ **已实现，但换来新瓶颈** | `h` 不再物化、状态零拷贝；代价是 group 级 barrier + stage2 只占 `Nv` 个核 |
| P5 低风险打底 | 约 3% | 🟡 **部分实现** | 掩码预计算复用、VF 融合已做；`UB_REST_BYTES`/`P_NUM`/`paraNum` 仍硬编码 |

### 2.2 逐条依据

**P1 —— 已消解，但同类问题换了形式残留。**
算子不要求调用方做任何转置；`query/key` 的 head 复制（`Nv % Nk == 0`）也不需要真复制，AIC 侧直接给同一个 GM 区块改指针偏移即可（`stage1:330` 的 `nIdBatch_[i] * nk_ / nv_ * dk_`）。

**残留**：为了给 matmul 提供行连续的算子，AIV 仍然把 `query`/`key` 拷成 `queryConGm_`/`keyConGm_`（`stage1:328-336` 的 `QKPreProcess`），并且 `QPrimeCompute`/`GBKCompute` 之后又从 GM 把同一份数据读回来（`stage1:591`、`:522`）。这笔"紧凑化副本"实测约 **7u**（见 §3），是 P1 在融合路径下的真正等价物——但它可以在**全块时走 stride 直读、只对尾块物化**来消掉，即 F2a。

**P2 —— 已消解。**
`transpose_state_layout` 这个在 op_api 里被明确拒绝的开关（`"reserved and only false is supported"`）在融合算子路径上已经不需要了：出侧靠 fixpipe 的 `dstStride`，状态池靠布局天然对齐。原文 P2 的 8.00u 与 P4 的状态池部分**都不再成立**。

**P3 —— 粒度统一了，但流量没省。**
原文担心"②③④ 并行粒度完全不同、不能直接融合"。在融合算子里这个问题确实被解决了：三者的任务空间统一成 `(Nv, chunk)`，由同一套 `Process()` 均分（`stage1:214-244`）。

但"融合"并没有让中间量不进 GM：`kkWsGm_`、`attnWsGm_`、`gBKWsGm_`、`vBetaWsGm_`、`queryConGm_`、`keyConGm_` 全部落在 `stageWorkspaceSz` 里，AIV 写、AIC 读。原因是硬约束——**cube 的操作数只能来自 GM/L1，vector 的生产结果必须先落 GM**。粗算这笔约 **14u**（§3），比原文 P3 的 4.25u 还多，因为现在连 `q/k` 的预处理副本和 `gBK`/`vBeta` 都算进来了。

**P4 —— 已实现，但引入了新的串行瓶颈。**
已经做到的：
- `h` **完全不物化**（原文 P4 最大的一笔 22.0u 里的主体）：只维护每 head 一份 `(Dv,Dk)` 的 running state（`stage2:105-106`），inter-chunk 贡献 `q_prime @ state` 直接 atomic 累加进 `out`（`stage2:210`）。
- 状态池零拷贝（`gdn.py:609-612`）。
- `v_new` 也不物化：`CalVPrime` 用 `enAtomic` 就地累加到 `vInner`（`stage2:190`）。

新增的代价：
1. **group 级全局 barrier**：每处理一个 group 要 3 次 `SyncAll<false>`（`arch22.h:224/227/230`），group 数 = $\lceil T / \text{maxGroupLength} \rceil$。
2. **stage2 的并行度只有 `Nv`**：`nvPerCore = ceil(Nv / aiCoreNum)`（`stage2:97-100`），`Nv=16`、40 核时有 24 个核在 stage2 全程空转；而 stage2 又是唯一不可并行的串行链（chunk 间有 state 依赖）。
3. 按算术量估，**stage2 的单核负载约为 stage1 的 20 倍量级**（推导见 §3.3）。

**P5 —— 部分实现。**

已做到的：掩码一次构造、全流程复用（`InitMask` `arch22.h:107-139`，`MASK_NUM=4`）；arch35 把 cumsum/exp/gamma、逆矩阵对角块、masked-qkt 缩放、state 缩放都做成了 VF；**编译期内核，没有 Python 侧 probe，也就没有首 token 抖动**（但 vLLM 侧的 probe 另有问题，见 §6）。

未做到的：`UB_REST_BYTES = 140KB`、`P_NUM = 2`、`STAGE_ONE_PARA_NUM = 4`、`INVERSE_SHAPE = 32`、`INVERSE_COUNT = 5` 全是硬编码。`UB_REST_BYTES` 尤其值得注意——它等价于把"UB 预算"写死在 kernel 里，换平台（950 vs A2）不会自动适配，**这正是原文 P5 里 `LARGE_BLOCK_T = 1216` 那类问题的同一家族，只是从 host 侧搬到了 kernel 侧**。

---

## 3. 流量复算：为什么"流量"这个标尺要降权

### 3.1 复算表

单位 `u` 与原文一致：一个 bf16 `[T,H,D]` 张量的每 token 字节 = $H \cdot D \cdot 2 = 4096$ B（ $H = N_v = 16$， $D = 128$）。

| 项 | 说明 | u/token |
|---|---|---|
| 输入 `q`/`k`/`v` + `g`/`beta` | 各 1.0u（按 $N_k = N_v$ 计） | 3.03 |
| **`queryCon` / `keyCon` 副本** | AIV 紧凑化落盘，AIC 读；`keyCon` 还会被 `QPrime`/`GBK` 二次读 | 7.0 |
| **`kkWs`** | `kk = k k^{\top}` 落盘再读 | 1.0 |
| **`attnWs`** | $(I+L)^{-1}$ 落盘、求逆过程读写、两次作为 matmul 的 A | 2.25 |
| **`gBKWs` / `vBetaWs`** | AIV 产出、AIC 读作 B | 4.0 |
| 掩码读 | stage1 每任务读一次 + stage3 每任务读一次（64×64 fp32） | 1.5 |
| stage1 产物 `kCumDecay`/`vInner`/`qPrime`/`kg`/`qkt` | 写 + 读；`vInner` 读 3 次（含一次 RMW） | 12.1 |
| **state 读写** | 每 chunk 搬 $D_v \times D_k$，含 UB 内 cast 与 AIC 复用 | 6.0 |
| `out` | 两次 atomic 累加（stage2 inter + stage3 intra）→ 两次 RMW | 4.0 |
| **合计** | | **≈ 40.8** |

对照原文多算子路径的 **37.45u**：**同量级**（差异落在估算误差内）。

### 3.2 但这个数字的含义变了

融合算子把工作集从 $\mathcal{O}(T)$ 压到了 $\mathcal{O}(\text{maxGroupLength})$： $N_v=16$、 $c=64$、 $D=128$、40 核时 group workspace 约 94 MB，中间量基本落在 L2 而不是 HBM。所以：

> **"访存流量"在融合算子里已经不是主要矛盾。** 剩下的真实成本是两样——**跨核同步次数**（每 group 3 次全局 barrier）与**阶段 2 的串行深度**（ $T/c$ 步，且只占 $N_v$ 个核）。

原文 §6 的流量排序不能直接搬过来用。

### 3.3 stage2 才是关键路径

按 $(M,N,K)$ 计 MAC，**取同一轮（一个 chunk group）的口径**（ $N_v = 16$、 $c = 64$、 $D_v = D_k = 128$、40 核、`maxGroupLength = 5120` 即每轮 80 个 chunk）：

- **stage1**：任务数 $N_v \times 80 = 1280$，40 核均分 → 32 任务/核；每任务 4 次 64 阶 matmul ≈ 2.1M MAC → **≈ 67 MMAC/核**
- **stage2**：只有 16 核活跃（`nvPerCore = ceil(Nv/40) = 1`，核 16–39 的 `nvStart ≥ Nv`，循环不执行），每核**串行**走完 80 个 chunk；每 chunk 3 次 matmul（64×128×128）≈ 3.15M MAC → **≈ 252 MMAC/核**
- **stage3**：任务空间同 stage1（1280 个任务 / 40 核），每任务一次 64×64×128 ≈ 0.52M MAC → **≈ 17 MMAC/核**

**比值 252 : 67 ≈ 3.8 倍**。即 stage1/stage3 把 40 个核铺满，stage2 只用 16 个核却要扛 3.8 倍的活——折进关键路径后 **stage2 占每轮 cube 时间的 75%**。这是当前形态最该动的地方。

> 口径说明：这里必须是**同一轮**比同一轮。若拿「stage1 单轮 67M」去比「stage2 全程 1.61G」，会得到虚高的 24 倍——1.61G 是全部 512 个 chunk 的量，跨了约 6.4 轮。

---

## 4. 融合算子自身的优化路径（按预期收益排序）

排序依据：**资源在关键路径中的占比 × 可压缩倍数 × 改动门槛**（占比见 §4.1 的理论上限分析；注意排序依据是"占比"而不是"省下的绝对量"）。

> 编号 **F1–F5 是稳定标识**（§6 会引用），按「端到端预期收益」排；而**实际动手顺序**还要考虑门槛与相互依赖，见 §4.1 末尾——两者不一致是正常的。

| | 路径 | 攻击的资源 | 关键路径占比 | 改动层 | 门槛 |
|---|---|---|---|---|---|
| **F1** | stage2 按 $D_v$ 切分 | stage2 的并行宽度 | $\beta = 75\%$ | kernel | 中 |
| **F2** | 消掉核内预处理物化 | stage1 的 GM / L2 流量 | $\alpha = 20\%$ | kernel | 中 |
| **F3** | matmul 流水化与跨架构对齐 | AIC 流水线利用率 | $\alpha+\beta+\gamma = 100\%$ | kernel | 中高 |
| **F4** | group 长度 $p$ 与 barrier 数扫描 | 每轮 3 次 `SyncAll` | $\delta \approx 10\text{–}15\%$ | tiling | 极低 |
| **F5** | 死 workspace 清理与常量收敛 | workspace / 掩码 / 常量 | 1–3% | tiling / kernel | 极低 |

### 4.1 收益的理论上限（先定分母，再看每条路径能拿走多少）

**分母 = 每轮（一个 chunk group）的关键路径。** 以 $T = 32\text{k}$、 $c = 64$、 $N_v = N_k = 16$、 $D_v = D_k = 128$、`aiCoreNum = 40`、`maxGroupLength = 5120`（每轮 80 个 chunk）计，先取 cube 的 MAC 作为第一个可算的口径：

| 阶段 | 每核 MAC | 并行宽度 | 占每轮 cube 关键路径 |
|---|---|---|---|
| stage1（`Nv × 80 = 1280` 个任务，40 核均分） | $1280/40 \times 2.1\text{M} = 67.2\text{M}$ | 40 | $\alpha$ = 20% |
| **stage2（每核串行 80 个 chunk）** | $80 \times 3.15\text{M} = 252\text{M}$ | **16** | $\beta$ = **75%** |
| stage3（任务空间同 stage1） | $1280/40 \times 0.52\text{M} = 16.8\text{M}$ | 40 | $\gamma$ = 5% |
| **合计** | **336M** | | 100% |

> 这是 **cube 口径**的分解。真实分母还要加上 AIV 侧的向量工作量（stage1 最重）、MTE2/MTE3 侧与 3 次 `SyncAll`。所以下面每条路径给的百分比都是**上限**，不是预测。

**每条路径能拿走多少**

| 路径 | 攻击的资源 | 资源在分母中的占比 | 可压缩倍数上限 | 端到端上限 | 主要折损 |
|---|---|---|---|---|---|
| **F1** | stage2 的**并行宽度** | $\beta$ = 75% | $1/k$， $k \le 2$ | **37.5%** | $N$ 降到 64 后 cube 效率、行段 stride |
| **F2** | stage1 的 GM / L2 流量 | $\alpha$ = 20%（且仅当 stage1 是 MTE 受限时成立） | 22% 流量 | **0–20%** | stage1 若 AIV 受限则趋近 0 |
| **F3a** | AIC 流水线利用率 $u$ | 100%（三个阶段都吃） | $1 - u / u'$， $u \approx 0.5 \to u' \approx 0.85$ 即 41% | **15–40%** | 只对 arch35 成立，**A2/A3 拿不到** |
| **F3b** | arch22 的 3u 重复 L1 搬运 | 隐含在 $\beta$ 里 | 最多去掉这 3u | **2–8%** | 只影响 arch22 |
| **F4** | 每轮 3 次 `SyncAll` | $\delta$（估 10–15%） | 100% | **≤ $\delta$** | $p$ 变大会顶爆 L2 |
| **F5** | workspace / 掩码 / 常量 | 1–3% | 100% | **1–3%** | 纯保险 |

**三条结构性约束（比百分比本身更重要）**

1. **并行宽度上限 $k \le \lfloor \text{aiCoreNum} / N_v \rfloor$。** 40 核、 $N_v = 16$ 时 $k$ 最多取 2（ $D_v$ 一切两半、 $s = 64$），所以 **F1 的理论天花板就是「stage2 时间减半」，即端到端 −37.5%**。要再往上必须同时切 $D_k$——而 $D_k$ 切分需要完整的 `vNew`，串行代价更高。
2. **F1 的收益强依赖 $N_v$**：

   | | $N_v = 4$ | $N_v = 16$ | $N_v = 64$ |
   |---|---|---|---|
   | stage2 活跃核 | 4 | 16 | 32（`nvPerCore = 2`） |
   | stage2 占 cube 关键路径 | 约 92% | 75% | 约 60% |
   | F1 的 $k$ 上限 | 10 | **2** | **0（ $N_v > 40$，F1 直接失效）** |

   $N_v$ 很小时 stage2 是绝对瓶颈、但 F1 **能**开出很宽的并行； $N_v$ 大到接近核数时 F1 不适用、问题也自动缓解。**最难受的是中间地带（ $N_v$ 与核数同量级），而这正是当前配置。**
3. **收益不可乘性叠加。**
   - F2 与 F1 互补：F1 之后 stage1 的占比从 20% 升到 32%，F2 的上限随之从 20% 抬到 32%——这条路径「越做越值钱」。
   - F3a 与 F1 都作用在 cube 上，只能相乘不能相加。
   - **F4 的最优 $p$ 在 F1 之后会变**（stage1 的每核任务数与 stage2 的活跃核数都变了），扫参必须在 F1 落地后重跑一次。

**组合上限**

- 只做 F1：**−37.5%**（理论上限；受 cube 效率折损，现实约 −30%）
- F1 + F3a： $1 - (1 - 0.375)(1 - 0.30) \approx$ **−56%**
- 再加 F4 与 F5：**约 −60% 至 −65%**（理论上限）
- 现实区间（含 $N$ 变小、瓶颈向 AIV/MTE 转移等折损）：**−25% 至 −40%**

**推荐执行顺序（与编号无关）**

1. **F5 先做**——所有路径的前置与保险，风险最低。
2. **F1**——唯一一条确定砍在关键路径最大占比（75%）上的路径；先做它，后面的账才算得准。
3. **F4**——成本最低，但**必须在 F1 之后重扫** $p$。
4. **F2**——当前并不在关键路径上；F1 落地后 stage1 占比抬升，它才升值。
5. **F3**——F3b 落在 arch22（当前主力，A2/A3），F3a 只在 arch35（Ascend 950）上兑现，两项要分开看。

> 原来的排序按「绝对资源占比」，会让 F2（省 22% 流量）看起来排第二；但流量不是当前的关键路径，F2 的端到端上限只有 0–20%，且大概率靠近下界。**排序依据应该是「资源在关键路径中的占比」，不是「省下的绝对量」。**

### F1 · stage2 按 $D_v$ 切分（收益上限最高）

**原理**：stage2 的 `nvPerCore = ceil(Nv / aiCoreNum)` 让并行度等于 $N_v$。但 state 的三步计算**对 $D_v$ 维是完全可分的**——取定 $D_v$ 的一段 $s$：

- `attnInter[:, s] = qPrime @ state[s, :]^T`：只需 state 的 $s$ 行
- `vNew[:, s] = vInner[:, s] + kCumDecay @ state[s, :]^T`：同样只需 state 的 $s$ 行
- `stateOut[s, :] = vNew[:, s]^T @ kg`：只需 `vInner/vNew` 的 $s$ 列 + 完整的 `kg`（ $C \times D_k$）

三段都不需要跨 $s$ 通信，**只有 chunk 维必须串行**。

**做法**：

- 把 `Process()` 的任务空间从 $N_v$ 改成 $N_v \times \lceil D_v / s \rceil$（`stage2:97-100`）； $s$ 取 32 或 64 以贴合 `MM_BLOCK_CUBE = 16` 与 `MM_MAX_DIM = 128`。
- `CalVPrime` / `CalAttnInter` / `CalStateNew` 的 $N$ 维改成 $s$，C 矩阵的 `dstStride` 保持 $N_v \cdot D_v$（`stage2:264-265, 272, 279`）；AIV 侧的 state 读/缩放/写（`ProcessAiv*`）按行段切。
- 注意 `state` 的行段访问要走 stride（`stateStride1`），`matmul_basic.h` 的 `bGmRowStride` / `Nd2NzParams.srcDValue` 已支持。

**预期收益**：stage2 单核负载降 $\min(k, \text{aiCoreNum}/N_v)$ 倍。 $N_v = 16$、40 核时 $k$ 取 2 就能从 16 核拉到 32 核，**理论上限正好是端到端 −37.5%**（stage2 占 75%、被压一半），再往上需要同时切 $D_k$。**若实测确认 stage2 是主瓶颈，这条应排第一。**

**风险**： $N$ 变小后 Mmad 效率下降（ $N$ 小于 64 时 cube 利用率明显掉），需要用实测权衡 $k$ 的取值；state 的行段 stride 要先确认 `Nd2Nz` 的 `srcDValue` 语义。

### F2 · 消掉核内预处理物化（纯 kernel 局部改动）

**F2a · `queryCon`/`keyCon` 全块走 stride 直读。**
`query`/`key` 本身就是 `(T,Nk,Dk)` 行连续，matmul 侧支持 `srcDValue` 行步长；`Nv/Nk > 1` 的 head 复制也可以直接改指针而非真拷。唯一硬需求是**尾块补零**（partial chunk 需要零行）——只对每条序列的最后一个 chunk 物化即可。
省下 `queryCon`/`keyCon` 的写 + 2 次读 ≈ **7.0u 的绝大部分**（长序列下约 6.9u，因为尾块占比 $1/\text{chunkNum}$ 很小）。

**F2b · 把 `beta` 与 `exp(g)` 折进 `attn`，直接用原张量作 B。**

$$
\begin{aligned}
\text{vInner} &= A \, (v \odot \beta) &= (A \odot \beta_{\text{row}}) \, v \\
\text{kCumDecay} &= A \, (\text{gBK}) &= (A \odot (\!-\beta \cdot e^{g})_{\text{row}}) \, k
\end{aligned}
$$

其中 $A = (I+L)^{-1}$， $\odot$ 表示按行广播缩放。这样**可以直接删掉 `vBetaWs` 与 `gBKWs` 两个 buffer**（省 4.0u），代价是 $A$ 需要按两种行缩放各出一份（ $A$ 只有 0.5u 量级，+2.0u），净省约 2.0u，同时**省掉两条 AIV 计算通路**（`VBetaCompute`、`GBKCompute` 里的乘加链）。

**预期收益**：F2a + F2b 合计约 **9u ≈ 总流量的 22%**，且完全是 kernel 内部改动，不动算子接口、不动 vLLM 侧。**但要注意口径**：22% 是「流量」的占比，stage1 在 cube 关键路径上只占 20%（见 §4.1），所以端到端上限是 0–20%——**若 stage1 实际是 AIV 受限而不是 MTE 受限，这条的端到端收益会趋近 0**。这也正是它排在 F1 之后的原因。

**风险**：缩放顺序改变会动舍入路径，必须过 §5 Step 3 的精度基线；`value`/`key` 直读要求 matmul 侧支持 $B$ 的行步长（arch35 的 basic matmul 已具备）。

### F3 · matmul 流水化与跨架构对齐

**F3a · arch35 的 basic matmul 目前是完全串行的。**
`CGDRMatmulBasic` 只有一组 L1A/L1B + 一对 L0A/L0B + 单 L0C，并用**同一个 `FLAG_ZERO`** 串起 GM→L1→L0→Mmad→fixpipe（`matmul_basic.h:87-121`），意味着下一块的 MTE2 无法与当前 Mmad 重叠；`Execute` 还用 `cmatrixInitVal=true` + `SetAtomicAdd` 做跨调用累加（`:173-195`），每次累加都要对目标做一次读改写。

做法：(a) L1A/L1B 双缓冲（当前只用 2×32KB，L1 余量充足），让 MTE2 与 Mmad 重叠；(b) 打开多 flag / `dbL0C`，让 fixpipe 与下一次 Mmad 重叠；(c) `CalVPrime`/`CalAttnInter` 的累加尽量在 L0C 内完成，只在真的跨阶段累加时才用 atomic。

**F3b · arch22 向 arch35 对齐。**
arch22 走 `MatmulImpl`，没有 arch35 的两个操作数复用：`CalVPrime` 与 `CalAttnInter` 各搬一次 state（多 2.0u），`kk` 的 A/B 各搬一次 `keyCon`（多 1.0u），另有 AIV 侧多轮 `PipeBarrier<PIPE_V>` 的串行。**A2/A3 是当前主力在跑的路径**，把 `BSource::SameAsA`/`Reuse` 与 VF 融合下沉回去，收益直接落在主战场。

**预期收益**：matmul 是 AIC 侧的全部工作量，流水化通常能拿 15–30% 的 AIC 时间（按 $1 - u/u'$ 估， $u \approx 0.5$ 时上限约 40%）；它是**唯一同时作用于 stage1/2/3 的路径**，所以上限最高。但两个前提：**只在 AIC 是瓶颈时兑现**（F1 判断完成后才知道），而且 **F3a 只对 arch35 成立——A2/A3 上这部分收益拿不到**，A2/A3 只能拿 F3b 的 2–8%。

### F4 · group 长度 $p$ 与 barrier 数扫描（成本最低）

**原理**：`P_NUM = 2` 硬编码，`maxGroupLength = p × aiCoreNum × c`。 $p$ 越大 → group 数越少 → barrier 越少，但 workspace 与 L2 压力越大（ $p=2$、 $N_v=16$、40 核时约 94 MB）； $p$ 越小 → barrier 越多。

**做法**：把 `P_NUM` 提成 tiling 参数，按"workspace ≤ L2 的一定比例"与"group 数 ≥ 一定值"两个约束扫 $\lbrace 1, 2, 4, 8 \rbrace$。

**预期收益**：上限就是 $\delta$ 本身（每轮 3 次全局 barrier 的开销占比，未实测前估 10–15%），取决于实测的 barrier + 空转占比（用 §5 Step 2 的 timeline 读）。**改一个常量 + 扫参，建议第一批就做；但注意 F1 落地后 stage1 的每核任务数与 stage2 的活跃核数都会变， $p$ 的最优值必须重扫一次。**

### F5 · 死 workspace 清理与常量收敛（低风险打底）

- **`attnInter_` 全流程未使用**：arch22 与 arch35 都声明了并为其推了偏移量（`arch22.h:177`、`arch35.h:127`），但 stage2 是把 inter-chunk 结果直接 atomic 写进 `out`，**这个 buffer 既没写过也没读过**。tiling 里的注释自己写着 `// attnInter (BF16, arch22 compat)`（`:141`）。
  两个可选动作：直接退掉（省约 1.0u 的 group workspace），或者**改用它承接 stage2 的 inter 贡献**，从而把 `out` 从"两次 atomic RMW"降到"一次 atomic + 一次普通写"。
- **`highState_` 在 arch22 从未使用**：只在 `arch22.h:186` 设了 GlobalBuffer，之后没有任何读写，却按 `fp32 × B × Nv × Dv × Dk` 分配（`tiling.cpp:146-149`； $B=8$、 $N_v=16$、 $D=128$ 时 8 MB）。A2/A3 必经 arch22，这笔该退。
- **掩码读流量**：两张 64×64 fp32 掩码在每个 `(Nv, chunk)` 任务被读一次，合计约 1.5u。掩码只含 0/1，可考虑降到 uint8 / bool，或在有 `g` 时复用 gamma 通路（arch35 的 `GammaCompute` 已经把掩码乘进 gamma 了，stage1 那次乘可以省）。
- **常量收敛**：`UB_REST_BYTES = 140KB`、`P_NUM = 2`、`STAGE_ONE_PARA_NUM = 4`、`INVERSE_SHAPE = 32`、`INVERSE_COUNT = 5`。建议至少把 `UB_REST_BYTES` 改成按平台查询推导（`GetCoreMemSize` 在 tiling 里已经在用了，`tiling.cpp:174`）。

**预期收益**：单项 1–3%，但**它是 F1–F4 的前置与保险**（尤其 F1 会改 workspace 尺寸）。

### 已评估、建议不做的一项

**把 stage1 的逆矩阵从 32×32 分块改成 64×64 级数展开**：原文 §6 已经否掉过一次（算力多 8.4 倍、伤精度），融合算子里结论不变——而且 arch35 已经把对角块前代换做成了 `InverseAIVVF` 的 VF 调用（`stage1:495`），向量侧不再是瓶颈，改造收益只会更低。

---

## 5. 动手前的测试步骤

> 与原文 §6.3 同构，但**打点位置换成融合算子**。所有收益排序成立的前提是先做完这套。

**Step 0 · 冻结基线环境**（每组数字附这一行）

```bash
# CANN 侧
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg   # 或 ascend_toolkit_install.info
python -c "import torch, torch_npu; print(torch_npu.__version__)"

# 融合算子来源（确认是本地构建还是已装包）
cd D:/Code/ops-transformer-master && git rev-parse --short HEAD
```

**Step 1 · 端到端基线（后面所有百分比的分母）**

融合算子自带 benchmark 入口（`tests/pytest/test_run.sh`，含 msprof 采集与耗时回填）：

```bash
cd attention/chunk_gated_delta_rule/tests/pytest
bash test_run.sh single prof          # msprof 采集，durations 回填 output/result_*.csv
bash test_run.sh single graph prof    # aclgraph 模式（const graph 不支持，算子未在 GE 注册）
```

**Step 2 · 分阶段耗时占比（最关键的一步）**

- 打点位置：
  - `op_kernel/.../chunk_gated_delta_rule_arch22.h`（或 arch35）`:223/226/229` —— stage1 / stage2 / stage3 的调用点
  - `:224/227/230` —— **三次 `SyncAll<false>` 的 barrier 时间必须单独量出来**，这是 F4 的唯一证据
  - `stage2:209 / :210 / :214` —— stage2 三个 matmul 分别打点，用来验证 §3.3 的"stage2 是 24 倍负载"判断
- 两种手段交叉验证：
  - 轻量：kernel 内 `GetSystemCycle()` / host 侧 `torch.npu.Event` 包住各段（可复用原文 §6.3 的 `seg_us` 骨架）
  - 权威：`msprof` 出算子级 timeline + `op_summary`，用 AI Core 的 `aiv_time`/`aic_time` 占比验证
- **产出表**：

| 观测量 | 实测 | 备注 |
|---|---|---|
| 算子总耗时 | | |
| stage1 耗时 | | |
| **stage2 耗时** | | 关键路径候选 |
| stage3 耗时 | | |
| **三次 SyncAll 合计** | | F4 的判据 |
| AIC 利用率 / AIV 利用率 | | 混合核是否失衡 |
| stage2 期间的活跃核数 | | 应等于 $N_v$，用来直接验证 §3.3 |

**Step 3 · 正确性基线（改之前先固化输出）**

```bash
cd attention/chunk_gated_delta_rule/tests/pytest
bash test_run.sh single                # 带 CPU golden 精度对比
bash test_run.sh rdv                   # RDV 全量用例（含 fp32 state、非连续 state）
bash test_run.sh random 100            # 随机 shape 扫描（含 partial chunk）

# TTK 三方交叉校验（NPU vs golden vs benchmark）
python3 -m ttk e2e -i $CSV_E2E --plugin $ASSETS --aclgraph -o $OUT
python3 -m ttk aclnn -i $CSV_ACLNN --plugin $ASSETS -o $OUT
```

- **把改前的 `out` 与 `final_state` 存盘**（`torch.save`），改完逐项对比。
- 判据：**F2b 这类"改缩放顺序"的改动允许有数值差异，但必须落在算子自带的三方交叉阈值内**（`compare.py`：`CV_MAX_RE=5.0`、`CV_AVER_RE=1.5`、`CV_RMSE=1.5`、`CV_SMALL_VAL=2.0`，分母 floor $2^{-8}$）；F4/F5 这类纯参数/清理改动应当**逐位一致**。

**Step 4 · 判据表（先定阈值再看数据）**

| 观测量 | 阈值 | 决策 |
|---|---|---|
| stage2 占比 | > 40% | **F1 立即做**，且优先于 F2/F3 |
| stage2 期间活跃核数 | $= N_v \ll \text{aiCoreNum}$ | 确认 F1 的并行度缺口成立 |
| 三次 SyncAll 合计占比 | > 10% | **F4 优先**，扫 $p \in \lbrace1,2,4,8\rbrace$ |
| AIC 利用率低于 AIV | 差值 > 20% | **F3a 优先**（arch35 串行 matmul 在拖后腿） |
| stage1 中 `QKPreProcess` + `QPrime`/`GBK` 的 MTE 占比 | > 25% | **F2 优先** |
| group workspace 占 L2 | > 60% | $p$ 需要下调，F4 先做 |

**Step 5 · 变量矩阵扫描**

$T \in \lbrace 512, 2048, 8192, 32768 \rbrace$ × $N_v \in \lbrace 4, 16, 64 \rbrace$（ $N_v$ 是 stage2 并行度的唯一变量，必须扫）× `state_dtype` 取 bf16 与 fp32 两种（fp32 路多一份 `curStateBf16`/`vInnerBf16` 往返）。

重点盯两个拐点：

- **低 $N_v$ 场景**： $N_v = 4$ 时 stage2 是否塌陷（4/40 核活跃，预测占比会显著恶化）
- **长序列场景**： $T = 32768$ 时 group 数只有 7，barrier 占比是否反而下降（验证 F4 的权衡方向）

**Step 6 · 单点微基准**

- 一次 `GM → L1 → L0 → Mmad → fixpipe` 的 basic matmul 到底多少周期？`matmul_basic.h` 的串行程度直接决定 F3a 的收益。
- 单次 `attnWs` 的 stride 直读 vs 物化后直读的耗时差（F2a 的判据）。
- 单次 `SyncAll<false>` 在目标平台的开销（F4 的判据）。

**Step 7 · 每条路径交付后的回归口径**

| 维度 | 必跑 |
|---|---|
| 数值 | Step 3 全部用例 + 与存盘基线的逐项对比 |
| 性能 | Step 1 的算子耗时 **和** Step 2 的分阶段占比，两者都要（防"局部变快、整体变慢"） |
| 平台 | 至少 950（arch35）+ A2/A3（arch22）双平台——F3b 与 arch35 的 `stateIsFp32` 路径都是平台相关的 |

---

## 6. 与 vLLM 侧的联动

1. **`gdn.py:148-150` 的 TODO 已过期。** 注释写"A5 专用实现未在正式发布包中提供，调用必然报错"，但 `arch35`（Ascend 950）内核已经存在且完整（含 fp32 state 路径）。建议在新 CANN 包到位后删掉该判断块，**否则 A5 上 `_probe_fused_chunk()` 会被无条件否决，融合路径永远走不到**。

2. **`_probe_fused_chunk()` 的首 token 抖动仍在。** 探测是一次真实 smoke call（`gdn.py:151-178`），虽然结果缓存在类上，但第一次 layer 仍要付一次同步开销。原文 P5 提到的"探测预热前移"在融合算子侧没做——因为它没法做（探的是 torch_npu 接口，不是内核）。

3. **PCP 场景仍然只能用 Triton 路径。** 融合算子里没有任何 Hccl/AllGather 调用（`op_kernel/` 全目录已确认），是单卡算子；`gdn.py:607` 的 `world_size == 1` 限定依然必要。

4. **原文 §6 里 P4 的立项理由需要重写。** 原文说 P4 只在两种情况有意义——"A5 上没有融合算子"与"PCP > 1"。现在前者已不成立（arch35 已实现），**P4 的理由只剩 PCP > 1 一条**；而在 `world_size == 1` 的通用路径上，应该把精力放在 F1–F5（融合算子自身）而不是重造一个内核。

---

## 7. 附：本文引用的关键位置

| 内容 | 位置 |
|---|---|
| 三阶段调度与 group barrier | `op_kernel/arch22/chunk_gated_delta_rule_arch22.h:201-233`、`arch35/..._arch35.h:153-186` |
| stage1 任务划分 | `arch22/...stage1_arch22.h:214-244` |
| stage2 并行度（仅 $N_v$） | `arch22/...stage2_arch22.h:91-138` |
| stage3 任务划分 | `arch35/...stage3_arch35.h:97-129` |
| 手写 basic matmul（单 flag 串行） | `arch35/chunk_gated_delta_rule_matmul_basic.h:70-122` |
| 操作数复用 `SameAsA` / `Reuse` | `matmul_basic.h:40-44`、`stage2_arch35.h:264`、`stage1_arch35.h:288` |
| 逆矩阵（32×32 对角 + 两次块合并） | `arch22/...stage1_arch22.h:536-580, 789-825`、`arch35/...:469-501, 710-728` |
| tiling 常量与 workspace | `op_host/chunk_gated_delta_rule_tiling.cpp:50-62, 117-159` |
| 算子接口与约束 | `docs/aclnnChunkGatedDeltaRule.md`、`README.md` |
| vLLM 融合路径调用 | `vllm_ascend/ops/gdn.py:133-235, 607-626` |
