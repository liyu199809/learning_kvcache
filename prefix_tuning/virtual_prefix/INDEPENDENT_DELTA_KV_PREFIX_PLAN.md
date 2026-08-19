# Qwen3.5 Independent Delta K/V Prefix 设计与实施计划

## 1. 目标

当前 Delta Virtual Prefix 在每个 Gated DeltaNet 层中学习一个 hidden-space Prefix

$$
P_l\in\mathbb{R}^{M\times H},\qquad M=2048,\ H=2560.
$$

但是当 $P_l$ 全零初始化时，Delta state 的双线性写入项在零点缺少有效的一阶梯度。真实 step-1 checkpoint 显示：24 个层中，每层 2048 个 Prefix token 只有最后 3 行发生更新。

新方案将 hidden-space $P_l$ 替换为每层独立的 Delta-space Prefix：

$$
\left(\bar K_l^P,\bar V_l^P,B_l^P,A_l^P\right),
$$

通过“K 为零、V 随机”的非对称初始化，使初始 Prefix state 仍为零，但 K 分支从第一步就获得非零梯度。

## 2. 原始 Qwen3.5 Gated DeltaNet

Qwen3.5-4B 的 32 个 decoder layer 中有 24 个 Gated DeltaNet 层。对于某个 DeltaNet 层的输入

$$
X_l=(x_1,\ldots,x_T),
$$

原模型首先计算

$$
[\bar q_t,\bar k_t,\bar v_t]=W_{qkv}x_t,
\qquad
\beta_t=\sigma(W_bx_t),
\qquad
a_t=W_ax_t.
$$

$\bar q_t,\bar k_t,\bar v_t$ 通过 kernel size 4 的 depthwise causal convolution 和 SiLU 后得到 $q_t,k_t,v_t$，Q/K 随后进行 L2 normalization。每个 value head 的 log-space 遗忘门为

$$
g_t=-\exp(A_{\log})\operatorname{softplus}(a_t+dt_{\mathrm{bias}}),
\qquad
\alpha_t=\exp(g_t)\in(0,1).
$$

FLA 实际执行的 recurrent gated delta rule 是

$$
\widetilde S_t=\alpha_tS_{t-1},
$$

$$
e_t=v_t-k_t\widetilde S_t,
$$

$$
S_t=\widetilde S_t+\beta_tk_t^\top e_t,
\qquad
o_t=q_tS_t.
$$

合并后为

$$
\boxed{
S_t=
\alpha_tS_{t-1}
+\beta_tk_t^\top
\left(v_t-k_t\alpha_tS_{t-1}\right)
}.
$$

其中：

- $S_t\in\mathbb{R}^{d_k\times d_v}$ 是 recurrent state；
- $\alpha_t$ 决定保留多少旧状态；
- $k_t\widetilde S_t$ 是当前状态对 $v_t$ 的预测；
- $e_t$ 是预测误差；
- $\beta_t$ 决定将误差写回状态的强度。

## 3. 当前 hidden-space Prefix Tuning

当前方案在每个 DeltaNet 层中学习独立参数

$$
P_l=(p_{l,1},\ldots,p_{l,M})\in\mathbb{R}^{M\times H},
\qquad M=2048,\ H=2560.
$$

数学上等价于该层处理

$$
X_l'=[P_l;X_l],
$$

且 $P_l$ 不进入 `input_ids` 或 position ids。

要理解实现方式，先注意一个 Gated DeltaNet 层在处理任意序列时，跨 token 携带两类内部状态：

1. **Recurrent state** $S_t\in\mathbb{R}^{d_k\times d_v}$：由第 2 节的 delta rule 逐步累积所有历史 token。计算第 $t$ 个 token 需要 $S_{t-1}$，因此用户序列的第一个 token 也需要一个初始 state。
2. **Convolution history**：第 2 节中进入 recurrence 的 $k_t,v_t$ 并非 $W_{qkv}$ 的原始输出，而是先经过 kernel size 4 的 depthwise causal convolution 和 SiLU。这意味着第 $t$ 个 token 的 $k_t,v_t$ 依赖于前 3 个 token 的 raw projected values（conv 之前的 $\bar k,\bar v$）。因此用户序列前 3 个 token 的计算需要紧邻其前的 3 个 token 的 projected values 作为 convolution 输入历史。

基于此，实现上让 $P_l$ 经过原模型冻结的 $W_{qkv},W_a,W_b$、causal convolution 和 Delta recurrence 完整前向。这次前向的输出 token $o_t$ 全部丢弃，只保留它遗留下来的上述两类状态：

- **Prefix 最终 recurrent state** $S_l^P$：即处理完全部 $M$ 个 Prefix token 后的 $S_M$，是整个 Prefix 信息的累积摘要；
- **Convolution history** $C_l^P\in\mathbb{R}^{3\times H}$：即 Prefix 最后 3 个 token 的 raw projected values。之所以恰好是最后 3 个，是因为 kernel size 为 4，用户序列的第 1 个 token 的 $k,v$ 依赖它前面的 3 个 token（即 Prefix 的最后 3 个）的 projected values。

用户序列随后以

$$
S_{l,0}^{\mathrm{user}}=S_l^P
$$

作为 recurrent 初始状态，并以 $C_l^P$ 作为 convolution history。这等价于在冻结基座内部“预填充”了 $P_l$：对用户 token 而言，效果与序列前面真的存在 2048 个 Prefix token 完全一致，但不占用 `input_ids`、position ids 或 context 长度。packed batch 中，同一个 $(S_l^P,C_l^P)$ 复制给每个样本，样本间通过 `cu_seqlens` 隔离。

当前共有

$$
24\times2048\times2560=125,829,120
$$

个可训练参数，原始 Qwen3.5 基座全部冻结。

## 4. 为什么全零初始化会导致优化退化

当前 prepared model 将

$$
P_l=0.
$$

因为 $W_{qkv},W_a,W_b$ 没有 bias，所以 Prefix 初始时有

$$
k_t^P=v_t^P=0,
\qquad
S_t^P=0.
$$

在 $S_{t-1}^P=0$ 时，写入项退化为

$$
\Delta S_t^P
=
\beta_t^P k_t^{P\top}v_t^P.
$$

由于 K 和 V 都由同一个 $P_l$ 线性生成并同时从零开始，所以

$$
\frac{\partial\Delta S_t^P}{\partial P_l}
=
\beta_t^P
\left[
\frac{\partial k_t^{P\top}}{\partial P_l}v_t^P
+k_t^{P\top}\frac{\partial v_t^P}{\partial P_l}
\right]
=0.
$$

因此，在初始点，recurrent Prefix 对 $P_l$ 缺少有效的一阶梯度。只有 Prefix 最后 3 个 raw projected values 通过 kernel size 4 的 convolution history 直接连到用户 token，所以它们可以在第一步获得梯度。

已有实验证据：

- step-1 checkpoint 中 24 个层每层都只有最后 3/2048 行发生变化；
- 全部 Prefix 合计只有 72/49,152 个 token row 发生变化；
- Prefix 前 13 步 grad norm 约为 `0.0041～0.0071`；
- LoRA r=64 同阶段 grad norm 约为 `1.0～4.2`。

这不只是两种参数化的 norm 尺度不同，也表明当前的 2048-token Prefix 在训练初期实际只有极少数 token 参与优化。

## 5. 新方法：Independent Delta K/V/beta/a Prefix

### 5.1 参数化

新方法不再通过一个 hidden-space $P_l$ 同时生成 K、V、beta 和 a，而是在每个 DeltaNet 层直接学习

$$
\bar K_l^P\in\mathbb{R}^{M\times d_k},
\qquad
\bar V_l^P\in\mathbb{R}^{M\times d_v},
$$

$$
B_l^P\in\mathbb{R}^{M\times H_v},
\qquad
A_l^P\in\mathbb{R}^{M\times H_v}.
$$

对 Qwen3.5-4B：

$$
d_k=2048,
\qquad
d_v=4096,
\qquad
H_v=32.
$$

对应代码参数为

```python
self.prefix_k = nn.Parameter(torch.zeros(M, 2048))
self.prefix_v = nn.Parameter(torch.empty(M, 4096))
self.prefix_beta_logits = nn.Parameter(torch.zeros(M, 32))
self.prefix_a = nn.Parameter(torch.zeros(M, 32))
```

$\bar K_l^P,\bar V_l^P$ 表示 causal convolution 之前的 projected K/V。Prefix query 可设为零，因为我们只需要 Prefix 的最终 recurrent state，Prefix 自身的 output 会被丢弃。

beta 使用无约束 logits 参数化：

$$
\beta_l^P=\sigma(B_l^P),
$$

从而始终满足 $0<\beta_l^P<1$。a 使用原模型的稳定衰减参数化：

$$
g_l^P
=
-\exp(A_{\log})
\operatorname{softplus}(A_l^P+dt_{\mathrm{bias}}),
$$

$$
\alpha_l^P=\exp(g_l^P)\in(0,1).
$$

原模型的 $A_{\log}$ 和 $dt_{\mathrm{bias}}$ 保持冻结，不改变原 checkpoint 的参数路径或 shape。

### 5.2 初始化

默认初始化设计为

$$
\bar K_l^P=0,
\qquad
\bar V_l^P\sim\mathcal N(0,\sigma^2),
\qquad
B_l^P=0,
\qquad
A_l^P=0.
$$

因此

$$
K_l^P=0,
\qquad
\beta_l^P=0.5,
$$

$$
g_l^P
=
-\exp(A_{\log})
\operatorname{softplus}(dt_{\mathrm{bias}}).
$$

在 K 为零时，无论 V、beta 和 a 为何值，Prefix state 都保持为零：

$$
S_l^P=0.
$$

为了避免随机 V 通过 convolution history 改变用户前几个 token，初始化时必须将最后 $K_{\mathrm{conv}}-1=3$ 行设为零：

```python
nn.init.normal_(self.prefix_v, std=prefix_init_std)
with torch.no_grad():
    self.prefix_v[-(self.conv_kernel_size - 1):].zero_()
```

Prefix K 必须由代码显式设为严格的零，不能只构造“非常小的 K”，因为 Qwen3.5 会对 K 做 L2 normalization，微小非零误差可能被放大。

$\sigma$ 不直接固定为 `0.02`。先比较 `1e-4`、`3e-4`、`1e-3` 三档，根据 K 梯度、是否触发 grad clipping 以及第二步的状态变化选择。

### 5.3 梯度动力学

在初始 $S_{t-1}^P=0$ 时，

$$
\Delta S_t^P
=
\beta_t^P K_t^{P\top}V_t^P
=0.
$$

但 K 分支的梯度为

$$
\frac{\partial\Delta S_t^P}{\partial K_t^P}
=
\beta_t^P V_t^P
\neq0.
$$

所以第一步：

- `prefix_k` 通过 recurrent state 获得非零梯度；
- `prefix_v` 的 recurrent 梯度为零，但最后 3 行可通过用户 convolution 路径获得梯度；
- `prefix_beta_logits` 和 `prefix_a` 因当前写入项为零，其 recurrent 梯度也为零。

其中 g 和 alpha 本身不是被优化的参数，它们由可学习的 $A_l^P$（即 `prefix_a`）经过固定非线性变换得到，优化器只更新 $A_l^P$，冻结的 $A_{\log}$ 和 $dt_{\mathrm{bias}}$ 只作为常数缩放和平移进入变换：

$$
A_l^P
\;\xrightarrow{\;+\,dt_{\mathrm{bias}}\;}\;
\operatorname{softplus}
\;\xrightarrow{\;\times[-\exp(A_{\log})]\;}\;
g_l^P
\;\xrightarrow{\;\exp\;}\;
\alpha_l^P.
$$

梯度沿该链自动回传：

$$
\frac{\partial\mathcal{L}}{\partial A_l^P}
=
\frac{\partial\mathcal{L}}{\partial\alpha_l^P}\,
\alpha_l^P\,
\Big[-\exp(A_{\log})\,
\sigma\big(A_l^P+dt_{\mathrm{bias}}\big)\Big],
$$

而 $\partial\mathcal{L}/\partial\alpha_l^P$ 来自 recurrence。$\alpha_t$ 在状态更新

$$
S_t=\alpha_tS_{t-1}+\beta_tk_t^\top\big(v_t-k_t\alpha_tS_{t-1}\big)
$$

中出现两处，且都乘以旧状态 $S_{t-1}$，因此 $\partial S_t/\partial\alpha_t\propto S_{t-1}$。这从梯度角度解释了上一条结论：初始时 $S_{t-1}^P=0$，$\alpha_t$ 不影响任何输出，故 `prefix_a` 第一步的 recurrent 梯度为零；状态非零后 a 分支才开始参与优化。该参数化同时保证对任意 $A_l^P$ 都有 $\operatorname{softplus}>0\Rightarrow g_l^P<0\Rightarrow\alpha_l^P\in(0,1)$，无需投影或裁剪即可维持遗忘门稳定。beta 分支同理：$B_l^P$（即 `prefix_beta_logits`）经 $\sigma$ 优化，初始 logits 为零对应 $\beta_l^P=0.5$。

第一步之后 $K_l^P\neq0$，因此 V、beta 和 a 开始获得 recurrent 梯度。该初始化与 LoRA 的“A 随机、B 为零”性质类似：初始函数增量为零，但零初始化的分支从第一步就有梯度。

### 5.4 参数量

对 $M=2048$，每层参数量为

$$
2048\times(2048+4096+32+32)
=12,713,984.
$$

24 层合计

$$
\boxed{305,135,616}
$$

个可训练参数。其中 beta/a 只占

$$
24\times2048\times(32+32)
=3,145,728,
$$

参数增长主要来自独立 K/V。

计划默认先以 $M=2048$ 验证算法正确性。在正式长训前需要确认最终参数预算：

- 保持 $M=2048$：305,135,616 个参数；
- 对齐当前约 1.26 亿参数：使用 $M\approx845$。

## 6. 实施设计

### 6.1 Transformers remote model

1. 新建独立的 Delta K/V Prefix GDN 类，不覆盖当前 hidden-prefix 类。
2. 保留原始 `in_proj_qkv`、`conv1d`、`A_log`、`dt_bias`、`out_proj` 等所有基座参数名称和 shape，保证原 Qwen3.5 checkpoint 可以原样加载。
3. 新增 `prefix_k`、`prefix_v`、`prefix_beta_logits`、`prefix_a` 四类参数。
4. Prefix Q 使用零 tensor；K/V 经过原模型冻结的 depthwise convolution。
5. 用独立 beta/a 计算 Prefix recurrence，但继续使用冻结的 `A_log` 和 `dt_bias`。
6. padded、packed、prefill 和 decode 统一使用相同的 Prefix recurrent/conv state 语义。
7. Prefix state 依然不进入 `input_ids`、position ids 或 context token budget。

### 6.2 Prepared model

1. 新建模型目录，不删除或覆盖现有 hidden-prefix M=2048 模型。
2. `model_type` 继续使用 `qwen3_5`，只更换 custom architecture 名称和 `auto_map` remote modeling 文件。
3. 4B 基座 safetensors 继续使用 hardlink，新增独立 Delta Prefix safetensors。
4. config metadata 记录 M、K/V/head 维度、初始化 std、conv tail 规则和参数总量。
5. 现有 hidden-prefix checkpoint 不能直接 resume 到新参数结构，新方法从新的 prepared model 开始训练。

### 6.3 verl

1. trainable regex 只匹配四类 Delta Prefix 参数。
2. $M=2048$ 时严格校验 96 个 trainable tensor 和 305,135,616 个 elements。
3. optimizer 仍只接收 `requires_grad=True` 参数，基座不产生参数梯度或 optimizer state。
4. rollout 热同步仅传输 96 个 Prefix tensor。
5. trainable-only checkpoint metadata 升级为新 Prefix type，严格记录参数名称、shape、dtype、M 和总参数量。
6. 新建独立 OPSD launcher，不改动现有 hidden-prefix、LoRA 或 full launcher。

### 6.4 vLLM

1. 为新 architecture 注册独立 native vLLM 实现或扩展现有 plugin。
2. vLLM 使用与 Transformers 完全相同的 K/V/beta/a 初始化和 state recurrence。
3. `load_weights()` 严格校验 96 个 Prefix tensor 全部加载。
4. Prefix state cache 以参数 version、device 和 dtype 为 key；热更新后自动失效并清理旧 request cache。
5. Prefix state 在同一权重版本只计算一次，新 request 复制计算好的 recurrent/conv state。

## 7. 验证与验收

### 7.1 数学与梯度测试

1. 初始 Prefix K 严格为零，Prefix recurrent state 严格为零。
2. 最后 3 个 raw V 为零，初始 convolution history 严格为零。
3. 初始模型与原 Qwen3.5 短序列 logits 在既定容差内对齐。
4. 第一步 24 层的所有 `prefix_k` token row 获得有限、非零梯度。
5. 第一步的 V/beta/a recurrent 梯度符合预期；K 更新后，第二步 V、beta、a 都获得非零梯度。
6. 比较三档 V 初始化 std，确保 K 梯度既不消失也不在每步都触发 clipping。

### 7.2 Transformers 行为

1. padded 与 packed/remove-padding 输出在既定容差内一致。
2. packed batch 中修改前一段输入不能改变后一段输出。
3. prefill/decode cache 与一次性 forward 结果一致。
4. gradient checkpointing 下 Prefix state 重算正确，不存在 cache mutation。

### 7.3 FSDP2 与 checkpoint

1. 只存在 96 个 trainable tensor，参数总量严格符合配置。
2. optimizer step 后 Prefix 发生变化，抽样基座参数 bitwise 不变。
3. checkpoint 只包含 Prefix model shard、Prefix optimizer state、scheduler 和 RNG，不包含 4B 基座。
4. save/resume 后 Prefix、optimizer、scheduler 和 global step 完整恢复。

### 7.4 vLLM 与 OPSD

1. Transformers 与 native vLLM 在 prefill/decode 上对齐。
2. 热同步仅传输 96 个 Prefix tensor，更新后重算 Prefix state 并清理旧 cache。
3. 完成 6 actor GPU + 2 teacher GPU 的 1～2 step OPSD smoke。
4. 检查 rollout、teacher top-k、actor loss、grad norm、reward、checkpoint 和下一轮热同步，无 NaN/OOM。
5. 对比 hidden-prefix 与新方法的每行梯度覆盖率、grad norm、前几步参数变化和训练 loss。

## 8. 实施顺序

1. 先在小配置 DeltaNet 上实现 K/V/beta/a Prefix，验证两步梯度动力学。
2. 接入 Transformers 完整 Qwen3.5-4B，完成 padded/packed/prefill/decode 测试。
3. 扩展 prepared-model builder 并创建新模型目录。
4. 扩展 native vLLM plugin、Prefix state cache 和 partial weight loading。
5. 更新 verl trainable selection、checkpoint/exporter 和独立 launcher。
6. 完成 2 GPU FSDP2 测试和 8 GPU OPSD smoke。
7. 根据 V 初始化 std 对比结果确定正式配置。
8. 正式长训前最终确认 $M=2048$ 还是参数对齐的 $M\approx845$。

## 9. 不变的约束

- 不修改已安装的 Transformers 包。
- 不改变原始 Qwen3.5 基座参数名称或 shape。
- `model_type` 继续为 `qwen3_5`。
- OPSD teacher 继续使用原始 Qwen3.5-4B。
- Prefix 不进入 token/context budget。
- 不覆盖当前 hidden-prefix model 和 checkpoint。
- 实施和验证完成前不自动提交 Git。
