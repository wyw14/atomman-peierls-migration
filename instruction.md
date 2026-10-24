# 解离位错芯的热激活迁移

为 Atomman 的半离散 Peierls–Nabarro 模型增加局部迁移势垒与谐近似逃逸率计算。实现 `/app/src/peierls_migration/__init__.py` 中的 `solve(model, settings)`；可在该包内增加模块。`factory.py` 构造原生 `atomman.defect.SDVPN` 对象。此应用使用合成材料参数，研究有限滑移条带中解离位错的一个分位错移动一格时的局部迁移。

## 能量、坐标与边界

使用 Atomman 1.5.4 的标准单位制。长度为 Å，面能为 eV/Å²，应力为 eV/Å³，以下能量为每单位直位错长度的 eV/Å。位错坐标为 `(m,n,ξ)=(x,y,z)`，`model.transform` 为单位阵，GammaSurface 的 Box 为单位盒。两个原始滑移基矢为 `a1=(b,0,0)`、`a2=(s b,0,t b)`，`b,t>0`；`model.burgers=a1`。两个分数位移记为 `u_i=(q_i,r_i)`，物理 disregistry 是 `δ_i=q_i a1+r_i a2`，法向分量恒为零。

有 `n` 个自由节点和两个固定端点，`i=0,…,n+1`，`x_i=a(i−(n+1)/2)`，`a=settings["spacing_A"]`。固定 `u_0=(0,0)`、`u_{n+1}=(1,0)`。自由变量按节点排列为 `(q_1,r_1,…,q_n,r_n)`。

GSFE 的定义是 `model.gamma.data` 在完整均匀奇数网格上的唯一周期三角插值 `Γ(q,r)`，采用每轴频率 `−(M−1)/2,…,(M−1)/2`。不是 GammaSurface 默认的非周期 RBF 插值。数据来自

`Γ=A[1−cos(4πq)+ε(1−cos(2πq))+κ(1−cos(2πr))+c sin(4πq) sin(2πr)]+Γ0`。

`A>0`，`ε>0`；`q≈1/2` 是正层错能的局部稳定相。模型已经包含表格、晶格基矢、Stroh 各向异性能量张量及应力张量，不另行传入这些 Fourier 系数。

定义

`E_λ(u)=λ E_elastic(u)+a Σ_{i=0}^{n+1} Γ(q_i,r_i)+E_stress(u)`, `0≤λ≤1`。

`E_elastic` 和 `E_stress` 分别与原生 `SDVPN.elastic_energy(x,δ)`、`SDVPN.stress_energy(x,δ)` 相同，使用 `cdiffelastic=False`、`fullstress=True`、`cdiffstress=False`。`alpha` 和 `beta` 为零；不计与构型无关的 longrange 常数。具体地，`ρ_j=(δ_{j+1}−δ_j)/a`，`j=0,…,n`，

`E_elastic=(1/(4π)) Σ_{j,k} χ_{jk} ρ_jᵀ K ρ_k`, `K=model.K_tensor`，

`χ_{jk}=3a²/2+2ψ(j−k)−ψ(j−k−1)−ψ(j−k+1)`，

`ψ(d)=(d a)² log(|d|a)/2` 当 `d≠0`，`ψ(0)=0`。

`E_stress=−(1/2) Σ_j (x_{j+1}²−x_j²) ρ_j·τ_{y,:}`，`τ=model.tau`。

## 所求的三个分支

以下定义用于选定局部迁移，不要求搜索所有位错构型的全局最低势垒。

设 `g(u)=Γ(u)+τ_{y,:}·(q a1+r a2)`。`v0` 和 `vh` 是分别在 `(0,0)`、`(1/2,0)` 附近距离小于 `0.08` 的唯一局部极小；`v1=v0+(1,0)`。`w0`、`wh` 是分别在 `(1/4,0)`、`(3/4,0)` 附近距离小于 `0.12` 的唯一单负模临界点。距离使用分数坐标的欧氏范数；这些局部点存在且满足所述性质。

用从零开始的自由节点下标 `j=0,…,n−1`。记 `h=n//2`、`d=settings["plateau_width"]//2`、`l=h−d`、`r=h+d`。

- `partial="leading"`：分支 A 在 `λ=0` 时，`j<l` 为 `v0`，`l≤j≤r` 为 `vh`，`j>r` 为 `v1`。分支 B 只把 `j=r` 改成 `v1`；分支 S 只把该点改成 `wh`。活跃节点 `j*=r`。
- `partial="trailing"`：分支 B 在 `λ=0` 时取上述完整 `vh` 平台。分支 A 只把 `j=l` 改成 `v0`；分支 S 只把该点改成 `w0`。活跃节点 `j*=l`。

所求 `u_A,u_B,u_S` 是这些临界点在 `E_λ` 下连续分支的 `λ=1` 终值。输入保证三个分支在整个闭区间上唯一延续、Hessian 非奇异，A/B 的 Hessian 正定，S 恰有一个负特征值。该承诺包括端点附近，不把近零数值阈值当成物理分支。输入还保证两个最终势垒为正，S 的归一化负模在活跃节点的 q 分量非零。

## 谐近似动力学与输出

令 `H_A,H_B,H_S` 是 `E_1/(1 eV/Å)` 对上述自由分数坐标的 Hessian；它们的数值与用 eV/Å 表示的 Hessian 相同。定义 `M_n=I_n⊗settings["mobility"]`，过阻尼动力学为

`du/dt=−ν M_n ∇[E_1/(1 eV/Å)]`, `ν=settings["frequency_s"]`。

这是分数 disregistry 坐标的有效过阻尼模型，平衡测度相对于 `d^{2n}u` 定义。温度 T 下的 FDT 噪声增量协方差为 `2ν M_n kB T/[L(1 eV/Å)] dt`；不将这组有效自由度视为原子声子坐标或实验测定的尝试频率。

设 `g+` 是 `−M_n H_S` 的唯一正特征值，`L=settings["line_length_A"]`，`T=settings["temperature_K"]`，`kB=8.617333262145×10⁻⁵ eV/K`。给出的整段直位错沿 ξ 方向保持一致，长度 L 将每单位长度势垒转为能量。所求两个谐近似逃逸率为

`log k_j=log(ν g+/(2π))+(log det H_j−log |det H_S|)/2−L[E_1(u_S)−E_1(u_j)]/(kB T)`，`j=A,B`。

返回恰好具有以下键的字典；数组可为数值列表或 NumPy 数组，所有数必须有限：

| 键 | 形状与含义 | 允许误差 |
|---|---|---|
| `minima` | `(2,n,2)`，依次 A、B 的分数位移 | 绝对 `2×10⁻⁶` |
| `saddle` | `(n,2)`，S 的分数位移 | 绝对 `2×10⁻⁶` |
| `barriers_eV_per_A` | `(2,)`，S−A、S−B | 绝对 `10⁻⁷` 加相对 `2×10⁻⁶` |
| `negative_eigenvalue` | `H_S` 的唯一负特征值 | 绝对 `10⁻⁷` 加相对 `8×10⁻⁶` |
| `unstable_mode` | `(n,2)`，`H_S` 的单位负特征向量，活跃节点 q 分量取正 | 绝对 `3×10⁻⁶` |
| `growth_rate_s` | `ν g+`，单位 s⁻¹ | 相对 `8×10⁻⁶` |
| `log_rates_s` | `(2,)`，A、B 的自然对数逃逸率 | 绝对 `2×10⁻⁴` |

使用完整的 `2n` 个自由变量计算 Hessian 和行列式；边界固定变量不计入。返回对数，不要求把极小逃逸率转成可能下溢的浮点率。

## 支持域与调用约定

`settings` 包含 `n`、`plateau_width`、`partial`、`spacing_A`、`temperature_K`、`line_length_A`、`frequency_s`、`mobility`。`n` 是 13 至 49 的奇整数；平台宽度是 3 至 9 的奇整数，且 `n−plateau_width≥8`。`partial` 只能是 `leading` 或 `trailing`。间距、温度、长度与频率均为有限正数；mobility 是有限、对称、正定的 `2×2` 数组。整数不接受布尔值。缺少必需字段、这些类型/取值不满足要求时抛出 `ValueError`；不要求异常文本。

科学输入由给定 factory 构造，GSFE 每轴网格为 9 或 11 点。材料系数满足 `0.012≤A≤0.12 eV/Å²`、`0.04≤ε≤0.2`、`0.55≤κ≤0.85`、`|c|≤0.3`、`|s|≤0.4`、`0.8≤t≤1.3`、`2≤b≤3 Å`；弹性常数为正定各向异性晶体。科学输入还满足 `1.5≤a≤3 Å`、`120≤T≤800 K`、`5≤L≤50 Å`、`10¹⁰≤ν≤10¹³ s⁻¹`，mobility 的两个特征值在 `[0.1,4]`。这些科学范围与上述基本参数校验分开；不要求检测分支存在性承诺之外的输入。只验收满足上述局部分支承诺的科学输入；求解失败不能以另选一条分支替代。

不得修改调用者的 model、其 gamma 数据或 settings。重复调用应得到容差内相同的结果。可使用容器中已有的 NumPy、SciPy、Atomman 及其依赖；不需要网络数据或外部材料数据库。每次调用限 180 秒，资源为 2 CPU、4 GiB 内存。
