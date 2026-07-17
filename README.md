# AILSVRG-ADMM：图引导稀疏 Logistic 回归实验

本项目实现以下十种方法，用于求解

\[
\min_{x,y}\;\frac1n\sum_{i=1}^n\log(1+\exp(-b_i a_i^\top x))+\mu\lVert y\rVert_1
\quad\text{s.t.}\quad Dx-y=0.
\]

- STOC-ADMM、SAG-ADMM、SAGA-ADMM、SVRG-ADMM、ASVRG-ADMM、SPIDER-ADMM；
- AILSVRG-ADMM；
- AILSVRG-ADMM-NoMom、AILSVRG-ADMM-Fixed-p、AILSVRG-ADMM-WithCorr。

其中，`AILSVRG-ADMM-NoMom` 去除惯性外推，`AILSVRG-ADMM-Fixed-p` 使用成本匹配的固定快照刷新概率，`AILSVRG-ADMM-WithCorr` 保留旧版的 \(x\) 校正项。

`--algorithms` 既接受完整算法名，也接受严格区分大小写的规范短名。短名仅由完整名称移除第一个 `-ADMM` 得到，例如 `SVRG`、`ASVRG`、`SPIDER`、`AILSVRG` 和 `AILSVRG-NoMom`；`ASVR`、`AIL`、`SPI` 或小写名称不会进行模糊匹配。短名在解析后立即转换为完整名称，因此日志、图例、CSV 和元数据始终显示完整算法名。重复输入的全名和短名会自动去重。

代码兼容 Python 3.10，所有算法使用未缩放乘子。项目不包含参考工程中的额外 L2 正则。

## 1. 在 PyCharm 中运行

在 PyCharm 中将解释器设为 conda 环境 `l2o`，然后打开 **Terminal**，按顺序执行：

```powershell
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
Set-Location -LiteralPath "E:/Gits/AilAdmm_GGLR"
conda activate l2o
python -m pip install -r requirements.txt
python scripts/check_solver.py
python main.py --preflight --data-dir datasets
python main.py --data-dir datasets
```

最后一条命令才会运行正式实验。它开始时会清空 `results/`，结束后其中只保留本次运行生成的 PDF 与 SVG 图像。

CLARABEL（Apache 2.0）是首选免费凸锥内点求解器；ECOS（GPLv3）是仅在 CLARABEL 严格认证失败时启用的开源指数锥内点法备用求解器。二者均无需许可证。检查两个求解器和高精度参数：

## 2. 数据目录

默认从项目内的 `datasets` 读取。先检查数据文件、稀疏矩阵维度和求解器：

```powershell
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
Set-Location -LiteralPath "E:/Gits/AilAdmm_GGLR"
conda activate l2o
python main.py --preflight --data-dir datasets
```

程序识别：

| 数据集 | 训练文件 | 测试文件 |
| --- | --- | --- |
| a9a | `a9a` | `a9a.t` |
| w8a | `w8a` | `w8a.t` |
| ijcnn1 | `ijcnn1.bz2` | `ijcnn1.t.bz2` |
| madelon | `madelon` | `madelon.t` |

若测试文件不存在，将从训练集进行固定种子、分层的 80/20 划分。`MaxAbsScaler` 只在训练集拟合。

## 3. 参考最优值

`utils/optimizer.py` 用 CVXPY 把 logistic 损失转换为指数锥模型。首选 CLARABEL 的默认精度为：

```text
tol_gap_abs = 1e-9
tol_gap_rel = 1e-9
tol_feas    = 1e-9
max_iter    = 2000
```

只有状态严格为 `OPTIMAL`，且相对可行性误差、重新计算的目标值和独立 KKT residual 全部通过认证时，参考解才会用于绘图。参考解缓存和诊断只存于系统临时目录，进程结束后自动删除。认证失败时程序终止该数据集，不会用 `0.0` 或算法运行中的最小值代替 \(F^\star\)。

程序依次使用三个 CLARABEL 配置档：`1e-9 + auto`、`1e-9 + faer`，以及 `1e-8 + faer` 的完整精度兜底档。后两档启用迭代精化并把精化上限提高到 20。若三个档位均不能严格认证，则使用 ECOS 的 `abstol/reltol/feastol=1e-9`、`max_iters=10000` 配置。

两个求解器都只接受严格的 `OPTIMAL`；`OPTIMAL_INACCURATE` 即使有变量值也不会用于绘图。ECOS 备用解还必须通过相同的原始可行性、独立 KKT、目标一致性和有限性认证；若 CLARABEL 最后一个 `OPTIMAL_INACCURATE` 目标存在，ECOS 目标还必须在相对 `1e-7` 以内与其一致。否则程序在控制台报告失败并拒绝该数据集的 gap 图。用于判断 \(y_i=0\) 的活跃集阈值与本次锥可行性容差联动，避免把求解器容差量级的数值噪声误判为非零坐标。

只计算参考解：

```powershell
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
Set-Location -LiteralPath "E:/Gits/AilAdmm_GGLR"
conda activate l2o
python main.py --data-dir datasets --reference-only
```

`--reference-only` 仅在控制台输出认证结果，不会创建或清空 `results/`。参考解每次均在临时目录重新计算，`--recompute-reference` 仅为旧命令兼容保留。

## 4. 正式实验

完整运行四个数据集：

```powershell
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
Set-Location -LiteralPath "E:/Gits/AilAdmm_GGLR"
conda activate l2o
python main.py --data-dir datasets --datasets a9a w8a ijcnn1 madelon
```

先检查解析后的配置而不读取数据或运行算法：

```powershell
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
Set-Location -LiteralPath "E:/Gits/AilAdmm_GGLR"
conda activate l2o
python main.py --dry-run --data-dir datasets
```

常用覆盖参数：

```text
--algorithms AILSVRG AILSVRG-NoMom
--seeds 2026 2027 2028
--max-iter 2000
--eval-every 20
```

输出位置和格式不可覆盖：正式运行始终写入项目根目录的 `results/`，并且固定输出 PDF 与 SVG。

## 5. 指标与公平计时

- Optimality Gap 使用可行化目标 \(f(x_t)+\mu\lVert Dx_t\rVert_1-F^\star\)。
- Primal Residual 为 \(\lVert Dx_t-y_t\rVert\)。
- KKT residual 是原始可行性、\(x\)-stationarity 与 \(y\)-stationarity 三个平方残差之和。
- Test Logistic Loss 为测试集上的平均 logistic 损失，只用于评估，不参与训练或参数更新。
- Test Accuracy 使用阈值 `score >= 0` 判为 `+1`。
- `algorithm_time` 包括算法必需的梯度表、快照全梯度和更新，不包括统一指标计算与绘图。
- `ifo_count` 统计样本梯度计算；同一样本在两个点求梯度计为两次 IFO。
- ASVRG 的乘子更新使用 `z`，因此输出和 KKT 指标也使用受约束的 `z` 变量。

## 6. 特征图

图只由缩放后的训练集生成：绝对 Pearson 相关性、每个特征 5 个近邻、对称化，再补最大生成树保证连通。Incidence matrix 每行只有一个 `+1` 和一个 `-1`，以 CSR 稀疏格式在内存中构建。测试数据不参与建图。

步长中的 \(\lVert D\rVert_2^2\) 使用 `scipy.sparse.linalg.eigsh` 计算 \(D^\top D\) 的最大特征值，并增加 `1e-6` 相对安全裕量。不能用全 1 向量初始化 incidence matrix 的幂迭代，因为恒有 \(D\mathbf 1=0\)，这会把非零图的谱范数错误计算为 0。

每个算法在每次更新后检查梯度、原始变量和未缩放乘子。发现 `NaN`、`Inf` 或浮点溢出时，该次算法/种子立即停止，诊断仅输出到控制台，也不会把失败运行加入绘图统计。只要有一次运行失败，程序最终返回非零退出码。

## 7. 矢量图

默认输出 PDF 和 SVG：

- `svg.fonttype = "none"`，文本在 SVG 中保持可编辑；
- `pdf.fonttype = 42`，PDF 嵌入 TrueType 字体；
- 曲线、坐标轴和图例均保持矢量，不使用 rasterization。

每个数据集有 5 个纵轴指标，默认只输出 iterations 与 time，共 10 组图、20 个 PDF/SVG 文件。每个算法在每张图中只显示一条逐检查点中位数曲线，不绘制 seed 细线、均值线、标准差带或其他填充区域。Optimality Gap、Primal Residual 和 KKT Residual 使用对数纵轴；Test Logistic Loss 和 Test Accuracy 使用线性纵轴。文件直接位于 `results/` 根目录，命名为 `gglr_<数据集>_<指标>_vs_<横轴>.<格式>`，例如 `gglr_a9a_kkt_residual_vs_iterations.svg` 和 `gglr_a9a_test_logistic_loss_vs_time.pdf`。

IFO 图由 `config.py` 中的绘图开关控制：

```python
PLOT_SETTINGS = {
    "include_ifo_plots": False,
}
```

默认的 `False` 不创建任何 `_vs_ifo.pdf` 或 `_vs_ifo.svg` 文件。需要按 IFO 预算比较时，将该值改为 `True`；程序会额外生成 5 组 IFO 图，此时每个数据集共 15 组图、30 个 PDF/SVG 文件。time 和 IFO 图（开启时）会先将各 seed 插值到共同可达的横轴预算。

## 8. 输出目录

`results/` 是唯一的持久化输出目录。每次正式实验开始时，程序会先清空其旧内容；运行后它只包含当前实验生成的 `.pdf` 和 `.svg` 文件，不包含数据集子目录、CSV、参考解、图结构、元数据或失败日志。

```text
results/
├── gglr_a9a_kkt_residual_vs_iterations.pdf
├── gglr_a9a_kkt_residual_vs_iterations.svg
└── ...
```

## 9. 测试

本项目交付时不运行四个真实数据集的正式实验。可运行的测试只使用公式级小矩阵或人工曲线：

```powershell
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
Set-Location -LiteralPath "E:/Gits/AilAdmm_GGLR"
conda activate l2o
python -m unittest discover -s tests -v
```
