# Stellarator Diagnostics

一套统一、适合超算批处理的 VMEC 与 DESC 平衡结果分析工具。输入 VMEC
`wout_*.nc` 或 DESC `.h5`，输出 HTML 报告、PNG 图片、CSV 表和完整 JSON
数据。BOOZ_XFORM 的 `boozmn_*.nc` 可单独生成 Boozer 坐标下的磁场强度图。

## 功能

- 自动识别 VMEC/DESC 文件，DESC `EquilibriumFamily` 可选择成员；
- 标量摘要：长宽比、体积、主/副半径、磁场、beta、电流、收敛残差；
- 剖面：iota、压力、磁剪切、电流、`dV/ds`；
- 自动定位分母不超过 12 的低阶有理面，并写入 JSON；
- 磁井：VMEC `vp`（或 `gmnc(0,0)`）边界外推定义；
- Mercier：`D_Mercier` 及 shear/well/current/geodesic 分项，排除轴附近后的
  最小值和负值占比；
- 几何：依据 `NFP` 在半个场周期内绘制四个环向截面，每个截面含 9 个闭合磁面；
- 磁场：VMEC/DESC 原生角坐标下 `|B|`，以及多个 `s` 位置的无填充
  BOOZ_XFORM Boozer 等值线；
- 三维完整环面以 `|B|` 为表面颜色；
- iota 图按 `NFP` 标记允许的低阶 `(m,n)` 共振；
- 沿多条直线磁力线的 `|B|` 轨迹，用于快速观察 ripple 与局部磁阱；
- VMEC–DESC 或不同构型间的剖面/标量对比；
- NEO `neo_out`：$\epsilon_{\rm eff}^{3/2}$ 与 $\epsilon_{\rm eff}$ 径向诊断；
- DKES `results`：$L_{11}$、$L_{31}$、$L_{33}$ 单能输运系数、变分上下界
  及相对收敛带；
- COBRAVMEC `cobra_grate`：各磁力线的无限-$n$ 理想气球模本征值及径向包络，
  并按官方约定标出负本征值不稳定区；
- 大批文件扫描，单个坏文件不会终止整个任务；
- Python API 与 `stell-diag` CLI。

## 安装

基础 VMEC 分析：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

在已有 DESC 环境中安装本项目即可；或安装可选依赖：

```bash
pip install -e ".[desc]"
```

## 最常用命令

```bash
# 单个 VMEC 或 DESC 文件的完整报告
stell-diag analyze wout_case.nc -o diagnostics/case
stell-diag analyze eqfam_case.h5 -o diagnostics/desc --family-index -1

# 快速查看标量
stell-diag summary wout_case.nc

# VMEC 与 DESC（或多个同类构型）对比
stell-diag compare wout_case.nc eq_case.h5 -o comparison

# 批量扫描；建议在登录节点只提交轻量后处理作业
stell-diag scan "runs/*/wout_*.nc" -o scan_summary.csv

# 使用成熟 booz_xform 对 wout 做变换
pip install -e ".[boozer]"
stell-diag xboozer wout_case.nc -o boozmn_case.nc \
  --surfaces 0.15 0.30 0.45 0.60 0.75 0.90 1.0 --mboz 54 --nboz 32

# 多磁面、无填充 Boozer 等值线
stell-diag boozer boozmn_case.nc \
  --surfaces 0.15 0.30 0.45 0.60 0.75 0.90 1.0 -o boozer_surfaces

# 将 Boozer 图并入完整报告
stell-diag analyze wout_case.nc --boozmn boozmn_case.nc -o diagnostics/case

# 单独分析已经计算好的 NEO、DKES、COBRA 输出
stell-diag neo neo_out.case -o diagnostics/neo
stell-diag dkes results.case -o diagnostics/dkes
stell-diag cobra cobra_grate.case -o diagnostics/cobra

# 全部并入同一份 VMEC 报告
stell-diag analyze wout_case.nc -o diagnostics/case \
  --boozmn boozmn_case.nc --neo-out neo_out.case \
  --dkes-results results.case --cobra-grate cobra_grate.case
```

报告目录包含：

```text
report.html              浏览器可打开的总报告
summary.csv              单行标量摘要
diagnostics.json         完整机器可读结果
profiles.png             径向剖面
iota.png                 iota 与 NFP 允许的低阶有理面
stability.png            Mercier 及分项
cross_sections.png       多磁面截面
fieldline_traces.png     多条磁力线上的 |B|
fieldline_long.png       一条磁力线连续 200 个场周期的 |B|
surface_3d.png           以 |B| 着色的完整三维磁面
surface_top.png          以 |B| 着色的三维磁面俯视图
boozer/boozer_s*.png     各磁面分别输出的无填充 Boozer 等值线
neo/neo_effective_ripple.png
dkes/dkes_coefficients.png
dkes/dkes_convergence.png
cobra/cobra_ballooning.png
{neo,dkes,cobra}/*_normalized.csv
<quantity>.csv           各剖面的原始数据
```

## Python API

```python
from stellarator_diagnostics import analyze, compare, scan

eq, report_path = analyze("wout_case.nc", "diagnostics/case", surface_s=0.5)
print(eq.scalars["iota_axis"], eq.scalars["iota_edge"])
surface = eq.surface(s=1.0)
field = eq.field_map(s=0.5)

# 外部程序结果可选；只读取真实输出，不自动伪造缺失结果
eq, report_path = analyze(
    "wout_case.nc",
    "diagnostics/case",
    boozmn="boozmn_case.nc",
    neo_out="neo_out.case",
    dkes_results="results.case",
    cobra_grate="cobra_grate.case",
)
```

项目特定的验收条件可参考 `examples/custom_analysis.py`。它特别示范了
NFP=4、A≈8、iota≈0.68–0.79 构型的自动检查。

## 物理与数值约定

1. 径向坐标统一记为归一化环向磁通 `s`。DESC 原生 `rho` 会转换为
   `s=rho²`。
2. VMEC 的全网格与半网格不会混用。`bmnc`、Mercier 分项和部分剖面使用
   半网格插值。
3. 磁井标量使用

   \[
   W = \frac{V'(0)-V'(1)}{V'(0)}.
   \]

   优先读取 `vp`；缺失时从 `4π²|gmnc_{00}|` 得到 `V'(s)`。半网格端点使用
   线性外推。
4. `D_Mercier` 轴上常受坐标奇性/离散误差影响，摘要默认报告 `s≥0.05`
   区域的最小值；论文图只展示此分析区，并将纵轴固定为 `[-4,4]`。
5. `wout` 中的角度是 VMEC 角，不是 Boozer 角。程序不会把 VMEC
   `theta-zeta` 图放入默认报告或误标为 Boozer 图；Boozer 图只由成熟
   `booz_xform` 或 STELLOPT `xbooz_xform` 产生的 `boozmn` 生成。
6. VMEC 和 DESC 的同名稳定性量可能采用不同归一化。比较符号和径向结构前，
   应先核对版本与定义；本工具不擅自重标定。
7. NEO、DKES、COBRA 是独立求解器。本包负责可靠读取、质量诊断和统一绘图，
   不把 `wout` 后处理值冒充它们的计算结果。NEO 需要 `boozmn`；DKES 使用
   Boozer/直线磁力线坐标输入；COBRAVMEC 直接基于 VMEC 平衡。

## DESC 兼容性

DESC 量名会随版本演进。适配器对每一组可选量独立计算：某一量在当前版本不存在
时，会跳过它而保留其余报告。三维与 Boozer 专用高级图仍可直接使用 DESC 官方
`desc.plotting.plot_3d`、`plot_boozer_surface` 和 `plot_boozer_modes`。

## 公开代码参考

设计参考但没有复制这些项目的实现：

- [USTCstellarators/VMECdash](https://github.com/USTCstellarators/VMECdash)：
  VMEC 摘要、剖面、截面、三维磁面和磁力线的组织方式；
- [PlasmaControl/DESC](https://github.com/PlasmaControl/DESC)：
  官方 `compute`、`LinearGrid` 和 plotting API；
- [singh-jaydeep/desc-eq-visualizer](https://github.com/singh-jaydeep/desc-eq-visualizer)：
  DESC 预计算与多维结果展示思路；
- [hiddenSymmetries/simsopt](https://github.com/hiddenSymmetries/simsopt)：
  VMEC 网格处理、标量诊断和磁井定义。
- [STELLOPT NEO](https://princetonuniversity.github.io/STELLOPT/NEO)：
  `neo_out` 三种输出格式和有效波纹定义；
- [STELLOPT DKES](https://princetonuniversity.github.io/STELLOPT/DKES.html)：
  19 列 `results` 输出及变分上下界；
- [STELLOPT COBRAVMEC](https://princetonuniversity.github.io/STELLOPT/COBRAVMEC)：
  `cobra_grate` 分块格式及负本征值不稳定约定。

## 测试

```bash
pip install -e ".[dev]"
pytest -q
ruff check src tests
```

测试使用程序生成的小型 NetCDF，不依赖外部平衡文件。
