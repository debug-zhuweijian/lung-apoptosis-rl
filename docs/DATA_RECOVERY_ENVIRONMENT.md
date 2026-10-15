# 数据恢复附加环境

2026-10-15 验证：在 Windows x64、Python 3.13.14 的现有 `study` 环境上，通过独立依赖目录运行公共来源恢复入口。训练环境的 `requirements-runtime.txt`、`requirements-lock.txt` 和已安装包保持不变。

## 依赖范围

[requirements-data-recovery.txt](../requirements-data-recovery.txt) 是现有训练环境的附加清单，不是完整 Python 环境。它覆盖 DOCX、XLSX、PDF、HDF5，以及复用来源抽取器直接导入的网络/HTML 库；`numpy==2.5.1` 和 `typing-extensions==4.16.0` 由现有训练锁提供。

| 用途 | 本次实际验证版本 |
|---|---|
| DOCX 原表读取 | python-docx 1.2.0；lxml 6.1.1 |
| XLSX 源数据读取 | openpyxl 3.1.5；et-xmlfile 2.0.0 |
| PDF 补充材料读取 | PyMuPDF 1.28.2 |
| HDF5 表达文件读取 | h5py 3.16.0；使用基础环境 NumPy |
| 复用来源抽取器的直接导入 | requests 2.34.2；beautifulsoup4 4.15.0 |
| 网络/HTML 传递依赖 | charset-normalizer 3.4.9、idna 3.18、urllib3 2.7.0、certifi 2026.6.17、soupsieve 2.9 |

这些 13 个附加包的固定版本均从 PyPI 安装到隔离目录，已逐项核对包元数据中的 10 条当前生效依赖约束。`python-docx` 的导入名是 `docx`；`PyMuPDF` 提供历史抽取器使用的 `fitz`。不要安装名称相近的 `docx` 或 `fitz` 包来替代。当前 PyMuPDF 会提示 `fitz` 接口弃用，本次固定版本仍通过读取检查。

## 在独立目录安装

在仓库根目录执行。以下目标应是新目录；后续升级另选版本目录，避免覆盖已有恢复环境。

```powershell
$recoveryPython = 'F:/anaconda/envs/study/python.exe'
$recoveryDeps = Join-Path (Get-Location) 'experiments/data-recovery-env-v1/site-packages'
& $recoveryPython -m pip install --no-deps --only-binary=:all: `
  --index-url https://pypi.org/simple `
  --target $recoveryDeps -r requirements-data-recovery.txt
```

`--target` 将包写入独立目录；`--no-deps` 防止解析器替换基础环境中的数值计算库。清单已列出基础训练锁之外的附加闭包，因此该命令的前提是基础环境已具备训练锁中的包。新电脑应先在独立 Python 环境重建基础运行时，再执行附加安装；本次没有验证全新机器上的完整训练环境重建。

## 只为恢复进程启用附加依赖

```powershell
$previousPythonPath = $env:PYTHONPATH
try {
  $env:PYTHONPATH = $recoveryDeps
  if ($previousPythonPath) {
    $env:PYTHONPATH += [IO.Path]::PathSeparator + $previousPythonPath
  }
  & $recoveryPython scripts/recover_public_expression.py --help
  & $recoveryPython scripts/recover_public_expression_bundle.py --help
  & $recoveryPython scripts/recover_public_function_links.py --help
  & $recoveryPython scripts/audit_public_crosslinks.py --help
  # 在此环境内，按各入口说明提供已核来源目录与新输出目录。
} finally {
  $env:PYTHONPATH = $previousPythonPath
}
```

第一项是按具体来源恢复的入口；`recover_public_expression_bundle.py` 是公共来源锁驱动的批量恢复入口。两者保留各自的输入合同，不能仅因环境就绪而交换产物或覆盖冻结训练资产。源文件、基因轴、测量单位、背景及配对边界仍由各恢复器和消费者验证。

## 本次验证及边界

- 四个入口均从仓库之外的工作目录执行 `--help`，返回码全部为 0；此前功能恢复入口的 `ModuleNotFoundError: docx` 已在附加环境中消除。
- 13 个附加包的实际版本符合清单；10 条生效传递依赖约束通过。
- DOCX 文本、XLSX 数值、PDF 文本、HDF5 数值分别完成临时文件写入和读取比较。
- 本机验证目录为 `experiments/20261015-cloud-review/pr1_reviewer/recovery_deps`，验证记录为同级 `recovery_environment_validation.json`、`recovery_cli_import_validation.json` 及两个 pip 安装报告。它们位于执行项目的本地产物目录，不随代码提交；集成工作树不包含这些数据文件。

以上证明附加环境可导入入口并读取相应文件格式；不等于完整源数据恢复、冻结模型兼容、正式训练或科研效果验收。安装没有改变模型、奖励、划分或 `whole_project_qualified` 判据。
