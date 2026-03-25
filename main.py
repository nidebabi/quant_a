import subprocess
import sys

python_exe = sys.executable

steps = [
    [python_exe, "download_data.py"],
    [python_exe, "build_features.py"],
    [python_exe, "build_labels.py"],
    [python_exe, "rank_stocks.py"],
    [python_exe, "backtest.py"],
]

for step in steps:
    print("=" * 60)
    print("运行:", " ".join(step))
    ret = subprocess.run(step)
    if ret.returncode != 0:
        sys.exit(ret.returncode)

print("=" * 60)
print("全部完成")