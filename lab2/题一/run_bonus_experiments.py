"""一键复现本轮四组补充实验，原有样例和实验仍可分别运行。"""

import argparse
import subprocess
import sys
from pathlib import Path


STUDIES = {
    "bfs": "run_bfs_study.py",
    "reopen": "run_reopen_study.py",
    "connectivity": "run_connectivity_study.py",
    "audit": "run_evidence_audit.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("all", *STUDIES), default="all")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    selected = STUDIES if args.study == "all" else {args.study: STUDIES[args.study]}
    for name, filename in selected.items():
        print(f"\n正在运行补充实验：{name}", flush=True)
        subprocess.run([sys.executable, str(base / filename)], cwd=str(base), check=True)
    print("\n所选补充实验已完成，结果分别保存在 results/bonus_* 中。")


if __name__ == "__main__":
    main()
