"""Serial entry point for the three added Q3 studies; never times in parallel."""
import subprocess
import sys
from pathlib import Path


def main():
    base = Path(__file__).resolve().parent
    for filename in ('run_sampling_confirmation.py', 'run_correlation_study.py', 'run_capacity_study.py'):
        print(f'Running {filename}', flush=True)
        subprocess.run([sys.executable, '-B', str(base / filename)], cwd=base, check=True)


if __name__ == '__main__':
    main()
