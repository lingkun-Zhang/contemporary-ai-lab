"""Plot MLP loss and accuracy from saved epoch records."""
import argparse
from pathlib import Path

from record_all_experiments import plot_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new plot directory to preserve previous figures.")
    histories = sorted((args.results / "runs").glob("*/history.csv"))
    if not histories:
        raise ValueError("No histories found.")
    import shutil
    for history in histories:
        destination = args.output / history.parent.name
        destination.mkdir(parents=True, exist_ok=False)
        shutil.copy2(history, destination / "history.csv")
        plot_history(destination)
    print(f"Saved {len(histories)} train/validation curve pairs to {args.output}.")


if __name__ == "__main__":
    main()
