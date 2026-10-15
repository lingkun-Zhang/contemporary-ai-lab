"""从公开的完整逐轮记录重画报告两张损失图，不训练或修改报告。"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LAB = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw(histories, labels, destination, zoom_type):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({"font.size": 8.1, "axes.labelsize": 8.1, "xtick.labelsize": 7.6,
                        "ytick.labelsize": 7.6, "legend.fontsize": 7.1})
    fig, ax = plt.subplots(figsize=(3.4, 2.36), layout="constrained")
    colors = ("#1C5C91", "#BE4D27")
    for history, label, color in zip(histories, labels, colors):
        for phase, linestyle in (("train", "-"), ("val", "--")):
            ax.plot(history.epoch, history[f"{phase}_loss"], color=color, linestyle=linestyle,
                    marker=".", markersize=2, linewidth=.9,
                    label=label + (" 训练" if phase == "train" else " 验证"))
    ax.set(xlabel="训练轮次", ylabel="平均交叉熵（不含L2项）",
           xlim=(1, histories[0].epoch.max()), ylim=(0, 2.4))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, borderaxespad=0,
              frameon=False, columnspacing=.8, handlelength=1.7)
    ax.grid(alpha=.2, linewidth=.5)
    # Insets occupy the empty upper-right part of the full-range plot, rather
    # than hiding either original learning curve.
    bounds = [.60, .50, .36, .42] if zoom_type == "lr" else [.59, .40, .38, .42]
    inset = ax.inset_axes(bounds)
    limits = (20, 30) if zoom_type == "lr" else (80, 100)
    selected = histories[:1] if zoom_type == "lr" else histories
    values = []
    for history, color in zip(selected, colors):
        local = history.loc[history.epoch.between(*limits)]
        values.extend(local.val_loss.to_list())
        inset.plot(local.epoch, local.val_loss, color=color, marker=".", markersize=3, linewidth=.85)
    inset.set_xlim(*limits)
    padding = .00045 if zoom_type == "lr" else .0012
    inset.set_ylim(min(values) - padding, max(values) + padding)
    inset.set_title("验证损失局部放大", fontsize=6.5, pad=2)
    inset.tick_params(labelsize=5.8, length=2, pad=1)
    inset.set_xticks([20, 25, 30] if zoom_type == "lr" else [80, 90, 100])
    inset.set_yticks([.263, .266] if zoom_type == "lr" else [.26, .28, .29])
    inset.grid(alpha=.2, linewidth=.4)
    fig.savefig(destination, dpi=450)
    plt.close(fig)
    return {"epochs": list(limits), "metric": "val_loss", "run_count": len(selected),
            "y_range": [min(values) - padding, max(values) + padding], "data_changed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=LAB / "results/reproduction_20261014")
    parser.add_argument("--output", type=Path, default=LAB / "local_plots/report")
    args = parser.parse_args()
    records, output = args.results.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Output already exists; choose a new directory.")
    groups = [
        ("learning_rate", ["0.001", "0.0001"], "lr", [
            ("mlp_lr001_e30", dict(max_features=5000, hidden_units=50, epochs=30, alpha=.001, learning_rate_init=.001)),
            ("mlp_lr0001_e30", dict(max_features=5000, hidden_units=50, epochs=30, alpha=.001, learning_rate_init=.0001))]),
        ("vocabulary", ["5000词", "10000词"], "vocab", [
            ("mlp_width100_s42", dict(max_features=5000, hidden_units=100, epochs=100, alpha=.005, learning_rate_init=.0001)),
            ("mlp_vocab10000", dict(max_features=10000, hidden_units=100, epochs=100, alpha=.005, learning_rate_init=.0001))]),
    ]
    loaded_groups = []
    for name, labels, zoom, configurations in groups:
        histories, inputs = [], []
        for run_id, expected in configurations:
            directory = records / "runs" / run_id
            config_file, history_file = directory / "config.json", directory / "history.csv"
            config = json.loads(config_file.read_text(encoding="utf-8-sig"))
            common = dict(model="MLP", seed=42, ngram_range=[1, 1], sublinear_tf=False,
                          weight_class=None, relative_weight=1.0, batch_size=64)
            for key, value in {**common, **expected}.items():
                if config[key] != value:
                    raise ValueError(f"{run_id}: unexpected {key}")
            history = pd.read_csv(history_file)
            if history.epoch.tolist() != list(range(1, expected["epochs"] + 1)):
                raise ValueError("Incomplete history: " + run_id)
            histories.append(history)
            for path in (config_file, history_file):
                inputs.append({"file": Path(os.path.relpath(path, output)).as_posix(), "sha256": sha(path)})
        loaded_groups.append((name, labels, zoom, histories, inputs))
    output.mkdir(parents=True)
    figures = []
    for name, labels, zoom, histories, inputs in loaded_groups:
        image = output / (name + ".png")
        inset = draw(histories, labels, image, zoom)
        figures.append({"file": image.name, "sha256": sha(image), "input_files": inputs,
                        "inset": inset, "smoothed": False, "points_removed": 0})
    record = {"figures": figures, "training_performed": False, "stored_records_modified": False,
              "renderer_sha256": sha(Path(__file__)), "note": "Portable renderer; font and library versions may change image bytes."}
    (output / "figure_sources.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figures": len(figures), "training_performed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
