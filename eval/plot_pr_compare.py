import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# 你只需要改这里：要对比的实验目录
COMPARE_RUNS = [
    # {"name": "Only Image", "dir": "eval_output/0label/NUDT-SIRST"},
    {"name": "Image+Text", "dir": "eval_output/0304_ST_exp1/SIRST"},
]

SAVE_PATH = "eval_output/figs/pr_curve_SIRST_0305.png"
TITLE = "SIRST PR Curve "


def plot_pr_compare(runs, save_path, title="PR Curve Comparison"):
    plt.figure(figsize=(8, 6))

    for r in runs:
        run_dir = Path(r["dir"])
        npz_path = run_dir / "pr_curve_data.npz"
        if not npz_path.exists():
            print(f"[WARN] missing: {npz_path}, skip")
            continue

        data = np.load(npz_path)
        px = data["px"]           # recall grid
        py_mean = data["py_mean"] # mean precision curve (single-class就等于该类)

        label = r["name"]

        # 可选：从 metrics.json 里读 AP50 显示在 legend 上
        mjson = run_dir / "metrics.json"
        if mjson.exists():
            try:
                mj = json.loads(mjson.read_text(encoding="utf-8"))
                ap50 = mj.get("metrics", {}).get("AP50", None)
                if ap50 is not None:
                    label = f"{label} (AP50={ap50:.3f})"
            except Exception:
                pass

        plt.plot(px, py_mean, linewidth=1.5, label=label)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved PR compare plot to: {save_path}")


if __name__ == "__main__":
    plot_pr_compare(COMPARE_RUNS, SAVE_PATH, TITLE)