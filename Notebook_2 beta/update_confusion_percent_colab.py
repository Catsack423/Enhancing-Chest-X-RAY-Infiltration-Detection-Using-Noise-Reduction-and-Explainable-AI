# Run after the setup cell of your EXISTING CNN notebook. Keeps the current run config.
from pathlib import Path
import ast, importlib
import shared.reporting as reporting
module_path = Path(reporting.__file__)
old = module_path.read_text(encoding="utf-8")
node = next(n for n in ast.parse(old).body if isinstance(n, ast.FunctionDef) and n.name == "plot_confusion")
lines = old.splitlines(keepends=True)
replacement = 'def plot_confusion(matrix, path, title, ax=None):\n    own = ax is None\n    if own:\n        fig, ax = plt.subplots(figsize=(4.5, 4))\n    matrix = np.asarray(matrix)\n    totals = matrix.sum(axis=1, keepdims=True)\n    percentages = np.divide(matrix * 100.0, totals, out=np.zeros(matrix.shape, dtype=float), where=totals != 0)\n    ax.imshow(percentages, cmap="Blues", vmin=0, vmax=100)\n    ax.set_xticks([0, 1], ["Normal", "Infiltration"])\n    ax.set_yticks([0, 1], ["Normal", "Infiltration"])\n    ax.set_xlabel("Predicted (count / % of actual class)")\n    ax.set_ylabel("Actual")\n    ax.set_title(title)\n    for (y, x), value in np.ndenumerate(matrix):\n        ax.text(x, y, f"{value:,}\\n({percentages[y, x]:.2f}%)", ha="center", va="center", color="black", bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"})\n    if own:\n        Path(path).parent.mkdir(parents=True, exist_ok=True)\n        fig.tight_layout()\n        fig.savefig(path, dpi=150)\n        plt.close(fig)'
module_path.write_text("".join(lines[:node.lineno-1]) + replacement + "\n" + "".join(lines[node.end_lineno:]), encoding="utf-8")
importlib.reload(reporting)
refresh_reports = reporting.refresh_reports
status = refresh_reports(exp)
display(status)
# Also regenerate each individual condition image from its measured confusion matrix.
import json
for condition in exp.condition_ids:
    folder = exp.results("CNN", condition)
    if (folder / "complete.json").is_file():
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        reporting.plot_confusion(metrics["confusion_matrix"], folder / "confusion_matrix.png", condition)
from IPython.display import Image, display
display(Image(filename=str(exp.output_root / "CNN/comparison/confusion_matrices_all.png")))
print("Saved to:", exp.output_root)
