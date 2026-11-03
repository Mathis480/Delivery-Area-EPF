"""
Shared figure style for the paper, matching the existing figures in Contribution/SVG
(STIX serif font, full axis frame, dashed light-gray grid, bold labels and panel titles).
Model colors are fixed here and must be used identically in every figure.
"""
import matplotlib

MODEL_COLORS = {
    "Naive": "#000000",
    "LASSO": "#2ca02c",
    "LASSO+N": "#98df8a",
    "cSVR (filter)": "#7f7f7f",
    "cSVR": "#ff7f0e",
    "FA-NN": "#1f77b4",
}
MODEL_LINESTYLES = {"Naive": "--", "cSVR (filter)": ":"}
SEQUENTIAL_CMAP = "viridis"  # as in Figure 2 (trade counts)

TEXT_WIDTH_IN = 6.3  # elsarticle preprint text width


def apply() -> None:
    matplotlib.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.titlesize": 10,
        "axes.edgecolor": "#000000",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#b0b0b0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "lines.linewidth": 1.4,
        "legend.frameon": True,
        "legend.edgecolor": "#cccccc",
        "legend.fancybox": False,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
    })
