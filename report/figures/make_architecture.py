"""Draw the MorphoCLIP architecture figure (report/figures/architecture.pdf).

Run from the repo root: uv run python report/figures/make_architecture.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).with_name("architecture.pdf")

IMG = "#1b7f79"
TXT = "#4b4fbf"
LOSS = "#c47a10"
FROZEN = "#efefef"
INK = "#222222"
GREY = "#555555"


def box(ax, x, y, w, h, title, lines, color, *, frozen=False, dashed=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            linewidth=1.5,
            edgecolor=color,
            facecolor=FROZEN if frozen else "white",
            linestyle="--" if dashed else "-",
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.22,
        title,
        ha="center",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color=INK,
    )
    for i, line in enumerate(lines):
        ax.text(
            x + w / 2,
            y + h - 0.62 - 0.30 * i,
            line,
            ha="center",
            va="top",
            fontsize=7.6,
            color="#444444",
        )
    if frozen:
        ax.text(
            x + w / 2,
            y + 0.1,
            "frozen",
            ha="center",
            va="bottom",
            fontsize=7,
            style="italic",
            color="#666666",
        )


def arrow(ax, p0, p1, color=GREY):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.2,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(14.0, 5.4))
    ax.set_xlim(0, 14.0)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    # Bands
    ax.add_patch(
        FancyBboxPatch(
            (0.15, 2.85),
            11.75,
            2.4,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            linewidth=1.1,
            edgecolor=IMG,
            facecolor="#eef7f6",
            linestyle="--",
        )
    )
    ax.text(
        0.35,
        5.12,
        "IMAGE BRANCH   (training unit: one well = S sites x 5 channels)",
        fontsize=9,
        fontweight="bold",
        color=IMG,
        va="top",
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.15, 0.45),
            8.3,
            2.1,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            linewidth=1.1,
            edgecolor=TXT,
            facecolor="#eeeffa",
            linestyle="--",
        )
    )
    ax.text(
        0.35,
        2.42,
        "TEXT BRANCH   (one prompt per perturbation)",
        fontsize=9,
        fontweight="bold",
        color=TXT,
        va="top",
    )

    # Image row
    yb, hb = 3.05, 1.7
    xs = [0.4, 2.55, 4.7, 6.95, 8.75, 10.3]
    ws = [1.85, 1.85, 1.95, 1.5, 1.25, 1.45]
    box(
        ax,
        xs[0],
        yb,
        ws[0],
        hb,
        "Site image",
        ["5 channels", "1080 x 1080 to 224", "each channel as RGB"],
        IMG,
    )
    box(
        ax,
        xs[1],
        yb,
        ws[1],
        hb,
        "DINOv3 ViT-L/16",
        ["300M params", "CLS per channel", "5 x 1024, cached"],
        IMG,
        frozen=True,
    )
    box(
        ax,
        xs[2],
        yb,
        ws[2],
        hb,
        "CrossChannelFormer",
        ["1 layer, 4 heads", "channel embed, agg. token", "5 x 1024 to 1024 / site"],
        IMG,
    )
    box(
        ax,
        xs[3],
        yb,
        ws[3],
        hb,
        "Site pooling",
        ["masked mean", "over S sites", "1024 per well"],
        IMG,
    )
    box(ax, xs[4], yb, ws[4], hb, "Proj. head", ["1024 to 512", "L2 norm"], IMG)
    box(
        ax,
        xs[5],
        yb,
        ws[5],
        hb,
        "Plate offset",
        ["optional", "minus plate offset", "re-normalize"],
        IMG,
        dashed=True,
    )
    ym = yb + hb / 2
    for i in range(5):
        arrow(ax, (xs[i] + ws[i], ym), (xs[i + 1], ym))

    # Text row
    yt, ht = 0.65, 1.55
    box(
        ax,
        0.4,
        yt,
        1.85,
        ht,
        "Prompt template",
        ["compound / CRISPR / ORF", "cell line, target,", "function, SMILES"],
        TXT,
    )
    box(
        ax,
        2.75,
        yt,
        2.3,
        ht,
        "BioClinical ModernBERT",
        ["150M params", "CLS, 768-d, cached"],
        TXT,
        frozen=True,
    )
    box(ax, 5.55, yt, 1.25, ht, "Proj. head", ["768 to 512", "L2 norm"], TXT)
    yn = yt + ht / 2
    arrow(ax, (2.25, yn), (2.75, yn))
    arrow(ax, (5.05, yn), (5.55, yn))

    # Losses
    lx, lw = 12.25, 1.6
    box(ax, lx, 3.35, lw, 1.3, "CWCL", ["image vs text", "soft labels"], LOSS)
    box(ax, lx, 1.55, lw, 1.3, "Replicate loss", ["image vs image", "same perturbation"], LOSS)
    xe = xs[5] + ws[5]
    arrow(ax, (xe, ym + 0.1), (lx, 4.0), color=IMG)
    arrow(ax, (xe, ym - 0.1), (lx, 2.2), color=IMG)
    arrow(ax, (6.8, yn), (lx, 3.75), color=TXT)

    ax.text(
        7.0,
        0.12,
        "Trained: CrossChannelFormer, two projection heads, and logit scale (about 14M parameters). "
        "Frozen: DINOv3 and ModernBERT, run once and cached.",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#333333",
    )

    fig.savefig(OUT, bbox_inches="tight")
    matplotlib.pyplot.close(fig)


if __name__ == "__main__":
    main()
