import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
})

# ── colour palette ────────────────────────────────────────────────────────────
C_NEUTRAL = "#F0F0F0"
C_TRAIN   = "#FAD7A0"   # warm yellow  – trainable
C_FROZEN  = "#AED6F1"   # cool blue    – frozen
C_HEAD_FT = "#F1948A"   # salmon       – head trained in fine-tune
C_EDGE    = "#555555"
C_TEXT    = "#1C1C1C"
C_SUB     = "#666666"
C_ARROW   = "#444444"

# ── geometry ──────────────────────────────────────────────────────────────────
BW, BH  = 0.58, 0.10    # block width / height (axes-fraction units)
GAP     = 0.055          # vertical gap between blocks
CX      = 0.50           # horizontal centre of each panel

def block_y(rank, n_blocks=5):
    """Bottom y of block at 'rank' (0 = top), given n_blocks stacked."""
    total = n_blocks * BH + (n_blocks - 1) * GAP
    top   = (1.0 - total) / 2 + total
    return top - (rank + 1) * BH - rank * GAP

def draw_block(ax, rank, label, sub, color):
    x  = CX - BW / 2
    y  = block_y(rank)
    pad = 0.015
    box = FancyBboxPatch(
        (x, y), BW, BH,
        boxstyle=f"round,pad={pad}",
        facecolor=color, edgecolor=C_EDGE, linewidth=1.1,
        zorder=2,
    )
    ax.add_patch(box)
    cy = y + BH / 2
    if sub:
        ax.text(CX, cy + BH * 0.14, label,
                ha="center", va="center",
                fontsize=10, fontweight="bold", color=C_TEXT, zorder=3)
        ax.text(CX, cy - BH * 0.18, sub,
                ha="center", va="center",
                fontsize=7.8, color=C_SUB, zorder=3)
    else:
        ax.text(CX, cy, label,
                ha="center", va="center",
                fontsize=10, fontweight="bold", color=C_TEXT, zorder=3)

def draw_arrow(ax, from_rank, to_rank):
    """Draw a clean downward arrow between two blocks."""
    x  = CX
    y0 = block_y(from_rank)          # bottom of upper block
    y1 = block_y(to_rank) + BH       # top of lower block
    ax.annotate(
        "", xy=(x, y0), xytext=(x, y1),
        arrowprops=dict(
            arrowstyle="->, head_width=0.25, head_length=0.4",
            color=C_ARROW, lw=1.2,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=1,
    )

def draw_freeze_brace(ax, from_rank, to_rank):
    """Vertical bar with 'frozen' label beside the frozen blocks."""
    bx   = CX - BW / 2 - 0.07
    y_lo = block_y(to_rank)
    y_hi = block_y(from_rank) + BH
    ax.plot([bx, bx], [y_lo, y_hi], color="#2980B9", lw=2.2, solid_capstyle="round")
    ax.text(bx - 0.025, (y_lo + y_hi) / 2, "frozen",
            ha="right", va="center", fontsize=8.5,
            color="#2980B9", fontweight="bold", rotation=90)

def make_panel(ax, title, colors, show_brace):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", color=C_TEXT, pad=10)

    blocks = [
        ("Input",              "N layers × 18 features",  colors[0]),
        ("Layer Tokenizer",    "Linear  ·  pos-emb  ·  CLS",  colors[1]),
        ("Transformer Encoder","2 layers  ·  8 heads  ·  d = 512", colors[2]),
        ("Regression Head",    "Linear  512 → 2",         colors[3]),
        ("Output",             "Latency  ·  Area",        colors[4]),
    ]

    for rank, (label, sub, color) in enumerate(blocks):
        draw_block(ax, rank, label, sub, color)

    for r in range(len(blocks) - 1):
        draw_arrow(ax, r + 1, r)   # arrow from block below to block above (y decreases downward)

    if show_brace:
        draw_freeze_brace(ax, 1, 2)   # ranks 1 and 2 = tokenizer + transformer

    # legend
    handles = [
        mpatches.Patch(facecolor=colors[1], edgecolor=C_EDGE,
                       label="Frozen" if show_brace else "Trainable"),
        mpatches.Patch(facecolor=colors[3], edgecolor=C_EDGE,
                       label="Trainable"),
    ]
    if show_brace:
        ax.legend(handles=handles, loc="lower right", fontsize=8.5,
                  framealpha=0.92, edgecolor="#ccc", handlelength=1.2)

# ── build figure ──────────────────────────────────────────────────────────────
fig, (ax_pre, ax_ft) = plt.subplots(1, 2, figsize=(10, 6.5))
fig.patch.set_facecolor("white")
fig.subplots_adjust(wspace=0.18)

make_panel(
    ax_pre,
    "Pre-training\n(base model – all processes)",
    colors=[C_NEUTRAL, C_TRAIN, C_TRAIN, C_TRAIN, C_NEUTRAL],
    show_brace=False,
)

make_panel(
    ax_ft,
    "Fine-tuning\n(GF22nm – head only)",
    colors=[C_NEUTRAL, C_FROZEN, C_FROZEN, C_HEAD_FT, C_NEUTRAL],
    show_brace=True,
)

out = "new_results_plots/model_finetune_diagram.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"Saved: {out}")
