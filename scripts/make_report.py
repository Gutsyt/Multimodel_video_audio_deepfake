"""Draw the README figures and fill its results section from the REAL training artefacts.

    python scripts/make_report.py --cache CACHE [--e2e E2E_JSON_DIR]

Reads the cached embeddings (same splits as train_heads.py), the trained heads in avdd/assets and,
optionally, the end-to-end JSON files written by evaluate_folder.py. Writes docs/figures/*.svg and
replaces the block between the RESULTS markers in README.md. Nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).parent))
from train_heads import _group_scores, _load, split_of

from avdd.common.head import ASSETS, LinearHead

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs" / "figures"
REAL, FAKE, MID, INK, INK2, GRID = "#2a6fb0", "#d2462b", "#b9b3a5", "#15171c", "#5c6068", "#e6e2d9"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "svg.fonttype": "none",
    "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
})
SETS = {
    "visual": (["vis_ff", "vis_wdf", "vis_celeb"], {"in_domain_test": ("Held-out test", ["vis_ff", "vis_wdf", "vis_celeb"]),
               "wilddeepfake_test": ("WildDeepfake test", ["vis_wdf_test"]), "celebdf_test": ("Celeb-DF test", ["vis_celeb_test"])}),
    "audio": (["aud_ASVspoof2019_LA", "aud_ASVspoof5", "aud_InTheWild"], {"in_domain_test": ("Held-out test (unseen speakers)", ["aud_ASVspoof2019_LA", "aud_ASVspoof5", "aud_InTheWild"])}),
}
NAME = {"visual": "Face check", "audio": "Voice check"}


def scores(cache: Path, kind: str, names: list[str], test_split: bool) -> tuple[Any, Any]:
    head = LinearHead(ASSETS / f"{kind}_head.npz")
    x, y, g = _load(cache, names, int(head.meta["layer"]))
    if test_split:
        keep = np.array([split_of(k) == "test" for k in g])
        x, y, g = x[keep], y[keep], g[keep]
    p = head.prob(x)
    return _group_scores(p, y, g) if kind == "visual" else (p, y)


def num(v: float | None, spec: str) -> str:
    """Format a metric; a metric that could not be computed (single-class set) shows as a dash."""
    return "–" if v is None or np.isnan(v) else format(v, spec)


def save(fig: Any, name: str) -> str:
    fig.tight_layout()
    fig.savefig(FIG / name, format="svg")
    plt.close(fig)
    return f"docs/figures/{name}"


def fig_data(cache: Path) -> tuple[str, list[list[str]]]:
    rows = []
    for f in sorted(cache.glob("*.npz")):
        y = np.load(f)["y"]
        rows.append((f.stem, int((y == 0).sum()), int((y == 1).sum())))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.2))
    lab = [r[0].replace("vis_", "face · ").replace("aud_", "voice · ") for r in rows]
    yy = np.arange(len(rows))
    ax.barh(yy - 0.19, [r[1] for r in rows], 0.36, color=REAL, label="genuine")
    ax.barh(yy + 0.19, [r[2] for r in rows], 0.36, color=FAKE, label="fake")
    for i, r in enumerate(rows):
        ax.text(r[1], i - 0.19, f" {r[1]:,}", va="center", fontsize=8.5, color=INK2)
        ax.text(r[2], i + 0.19, f" {r[2]:,}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(yy, lab)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("samples embedded (face crops or 4 s speech windows)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    return save(fig, "data.svg"), [[r[0], f"{r[1]:,}", f"{r[2]:,}"] for r in rows]


def fig_layers() -> str:
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.8))
    for ax, kind in zip(axs, ("visual", "audio"), strict=True):
        meta = LinearHead(ASSETS / f"{kind}_head.npz").meta
        best: dict[int, float] = {}
        for s in meta["metrics"]["search"]:
            best[s["layer"]] = max(best.get(s["layer"], 0), s["val_auc"])
        ks = sorted(best)
        ax.plot(ks, [best[k] for k in ks], color=INK, lw=2, marker="o", ms=6)
        ax.scatter([meta["layer"]], [best[meta["layer"]]], s=90, color=INK, zorder=3)
        ax.annotate(f"chosen: layer {meta['layer']}", (meta["layer"], best[meta["layer"]]), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9, color=INK2)
        ax.set_title(NAME[kind], loc="left", fontsize=11, color=INK)
        ax.set_xlabel("transformer layer used as feature")
        ax.set_ylabel("validation AUC")
        ax.set_xticks(ks)
    return save(fig, "layers.svg")


def fig_roc(cache: Path) -> str:
    panels = [(k, t, lbl, names) for k, (_, tests) in SETS.items() for t, (lbl, names) in tests.items()]
    cols = 2
    rows = -(-len(panels) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(8, 3.6 * rows), squeeze=False)
    for ax in axs.flat[len(panels):]:
        ax.set_visible(False)
    for ax, (kind, t, lbl, names) in zip(axs.flat, panels, strict=False):
        p, y = scores(cache, kind, names, t == "in_domain_test")
        meta = LinearHead(ASSETS / f"{kind}_head.npz").meta
        fpr, tpr, _ = roc_curve(y, p)
        m = meta["metrics"]["tests"][t]
        ax.plot([0, 1], [0, 1], color=GRID, lw=1.2, ls="--")
        ax.plot(fpr, tpr, color=INK, lw=2)
        op = (float((p[y == 0] >= meta["t_fake"]).mean()), float((p[y == 1] >= meta["t_fake"]).mean()))
        ax.scatter([op[0]], [op[1]], s=60, color=FAKE, zorder=3, edgecolor="white", linewidth=2)
        ax.annotate(f"at the flag threshold:\n{op[0]:.1%} of genuine flagged\n{op[1]:.0%} of fakes caught", op, xytext=(0.38, 0.12),
                    textcoords="axes fraction", fontsize=8.5, color=INK2, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
        ax.set_title(f"{NAME[kind]} · {lbl}\nAUC {num(m['auc'], '.3f')} · EER {num(m['eer'], '.1%')}", loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("genuine flagged (FPR)")
        ax.set_ylabel("fakes caught (TPR)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
    return save(fig, "roc.svg")


def fig_scores(cache: Path) -> str:
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, kind in zip(axs, ("visual", "audio"), strict=True):
        p, y = scores(cache, kind, SETS[kind][1]["in_domain_test"][1], True)
        meta = LinearHead(ASSETS / f"{kind}_head.npz").meta
        bins = np.linspace(0, 1, 26)
        ax.hist(p[y == 0], bins, color=REAL, alpha=0.85, label=f"genuine (n={int((y == 0).sum())})", rwidth=0.9)
        ax.hist(p[y == 1], bins, color=FAKE, alpha=0.65, label=f"fake (n={int((y == 1).sum())})", rwidth=0.9)
        ax.axvline(meta["t_real"], color=INK, lw=1.2, ls=":", label=f"authentic at or below {meta['t_real']:.2f}")
        ax.axvline(meta["t_fake"], color=INK, lw=1.2, ls="--", label=f"manipulated at or above {meta['t_fake']:.2f}")
        ax.set_title(f"{NAME[kind]} · held-out test", loc="left", fontsize=11, color=INK)
        ax.set_xlabel("calibrated P(fake)")
        ax.set_ylabel("count")
        ax.legend(loc="upper center", fontsize=8.5)
    return save(fig, "scores.svg")


E2E_LABEL = {"A_ff": "FaceForensics++ videos", "B_wdf": "WildDeepfake videos", "C_itw": "In-the-Wild speech", "D_av": "Genuine video + genuine speech"}


def fig_e2e(e2e: Path) -> tuple[str, list[list[str]]]:
    rows: list[tuple[str, dict[str, float], int]] = []
    for f in sorted(e2e.glob("*.json")):
        data = json.loads(f.read_text())
        for lab in ("real", "fake"):
            vs = [d["verdict"] for d in data if d["label"] == lab]
            if vs:
                rows.append((f"{E2E_LABEL.get(f.stem, f.stem)} · {'genuine' if lab == 'real' else 'fake'}",
                             {v: vs.count(v) / len(vs) for v in ("likely_authentic", "inconclusive", "likely_manipulated", "insufficient_evidence")}, len(vs)))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 1.3))
    for i, (_, frac, _) in enumerate(rows):
        left = 0.0
        for v, col, hatch in (("likely_authentic", REAL, None), ("inconclusive", MID, "///"), ("likely_manipulated", FAKE, None), ("insufficient_evidence", "#e9e6df", "..")):
            w = frac[v]
            if w > 0:
                ax.barh(i, w - 0.004, left=left, color=col, hatch=hatch, edgecolor="white", linewidth=0)
                if w >= 0.07:
                    ax.text(left + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=8.5, color="white" if col in (REAL, FAKE) else INK)
            left += w
    ax.set_yticks(range(len(rows)), [f"{r[0]} (n={r[2]})" for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=REAL, label="likely authentic"), Patch(facecolor=MID, hatch="///", label="inconclusive"),
                       Patch(color=FAKE, label="likely manipulated"), Patch(facecolor="#e9e6df", hatch="..", label="not enough to judge")],
              loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=4, fontsize=8.5)
    table = [[r[0], str(r[2])] + [f"{r[1][v]:.1%}" for v in ("likely_authentic", "inconclusive", "likely_manipulated", "insufficient_evidence")] for r in rows]
    return save(fig, "e2e.svg"), table


def md_table(head: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--e2e", type=Path)
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    data_fig, data_rows = fig_data(a.cache)
    metric_rows = []
    for kind in ("visual", "audio"):
        meta = LinearHead(ASSETS / f"{kind}_head.npz").meta
        for t, m in {"validation": meta["metrics"]["validation"], **meta["metrics"]["tests"]}.items():
            lbl = "Validation (thresholds set here)" if t == "validation" else SETS[kind][1][t][0]
            metric_rows.append([NAME[kind], lbl, f"{m['n_real']:,} / {m['n_fake']:,}", num(m["auc"], ".3f"), num(m["eer"], ".1%"),
                                f"{m['genuine_flagged']:.1%}", f"{m['genuine_cleared']:.1%}", f"{m['fake_flagged']:.1%}"])
    th = [[NAME[k], str(LinearHead(ASSETS / f"{k}_head.npz").meta["layer"]), f"{LinearHead(ASSETS / f'{k}_head.npz').t_real:.3f}",
           f"{LinearHead(ASSETS / f'{k}_head.npz').t_fake:.3f}"] for k in ("visual", "audio")]
    parts = [
        "### Training data actually used\n", f"![Samples per source]({data_fig})\n",
        md_table(["Cache", "Genuine", "Fake"], data_rows),
        "\n\n### Which layer of each backbone\n", f"![Validation AUC per layer]({fig_layers()})\n",
        md_table(["Check", "Layer", "t_real (authentic at or below)", "t_fake (manipulated at or above)"], th),
        "\n\n### Detection quality\nScores are aggregated per video / sequence for the face check (median over frames), per clip for the voice check.\n\n",
        f"![ROC curves]({fig_roc(a.cache)})\n", f"![Score distributions]({fig_scores(a.cache)})\n",
        md_table(["Check", "Set", "Genuine / fake", "AUC", "EER", "Genuine flagged", "Genuine cleared", "Fakes flagged"], metric_rows),
    ]
    if a.e2e and any(a.e2e.glob("*.json")):
        e2e_fig, e2e_rows = fig_e2e(a.e2e)
        parts += ["\n\n### End to end: held-out files through the real app pipeline\nEach file was run by `scripts/evaluate_folder.py`, i.e. the same probe → face/voice → verdict code the server runs.\n\n",
                  f"![Verdicts per held-out set]({e2e_fig})\n",
                  md_table(["Set", "Files", "Likely authentic", "Inconclusive", "Likely manipulated", "Insufficient evidence"], e2e_rows)]
    readme = ROOT / "README.md"
    body = "<!-- RESULTS:START (generated by scripts/make_report.py; do not edit by hand) -->\n" + "\n".join(parts) + "\n<!-- RESULTS:END -->"
    readme.write_text(re.sub(r"<!-- RESULTS:START.*?<!-- RESULTS:END -->", lambda _: body, readme.read_text(), flags=re.DOTALL))
    print("README results updated;", ", ".join(p.name for p in FIG.glob("*.svg")))


if __name__ == "__main__":
    main()
