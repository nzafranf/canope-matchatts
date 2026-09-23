import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from modules.metrics import (canvas_at_canonical, levenshtein, onehot_at_canonical,
                             to_ids, fit_linear_probe, _pooled)

S1, S2, MUTED = "#2a78d6", "#eb6834", "#8a8985"
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
DIV = LinearSegmentedColormap.from_list("div", [S1, "#f2f1ec", S2])
SEQ = LinearSegmentedColormap.from_list("seq", ["#eef4fc", S1, "#12365f"])

_RC = {
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 8.5, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
}

@torch.no_grad()
def canonicalization_table(model, clean_texts, corrupt_fn, device,
                           levels=(1, 3, 5), n_examples=6, chunk=8):
    model.eval()
    rows = ["| lv | masukan (rusak) | keluaran model | referensi | CER | identitas |",
            "|---|---|---|---|---|---|"]
    texts = clean_texts[:n_examples]
    for lv in levels:
        srcs = [corrupt_fn(t, lv) for t in texts]
        hyps = []
        for i in range(0, len(srcs), chunk):
            ids, mask = to_ids(srcs[i:i + chunk], device)
            try:
                seqs, _, _ = model.generate(ids, mask)
                hyps += [bytes(s).decode("utf-8", errors="replace") for s in seqs]
            except RuntimeError:
                hyps += ["<gagal>"] * len(srcs[i:i + chunk])
        for s, h, t in zip(srcs, hyps, texts):
            cm = levenshtein(h, t) / max(len(t), 1)
            ci = levenshtein(s, t) / max(len(t), 1)
            mark = "" if cm <= ci else " ⚠"
            rows.append("| lv%d | `%s` | `%s` | `%s` | %.3f%s | %.3f |"
                        % (lv, s.replace("|", "\\|"), h.replace("|", "\\|"),
                           t.replace("|", "\\|"), cm, mark, ci))
    return "\n".join(rows)

def _fig_recall(model, clean_texts, corrupt_fn, device, levels, epochs):
    n_tr = max(4, int(0.7 * len(clean_texts)))
    tr, te = clean_texts[:n_tr], clean_texts[n_tr:]
    tl = [len(t.encode("utf-8")) for t in tr]
    el = [len(t.encode("utf-8")) for t in te]
    ab = sorted({b for t in clean_texts for b in t.encode("utf-8")})
    b2c = {b: i for i, b in enumerate(ab)}
    ytr = torch.tensor([b2c[b] for t in tr for b in t.encode("utf-8")])
    yte = torch.tensor([b2c[b] for t in te for b in t.encode("utf-8")])
    maj = float(torch.bincount(ytr, minlength=len(ab)).max() / ytr.numel())
    augs = {lv: [corrupt_fn(t, lv) for t in te] for lv in levels}

    res = {}
    for tag, fn in (("canvas", lambda x, l: canvas_at_canonical(model, x, l, device)),
                    ("onehot", lambda x, l: onehot_at_canonical(x, l, device))):
        tests = {"lv%d" % lv: (fn(augs[lv], el), yte) for lv in levels}
        res[tag] = fit_linear_probe(fn(tr, tl), ytr, tests, len(ab), device,
                                    epochs=epochs)
    cv = [res["canvas"]["lv%d" % lv] for lv in levels]
    oh = [res["onehot"]["lv%d" % lv] for lv in levels]

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].fill_between(levels, oh, cv, color=S1, alpha=0.10)
    ax[0].plot(levels, cv, color=S1, lw=2, marker="o", ms=6, label="kanvas")
    ax[0].plot(levels, oh, color=S2, lw=2, marker="s", ms=6, label="one-hot")
    ax[0].axhline(maj, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax[0].set_ylim(0, 1.05)
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_xlabel("tingkat korupsi")
    ax[0].set_ylabel("akurasi byte bersih")
    ax[0].set_title("probe per posisi", loc="left")
    mg = [c - o for c, o in zip(cv, oh)]
    ax[1].bar(levels, mg, color=[S1 if v > 0 else S2 for v in mg], width=0.6)
    ax[1].axhline(0, color=INK2, lw=1.1)
    for x, v in zip(levels, mg):
        ax[1].annotate("%+.3f" % v, (x, v), ha="center", fontsize=7.5,
                       xytext=(0, 4 if v >= 0 else -11),
                       textcoords="offset points", color=INK2)
    ax[1].set_xlabel("tingkat korupsi")
    ax[1].set_title("margin = kanvas − one-hot", loc="left")
    fig.tight_layout()
    return fig

@torch.no_grad()
def _fig_canvas(model, text, device):
    ids, mask = to_ids([text], device)
    L = torch.tensor([len(text.encode("utf-8"))], dtype=torch.long, device=device)
    h, mk = model.encode(ids, mask)
    zc, _ = model.decoder(h, L, source_mask=mk, strict_max=False)
    lg = model.head(zc)
    Z = zc[0].float().cpu().numpy()
    Zr = Z - Z.mean(0, keepdims=True)
    order = np.argsort(-Zr.std(0))[:160]
    pr = F.softmax(lg[0].float(), -1).cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, M, t, cm, nm in (
            (ax[0], Z[:, order].T, "zc mentah", DIV, True),
            (ax[1], Zr[:, order].T, "zc − rerata posisi", DIV, True),
            (ax[2], pr[:, :128].T, "P(byte | posisi)", SEQ, False)):
        if nm:
            v = float(np.abs(M).max()) or 1.0
            im = a.imshow(M, aspect="auto", cmap=cm, norm=TwoSlopeNorm(0, -v, v),
                          interpolation="nearest")
        else:
            im = a.imshow(M, aspect="auto", cmap=cm, vmin=0, vmax=1,
                          interpolation="nearest")
        a.set_title(t, loc="left")
        a.set_xlabel("posisi kanvas")
        a.grid(False)
        fig.colorbar(im, ax=a, fraction=0.035)
    ax[0].set_ylabel("dimensi (diurut variansi)")
    ax[2].set_ylabel("kode byte")
    fig.tight_layout()
    return fig

@torch.no_grad()
def _fig_geometry(model, clean_texts, device, text):
    ids, mask = to_ids([text], device)
    L = torch.tensor([len(text.encode("utf-8"))], dtype=torch.long, device=device)
    h, mk = model.encode(ids, mask)
    zc, _ = model.decoder(h, L, source_mask=mk, strict_max=False)
    zn = F.normalize(zc[0].float(), dim=-1)
    S = (zn @ zn.t()).cpu().numpy()
    off = S[~np.eye(len(S), dtype=bool)]

    qc = _pooled(model, clean_texts, device)
    x = qc - qc.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(x.double())
    p = (sv ** 2) / (sv ** 2).sum().clamp(min=1e-12)
    er = float(torch.exp(-(p * torch.log(p.clamp(min=1e-12))).sum()))
    ceil_ = min(len(clean_texts) - 1, qc.shape[1])
    spec = p.tolist()[:30]

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    im = ax[0].imshow(S, cmap=SEQ, vmin=float(off.min()), vmax=1.0,
                      interpolation="nearest")
    ax[0].set_title("kemiripan antar posisi kanvas", loc="left")
    ax[0].set_xlabel("j")
    ax[0].set_ylabel("i")
    ax[0].grid(False)
    fig.colorbar(im, ax=ax[0], fraction=0.04)
    ax[1].bar(range(1, len(spec) + 1), spec, color=S1, width=0.7)
    ax[1].axhline(1.0 / max(ceil_, 1), color=S2, lw=1.6, ls=(0, (4, 3)))
    ax[1].annotate("isotropik (1/%d)" % ceil_, (len(spec), 1.0 / max(ceil_, 1)),
                   ha="right", xytext=(-2, 5), textcoords="offset points",
                   color=INK2, fontsize=7.5)
    ax[1].annotate("effective rank %.1f / %d" % (er, ceil_), (0.4, 0.85),
                   xycoords="axes fraction", color=INK, fontsize=9, weight="bold")
    ax[1].set_xlabel("komponen utama ke-")
    ax[1].set_ylabel("proporsi variansi")
    ax[1].set_title("spektrum embedding kalimat", loc="left")
    fig.tight_layout()
    return fig

def preview_figures(model, clean_texts, corrupt_fn, device,
                    levels=(0, 1, 2, 3, 4, 5), probe_epochs=300, max_items=96):
    model.eval()
    texts = clean_texts[:max_items]
    with plt.rc_context(_RC):
        figs = {
            "viz/recall": _fig_recall(model, texts, corrupt_fn, device, list(levels),
                                      probe_epochs),
            "viz/canvas": _fig_canvas(model, texts[0], device),
            "viz/geometry": _fig_geometry(model, texts, device, texts[0]),
        }
    return figs
