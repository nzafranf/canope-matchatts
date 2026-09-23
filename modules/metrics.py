
import torch
import torch.nn.functional as F

from modules.model import gaussian_resample, PAD_CLASS
from modules.utils import length_to_mask


def to_ids(texts, device, max_len=1024):
    seqs = [list(t.encode("utf-8"))[:max_len] or [32] for t in texts]
    L = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), L, dtype=torch.long)
    mask = torch.zeros(len(seqs), L, dtype=torch.bool)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = True
    return ids.to(device), mask.to(device)


def levenshtein(a, b) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@torch.no_grad()
def canvas_at_canonical(model, texts, canon_lens, device, chunk=8):
    feats = []
    for i in range(0, len(texts), chunk):
        bt, bl = texts[i:i + chunk], canon_lens[i:i + chunk]
        ids, mask = to_ids(bt, device)
        L = torch.tensor(bl, dtype=torch.long, device=device)
        h, _ = model.encode(ids, mask)
        zc, _ = model.decoder(h, L, source_mask=mask, strict_max=False)
        for b in range(len(bt)):
            feats.append(zc[b, : bl[b]].float().cpu())
    return torch.cat(feats)


@torch.no_grad()
def onehot_at_canonical(texts, canon_lens, device, tau_r=0.3, chunk=8):
    feats = []
    for i in range(0, len(texts), chunk):
        bt, bl = texts[i:i + chunk], canon_lens[i:i + chunk]
        ids, mask = to_ids(bt, device)
        L = torch.tensor(bl, dtype=torch.long, device=device)
        oh = F.one_hot(ids, num_classes=256).float()
        c, _ = gaussian_resample(oh, mask, L, tau_r=tau_r, target_max=int(L.max()))
        for b in range(len(bt)):
            feats.append(c[b, : bl[b]].float().cpu())
    return torch.cat(feats)


def fit_linear_probe(Xtr, ytr, tests, n_cls, device, epochs=400, lr=1e-2, wd=1e-4):
    Xtr, ytr = Xtr.to(device), ytr.to(device)
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp(min=1e-5)
    Xn = (Xtr - mu) / sd
    W = torch.zeros(Xn.shape[1], n_cls, device=device, requires_grad=True)
    b = torch.zeros(n_cls, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(Xn @ W + b, ytr).backward()
        opt.step()
    res = {}
    with torch.no_grad():
        res["train"] = float(((Xn @ W + b).argmax(1) == ytr).float().mean())
        for name, (Xe, ye) in tests.items():
            Xe = (Xe.to(device) - mu) / sd
            res[name] = float(((Xe @ W + b).argmax(1) == ye.to(device)).float().mean())
    return res


@torch.no_grad()
def _labels(texts, b2c):
    return torch.tensor([b2c[b] for t in texts for b in t.encode("utf-8")])


def decodability_probe(model, clean_texts, corrupt_fn, device,
                       train_frac=0.7, levels=(1, 3, 5), epochs=400):
    model.eval()
    n_tr = int(len(clean_texts) * train_frac)
    tr, te = clean_texts[:n_tr], clean_texts[n_tr:]
    tr_len = [len(t.encode("utf-8")) for t in tr]
    te_len = [len(t.encode("utf-8")) for t in te]

    all_bytes = sorted({b for t in clean_texts for b in t.encode("utf-8")})
    b2c = {b: i for i, b in enumerate(all_bytes)}
    n_cls = len(all_bytes)
    ytr, yte = _labels(tr, b2c), _labels(te, b2c)
    counts = torch.bincount(ytr, minlength=n_cls).float()
    majority = float(counts.max() / counts.sum())

    aug_sets = {f"lv{lv}": [corrupt_fn(t, lv) for t in te] for lv in levels}

    out = {"probe/majority_baseline": majority}
    for tag, feat_clean, feat_fn in (
        ("canvas", canvas_at_canonical(model, tr, tr_len, device),
         lambda ts: canvas_at_canonical(model, ts, te_len, device)),
        ("onehot", onehot_at_canonical(tr, tr_len, device),
         lambda ts: onehot_at_canonical(ts, te_len, device)),
    ):
        tests = {"clean": (feat_fn(te), yte)}
        for name, ts in aug_sets.items():
            tests[name] = (feat_fn(ts), yte)
        r = fit_linear_probe(feat_clean, ytr, tests, n_cls, device, epochs=epochs)
        for k, v in r.items():
            out[f"probe/{tag}_{k}"] = v

    for name in ["clean"] + list(aug_sets):
        out[f"probe/margin_{name}"] = (out[f"probe/canvas_{name}"]
                                       - out[f"probe/onehot_{name}"])
    return out


@torch.no_grad()
def decode_cer(model, clean_texts, corrupt_fn, device, levels=(0, 1, 3),
               max_items=64, chunk=8):
    model.eval()
    out = {}
    for lv in levels:
        srcs = [t if lv == 0 else corrupt_fn(t, lv) for t in clean_texts[:max_items]]
        refs = clean_texts[:max_items]
        tot_err = tot_len = 0
        n_fail = 0
        hyp_chars = ref_chars = 0
        for i in range(0, len(srcs), chunk):
            block = refs[i:i + chunk]
            ids, mask = to_ids(srcs[i:i + chunk], device)
            try:
                seqs, _, _ = model.generate(ids, mask)
            except RuntimeError:
                n_fail += len(block)
                for ref in block:
                    tot_err += len(ref)
                    tot_len += max(len(ref), 1)
                    ref_chars += len(ref)
                continue
            for s, ref in zip(seqs, block):
                try:
                    hyp = bytes(s).decode("utf-8", errors="replace")
                except Exception:
                    hyp = ""
                tot_err += levenshtein(hyp, ref)
                tot_len += max(len(ref), 1)
                hyp_chars += len(hyp)
                ref_chars += len(ref)
        out[f"cer/lv{lv}"] = tot_err / max(tot_len, 1)
        out[f"cer/fail_lv{lv}"] = n_fail / max(len(srcs), 1)
        out[f"cer/len_ratio_lv{lv}"] = hyp_chars / max(ref_chars, 1)
        out[f"cer/identity_lv{lv}"] = (
            sum(levenshtein(s, r) for s, r in zip(srcs, refs))
            / max(sum(max(len(r), 1) for r in refs), 1))
    return out


@torch.no_grad()
def _no_pad_frac(model, texts, device, ratio, chunk=8):
    n_no_pad = n = 0
    for i in range(0, len(texts), chunk):
        bt = texts[i:i + chunk]
        ids, mask = to_ids(bt, device)
        L = torch.tensor([max(1, int(len(t.encode("utf-8")) * ratio)) for t in bt],
                         dtype=torch.long, device=device)
        h, _ = model.encode(ids, mask)
        zc, _ = model.decoder(h, L, source_mask=mask, strict_max=False)
        pred = model.head(zc).argmax(-1)
        for b in range(len(bt)):
            row = pred[b, : int(L[b])]
            n_no_pad += int(not bool((row == PAD_CLASS).any()))
            n += 1
    return n_no_pad / max(n, 1)


@torch.no_grad()
def truncation_behaviour(model, clean_texts, device, short_ratio=0.7,
                         long_ratio=1.4, max_items=64, chunk=8):
    model.eval()
    texts = clean_texts[:max_items]
    short = _no_pad_frac(model, texts, device, short_ratio, chunk)
    long = _no_pad_frac(model, texts, device, long_ratio, chunk)
    return {
        "trunc/no_pad_frac_short": short,
        "trunc/no_pad_frac_long": long,
        "trunc/separation": short - long,
    }


@torch.no_grad()
def _pooled(model, texts, device, chunk=8):
    qs = []
    for i in range(0, len(texts), chunk):
        ids, mask = to_ids(texts[i:i + chunk], device)
        h, _ = model.encode(ids, mask)
        _, l_hat = model.length_predictor(h, mask, detach_input=True)
        L = torch.ceil(l_hat).long().clamp(min=1, max=model.max_length)
        zc, cmask = model.decoder(h, L, source_mask=mask, strict_max=False)
        pred = model.head(zc).argmax(-1)
        is_pad = (pred == PAD_CLASS)
        first_pad = torch.where(is_pad.any(1), is_pad.float().argmax(1),
                                torch.full_like(L, zc.shape[1]))
        cm = length_to_mask(first_pad.clamp(min=1), zc.shape[1]) & cmask
        m = cm.unsqueeze(-1).float()
        qs.append(((zc.float() * m).sum(1) / m.sum(1).clamp(min=1e-6)).cpu())
    return F.normalize(torch.cat(qs), dim=-1)


@torch.no_grad()
def embedding_geometry(model, clean_texts, corrupt_fn, device,
                       levels=(1, 3, 5), max_items=96):
    model.eval()
    texts = clean_texts[:max_items]
    qc = _pooled(model, texts, device)

    n = len(texts)
    eye = torch.eye(n, dtype=torch.bool)
    out = {}
    for lv in levels:
        qa = _pooled(model, [corrupt_fn(t, lv) for t in texts], device)
        sim = qa @ qc.t()
        out[f"geom/cos_same_lv{lv}"] = float(sim.diag().mean())
        out[f"geom/cos_diff_lv{lv}"] = float(sim[~eye].mean())
        out[f"geom/gap_lv{lv}"] = (out[f"geom/cos_same_lv{lv}"]
                                   - out[f"geom/cos_diff_lv{lv}"])

    sim_c = qc @ qc.t()
    out["geom/cos_diff_clean"] = float(sim_c[~eye].mean())
    x = qc - qc.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(x.double())
    p = (sv ** 2) / (sv ** 2).sum().clamp(min=1e-12)
    out["geom/effective_rank"] = float(torch.exp(
        -(p * torch.log(p.clamp(min=1e-12))).sum()))
    return out


@torch.no_grad()
def entropy_vs_corruption(model, clean_texts, corrupt_fn, device,
                          levels=(0, 1, 3, 5), max_items=64, chunk=8):
    model.eval()
    out = {}
    for lv in levels:
        texts = [t if lv == 0 else corrupt_fn(t, lv) for t in clean_texts[:max_items]]
        lens = [len(t.encode("utf-8")) for t in clean_texts[:max_items]]
        tot, cnt = 0.0, 0
        for i in range(0, len(texts), chunk):
            ids, mask = to_ids(texts[i:i + chunk], device)
            L = torch.tensor(lens[i:i + chunk], dtype=torch.long, device=device)
            h, _ = model.encode(ids, mask)
            zc, _ = model.decoder(h, L, source_mask=mask, strict_max=False)
            lg = model.head(zc).float()
            ent = -(F.softmax(lg, -1) * F.log_softmax(lg, -1)).sum(-1)
            m = length_to_mask(L, zc.shape[1])
            tot += float((ent * m).sum()); cnt += int(m.sum())
        out[f"entropy/lv{lv}"] = tot / max(cnt, 1)
    return out


@torch.no_grad()
def boundary_accuracy(model, clean_texts, corrupt_fn, device, levels=(0, 1, 3),
                      over=1.4, max_items=64, chunk=8):
    model.eval()
    out = {}
    for lv in levels:
        texts = clean_texts[:max_items]
        srcs = [t if lv == 0 else corrupt_fn(t, lv) for t in texts]
        hit0 = hit2 = err = n = 0
        for i in range(0, len(srcs), chunk):
            bt, br = srcs[i:i + chunk], texts[i:i + chunk]
            ids, mask = to_ids(bt, device)
            h, mk = model.encode(ids, mask)
            tl = [len(t.encode("utf-8")) for t in br]
            L = torch.tensor([max(1, int(v * over)) for v in tl],
                             dtype=torch.long, device=device)
            zc, cm = model.decoder(h, L, source_mask=mk, strict_max=False)
            is_pad = (model.head(zc).argmax(-1) == PAD_CLASS) & cm
            for b, true_l in enumerate(tl):
                pos = is_pad[b].nonzero()
                p = int(pos[0]) if len(pos) else int(cm[b].sum())
                d = abs(p - true_l)
                hit0 += int(d == 0)
                hit2 += int(d <= 2)
                err += d
                n += 1
        out[f"bound/exact_lv{lv}"] = hit0 / max(n, 1)
        out[f"bound/within2_lv{lv}"] = hit2 / max(n, 1)
        out[f"bound/mae_lv{lv}"] = err / max(n, 1)
    return out


@torch.no_grad()
def positionwise_alignment(model, clean_texts, corrupt_fn, device, levels=(1, 3),
                           max_items=64, chunk=8):
    model.eval()
    out = {}
    for lv in levels:
        texts = clean_texts[:max_items]
        tot = n = 0.0
        worst = 1.0
        for i in range(0, len(texts), chunk):
            bt = texts[i:i + chunk]
            lens = [len(t.encode("utf-8")) for t in bt]
            L = torch.tensor(lens, dtype=torch.long, device=device)
            za = []
            for src in (bt, [corrupt_fn(t, lv) for t in bt]):
                ids, mask = to_ids(src, device)
                h, mk = model.encode(ids, mask)
                zc, _ = model.decoder(h, L, source_mask=mk, strict_max=False)
                za.append(zc.float())
            cos = F.cosine_similarity(za[0], za[1], dim=-1)
            for b, l in enumerate(lens):
                c = cos[b, :l]
                tot += float(c.sum())
                n += l
                worst = min(worst, float(c.min()))
        out[f"align/pos_mean_lv{lv}"] = tot / max(n, 1.0)
        out[f"align/pos_worst_lv{lv}"] = worst
    return out


def retrieval_probe(model, clean_texts, corrupt_fn, device, levels=(1, 3),
                    dim=128, epochs=400, tau=0.05, train_frac=0.6):
    model.eval()
    texts = clean_texts
    n = len(texts)
    n_tr = max(4, int(train_frac * n))
    qc = _pooled(model, texts, device)
    qa = {lv: _pooled(model, [corrupt_fn(t, lv) for t in texts], device)
          for lv in levels}
    mu = qc[:n_tr].mean(0, keepdim=True)
    out = {}

    idx = torch.arange(n_tr, n)
    base_raw = F.normalize(qc - mu, dim=-1)
    for lv in levels:
        q = F.normalize(qa[lv][n_tr:] - mu, dim=-1)
        hit = ((q @ base_raw.t()).argmax(1) == idx).float().mean()
        out[f"retr/raw_top1_lv{lv}"] = float(hit)

    W = torch.nn.Linear(qc.shape[1], dim, bias=False).to(device)
    opt = torch.optim.Adam(W.parameters(), lr=3e-3, weight_decay=1e-4)
    anc = (qc[:n_tr] - mu).to(device)
    pos = {lv: (qa[lv][:n_tr] - mu).to(device) for lv in levels}
    tgt = torch.arange(n_tr, device=device)
    for ep in range(epochs):
        lv = levels[ep % len(levels)]
        a = F.normalize(W(anc), dim=-1)
        p = F.normalize(W(pos[lv]), dim=-1)
        loss = F.cross_entropy((p @ a.t()) / tau, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        b = F.normalize(W((qc - mu).to(device)), dim=-1)
        for lv in levels:
            q = F.normalize(W((qa[lv][n_tr:] - mu).to(device)), dim=-1)
            hit = ((q @ b.t()).argmax(1).cpu() == idx).float().mean()
            out[f"retr/head_top1_lv{lv}"] = float(hit)
            out[f"retr/gain_lv{lv}"] = (out[f"retr/head_top1_lv{lv}"]
                                        - out[f"retr/raw_top1_lv{lv}"])
    return out


def full_evaluation(model, clean_texts, corrupt_fn, device, probe_epochs=400,
                    heavy: bool = True):
    metrics = {}
    metrics.update(truncation_behaviour(model, clean_texts, device))
    metrics.update(embedding_geometry(model, clean_texts, corrupt_fn, device))
    metrics.update(entropy_vs_corruption(model, clean_texts, corrupt_fn, device))
    metrics.update(decode_cer(model, clean_texts, corrupt_fn, device))
    metrics.update(boundary_accuracy(model, clean_texts, corrupt_fn, device))
    metrics.update(positionwise_alignment(model, clean_texts, corrupt_fn, device))
    if heavy:
        metrics.update(decodability_probe(model, clean_texts, corrupt_fn, device,
                                          epochs=probe_epochs))
        metrics.update(retrieval_probe(model, clean_texts, corrupt_fn, device,
                                       epochs=probe_epochs))
    return metrics
