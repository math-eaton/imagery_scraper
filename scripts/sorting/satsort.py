#!/usr/bin/env python3
"""
satsort.py - order a directory of images by visual similarity, for animation.

Two stages:

  embed   Extract DINOv2 features for every image (slow, run once, cached to .npz).
          Stores the CLS token plus a 4x4 spatial grid of pooled patch tokens, so
          the feature mix can be changed later without re-embedding.

  order   Build an ordered manifest from a seed image (fast, re-run freely):
          kNN graph -> greedy nearest-neighbour walk -> 2-opt refinement.

Examples:
  python satsort.py embed /data/sat -o sat.npz
  python satsort.py order sat.npz --seed tile_0412.jpg -o order.txt
  python satsort.py order sat.npz --seed random --grid 4 --cls-weight 0.5 -o order_b.txt

Requirements:
  embed: torch, torchvision (CPU build is fine), pillow, numpy
  order: numpy only
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
GRID_STORED = 4  # spatial grid stored at embed time; order can use 1, 2 or 4


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# embed
# --------------------------------------------------------------------------- #

def list_images(root: Path, recursive: bool):
    it = root.rglob("*") if recursive else root.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in EXTS)


class ImageSet:
    """Map-style dataset for torch's DataLoader. Module-level and torch-free so
    loader workers can pickle it; the default collate turns arrays into tensors."""

    def __init__(self, paths, size):
        self.paths, self.size = paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        size = self.size
        ok = True
        try:
            with Image.open(self.paths[i]) as im:
                # JPEG: decode directly at reduced scale (much faster)
                im.draft("RGB", (size * 2, size * 2))
                im = im.convert("RGB")
                w, h = im.size
                s = size / min(w, h)
                im = im.resize((max(size, round(w * s)), max(size, round(h * s))),
                               Image.BICUBIC)
                w, h = im.size
                l, t = (w - size) // 2, (h - size) // 2
                im = im.crop((l, t, l + size, t + size))  # no-op for square tiles
                a = np.asarray(im, dtype=np.float32) / 255.0
        except Exception as e:  # unreadable file: keep slot, flag it
            print(f"WARN: could not read {self.paths[i]}: {e}", file=sys.stderr)
            a = np.zeros((size, size, 3), np.float32)
            ok = False
        a = (a - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(a.transpose(2, 0, 1)), i, ok


def cmd_embed(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    root = Path(args.image_dir).resolve()
    paths = list_images(root, args.recursive)
    if not paths:
        sys.exit(f"No images found in {root}")
    size = max(14, round(args.size / 14) * 14)  # DINOv2 patch size is 14
    if size != args.size:
        log(f"--size rounded to {size} (must be a multiple of 14)")
    log(f"{len(paths)} images in {root}, model {args.model}, input {size}px")

    if args.threads:
        torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    log(f"device={device}  torch threads={torch.get_num_threads()}  "
        f"loader workers={args.workers}")

    model = torch.hub.load("facebookresearch/dinov2", args.model,
                           pretrained=not args.no_weights)
    model.eval().to(device)

    loader = DataLoader(ImageSet(paths, size), batch_size=args.batch, num_workers=args.workers,
                        shuffle=False, pin_memory=(device == "cuda"))

    n = len(paths)
    cls_out = grid_out = None
    ok_out = np.ones(n, bool)
    t0 = time.time()
    done = 0
    with torch.inference_mode():
        for x, idx, ok in loader:
            out = model.forward_features(x.to(device))
            cls = out["x_norm_clstoken"]                 # B, C
            pt = out["x_norm_patchtokens"]               # B, N, C
            B, N, C = pt.shape
            h = w = int(round(math.sqrt(N)))
            pt = pt.reshape(B, h, w, C).permute(0, 3, 1, 2)
            g = F.adaptive_avg_pool2d(pt, GRID_STORED).permute(0, 2, 3, 1)  # B,4,4,C
            if cls_out is None:
                cls_out = np.zeros((n, C), np.float16)
                grid_out = np.zeros((n, GRID_STORED, GRID_STORED, C), np.float16)
            ii = idx.numpy()
            cls_out[ii] = cls.float().cpu().numpy()
            grid_out[ii] = g.float().cpu().numpy()
            ok_out[ii] = ok.numpy()
            done += B
            el = time.time() - t0
            eta = el / done * (n - done)
            print(f"\r  {done}/{n}  {done / el:5.1f} img/s  ETA {eta / 60:5.1f} min",
                  end="", file=sys.stderr, flush=True)
    log(f"\nembedded {n} images in {(time.time() - t0) / 60:.1f} min "
        f"({(~ok_out).sum()} unreadable)")

    rel = np.array([str(p.relative_to(root)) for p in paths])
    np.savez(args.out, cls=cls_out, grid=grid_out, ok=ok_out, paths=rel,
             root=str(root), model=args.model, size=size)
    log(f"saved {args.out}")


# --------------------------------------------------------------------------- #
# order: features
# --------------------------------------------------------------------------- #

def l2n(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def build_features(d, keep, cls_weight, grid, grid_weight, center, pca):
    blocks = []
    if cls_weight > 0:
        b = d["cls"][keep].astype(np.float32)
        if center:
            b -= b.mean(0)
        blocks.append(l2n(b) * cls_weight)
    if grid_weight > 0:
        G = d["grid"][keep].astype(np.float32)  # n, 4, 4, C
        n, gs, _, C = G.shape
        s = gs // grid
        G = G.reshape(n, grid, s, grid, s, C).mean(axis=(2, 4)).reshape(n, -1)
        if center:
            G -= G.mean(0)
        blocks.append(l2n(G) * grid_weight)
    if not blocks:
        sys.exit("cls-weight and grid-weight are both 0; nothing to compare")
    X = np.concatenate(blocks, axis=1)
    if pca and pca < X.shape[1]:
        X = X - X.mean(0)
        cov = X.T @ X
        _, vecs = np.linalg.eigh(cov)
        X = X @ vecs[:, ::-1][:, :pca]
    return np.ascontiguousarray(l2n(X), dtype=np.float32)


def knn(X, k, chunk=1024):
    """Exact top-k cosine neighbours (brute force, chunked). Returns idx, dist."""
    n = len(X)
    k = min(k, n - 1)
    idx = np.empty((n, k), np.int64)
    dist = np.empty((n, k), np.float32)
    for s in range(0, n, chunk):
        S = X[s:s + chunk] @ X.T
        r = np.arange(S.shape[0])
        S[r, s + r] = -np.inf
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, 1)
        o = np.argsort(-ps, axis=1)
        idx[s:s + chunk] = np.take_along_axis(part, o, 1)
        dist[s:s + chunk] = 1.0 - np.take_along_axis(ps, o, 1)
    return idx, dist


# --------------------------------------------------------------------------- #
# order: path construction
# --------------------------------------------------------------------------- #

def greedy_walk(X, nbr, start):
    n = len(X)
    visited = np.zeros(n, bool)
    nbr_l = nbr.tolist()
    order = np.empty(n, np.int64)
    cur = start
    visited[cur] = True
    order[0] = cur
    fallbacks = 0
    for t in range(1, n):
        nxt = -1
        for c in nbr_l[cur]:
            if not visited[c]:
                nxt = c
                break
        if nxt < 0:  # all listed neighbours used: search every unvisited image
            sims = X @ X[cur]
            sims[visited] = -np.inf
            nxt = int(np.argmax(sims))
            fallbacks += 1
        visited[nxt] = True
        order[t] = nxt
        cur = nxt
    return order, fallbacks


def path_steps(X, path):
    return 1.0 - np.einsum("ij,ij->i", X[path[:-1]], X[path[1:]])


def two_opt(X, path, nbr, nbr_d, max_passes, time_limit, eps=1e-7):
    """
    Neighbour-list 2-opt on an open path with the first node fixed.
    reverse(i, j) flips path[i..j] (1 <= i < j <= n-1), replacing edges
    (p[i-1],p[i]) and (p[j],p[j+1]) with (p[i-1],p[j]) and (p[i],p[j+1]).
    """
    n = len(path)
    path = path.copy()
    pos = np.empty(n, np.int64)
    pos[path] = np.arange(n)
    nbr_l = nbr.tolist()
    nbrd_l = nbr_d.tolist()
    Xd = X  # local alias

    def d(u, v):
        return 1.0 - float(Xd[u] @ Xd[v])

    def gain(i, j):
        a, b, c = path[i - 1], path[i], path[j]
        g = d(a, b) - d(a, c)
        if j < n - 1:
            e = path[j + 1]
            g += d(c, e) - d(b, e)
        return g

    def apply(i, j):
        seg = path[i:j + 1][::-1].copy()
        path[i:j + 1] = seg
        pos[seg] = np.arange(i, j + 1)

    t0 = time.time()
    total_moves = 0
    for p in range(1, max_passes + 1):
        moves = 0
        for a in path.tolist():  # snapshot of current order
            ia = int(pos[a])
            d_succ = d(a, path[ia + 1]) if ia < n - 1 else math.inf
            d_pred = d(path[ia - 1], a) if ia > 0 else -math.inf
            bound = max(d_succ, d_pred)
            for c, d_ac in zip(nbr_l[a], nbrd_l[a]):
                if d_ac >= bound:
                    break  # sorted list: no further candidate can help
                ic = int(pos[c])
                cands = []
                if ic > ia:
                    if ic > ia + 1:
                        cands.append((ia + 1, ic))      # new edge a-c on a's succ side
                    if ia >= 1 and ic - 1 > ia:
                        cands.append((ia, ic - 1))      # new edge a-c on a's pred side
                else:
                    if ia > ic + 1:
                        cands.append((ic + 1, ia))      # new edge c-a, drop a's succ edge
                    if ic >= 1 and ia - 1 > ic:
                        cands.append((ic, ia - 1))      # new edge c-a, drop a's pred edge
                best, bm = eps, None
                for i, j in cands:
                    g = gain(i, j)
                    if g > best:
                        best, bm = g, (i, j)
                if bm:
                    apply(*bm)
                    moves += 1
                    break
        total_moves += moves
        log(f"  2-opt pass {p}: {moves} moves, "
            f"length {path_steps(X, path).sum():.2f}, {time.time() - t0:.0f}s")
        if moves == 0 or time.time() - t0 > time_limit:
            break
    return path, total_moves


def resolve_seed(seed, rel_paths, rng):
    n = len(rel_paths)
    if seed is None or seed == "random":
        return rng.randrange(n)
    for i, p in enumerate(rel_paths):
        if p == seed:
            return i
    names = [os.path.basename(p) for p in rel_paths]
    hits = [i for i, b in enumerate(names) if b == seed]
    if not hits:
        hits = [i for i, b in enumerate(names) if os.path.splitext(b)[0] == seed]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        sys.exit(f"seed '{seed}' is ambiguous ({len(hits)} matches); use the relative path")
    if seed.isdigit() and int(seed) < n:
        return int(seed)
    sys.exit(f"seed '{seed}' not found")


def describe(steps, label):
    q = np.percentile(steps, [50, 95, 99])
    log(f"{label}: total {steps.sum():.2f}  mean {steps.mean():.4f}  "
        f"median {q[0]:.4f}  p95 {q[1]:.4f}  p99 {q[2]:.4f}  max {steps.max():.4f}")


def cmd_order(args):
    t0 = time.time()
    d = np.load(args.embeddings)
    root = Path(str(d["root"]))
    keep = np.flatnonzero(d["ok"])
    rel = d["paths"][keep].tolist()
    if len(keep) < len(d["ok"]):
        log(f"skipping {len(d['ok']) - len(keep)} unreadable images")
    n = len(keep)

    X = build_features(d, keep, args.cls_weight, args.grid, args.grid_weight,
                       not args.no_center, args.pca)
    log(f"{n} images, feature dim {X.shape[1]}")

    rng = random.Random(args.rng)
    start = resolve_seed(args.seed, rel, rng)
    log(f"seed: {rel[start]}")

    nbr, nbr_d = knn(X, args.k)
    log(f"kNN graph (k={nbr.shape[1]}) in {time.time() - t0:.1f}s")

    path, fb = greedy_walk(X, nbr, start)
    log(f"greedy walk done ({fb} fallbacks to full search)")
    describe(path_steps(X, path), "greedy ")

    if args.passes > 0:
        ko = min(args.opt_k, nbr.shape[1])
        path, mv = two_opt(X, path, nbr[:, :ko], nbr_d[:, :ko],
                           args.passes, args.time_limit)
        describe(path_steps(X, path), "2-opt  ")

    assert path[0] == start and len(np.unique(path)) == n

    steps = path_steps(X, path)
    worst = np.argsort(-steps)[:args.show_worst]
    if len(worst):
        log(f"largest transitions (frame -> frame+1):")
        for w in sorted(worst):
            log(f"  {w:6d}  {steps[w]:.4f}  {rel[path[w]]} -> {rel[path[w + 1]]}")

    out_paths = [rel[i] if args.relative else str(root / rel[i]) for i in path]
    Path(args.out).write_text("\n".join(out_paths) + "\n")
    log(f"wrote {args.out} ({n} entries) in {time.time() - t0:.1f}s total")

    if args.stats_csv:
        with open(args.stats_csv, "w") as f:
            f.write("frame,path,dist_to_next\n")
            for t, i in enumerate(path):
                dn = f"{steps[t]:.6f}" if t < n - 1 else ""
                f.write(f'{t},"{out_paths[t]}",{dn}\n')
        log(f"wrote {args.stats_csv}")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed", help="extract and cache DINOv2 features")
    e.add_argument("image_dir")
    e.add_argument("-o", "--out", default="embeddings.npz")
    e.add_argument("--model", default="dinov2_vits14",
                   choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"],
                   help="small is plenty here; base is ~3x slower")
    e.add_argument("--size", type=int, default=224,
                   help="input resolution, multiple of 14 (448 = finer detail, ~4x slower)")
    e.add_argument("--batch", type=int, default=32)
    e.add_argument("--workers", type=int, default=4, help="parallel image decoders")
    e.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    e.add_argument("--recursive", action="store_true")
    e.add_argument("--cpu", action="store_true", help="ignore CUDA even if present")
    e.add_argument("--no-weights", action="store_true", help=argparse.SUPPRESS)

    o = sub.add_parser("order", help="build an ordered manifest from a seed")
    o.add_argument("embeddings")
    o.add_argument("-o", "--out", default="order.txt")
    o.add_argument("--seed", default="random",
                   help="filename, relative path, stem, index, or 'random'")
    o.add_argument("--rng", type=int, default=None, help="RNG seed for --seed random")
    o.add_argument("--cls-weight", type=float, default=1.0,
                   help="weight of the global (CLS) descriptor")
    o.add_argument("--grid-weight", type=float, default=1.0,
                   help="weight of the spatial patch grid (layout/composition)")
    o.add_argument("--grid", type=int, default=2, choices=[1, 2, 4],
                   help="spatial grid size; 1 = mean of all patches, 4 = finest layout")
    o.add_argument("--no-center", action="store_true",
                   help="don't subtract the dataset mean before cosine")
    o.add_argument("--pca", type=int, default=0, help="reduce to N dims (0 = off)")
    o.add_argument("--k", type=int, default=30, help="neighbours kept in the kNN graph")
    o.add_argument("--opt-k", type=int, default=10, help="neighbours tried by 2-opt")
    o.add_argument("--passes", type=int, default=50, help="max 2-opt passes (0 = off)")
    o.add_argument("--time-limit", type=float, default=600, help="2-opt time cap (s)")
    o.add_argument("--relative", action="store_true", help="write relative paths")
    o.add_argument("--show-worst", type=int, default=10)
    o.add_argument("--stats-csv", help="also write frame,path,dist_to_next CSV")

    args = ap.parse_args()
    {"embed": cmd_embed, "order": cmd_order}[args.cmd](args)


if __name__ == "__main__":
    main()