"""Parameter counts and walltime benchmark for the few-shot bioacoustic models.

Measures, for each model on a single GPU in float32 (no JIT / torch.compile):
  - trainable parameter count
  - training step time: one 5-way 5-shot episode (5 queries per class ->
    50 segments), prototypical loss, backward + Adam step
  - inference time: forward on a batch of 128 segments under torch.no_grad()

Segments are 0.2 s -> 17 frames x 128 mel bins (config `features`).

Usage:
    python scripts/params_walltime.py [--device cuda:0] [--out results/params_walltime.md]
"""

import argparse
import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ProtoNet  # noqa: E402
from src.util import prototypical_loss  # noqa: E402

MODELS = {
    "proto": (),
    "hproto1": (1,),
    "hproto2": (2,),
    "hproto3": (3,),
    "hproto_all": (1, 2, 3, 4),
}

FRAMES, MELS = 17, 128  # 0.2 s at 22.05 kHz, hop 256
K_WAY, N_SHOT, N_QUERY = 5, 5, 5
EVAL_BATCH = 128


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_loop(fn, warmup, iters, device):
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


def benchmark(name, placement, device):
    torch.manual_seed(0)
    model = ProtoNet(hyper_placement=placement).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    n_samples = K_WAY * (N_SHOT + N_QUERY)
    x = torch.randn(n_samples, FRAMES, MELS, device=device)
    y = torch.arange(K_WAY, device=device).repeat_interleave(N_SHOT + N_QUERY)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()

    def train_step():
        opt.zero_grad(set_to_none=True)
        emb = model(x)
        loss, _ = prototypical_loss(emb, y, N_SHOT)
        loss.backward()
        opt.step()

    train_ms = time_loop(train_step, warmup=5, iters=50, device=device)

    model.eval()
    xe = torch.randn(EVAL_BATCH, FRAMES, MELS, device=device)

    def infer_step():
        with torch.no_grad():
            model(xe)

    infer_ms = time_loop(infer_step, warmup=10, iters=100, device=device)

    return dict(model=name, params=n_params,
                train_ms_per_episode=round(train_ms, 2),
                infer_ms_per_batch=round(infer_ms, 2),
                infer_ms_per_segment=round(infer_ms / EVAL_BATCH, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/params_walltime.md")
    args = ap.parse_args()
    device = torch.device(args.device)

    rows = []
    for name, placement in MODELS.items():
        rows.append(benchmark(name, placement, device))
        print(rows[-1])

    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    header = (
        f"# Parameter counts and walltime (few-shot bioacoustic models)\n\n"
        f"GPU: {gpu} | PyTorch {torch.__version__} (CUDA {torch.version.cuda}) | "
        f"float32, no JIT/torch.compile.\n"
        f"Train step = one {K_WAY}-way {N_SHOT}-shot episode with {N_QUERY} queries/class "
        f"({K_WAY * (N_SHOT + N_QUERY)} segments of {FRAMES} frames x {MELS} mels), "
        f"prototypical loss, forward+backward+Adam. Inference on {EVAL_BATCH} segments "
        f"under no_grad. Mean over 50 (train) / 100 (inference) iterations after warmup.\n\n"
    )
    table = "| Model | Params (M) | Train episode (ms) | Inference (ms/128 segs) | Inference (ms/segment) |\n"
    table += "|---|---|---|---|---|\n"
    for r in rows:
        table += (f"| {r['model']} | {r['params']/1e6:.3f} | {r['train_ms_per_episode']} "
                  f"| {r['infer_ms_per_batch']} | {r['infer_ms_per_segment']} |\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(header + table)
    with open(args.out.replace(".md", ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
