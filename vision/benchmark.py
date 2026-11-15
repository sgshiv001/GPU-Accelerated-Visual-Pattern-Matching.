"""Check trained CNN parity and measure CPU and native CUDA batch latency."""
import argparse
import csv
import ctypes as ct
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from model import DigitCNN, digits_split

ROOT = Path(__file__).resolve().parent
P = ct.c_void_p
I = ct.c_int
F = ct.POINTER(ct.c_float)
ARRAY = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")


class CUDA:
    def __init__(self, path):
        self.lib = ct.CDLL(str(path.resolve()))
        for name, result, args in (
            ("error", ct.c_char_p, []),
            ("device", I, [ct.POINTER(ct.c_char), I]),
            ("create", I, [I, ARRAY, I, ct.POINTER(P)]),
            ("destroy", None, [P]),
            ("forward", I, [P, ARRAY, ARRAY, I]),
            ("resident_ms", I, [P, I, I, F]),
        ):
            f = getattr(self.lib, f"vision_{name}")
            f.restype, f.argtypes = result, args

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.vision_error().decode("utf-8", errors="replace"))

    def device(self):
        name = ct.create_string_buffer(256)
        self.check(self.lib.vision_device(name, len(name)))
        return name.value.decode()


class Batch:
    def __init__(self, gpu, x, packed):
        self.gpu, self.x = gpu, np.ascontiguousarray(x, dtype=np.float32)
        self.y = np.empty((len(x), 10), dtype=np.float32)
        self.handle = P()
        gpu.check(gpu.lib.vision_create(len(x), packed, packed.size, ct.byref(self.handle)))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.gpu.lib.vision_destroy(self.handle)

    def forward(self, fused):
        self.gpu.check(self.gpu.lib.vision_forward(self.handle, self.x, self.y, fused))
        return self.y

    def resident_ms(self, fused, repeats):
        result = ct.c_float()
        self.gpu.check(self.gpu.lib.vision_resident_ms(self.handle, fused, repeats, ct.byref(result)))
        return result.value


def verify(batch, reference):
    errors = {}
    for fused, name in enumerate(("direct", "fused")):
        logits = batch.forward(fused)
        np.testing.assert_allclose(logits, reference, rtol=1e-4, atol=5e-5)
        np.testing.assert_array_equal(logits.argmax(1), reference.argmax(1))
        errors[name] = float(np.max(np.abs(logits - reference)))
    return errors


def wall_ms(run, repeats):
    start = perf_counter()
    for _ in range(repeats):
        run()
    return (perf_counter() - start) * 1000 / repeats


@torch.inference_mode()
def main(args):
    torch.set_num_threads(1)
    net = DigitCNN().eval()
    with np.load(args.results / "cnn_weights.npz", allow_pickle=False) as weights:
        params = {name: weights[name] for name in net.state_dict()}
    net.load_state_dict({key: torch.from_numpy(value) for key, value in params.items()})
    packed = np.concatenate([value.ravel() for value in params.values()])
    x, labels, _, test_ids = digits_split()
    heldout = x[test_ids]

    gpu = CUDA(args.library)
    name = gpu.device()
    if args.predict_id is not None:
        item = args.predict_id
        if item not in test_ids:
            raise SystemExit("Choose a held-out dataset sample ID from the demo.")
        with Batch(gpu, x[item:item + 1], packed) as batch:
            reference = net(torch.from_numpy(batch.x)).numpy()
            errors = verify(batch, reference)
            prediction = int(batch.y[0].argmax())
        if args.prediction_json:
            print(json.dumps({"id": item, "label": int(labels[item]), "prediction": prediction,
                              "device": name, "max_logit_error": errors["fused"]}))
            return
        print(f"{name}\nHeld-out sample #{item}: label {labels[item]}, fused CUDA CNN {prediction}")
        palette = " .:-=+*#%@"
        for row in x[item, 0]:
            print(" ".join(palette[round(float(p) * (len(palette) - 1))] for p in row))
        return
    if args.verify_only:
        groups = (heldout[:1], heldout[:37], heldout)
    else:
        groups = (heldout,)
    validation = {}
    for group in groups:
        with Batch(gpu, group, packed) as batch:
            validation[len(group)] = verify(batch, net(torch.from_numpy(group)).numpy())
    print(f"{name}: logits and predictions agree for {', '.join(map(str, validation))} images.")
    if args.verify_only:
        return

    rows = []
    for size in args.batches:
        # Above 360 we repeat held-out inputs for workload sizing, never for accuracy.
        batch_x = heldout[np.arange(size) % len(heldout)]
        tensor = torch.from_numpy(batch_x)
        with Batch(gpu, batch_x, packed) as batch:
            errors = verify(batch, net(tensor).numpy())
            cpu, direct, fused, direct_wall, fused_wall = [], [], [], [], []
            for i in range(args.warmup):
                net(tensor)
                batch.forward(0)
                batch.forward(1)
            for i in range(args.samples):
                cpu.append(wall_ms(lambda: net(tensor), args.repeats))
                # Alternate variant order to reduce warm-clock bias.
                for variant in ((0, 1) if i % 2 == 0 else (1, 0)):
                    (direct, fused)[variant].append(batch.resident_ms(variant, args.repeats))
                    (direct_wall, fused_wall)[variant].append(
                        wall_ms(lambda: batch.forward(variant), args.repeats))
            row = dict(batch=size, cpu_ms=median(cpu), direct_gpu_ms=median(direct),
                       fused_gpu_ms=median(fused), direct_wall_ms=median(direct_wall),
                       fused_wall_ms=median(fused_wall), direct_max_error=errors["direct"],
                       fused_max_error=errors["fused"])
            rows.append(row)
            print(f"{size:4d}: CPU {row['cpu_ms']:.4f} ms, direct GPU {row['direct_gpu_ms']:.4f} ms, "
                  f"fused GPU {row['fused_gpu_ms']:.4f} ms, fused incl. copies {row['fused_wall_ms']:.4f} ms",
                  flush=True)

    output = args.output or args.results
    output.mkdir(parents=True, exist_ok=True)
    with (output / "latency.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"gpu": name, "cpu": platform.processor(), "platform": platform.system(),
                "torch_threads": torch.get_num_threads(), "samples": args.samples,
                "repeats": args.repeats, "warmup": args.warmup, "held_out_parity": validation[360]}
    (output / "benchmark.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=ROOT.parent / "build" / "vision_cuda.dll")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, help="Write benchmark results to a separate directory")
    parser.add_argument("--prediction-json", action="store_true", help="Print a machine-readable prediction")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 128, 512, 1024])
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--predict-id", type=int, help="Run the native CNN on one held-out dataset sample")
    args = parser.parse_args()
    if args.prediction_json and args.predict_id is None:
        parser.error("--prediction-json requires --predict-id")
    if min(args.samples, args.repeats, args.warmup) < 1 or any(not 1 <= n <= 2048 for n in args.batches):
        parser.error("Samples, repeats and warmup must be positive; batches must be in 1..2048")
    with threadpool_limits(limits=1):
        main(args)
