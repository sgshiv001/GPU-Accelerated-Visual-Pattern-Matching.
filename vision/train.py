"""Fit both models without fitting or tuning on the held-out test set."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import sklearn
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

from model import DigitCNN, SEED, digits_split, hog, svm_pipeline

ROOT = Path(__file__).resolve().parent


def metrics(y, pred):
    return {
        "correct": int((y == pred).sum()),
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "confusion": confusion_matrix(y, pred, labels=range(10)).tolist(),
    }


def train(out, epochs):
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    x, y, train_ids, test_ids = digits_split()

    t0 = perf_counter()
    classical = svm_pipeline().fit(hog(x[train_ids]), y[train_ids])
    classical_ms = (perf_counter() - t0) * 1e3
    sp = classical.predict(hog(x[test_ids]))
    pca = classical.named_steps["pca"]

    net = DigitCNN().eval()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    train_x, train_y = torch.from_numpy(x[train_ids]), torch.from_numpy(y[train_ids])
    rng = torch.Generator().manual_seed(SEED)
    loss_fn = torch.nn.CrossEntropyLoss()
    losses = []
    t0 = perf_counter()
    net.train()
    for epoch in range(epochs):
        total = 0.0
        for idx in torch.randperm(len(train_ids), generator=rng).split(64):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(net(train_x[idx]), train_y[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        losses.append(total / len(train_ids))
        print(f"epoch {epoch + 1:2d}/{epochs}: train loss {losses[-1]:.5f}", flush=True)
    cnn_ms = (perf_counter() - t0) * 1e3
    net.eval()
    with torch.inference_mode():
        logits = net(torch.from_numpy(x[test_ids])).numpy()
    cp = logits.argmax(1)

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "cnn_weights.npz",
                        **{key: val.numpy() for key, val in net.state_dict().items()})
    report = {
        "seed": SEED, "train_count": len(train_ids), "test_count": len(test_ids),
        "epochs": epochs, "pca_components": int(pca.n_components_),
        "pca_variance": float(pca.explained_variance_ratio_.sum()),
        "parameters": sum(p.numel() for p in net.parameters()),
        "torch": torch.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__,
        "classical": {**metrics(y[test_ids], sp), "fit_ms": classical_ms},
        "cnn": {**metrics(y[test_ids], cp), "fit_ms": cnn_ms, "train_loss": losses},
        "predictions": [
            {"id": int(i), "label": int(y[i]), "svm": int(s), "cnn": int(c)}
            for i, s, c in zip(test_ids, sp, cp)
        ],
    }
    (out / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"HOG -> PCA ({pca.n_components_}) -> SVM: {report['classical']['accuracy']:.2%}")
    print(f"CNN: {report['cnn']['accuracy']:.2%} ({report['parameters']:,} parameters)")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, default=ROOT / "results")
    cli.add_argument("--epochs", type=int, default=20)
    args = cli.parse_args()
    if args.epochs < 1:
        cli.error("--epochs must be positive")
    with threadpool_limits(limits=1):
        train(args.out, args.epochs)
