"""Bundle measured results and held-out samples into a standalone HTML demo."""
import csv
import json
from pathlib import Path

from sklearn.datasets import load_digits

ROOT = Path(__file__).resolve().parent


def main():
    results = ROOT / "results"
    evaluation = json.loads((results / "evaluation.json").read_text(encoding="utf-8"))
    hardware = json.loads((results / "benchmark.json").read_text(encoding="utf-8"))
    with (results / "latency.csv").open(newline="", encoding="utf-8") as f:
        timings = [{k: int(v) if k == "batch" else float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    images = load_digits().images.astype(int)
    records = evaluation.pop("predictions")
    for record in records:
        record["pixels"] = images[record["id"]].flatten().tolist()
    # Escape markup boundaries so JSON can safely be embedded in a script tag.
    data = json.dumps({"metrics": evaluation, "samples": records, "timings": timings, "hardware": hardware},
                      separators=(",", ":")).replace("<", "\\u003c")
    template = (ROOT / "demo" / "template.html").read_text(encoding="utf-8")
    if template.count("/*__DEMO_DATA__*/") != 1:
        raise RuntimeError("Missing or duplicate demo data marker")
    target = ROOT / "demo.html"
    target.write_text(template.replace("/*__DEMO_DATA__*/", data), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
