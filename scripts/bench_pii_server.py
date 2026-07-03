import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8731/classify"
TOKEN = sys.argv[1] if len(sys.argv) > 1 else "testtoken123"
N_WARMUP = 10
N_RUNS = 200

texts = [json.loads(l)["text"] for l in open(Path(__file__).parent.parent / "eval" / "sa_names_eval.jsonl")]
texts = (texts * (N_RUNS // len(texts) + 1))[:N_RUNS]


def classify(text):
    req = urllib.request.Request(
        URL,
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


for t in texts[:N_WARMUP]:
    classify(t)

latencies = []
t_start = time.perf_counter()
for t in texts:
    t0 = time.perf_counter()
    classify(t)
    latencies.append(time.perf_counter() - t0)
total = time.perf_counter() - t_start

latencies_ms = sorted(x * 1000 for x in latencies)
n = len(latencies_ms)
print(f"N requests      = {n}")
print(f"Total wall time  = {total:.2f}s")
print(f"Throughput       = {n/total:.2f} req/s")
print(f"Mean latency     = {statistics.mean(latencies_ms):.1f} ms")
print(f"Median (p50)     = {latencies_ms[n//2]:.1f} ms")
print(f"p90              = {latencies_ms[int(n*0.90)]:.1f} ms")
print(f"p99              = {latencies_ms[int(n*0.99)]:.1f} ms")
print(f"Min / Max        = {latencies_ms[0]:.1f} / {latencies_ms[-1]:.1f} ms")
