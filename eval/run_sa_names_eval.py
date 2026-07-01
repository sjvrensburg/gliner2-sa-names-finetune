import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

URL = "http://127.0.0.1:8731/classify"
TOKEN = sys.argv[1] if len(sys.argv) > 1 else "testtoken123"
DATASET = Path(__file__).parent / "sa_names_eval.jsonl"
RESULTS_OUT = Path(__file__).parent / "sa_names_eval_results.json"


def classify(text):
    req = urllib.request.Request(
        URL,
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def spans_overlap(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def main():
    rows = [json.loads(l) for l in open(DATASET)]
    results = []
    for row in rows:
        resp = classify(row["text"])
        name_ents = [e for e in resp["entities"] if e.get("label") == "name"]
        hit = None
        for e in name_ents:
            if spans_overlap(e["start"], e["end"], row["expected_start"], row["expected_end"]):
                hit = e
                break
        exact = hit is not None and hit["start"] == row["expected_start"] and hit["end"] == row["expected_end"]
        results.append({**row, "detected": hit is not None, "exact": exact,
                        "score": hit["score"] if hit else None})

    by_group = defaultdict(lambda: {"n": 0, "detected": 0, "exact": 0})
    by_ctx = defaultdict(lambda: {"n": 0, "detected": 0, "exact": 0})
    by_group_ctx = defaultdict(lambda: {"n": 0, "detected": 0, "exact": 0})

    for r in results:
        for bucket, key in ((by_group, r["group"]), (by_ctx, r["context"]),
                            (by_group_ctx, (r["group"], r["context"]))):
            bucket[key]["n"] += 1
            bucket[key]["detected"] += int(r["detected"])
            bucket[key]["exact"] += int(r["exact"])

    print("=== Recall by name group ===")
    for g, s in sorted(by_group.items(), key=lambda kv: kv[1]["detected"] / kv[1]["n"]):
        print(f"  {g:22} detected={s['detected']:3}/{s['n']:3} ({s['detected']/s['n']:.1%})  exact-span={s['exact']}/{s['n']}")

    print("\n=== Recall by context ===")
    for c, s in sorted(by_ctx.items(), key=lambda kv: kv[1]["detected"] / kv[1]["n"]):
        print(f"  {c:18} detected={s['detected']:3}/{s['n']:3} ({s['detected']/s['n']:.1%})  exact-span={s['exact']}/{s['n']}")

    print("\n=== Worst (group, context) cells (< 100% detected) ===")
    for (g, c), s in sorted(by_group_ctx.items(), key=lambda kv: kv[1]["detected"] / kv[1]["n"]):
        rate = s["detected"] / s["n"]
        if rate < 1.0:
            print(f"  {g:22} / {c:18} detected={s['detected']}/{s['n']} ({rate:.0%})")

    print("\n=== Individual misses ===")
    for r in results:
        if not r["detected"]:
            print(f"  MISS  [{r['group']:22} / {r['context']:16}] {r['name']!r:22} in: {r['text']!r}")

    total = len(results)
    total_detected = sum(r["detected"] for r in results)
    print(f"\nOverall: {total_detected}/{total} ({total_detected/total:.1%}) recall, "
          f"{sum(r['exact'] for r in results)}/{total} exact-span")

    with open(RESULTS_OUT, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
