"""Round-6 delta training data (PLAN.md, Round 6): one-knob variants of round 5's combined soft data.

    uv run python scripts/build_round6_data.py            # every variant; refuses to overwrite an existing directory

Inputs (all frozen): evals/round5/longstate-v2/train.jsonl (2,800 buried records), evals/round6/longstate-v3/train.jsonl
(2,800, lengths 300 / 300 / 1,100 / 1,100), evals/round4/ambiguity-v1/{soft,hard}.jsonl (1,738 records; 150 softened MNLI
questions), evals/night2/dates_unknowable.jsonl (the released parents' delta data), and the open teacher's rows
runs/probes/qwen35-9b-semif-teacher2-decision-v7-train (for the lam / threshold variants, via scripts/build_soft_targets.py).
Each variant is written to evals/round6/<variant>/train.jsonl with a manifest carrying every input's sha256.

    soft-nomnli    long v2 + ambiguity soft with the MNLI records' targets removed (hard labels): is round 5's WANLI dip the MNLI softening?
    soft-lam03     long v2 + ambiguity targets at lam 0.3 (closer to the label)
    soft-thr08     long v2 + ambiguity targets at threshold 0.8 (only the most contested questions)
    soft-du        long v2 + ambiguity soft + the dates / unknowable records (does a delta on a delta forget night 2?)
    soft-longmix3  long v3 + ambiguity soft (more 4k and 6k records)
    soft-half      1,400 long v2 records (200 / 200 / 500 / 500, fixed seed) + ambiguity soft (half the long records, for 4B)
"""
import random, subprocess, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kev.suite import digest, read_jsonl, write_json, write_jsonl  # noqa: E402

LONG_V2, LONG_V3 = "evals/round5/longstate-v2/train.jsonl", "evals/round6/longstate-v3/train.jsonl"
SOFT, DU = "evals/round4/ambiguity-v1/soft.jsonl", "evals/night2/dates_unknowable.jsonl"
TEACHER = "runs/probes/qwen35-9b-semif-teacher2-decision-v7-train"
OUT = ROOT / "evals/round6"


def ambiguity(name, **rule):
    """build_soft_targets.py at a non-default rule, into evals/round6/<name>; returns its soft.jsonl path."""
    out = OUT / name
    if not out.exists():
        args = [f"--{k}={v}" for k, v in rule.items()]
        subprocess.run([sys.executable, str(ROOT / "scripts/build_soft_targets.py"), "--teacher", TEACHER, "--out", str(out), *args], check=True, cwd=ROOT)
    return f"evals/round6/{name}/soft.jsonl"


def without_mnli_targets(path):
    rows = read_jsonl(ROOT / path)
    for r in rows:
        if r["_meta"]["source"] == "mnli":
            for q in r["questions"].values(): q.pop("target", None)
    return rows


def half_long(path, seed="round6-soft-half"):
    rows, rng = read_jsonl(ROOT / path), random.Random(seed)
    counts = {1024: 200, 2048: 200, 4096: 500, 6144: 500}
    return [r for L, n in counts.items() for r in rng.sample([x for x in rows if x["_meta"]["length"] == L], n)]


def check_targets(rows):
    """Every soft target sums to 1 and keeps >= 0.5 on the label, except on the night-2 unknowable records (near-uniform
    targets by design)."""
    for r in rows:
        for q in r["questions"].values():
            t = q.get("target")
            if t is None: continue
            label = str(q["label"]).lower() if q["type"] == "noul" else str(q["label"])   # target keys are option keys (strings)
            if abs(sum(t.values()) - 1) > 1e-6 or (t[label] < 0.5 - 1e-9 and r["_meta"]["source"] != "night2_unknowable"):
                raise ValueError(f"bad target in {r['_meta']['id']}: {t} (label {label})")


def write_variant(name, parts, description):
    """parts: [(input path, rows)]; rows are written in order."""
    out = OUT / name
    if out.exists(): raise FileExistsError(out)
    rows = [r for _, rs in parts for r in rs]
    check_targets(rows)
    out.mkdir(parents=True)
    write_jsonl(out / "train.jsonl", rows)
    soft_q = sum("target" in q for r in rows for q in r["questions"].values())
    write_json(out / "manifest.json", {"version": name, "construction": description, "code_sha256": digest(Path(__file__)),
                                       "inputs": {p: {"sha256": digest(ROOT / p), "records_used": len(rs)} for p, rs in parts},
                                       "files": {"train.jsonl": {"sha256": digest(out / "train.jsonl"), "records": len(rows), "soft_target_questions": soft_q}},
                                       "sources": dict(Counter(r["_meta"]["source"] for r in rows))})
    print(f"{name}: {len(rows)} records, {soft_q} soft-target questions")


def main():
    long_v2, soft = read_jsonl(ROOT / LONG_V2), read_jsonl(ROOT / SOFT)
    lam03, thr08 = ambiguity("ambiguity-lam03", lam=0.3), ambiguity("ambiguity-thr08", threshold=0.8)
    write_variant("soft-nomnli", [(LONG_V2, long_v2), (SOFT, without_mnli_targets(SOFT))], "long v2 + ambiguity soft, MNLI records' targets removed")
    write_variant("soft-lam03", [(LONG_V2, long_v2), (lam03, read_jsonl(ROOT / lam03))], "long v2 + ambiguity targets at lam 0.3")
    write_variant("soft-thr08", [(LONG_V2, long_v2), (thr08, read_jsonl(ROOT / thr08))], "long v2 + ambiguity targets at threshold 0.8")
    write_variant("soft-du", [(LONG_V2, long_v2), (SOFT, soft), (DU, read_jsonl(ROOT / DU))], "long v2 + ambiguity soft + dates/unknowable")
    write_variant("soft-longmix3", [(LONG_V3, read_jsonl(ROOT / LONG_V3)), (SOFT, soft)], "long v3 (300/300/1100/1100) + ambiguity soft")
    write_variant("soft-half", [(LONG_V2, half_long(LONG_V2)), (SOFT, soft)], "1,400 long v2 records (200/200/500/500, seed round6-soft-half) + ambiguity soft")


if __name__ == "__main__":
    main()
