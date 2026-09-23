"""Freeze wanli-v2: a larger WANLI panel, disjoint from wanli-v1 (SemIf's 256), rendered exactly as v1.

    curl -sL https://huggingface.co/datasets/alisawuffles/WANLI/resolve/<REVISION>/test.jsonl -o /tmp/wanli-test.jsonl
    uv run python scripts/freeze_wanli_v2.py --source /tmp/wanli-test.jsonl --out evals/external/wanli-v2

Why: round 5's WANLI gate was a point estimate on 256 questions and turned on three of them. v2 draws ~1,000 more pairs
from the same pinned WANLI test split (CC-BY-4.0), excluding every pair whose source id, premise or hypothesis appears in
v1, balanced over the three labels. The rendering is v1's: the premise is the state, the question is "Assess the claim
using only the supplied evidence: <hypothesis>" over supported / insufficient / contradicted with v1's descriptions, the
option order is shuffled per row (seeded), and pairs sharing a premise share a group so the bootstrap keeps them together.
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from kev.suite import CONTEXT, digest, read_jsonl, record_digest, write_json, write_jsonl

REVISION = "61c95318fd71c55b6ba355d76253254615f387ec"
V1 = "evals/external/wanli-v1/development.jsonl"
LABELS = {"entailment": "supported", "neutral": "insufficient", "contradiction": "contradicted"}
DESCRIPTIONS = {"supported": "The evidence establishes the claim", "insufficient": "The evidence does not establish either",
                "contradicted": "The evidence establishes the opposite"}
PREFIX = "Assess the claim using only the supplied evidence: "
norm = lambda text: " ".join(text.casefold().split())
sha = lambda text: hashlib.sha256(text.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help=f"alisawuffles/WANLI test.jsonl at revision {REVISION}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_label", type=int, default=334)
    ap.add_argument("--seed", default="wanli-v2")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists(): raise FileExistsError(out)
    v1 = read_jsonl(V1)
    used_ids = {r["_meta"]["provenance"]["source_id"] for r in v1}
    used_text = {norm(r["state"]) for r in v1} | {norm(r["questions"]["decision"]["instructions"].removeprefix(PREFIX)) for r in v1}
    rows = read_jsonl(a.source)
    fresh = [r for r in rows if r["id"] not in used_ids and norm(r["premise"]) not in used_text and norm(r["hypothesis"]) not in used_text]
    rng = random.Random(a.seed)
    by_label = defaultdict(list)
    for r in fresh: by_label[r["gold"]].append(r)
    chosen = [r for gold in sorted(by_label) for r in rng.sample(by_label[gold], a.per_label)]
    rng.shuffle(chosen)
    records = []
    for r in chosen:
        keys = list(DESCRIPTIONS); rng.shuffle(keys)
        q = {"type": "choice", "instructions": PREFIX + r["hypothesis"], "criteria": {k: DESCRIPTIONS[k] for k in keys}, "label": LABELS[r["gold"]], "src": "wanli_nli"}
        rec = {"state": r["premise"], "questions": {"decision": q},
               "_meta": {"row": str(r["id"]), "source": "wanli", "repo": "alisawuffles/WANLI", "split": "test", "id": f"wanli/v2/{r['id']}",
                         "group_id": f"wanli/{sha(norm(r['premise']))[:20]}", "variant": "clean", "family": "evidence_interpretation",
                         "provenance": {"source": "WANLI", "source_id": r["id"], "source_pair_id": r.get("pairID"), "source_revision": REVISION,
                                        "source_official_split": "test", "original_label": r["gold"], "genre": r.get("genre"), "rights": "CC-BY-4.0"},
                         "text_sha256": sha(json.dumps(r["premise"], sort_keys=True, ensure_ascii=False).casefold())}}
        rec["_meta"]["row_sha256"] = record_digest({k: v for k, v in rec.items() if k != "_meta"})
        records.append(rec)
    assert not {norm(r["state"]) for r in records} & used_text and not {r["_meta"]["provenance"]["source_id"] for r in records} & used_ids
    out.mkdir(parents=True)
    files = {}
    for name, part in (("development.jsonl", records), ("train.jsonl", []), ("calibration.jsonl", []), ("test.jsonl", [])):
        write_jsonl(out / name, part); files[name] = {"sha256": digest(out / name), "records": len(part)}
    write_json(out / "manifest.json", {"version": 2, "external": {"repo": "https://huggingface.co/datasets/alisawuffles/WANLI", "revision": REVISION, "split": "test",
                                                                  "license": "CC-BY-4.0", "source_sha256": digest(a.source), "rows_in_source": len(rows)},
                                       "disjoint_from": {"path": V1, "sha256": digest(V1), "by": "source id, premise text, hypothesis text (case- and whitespace-normalised)"},
                                       "selection": {"seed": a.seed, "per_label": a.per_label, "eligible": len(fresh), "code_sha256": digest(Path(__file__))},
                                       "base_revisions": {}, "dataset_revisions": {"alisawuffles/WANLI": REVISION}, "holdout_sources": [], "trainable_sources": [],
                                       "eval_only_sources": ["wanli"], "context": CONTEXT, "files": files, "eval_only": True, "tasks": ["wanli_nli"],
                                       "protocol": {"note": "Rendered as wanli-v1 (SemIf's rendering); a guard panel for round 6: paired accuracy interval against the parent, margin -2 pp (n >= 500)."}})
    print({"eligible": len(fresh), "records": len(records), "labels": {g: sum(r["questions"]["decision"]["label"] == LABELS[g] for r in records) for g in LABELS},
           "groups": len({r["_meta"]["group_id"] for r in records})})


if __name__ == "__main__":
    main()
