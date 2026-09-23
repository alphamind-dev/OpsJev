"""Round-6 confirmation (PLAN.md, Round 6, rule 5): the selected candidate of one size against its released parent on the
fresh panels, read once. Rules 1 and 2 must hold; wanli-v2 is reported.

    uv run python scripts/round6_confirm.py --size 9b --candidate runs/r6-9b/00-trial-0 --out runs/r6-verdict

Reads runs/r6c-<size>-{cand,parent}-{long3,r6test,wanli2}/rows.json (one benchmarks pass per size). Each arm is served
at the temperature fitted on its own decision-v7 development rows (kev.metrics.served). Writes runs/r6-verdict/<size>.json.
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.metrics import metrics, paired_bootstrap, raw_row, recorded, served, tempered_row  # noqa: E402
from kev.suite import read_json, write_json  # noqa: E402

SAMPLES = 2000
PARENTS = {"9b": "runs/night2-9b-du/00-trial-0", "4b": "runs/night2-4b-du/00-trial-0", "08b": "runs/night2-08b-du2/00-trial-0"}
KEYS = ("n", "acc", "brier", "ece", "confident_error_rate", "coverage_at_5pct_error", "aurc")


def serve(trial, path, t):
    return [tempered_row(raw_row(recorded(r)), t) for r in read_json(path) if r["variant"] == "clean" and r["source"] != "unknowable"]


def boot(a, b, metric):
    x = paired_bootstrap(a, b, samples=SAMPLES, seed=0, metric=metric, aggregation="micro")
    return {"delta": x[f"micro_{metric}_delta"], "ci95": x["ci95"]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--size", required=True, choices=list(PARENTS)); ap.add_argument("--candidate", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = {"cand": a.candidate, "parent": PARENTS[a.size]}
    t = {k: served(read_json(Path(v) / "development/rows.json"), [])[0] for k, v in arms.items()}
    rows = {(k, suite): serve(v, f"runs/r6c-{a.size}-{k}-{suite}/rows.json", t[k]) for k, v in arms.items() for suite in ("long3", "r6test", "wanli2")}
    long = {k: [r for r in rows[k, "long3"] if r["source"] == "longstate"] for k in arms}
    rep = {"size": a.size, "candidate": a.candidate, "parent": arms["parent"], "temperature": t,
           "long3": {k: {"n": len(long[k]), "acc": metrics(long[k])["acc"]} for k in arms}, "long3_delta": boot(long["cand"], long["parent"], "acc"),
           "r6test": {k: {m: metrics(rows[k, "r6test"])[m] for m in KEYS} for k in arms},
           "r6test_delta": {m: boot(rows["cand", "r6test"], rows["parent", "r6test"], m) for m in ("acc", "brier", "confident_error_rate", "coverage_at_5pct_error", "aurc")},
           "wanli2": {k: metrics(rows[k, "wanli2"])["acc"] for k in arms}, "wanli2_delta": boot(rows["cand", "wanli2"], rows["parent", "wanli2"], "acc")}
    ld, sd = rep["long3_delta"], rep["r6test_delta"]
    rep["criteria"] = {"1_long_lower_above_0": ld["ci95"][0] > 0, "1_long_point_at_least_5pp": ld["delta"] >= 0.05,
                       "2_short_acc_lower_at_least_minus_1pp": sd["acc"]["ci95"][0] >= -0.01, "2_short_brier_upper_at_most_0.01": sd["brier"]["ci95"][1] <= 0.01,
                       "2_short_confident_errors_upper_at_most_1pp": sd["confident_error_rate"]["ci95"][1] <= 0.01}
    rep["passed"] = all(rep["criteria"].values())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); write_json(out / f"{a.size}.json", rep)
    f = lambda b: f"{b['delta']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]"
    print(f"== {a.size}: {a.candidate} vs parent  T {t}")
    print(f"  long3   acc {rep['long3']['parent']['acc']:.3f} -> {rep['long3']['cand']['acc']:.3f}  {f(ld)}  (n {rep['long3']['cand']['n']})")
    print("  r6test  " + "  ".join(f"{m} {f(v)}" for m, v in sd.items()))
    print(f"  wanli2  {rep['wanli2']['parent']:.3f} -> {rep['wanli2']['cand']:.3f}  {f(rep['wanli2_delta'])}")
    print("  criteria", rep["criteria"], "-> PASSED (locked read allowed)" if rep["passed"] else "-> failed: not a candidate")


if __name__ == "__main__":
    main()
