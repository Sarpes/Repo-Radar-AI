# -*- coding: utf-8 -*-
###############################################################
# tarihce.py | DS21tarihce0001 raw JSONL  ->  model-ready artifacts
###############################################################
# WHY THIS FILE REPLACES TWO
# --------------------------
# It takes over from fetch_commit_history.py (the ~5 h network job) and
# merge_commits.py (the JSONL -> parquet merge). Neither is needed any more:
# DS21tarihce0001 does the fetching, on a schedule, for BOTH endpoints.
#
#   /repos/{o}/{r}/stats/contributors   -> commit_weeks.jsonl  + commit_durum.jsonl
#   /repos/{o}/{r}/activity             -> push_events.jsonl   + push_durum.jsonl
#
# Keeping the tidy schema in ONE place is the point. It used to live in two
# files with two copies of the same column contract; whichever changed first
# would have drifted away silently.
#
# WHAT IT WRITES  (same names the chain already reads - no consumer changes)
# -------------------------------------------------------------------------
#   outputs/commit_weeks.parquet   full_name . week . commits . contributors
#   outputs/push_events.parquet    full_name . event_id . ts . kind . ref . actor
#   outputs/fetch_log.jsonl        one row per repo, BOTH statuses
#
# THE STATUS VOCABULARY TRANSLATION - the least obvious part of this file
# ----------------------------------------------------------------------
# DS21_Final_v2.py splits the commit-block NaN by CAUSE, and it does that by
# matching the log's `status` against two frozen sets. The new workflow writes
# a DIFFERENT vocabulary in a DIFFERENT field (`durum`). Left untranslated,
# nothing raises: `.isin(...)` simply matches nothing and both flags silently
# become zero, throwing away the two features the 2026-09-08 audit earned.
#
# So the translation below is not cosmetic. It is the difference between a
# model with cm_empty_repo/cm_fetch_failed and a model that thinks every repo
# was fetched successfully.
#
# PUSH IS KEPT AS RAW EVENTS, NOT AGGREGATED - deliberate
# ------------------------------------------------------
# score_today.py recomputes features at a DIFFERENT cutoff (today). Summarising
# at merge time would bake one cutoff into the file and make live scoring
# impossible. The commit side already works this way; push mirrors it.
#
# USAGE
#   python tarihce.py                    dry run - reports, writes NOTHING
#   python tarihce.py --apply            archive existing, then write
#   python tarihce.py --what commit      only the commit artifact
#   python tarihce.py --what push        only the push artifact
###############################################################

import argparse
import io
import json
import os
import shutil
import sys
import time
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import numpy as np
import pandas as pd

###############################################################
# PATHS
###############################################################
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CSV_PATH = os.path.join(PROJECT_ROOT, "Data", "github_ai_intelligence_2026.csv")

# n8n can only write under C:\Miuul (mounted as /data in the container).
FETCH_DIR = r"C:\Miuul\ds21final\tarihce"
F_COMMIT_WEEKS = os.path.join(FETCH_DIR, "commit_weeks.jsonl")
F_COMMIT_STATE = os.path.join(FETCH_DIR, "commit_durum.jsonl")
F_PUSH_EVENTS = os.path.join(FETCH_DIR, "push_events.jsonl")
F_PUSH_STATE = os.path.join(FETCH_DIR, "push_durum.jsonl")

OUT_COMMIT = os.path.join(OUTPUT_DIR, "commit_weeks.parquet")
OUT_PUSH = os.path.join(OUTPUT_DIR, "push_events.parquet")
OUT_LOG = os.path.join(OUTPUT_DIR, "fetch_log.jsonl")

# EK GIRDI - cek_eksikler.py'nin urettigi elle-tamamlama dosyalari.
# n8n bir repoyu 5 kez dener, sonra birakir; cekim bitince 236 repo `deneme=5`
# takili kaldi ve workflow onlari BIR DAHA denemeyecekti. cek_eksikler.py o
# 236'yi PAT ile tek tek cekti (12'si kurtarildi).
#
# NEDEN AYRI DOSYA: `C:\Miuul\ds21final\tarihce\*` YAZMA yasakli (n8n'in canli
# dizini). Ek kayitlar bu yuzden proje klasorune yaziliyor ve burada ana
# dosyalarin uzerine EKLENIYOR. Kayit secimi `latest_per_repo` ile zaman
# damgasina gore yapildigi icin bu kayitlar - bugun cekildikleri icin -
# n8n'inkileri OTOMATIK geceler; dosya sirasinin onemi yok.
EK_COMMIT_WEEKS = os.path.join(OUTPUT_DIR, "ek_commit_weeks.jsonl")
EK_COMMIT_STATE = os.path.join(OUTPUT_DIR, "ek_commit_durum.jsonl")

ARCHIVE_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "DS21_Final_ARSIV", "otomatik")

###############################################################
# CONTRACT WITH THE MODEL
###############################################################
# Kept here as constants AND cross-checked against model.pkl below, because a
# label window that is not actually covered is the one failure this file must
# never let through.
# AUDIT 2026-09-10 → docs/AUDIT_DEFTERI.md#tarihce-102: week-boundary fix.
CUTOFF_T = pd.Timestamp("2026-03-08")    # SUNDAY
HORIZON_DAYS = 182                       # 26 whole weeks

assert CUTOFF_T.dayofweek == 6, "CUTOFF_T must be a Sunday (GitHub weeks start Sunday)"
assert HORIZON_DAYS % 7 == 0, "HORIZON_DAYS must be a whole number of weeks"
COVERAGE_SLACK_DAYS = 7          # same slack DS21_Final_v2.py:797 allows

# tarihce `durum`  ->  the vocabulary DS21_Final_v2.py already matches on.
#   EMPTY_STATUS = {"error:204", "empty"}                        -> cm_empty_repo
#   FAIL_STATUS  = {"not_found", "timeout_202", "error:Connect"} -> cm_fetch_failed
# "error:fetch" is NEW and must be added to FAIL_STATUS in both consumers.
COMMIT_STATUS_MAP = {
    "ok":           "ok",
    "bos":          "empty",          # GitHub says: this repo genuinely has none
    "erisilemez":   "not_found",      # 404 - repo is gone
    "hesaplaniyor": "timeout_202",    # stats still being computed on GitHub's side
    "hata":         "error:fetch",    # our failure  <- NEW TOKEN
}

# Push statuses pass through untranslated: they have no legacy consumer, and
# their distinctions are the whole point (see the pa_* design note).
#   tam / api_bitti      -> zero is CORRECT: we looked, there was nothing
#   yok                  -> genuinely silent
#   tavan / derinlesiyor -> WE DID NOT LOOK - our limit, not the repo's
#   erisilemez           -> repo is gone
#   hata                 -> our failure
PUSH_LOOKED = {"tam", "api_bitti", "yok"}
PUSH_TRUNCATED = {"tavan", "derinlesiyor"}

BAR = "#" * 63


def header(text):
    print("\n" + BAR)
    print(f"# {text}")
    print(BAR)


def read_jsonl(path):
    """Yield parsed records, skipping blank and unparseable lines.

    Unparseable lines are counted, not silently dropped: the writer appends one
    repo at a time so a torn line means the run was interrupted mid-write, and
    that is worth seeing rather than absorbing.
    """
    ok = bad = 0
    if not os.path.exists(path):
        return [], 0, 0
    rows = []
    for line in io.open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
            ok += 1
        except ValueError:
            bad += 1
    return rows, ok, bad


def latest_per_repo(rows, when):
    """Collapse append-mode records to the most recent one per repo.

    Both state files are append-only, so a repo that was retried appears more
    than once. The LAST write wins - it carries the outcome of the most recent
    attempt, which is the only one that describes the data now on disk.
    """
    best = {}
    for r in rows:
        name = r.get("full_name")
        if not name:
            continue
        prev = best.get(name)
        if prev is None or str(r.get(when, "")) >= str(prev.get(when, "")):
            best[name] = r
    return best


###############################################################
# BUILDERS
###############################################################
def build_commit_weeks(rows):
    """JSONL (one row per repo, weeks nested) -> tidy (one row per repo-week).

    MERGE RULE: /stats/contributors returns the repo's ENTIRE lifetime in one
    response, so a repo's new data REPLACES its old data - it is never appended.
    Appending would double every week. Because this file rebuilds the parquet
    from the fetch directory in full, replacement is automatic; the rule only
    matters for the duplicate-record case handled by `latest_per_repo`.
    """
    best = latest_per_repo(rows, "cekilme")
    out = []
    empty_payload = 0
    for name, rec in best.items():
        weeks = rec.get("haftalar") or []
        if not weeks:
            empty_payload += 1
            continue
        for w, c, k in weeks:
            out.append((name, w, c, k))
    df = pd.DataFrame(out, columns=["full_name", "week", "commits", "contributors"])
    if len(df):
        df["week"] = pd.to_datetime(df["week"], unit="s")
    return df, len(best), empty_payload


def build_push_events(rows):
    """JSONL (one row per repo-PAGE, events nested) -> tidy (one row per event).

    DEDUPLICATION IS MANDATORY, and the key is the event id.

    Unlike the commit side, a repo appears once per fetched PAGE, and pages can
    overlap: the workflow deliberately refuses to trust ordering (it takes the
    whole page and stops on TIMESTAMP, not on a watermark), because the old
    "ids are strictly descending" assumption was measured wrong on 3.37% of
    records. That decision is correct and it makes duplicates expected, not
    exceptional - the previous fetch carried 8,861 of them.

    The id is kept in the output so the deduplication stays auditable and a
    re-merge stays idempotent.
    """
    seen = {}
    total = 0
    for rec in rows:
        name = rec.get("full_name")
        if not name:
            continue
        for ev in rec.get("olay") or []:
            total += 1
            eid = ev[0]
            # A given id belongs to exactly one repo; first sighting wins.
            if eid not in seen:
                seen[eid] = (name, eid, ev[1], ev[2], ev[3], ev[4])
    df = pd.DataFrame(list(seen.values()),
                      columns=["full_name", "event_id", "ts", "kind", "ref", "actor"])
    if len(df):
        df["ts"] = pd.to_datetime(df["ts"], unit="s")
    return df, total


def build_fetch_log(commit_state, push_state, commit_weeks, push_events):
    """One row per repo, BOTH statuses, in the vocabulary the model matches on.

    The commit `status` field deliberately keeps its legacy name and legacy
    tokens: build_fetch_flags() in DS21_Final_v2.py and score_today.py match
    against frozen sets, and this file's job is to speak their language.
    """
    names = set(commit_state) | set(push_state)

    if len(commit_weeks):
        g = commit_weeks.groupby("full_name")["week"]
        first_w, last_w, n_w = g.min(), g.max(), g.size()
    else:
        first_w = last_w = n_w = pd.Series(dtype="object")

    if len(push_events):
        pg = push_events.groupby("full_name")["ts"]
        p_earliest, p_n = pg.min(), pg.size()
    else:
        p_earliest = p_n = pd.Series(dtype="object")

    unmapped = set()
    out = []
    for name in sorted(names):
        c = commit_state.get(name, {})
        p = push_state.get(name, {})
        raw = c.get("durum")
        status = COMMIT_STATUS_MAP.get(raw)
        if raw is not None and status is None:
            unmapped.add(raw)
            status = f"error:unmapped:{raw}"
        row = {
            "full_name": name,
            "status": status,
            "n_weeks": int(n_w.get(name, 0)),
            "first_week": None if name not in first_w.index else str(first_w[name].date()),
            "last_week": None if name not in last_w.index else str(last_w[name].date()),
            "push_status": p.get("durum"),
            "push_events": int(p_n.get(name, 0)),
            "push_earliest": None if name not in p_earliest.index else str(p_earliest[name].date()),
            "push_pages": int(p.get("sayfa") or 0),
        }
        out.append(row)
    return out, unmapped


###############################################################
# GATES - these raise and stop the run. A comment is not protection.
###############################################################
def gate(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('   ' + detail) if detail else ''}")
    return bool(ok)


def run_gates(commit_weeks, push_events, log_rows, unmapped, want_commit, want_push,
              commit_state):
    header("GATES")
    ok = True

    # 1. COVERAGE - the label window must actually be observed.
    #    Without this a stale or partial fetch would turn live repos into
    #    negatives, which is precisely the failure the commit label exists to
    #    avoid. Mirrors DS21_Final_v2.py:797, one step earlier in the pipeline.
    if want_commit and len(commit_weeks):
        reach = commit_weeks["week"].max()
        need = CUTOFF_T + pd.Timedelta(days=HORIZON_DAYS - COVERAGE_SLACK_DAYS)
        ok &= gate("coverage: commit data reaches the label window",
                   reach >= need,
                   f"reaches {reach.date()}, needs {need.date()}")
    elif want_commit:
        ok &= gate("coverage: commit data present", False, "no rows at all")

    # 2. CORPUS - nothing may appear that is not in the CSV.
    csv_names = set(pd.read_csv(CSV_PATH, usecols=["full_name"])["full_name"])
    for label, df in (("commit", commit_weeks), ("push", push_events)):
        if not len(df):
            continue
        stray = set(df["full_name"]) - csv_names
        ok &= gate(f"corpus: every {label} repo is in the CSV",
                   not stray, f"{len(stray):,} stray" if stray else "")

    # 3. UNIQUENESS - deduplication actually happened.
    if want_push and len(push_events):
        dup = int(push_events["event_id"].duplicated().sum())
        ok &= gate("uniqueness: event_id is unique", dup == 0, f"{dup:,} duplicates")

    # 4. TYPES - the narrow integer types must be lossless.
    if want_commit and len(commit_weeks):
        c_max = int(commit_weeks["commits"].max())
        k_max = int(commit_weeks["contributors"].max())
        ok &= gate("types: commits fits int32", c_max <= np.iinfo("int32").max, f"max {c_max:,}")
        ok &= gate("types: contributors fits int16", k_max <= np.iinfo("int16").max, f"max {k_max:,}")
    if want_push and len(push_events):
        e_max = int(push_events["event_id"].max())
        ok &= gate("types: event_id fits int64", e_max <= np.iinfo("int64").max, f"max {e_max:,}")

    # 5. VOCABULARY - every commit status was translated.
    #    An unmapped token would silently zero cm_empty_repo / cm_fetch_failed.
    ok &= gate("vocabulary: every commit status is mapped",
               not unmapped, f"unmapped: {sorted(unmapped)}" if unmapped else "")

    # 6. NO REGRESSION - this file REBUILDS the parquet from the fetch directory
    #    rather than merging into the old one, so a run made while the fetch is
    #    still in flight would drop every repo not yet collected. Measured on the
    #    first dry run: 17,574 repos fetched vs 24,664 already in the file, i.e.
    #    applying then would have silently deleted 7,090 repos' history.
    #    The old merge_commits.py avoided this by preserving untouched rows; the
    #    rebuild design cannot, so the protection has to be a gate instead.
    #    FIRST FORM WAS WRONG, and measured wrong on 2026-09-10 05:10:
    #      "new_names >= old_names" can NEVER be satisfied, because repos get
    #      DELETED from GitHub between fetches. The old parquet was built when
    #      21 repos were unreachable; the new fetch finds 61. Those 40 are gone
    #      for good, so the new count is structurally lower and the gate would
    #      have blocked --apply forever.
    #    The right question is not "are there as many?" but "did we silently
    #    LOSE one?" - a repo may leave the parquet only for a reason GitHub
    #    gives us: it is gone (erisilemez), or it genuinely has no commits (bos).
    if want_commit and os.path.exists(OUT_COMMIT):
        old_names = set(pd.read_parquet(OUT_COMMIT, columns=["full_name"])["full_name"])
        new_names = set(commit_weeks["full_name"]) if len(commit_weeks) else set()
        # AUDIT 2026-09-11 → docs/AUDIT_DEFTERI.md#tarihce-369: `explained` genisletildi ve bu bir ZAYIFLATMA DEGIL,
        explained = {n for n, r in commit_state.items() if r.get("durum") != "ok"}
        lost = old_names - new_names - explained
        ok &= gate("no regression: no repo silently dropped from the commit parquet",
                   not lost,
                   f"{len(lost):,} unexplained  (old {len(old_names):,} · "
                   f"new {len(new_names):,} · gone/empty {len(explained):,})")
        if lost:
            print(f"        first 5: {sorted(lost)[:5]}")

    # 7. MODEL CONTRACT - the constants here still match the shipped model.
    model_path = os.path.join(OUTPUT_DIR, "model.pkl")
    if os.path.exists(model_path):
        try:
            import joblib
            m = joblib.load(model_path)
            ok &= gate("contract: cutoff_T matches model.pkl",
                       pd.Timestamp(m["cutoff_T"]) == CUTOFF_T, str(m["cutoff_T"]))
            ok &= gate("contract: horizon matches model.pkl",
                       int(m["horizon_days"]) == HORIZON_DAYS, str(m["horizon_days"]))
        except Exception as e:                                    # noqa: BLE001
            print(f"  SKIP  contract: could not read model.pkl ({e})")

    return ok


###############################################################
# WRITE
###############################################################
def archive_existing(paths, stamp):
    """Move current outputs aside before overwriting. There is no git here, so
    this is the only undo that exists."""
    dest = os.path.join(ARCHIVE_ROOT, stamp)
    moved = []
    for p in paths:
        if os.path.exists(p):
            os.makedirs(dest, exist_ok=True)
            shutil.move(p, os.path.join(dest, os.path.basename(p)))
            moved.append(os.path.basename(p))
    return dest, moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="archive existing outputs, then write. Without it: dry run.")
    ap.add_argument("--what", choices=["commit", "push", "all"], default="all")
    args = ap.parse_args()

    want_commit = args.what in ("commit", "all")
    want_push = args.what in ("push", "all")

    header("tarihce.py | DS21tarihce0001 -> model artifacts")
    print(f"  mode        {'APPLY (writes)' if args.apply else 'DRY RUN (writes nothing)'}")
    print(f"  what        {args.what}")
    print(f"  fetch dir   {FETCH_DIR}")
    print(f"  cutoff T    {CUTOFF_T.date()}   horizon {HORIZON_DAYS} d"
          f"   label window ends {(CUTOFF_T + pd.Timedelta(days=HORIZON_DAYS)).date()}")

    t0 = time.time()

    header("READING")
    cw_rows, cw_ok, cw_bad = read_jsonl(F_COMMIT_WEEKS)
    cs_rows, cs_ok, cs_bad = read_jsonl(F_COMMIT_STATE)
    pe_rows, pe_ok, pe_bad = read_jsonl(F_PUSH_EVENTS)
    ps_rows, ps_ok, ps_bad = read_jsonl(F_PUSH_STATE)
    for label, ok_n, bad_n in (("commit_weeks", cw_ok, cw_bad),
                               ("commit_durum", cs_ok, cs_bad),
                               ("push_events", pe_ok, pe_bad),
                               ("push_durum", ps_ok, ps_bad)):
        flag = "" if bad_n == 0 else f"   <- {bad_n} TORN LINE(S)"
        print(f"  {label:<14} {ok_n:>9,} records{flag}")

    # EK GIRDI - elle tamamlanan repolar (cek_eksikler.py). Ana listelerin
    # SONUNA ekleniyor, ama sirali oncelik degil ZAMAN DAMGASI karar veriyor
    # (latest_per_repo): bu kayitlar daha yeni oldugu icin n8n'inkileri geciyor.
    ek_cw, ek_cw_ok, ek_cw_bad = read_jsonl(EK_COMMIT_WEEKS)
    ek_cs, ek_cs_ok, ek_cs_bad = read_jsonl(EK_COMMIT_STATE)
    if ek_cw_ok or ek_cs_ok:
        cw_rows += ek_cw
        cs_rows += ek_cs
        print(f"\n  EK (cek_eksikler.py):")
        print(f"    ek_commit_weeks  {ek_cw_ok:>7,} records"
              f"{'' if ek_cw_bad == 0 else f'   <- {ek_cw_bad} TORN'}")
        print(f"    ek_commit_durum  {ek_cs_ok:>7,} records"
              f"{'' if ek_cs_bad == 0 else f'   <- {ek_cs_bad} TORN'}")

    commit_state = latest_per_repo(cs_rows, "cekim")
    push_state = latest_per_repo(ps_rows, "cekim")
    print(f"\n  repos with a commit state : {len(commit_state):,}")
    print(f"  repos with a push state   : {len(push_state):,}")

    header("COMMIT SIDE")
    commit_weeks, n_repo, n_empty = build_commit_weeks(cw_rows) if want_commit else (pd.DataFrame(), 0, 0)
    if want_commit:
        print(f"  repos with weekly data    : {n_repo:,}   (payload empty: {n_empty:,})")
        print(f"  tidy rows                 : {len(commit_weeks):,}")
        if len(commit_weeks):
            print(f"  week range                : {commit_weeks['week'].min().date()}"
                  f"  ->  {commit_weeks['week'].max().date()}")
            commit_weeks["commits"] = commit_weeks["commits"].astype("int32")
            commit_weeks["contributors"] = commit_weeks["contributors"].astype("int16")
            commit_weeks = commit_weeks.sort_values(["full_name", "week"]).reset_index(drop=True)
        cnt = pd.Series([r.get("durum") for r in commit_state.values()]).value_counts()
        print("\n  status:")
        for k, v in cnt.items():
            print(f"    {str(k):<14} {v:>7,}  ->  {COMMIT_STATUS_MAP.get(k, '?')}")

    header("PUSH SIDE")
    push_events, n_seen = build_push_events(pe_rows) if want_push else (pd.DataFrame(), 0)
    if want_push:
        dup = n_seen - len(push_events)
        print(f"  events seen               : {n_seen:,}")
        print(f"  after deduplication       : {len(push_events):,}   (removed {dup:,})")
        if len(push_events):
            print(f"  time range                : {push_events['ts'].min().date()}"
                  f"  ->  {push_events['ts'].max().date()}")
            print(f"  distinct repos            : {push_events['full_name'].nunique():,}")
            push_events = push_events.sort_values(["full_name", "ts"]).reset_index(drop=True)
            print("\n  activity kinds (top 8):")
            for k, v in push_events["kind"].value_counts().head(8).items():
                print(f"    {k:<22} {v:>9,}")
        cnt = pd.Series([r.get("durum") for r in push_state.values()]).value_counts()
        print("\n  status:")
        for k, v in cnt.items():
            tag = ("looked" if k in PUSH_LOOKED
                   else "TRUNCATED - we did not look" if k in PUSH_TRUNCATED else "")
            print(f"    {str(k):<14} {v:>7,}  {tag}")

    header("FETCH LOG")
    log_rows, unmapped = build_fetch_log(commit_state, push_state, commit_weeks, push_events)
    print(f"  rows (one per repo)       : {len(log_rows):,}")
    sc = pd.Series([r["status"] for r in log_rows]).value_counts(dropna=False)
    for k, v in sc.items():
        print(f"    status {str(k):<16} {v:>7,}")

    ok = run_gates(commit_weeks, push_events, log_rows, unmapped, want_commit, want_push,
                   commit_state)

    header("WRITE" if args.apply else "WRITE (skipped - dry run)")
    targets = []
    if want_commit:
        targets.append(OUT_COMMIT)
    if want_push:
        targets.append(OUT_PUSH)
    targets.append(OUT_LOG)

    if not args.apply:
        for p in targets:
            state = "exists, would be archived" if os.path.exists(p) else "new"
            print(f"  would write  {os.path.basename(p):<24} ({state})")
        print("\n  nothing was written. Re-run with --apply.")
    elif not ok:
        print("  REFUSED: a gate failed. Nothing was written.")
        return 1
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        dest, moved = archive_existing(targets, stamp)
        if moved:
            print(f"  archived to  {dest}")
            for m in moved:
                print(f"               {m}")
        if want_commit:
            commit_weeks.to_parquet(OUT_COMMIT, index=False)
            print(f"  wrote        {OUT_COMMIT}  ({len(commit_weeks):,} rows)")
        if want_push:
            push_events.to_parquet(OUT_PUSH, index=False)
            print(f"  wrote        {OUT_PUSH}  ({len(push_events):,} rows)")
        with io.open(OUT_LOG, "w", encoding="utf-8") as f:
            for r in log_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote        {OUT_LOG}  ({len(log_rows):,} rows)")

    print(f"\ndone in {time.time() - t0:.0f} s   ({'OK' if ok else 'GATE FAILURE'})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
