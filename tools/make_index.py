# -*- coding: utf-8 -*-
"""Regenerates the CONTENTS block at the top of DS21_Final_v2.py.

Line numbers in a table of contents rot the moment anything above them moves,
so they are never typed by hand. This script finds each anchor, works out how
much the new block shifts everything below it, and writes the corrected
numbers. Re-run it after any edit that changes line counts.
"""
import io, os, re, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

TARGET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DS21_Final_v2.py")
BAR = "#" * 63

# (indent, anchor regex, label)  -- None anchor = literal text, no line number
SPEC = [
    (0, r"^import os$",                        "imports, plotting defaults"),
    (0, r"^# PATHS$",                          "PATHS"),
    (0, r"^# HELPER FUNCTIONS$",               "HELPERS  header / check_df / grab_col_names /"),
    (None, None,                               "         build_asof_features / align_by_name"),
    (None, "",                                 ""),
    (None, "", "--- EXPLORATORY DATA ANALYSIS " + "-" * 25),
    (0, r'header\("STEP 1 \| ',                "STEP 1   Loading the dataset, general picture"),
    (0, r'header\("STEP 2 \| ',                "STEP 2   Variable capture (cat / num / cardinal)"),
    (0, r'header\("STEP 3 \| ',                "STEP 3   LEAKAGE AUDIT          <- core of the project"),
    (1, r'header\("STEP 3\.5 \| ',             "3.5    cost of leakage: R2 0.33 -> 0.88 -> 0.9999"),
    (0, r'header\("STEP 4 \| ',                "STEP 4   TARGET DEFINITION + backdating"),
    (1, r'header\("STEP 4\.2 \| ',             "4.2    why H=180 and not H=90"),
    (0, r'header\("STEP 5 \| ',                "STEP 5   Numerical analysis"),
    (1, r'header\("STEP 5\.4 \| ',             "5.4    sampling audit - NOT a random sample of GitHub"),
    (0, r'header\("STEP 6 \| ',                "STEP 6   Categorical analysis vs target"),
    (0, r'header\("STEP 7 \| ',                "STEP 7   Missing values (missingness is a feature)"),
    (0, r'header\("STEP 8 \| ',                "STEP 8   Correlation in log space"),
    (0, r'header\("STEP 9 \| ',                "STEP 9   Group structure - why GroupKFold(owner)"),
    (0, r'header\("STEP 10 \| ',               "STEP 10  README corpus - usable for RAG?"),
    (1, r'header\("STEP 10\.5 \| ',            "10.5   CONTRACT for STEP 11 + the guards"),
    (None, "", ""),
    (None, "", "--- FEATURE ENGINEERING  (25 rubric points) " + "-" * 11),
    (0, r'header\("STEP 11 \| ',               "STEP 11  overview"),
    (1, r'header\("STEP 11a \| ',              "11a    README structural features   (15)"),
    (1, r'header\("STEP 11b \| ',              "11b    commit-history features      (22)"),
    (1, r"^# 11b'  PUSH-ACTIVITY FEATURES",    "11b'   push-activity features       (12)"),
    (1, r'header\("STEP 11c \| ',              "11c    assembly + guards  ->  features.parquet"),
    (1, r'header\("STEP 11c\.4 \| ',           "11c.4  ABLATION - what each block earns"),
    (1, r'header\("STEP 11c\.5 \| ',           "11c.5  AUDIT - README block EXCLUDED (measured)"),
    (None, "", ""),
    (None, "", "--- MODEL " + "-" * 45),
    (0, r'header\("STEP 12 \| ',               "STEP 12  candidates, selection, calibration, SHAP"),
    (1, r'header\("STEP 12\.1b \| ',           "12.1b  selection rule (AUC -> fold-std tie -> calib)"),
    (1, r'header\("STEP 12\.2 \| ',            "12.2   per-slice: where the model earns its place"),
    (1, r'header\("STEP 12\.6 \| ',            "12.6   ->  model.pkl, repo_scores.parquet"),
    (0, r'header\("STEP 13 \| ',               "STEP 13  INTERPRETATION"),
    (1, r'header\("STEP 13\.1 \| ',            "13.1   direction check: model vs EDA"),
    (1, r'header\("STEP 13\.3 \| ',            "13.3   per-repo explanations (the product's why)"),
    (1, r'header\("STEP 13\.5 \| ',            "13.5   MODEL CARD"),
    (1, r'header\("STEP 13\.6 \| ',            "13.6   maturity axis + ranking floor"),
    (None, "", ""),
    (None, "", "--- NOT IN THIS FILE (separate entry points) " + "-" * 10),
    (None, None, "STEP 14  embeddings + index     reads features.parquet"),
    (None, None, "STEP 15  retrieval evaluation   recall@k"),
    (None, None, "STEP 16  RAG chain + re-ranking reads model.pkl + scores"),
    (None, None, "panel    Streamlit UI"),
    (None, None, "n8n      daily refresh flow"),
]

src = open(TARGET, encoding="utf-8").read().split("\n")

# locate the block being replaced: the second BAR..BAR group (the STEP CHAIN)
bars = [i for i, l in enumerate(src) if l.startswith(BAR)]
start, end = bars[3], bars[5]          # 0-1 = title banner, 3..5 = STEP CHAIN
old_len = end - start + 1

def find(pat, shift):
    for i, l in enumerate(src):
        if re.search(pat, l):
            return i + 1 + shift       # 1-based, plus the shift the new block adds
    raise SystemExit("anchor not found: " + pat)

def build(shift):
    out = [BAR, "# CONTENTS   line numbers regenerated by make_index.py",
           "#            jump with Ctrl+G in PyCharm", BAR]
    for indent, pat, label in SPEC:
        if pat is None and indent is None:
            out.append("#" + " " * 8 + label if label else "#")
        elif pat == "":
            out.append("# " + label if label else "#")
        else:
            n = find(pat, shift)
            out.append(f"# {n:>4}  " + "  " * indent + label)
    out.append(BAR)
    return out

block = build(0)                        # first pass: measure
block = build(len(block) - old_len)     # second pass: with the true shift
src[start:end + 1] = block
open(TARGET, "w", encoding="utf-8").write("\n".join(src))
print(f"replaced lines {start+1}-{end+1} ({old_len} lines) with {len(block)} lines")
print(f"shift applied to everything below: {len(block) - old_len:+d}")
