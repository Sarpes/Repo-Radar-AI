# -*- coding: utf-8 -*-
###############################################################
# n8n GIRDI DOSYALARI  |  refresh pipeline, Faz 1 + Faz 2
###############################################################
# n8n cannot read the project folder - only C:\Miuul (mounted as /data). This
# script writes the repo lists it needs there, in the exact shape the existing
# workflows expect.
#
# TWO CONSUMERS, TWO NAMING SCHEMES:
#   DS21readme0001 (already built) validates KADEME against a WHITELIST:
#       test · tier1 · tier2 · tier2a..tier2e
#   so the README chunks MUST reuse those names or the Code node throws.
#   DS21meta0001 (new, this refresh) is ours: meta1..meta5.
#
# safe_name is the README filename without ".md" - the cache-skip node keys on
# it, so getting it wrong silently refetches all 24,664 READMEs.
###############################################################

import io
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "Data", "github_ai_intelligence_2026.csv")
README_DIR = os.path.join(PROJECT_ROOT, "Data", "readme_raw")
OUT_DIR = r"C:\Miuul\ds21final\input"

N_CHUNKS = 5          # 24,664 / 5 ~ 4,933 -> ~1 h each, under the 5 h executionTimeout
N_SMOKE = 20


def header(text):
    print("\n" + "#" * 21 + f" {text} " + "#" * 21)


header("Kaynak")
df = pd.read_csv(CSV_PATH, usecols=["full_name"])
df["safe_name"] = df["full_name"].str.replace("/", "__", regex=False)
print(f"{CSV_PATH}\n  {len(df):,} repo")

# The convention is not assumed - it is checked against what is on disk. If
# this drifts, the cache-skip node stops recognising existing files and the
# README run silently becomes a full refetch.
header("safe_name sozlesmesi - diskle dogrulama")
if os.path.isdir(README_DIR):
    on_disk = {f[:-3] for f in os.listdir(README_DIR) if f.endswith(".md")}
    hit = df["safe_name"].isin(on_disk).sum()
    print(f"diskteki .md dosyasi : {len(on_disk):,}")
    print(f"safe_name eslesen    : {hit:,} / {len(df):,}  ({100*hit/len(df):.2f}%)")
    orphan = len(on_disk - set(df["safe_name"]))
    print(f"CSV'de karsiligi yok : {orphan}")
    if hit < 23_000:
        raise SystemExit("safe_name sozlesmesi TUTMUYOR - onbellek atlama calismaz, DUR.")
    print("-> sozlesme dogru")
else:
    print(f"{README_DIR} yok (Faz 2'de tasinmis olabilir) - dogrulama atlandi")

if df["safe_name"].duplicated().any():
    raise SystemExit("safe_name tekrar ediyor - dosya adlari cakisir, DUR.")

os.makedirs(OUT_DIR, exist_ok=True)


def write(name, frame):
    path = os.path.join(OUT_DIR, f"repos_{name}.csv")
    frame[["full_name", "safe_name"]].to_csv(path, index=False, encoding="utf-8")
    print(f"  repos_{name}.csv{'':<4} {len(frame):>7,} repo   {os.path.getsize(path)/1024:>7,.0f} KB")
    return len(frame)


header("README kademeleri  (DS21readme0001 - beyaz listedeki adlar)")
# tier2a..tier2e cover the WHOLE corpus this time. The original tier1/tier2
# split was about which repos were worth fetching first; the refresh needs all
# of them, and the cache-skip keeps the chunks independent and restartable.
total = write("test", df.head(N_SMOKE))
readme_names = ["tier2a", "tier2b", "tier2c", "tier2d", "tier2e"]
parts = [df.iloc[i::N_CHUNKS] for i in range(N_CHUNKS)]      # interleaved, not contiguous
covered = 0
for name, part in zip(readme_names, parts):
    covered += write(name, part)

header("Metadata kademeleri  (DS21meta0001 - yeni)")
meta_covered = 0
for i, part in enumerate(parts, 1):
    meta_covered += write(f"meta{i}", part)

header("Kapsama kontrolu")
# Interleaved slicing means every chunk is a random-ish sample of the corpus,
# so a partial run leaves a representative subset rather than "all the repos
# whose name starts with a".
union = pd.concat(parts)
print(f"parca toplami        : {covered:,}")
print(f"benzersiz full_name  : {union['full_name'].nunique():,}")
print(f"CSV satir sayisi     : {len(df):,}")
assert covered == len(df), "parcalar toplami CSV ile esit degil"
assert union["full_name"].nunique() == len(df), "parcalarda tekrar veya eksik var"
assert meta_covered == len(df), "meta parcalari eksik"
print("-> her repo tam olarak BIR parcada")

print(f"\nYazildi: {OUT_DIR}")
