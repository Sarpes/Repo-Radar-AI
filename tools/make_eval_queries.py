# -*- coding: utf-8 -*-
###############################################################
# DEGERLENDIRME SORGULARINI URET VE SABITLE
###############################################################
# NEDEN AYRI BIR DOSYA
# Sorgu seti bir BENCHMARK'tir, bir ara ciktisi degil. HEAD_CHARS ve chunk
# ayarlarini karsilastirabilmek icin BIR KEZ uretilip DEGISMEDEN kalmasi
# gerekiyor. DS21_RAG.py'nin icinde uretilirse her kosuda yeniden dogma
# riski var (dosya silinirse) ve o an hangi ayarla uretildigi belirsizlesir.
#
# KAYNAK: text_clean (tam temiz README), doc_primary DEGIL.
#   doc_primary indekslenen metindir. Ondan sorgu uretmek, test setini
#   test edilen seye baglar: HEAD_CHARS artinca hem indeks hem sorular
#   degisir ve iki ayar karsilastirilamaz.
#
# ⚠️ Bu betigin urettigi sorgu seti, 2026-09-07 oncesi olculen
#    0.408 / 0.581 / 0.646 ile KARSILASTIRILAMAZ. Sorgu yazari artik
#    README'nin 700 degil 4.000 karakterini goruyor; benchmark daha ZOR.
#    Yeni set icinde ayarlar birbiriyle karsilastirilabilir - amac bu.
#
# KULLANIM
#   python make_eval_queries.py              varsa dokunmaz, raporlar
#   python make_eval_queries.py --force      yeniden uretir (eskisini arsivler)
###############################################################
import argparse
import os
import shutil
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
import numpy as np
import pandas as pd

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(KOK, "Data", "github_ai_intelligence_2026.csv")
README_DIR = os.path.join(KOK, "Data", "readme_raw")
CIKTI = os.path.join(KOK, "outputs", "eval_queries.parquet")

GEN_MODEL = "qwen3:14b"
N_EN, N_TR = 60, 30
BANTLAR = [(0, 100, "<100"), (100, 1_000, "100-1K"),
           (1_000, 10_000, "1K-10K"), (10_000, 10 ** 9, "10K+")]
MIN_CHARS = 120
QUERY_SOURCE_CHARS = 4000

SYS_EN = (
    "A developer has a PROBLEM and is searching for a tool. You are given the tool. "
    "Write the one sentence that developer would type BEFORE they knew this tool existed.\n"
    "RULES:\n"
    "- Start with \"I need\", \"I want\", \"How do I\" or \"Looking for a way to\".\n"
    "- Describe the TASK they must accomplish, never the tool or its features.\n"
    "- Never name the tool, its author, or its repository.\n"
    "- 8 to 20 words. Plain sentence. No markdown. Output the sentence only.")
SYS_TR = (
    "A developer has a PROBLEM and is searching for a tool. You are given the tool. "
    "Write, IN TURKISH, the one sentence that developer would type BEFORE they knew "
    "this tool existed.\n"
    "RULES:\n"
    "- Write in natural Turkish, as a developer would actually type it.\n"
    "- Describe the TASK they must accomplish, never the tool or its features.\n"
    "- Never name the tool, its author, or its repository.\n"
    "- 8 to 20 words. Plain sentence. No markdown. Output the sentence only.")


def basl(t):
    print("\n" + "#" * 18 + f" {t} " + "#" * 18)


def temizle(ham):
    """DS21_RAG.py'deki clean_markdown ile AYNI olmak zorunda degil - burada
    yalnizca LLM'in okuyacagi metni uretiyoruz. Kaba temizlik yeterli:
    kod bloklari ve rozet satirlari soruyu bozar."""
    import re
    t = re.sub(r"```.*?```", " ", ham, flags=re.S)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[#*_>`|-]{2,}", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def uret(doc, system):
    import requests
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": GEN_MODEL, "system": system,
        "prompt": doc[:QUERY_SOURCE_CHARS], "stream": False,
        "options": {"temperature": 0.8, "num_predict": 60}, "think": False},
        timeout=300)
    return r.json().get("response", "").strip().replace("\n", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="varsa yeniden uret")
    args = ap.parse_args()

    if os.path.exists(CIKTI) and not args.force:
        ev = pd.read_parquet(CIKTI)
        print(f"{CIKTI} zaten var: {len(ev):,} sorgu")
        print(ev.groupby(["band", "lang"]).size().unstack(fill_value=0).to_string())
        print("\nyeniden uretmek icin: --force")
        return

    basl("Havuz")
    d = pd.read_csv(CSV, usecols=["full_name", "owner", "name", "stars"])
    metinler, uzunluk = {}, []
    for r in d.itertuples(index=False):
        p = os.path.join(README_DIR, f"{r.owner}__{r.name}.md")
        t = ""
        if os.path.exists(p):
            with open(p, encoding="utf-8", errors="replace") as fh:
                t = temizle(fh.read())
        metinler[r.full_name] = t
        uzunluk.append(len(t))
    d["clean_chars"] = uzunluk
    havuz = d[d["clean_chars"] >= MIN_CHARS]
    print(f"  CSV repo            : {len(d):,}")
    print(f"  README >= {MIN_CHARS} kar. : {len(havuz):,}   <- sorgu havuzu")
    print(f"  medyan temiz metin  : {havuz['clean_chars'].median():,.0f} karakter")
    print(f"  sorgu yazarinin gordugu: ilk {QUERY_SOURCE_CHARS:,} karakter")

    basl("Ornekleme")
    secim = []
    for lo, hi, bant in BANTLAR:
        alt = havuz[(havuz["stars"] >= lo) & (havuz["stars"] < hi)]
        n = min(N_EN + N_TR, len(alt))
        al = alt.sample(n, random_state=42)
        for i, r in enumerate(al.itertuples(index=False)):
            secim.append((r.full_name, r.stars, bant, "en" if i < N_EN else "tr"))
        print(f"  {bant:<8} havuz {len(alt):>6,}  secilen {n:>3}")
    print(f"  TOPLAM {len(secim)} sorgu   ~{len(secim)*5.5/60:.0f} dk beklenir")

    basl("Uretim")
    cikti, t0 = [], time.time()
    for k, (fn, stars, bant, lang) in enumerate(secim, 1):
        q = uret(metinler[fn], SYS_EN if lang == "en" else SYS_TR)
        cikti.append((fn, q, stars, bant, lang))
        if k % 30 == 0 or k == len(secim):
            hiz = k / (time.time() - t0)
            print(f"  {k:>4} / {len(secim)}  {hiz:4.2f} sorgu/sn  "
                  f"eta {(len(secim)-k)/hiz/60:4.1f} dk")
    ev = pd.DataFrame(cikti, columns=["full_name", "query", "stars", "band", "lang"])

    basl("Sizinti kapisi")
    sizan = [fn.split("/")[-1].lower() in q.lower()
             for fn, q in zip(ev["full_name"], ev["query"])]
    print(f"  repo adi sizan sorgu: {sum(sizan)}  -> atiliyor")
    bos = (ev["query"].str.len() < 15)
    print(f"  cok kisa / bos      : {int(bos.sum())}  -> atiliyor")
    ev = ev[~np.array(sizan) & ~bos].reset_index(drop=True)
    print(f"  KULLANILABILIR      : {len(ev):,}")
    print(ev.groupby(["band", "lang"]).size().unstack(fill_value=0).to_string())

    basl("Ornekler")
    for _, r in ev.groupby("lang").head(3).iterrows():
        print(f"  [{r['lang']}] {r['full_name']:<32} {r['query'][:78]}")

    if os.path.exists(CIKTI):
        yedek = CIKTI.replace(".parquet", f"_{time.strftime('%m%d_%H%M')}.parquet")
        shutil.move(CIKTI, os.path.join(KOK, "archive", os.path.basename(yedek)))
        print(f"\n  eski set arsivlendi: {os.path.basename(yedek)}")
    os.makedirs(os.path.dirname(CIKTI), exist_ok=True)
    ev.to_parquet(CIKTI, index=False)
    print(f"\n  {CIKTI}  ({len(ev):,} sorgu)")
    print(f"  toplam {(time.time()-t0)/60:.1f} dk")


if __name__ == "__main__":
    main()
