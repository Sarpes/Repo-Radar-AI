# -*- coding: utf-8 -*-
###############################################################
# govde_yukle.py | README GOVDE vektorlerini pgvector'e yukler
###############################################################
# NE YAPIYOR
# ----------
# build_body_index.py `README[900:4000]` icin 21.816 vektor uretip diske yazdi
# ama pgvector'e YUKLEYICISI YOKTU - dosya olarak var, canlida yok. Bu betik
# o boslugu kapatiyor.
#
#     repos        24.664 satir   README[:900] + kimlik + description + topics
#     repo_govde   21.816 satir   README[900:4000], CIPLAK metin
#
# Aramada iki vektorden MAX alinir:  skor(repo) = max(sim(q,bas), sim(q,govde))
# Ortalama DEGIL - ilgisiz bir govde skoru DUSURMEZ, sadece katki vermez.
#
# NEDEN AYRI TABLO, `repos`'a SUTUN DEGIL
# ---------------------------------------
# 1. `repos` 24.664, govde 21.816 - %11,5'inde README 900 karakteri asmiyor.
#    Sutun olsaydi 2.848 NULL tasirdi ve her sorgu onlari elemek zorunda kalirdi.
# 2. DS21_RAG.py `repos`'a TRUNCATE atiyor. Ayri tablo o pencereden ETKILENMEZ.
# 3. `repos` canli chatbot'un bagli oldugu tablo; deneysel bir indeks onun
#    semasini degistirmemeli.
#
# ⚠️ YABANCI ANAHTAR KASTEN YOK: `repos`'a FK koysaydik DS21_RAG'in TRUNCATE'i
# ya patlardi ya da CASCADE ile govdeyi SESSIZCE silerdi. Bunun yerine bu
# betik yuklemeden SONRA hizalamayi ACIKCA denetliyor.
#
# KULLANIM
#   python govde_yukle.py            kuru kosum - ne yapacagini soyler
#   python govde_yukle.py --apply    tabloyu kurar ve doldurur
#   python govde_yukle.py --drop     tabloyu siler (geri alma)
###############################################################
import argparse
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
SUFFIX = os.environ.get("DS21_SUFFIX", "_qwen8b")
EMB_DIM = int(os.environ.get("DS21_EMB_DIM", "4096"))

DOC = os.path.join(OUT, f"rag_documents{SUFFIX}.parquet")
EMB = os.path.join(OUT, f"embeddings{SUFFIX}_body.npy")
ROW = os.path.join(OUT, f"body_rows{SUFFIX}.npy")

PG = dict(host="localhost", port=5433, dbname="repoquest",
          user="ds21", password=os.environ.get("DS21_PG_PASSWORD", ""))

HEAD_CHARS = 900          # DS21_RAG.py:150 ile AYNI olmak zorunda
TABLO = "repo_govde"


def basl(t):
    print("\n" + "#" * 18 + f" {t} " + "#" * 18)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--drop", action="store_true")
    args = ap.parse_args()

    conn = psycopg2.connect(**PG)
    cur = conn.cursor()

    if args.drop:
        cur.execute(f"DROP TABLE IF EXISTS {TABLO}")
        conn.commit()
        print(f"{TABLO} silindi. `repos` el degmedi.")
        return 0

    basl("Girdi")
    for y in (DOC, EMB, ROW):
        if not os.path.exists(y):
            raise SystemExit(f"eksik: {y}\n  once: python build_body_index.py")

    docs = pd.read_parquet(DOC, columns=["full_name", "clean_chars"])
    E = np.load(EMB, mmap_mode="r")
    rows = np.load(ROW)
    print(f"  belge tablosu : {len(docs):,}")
    print(f"  govde vektoru : {E.shape}")
    print(f"  satir eslemesi: {rows.shape}")

    # 🚪 KAPI 1 - esleme hala gecerli mi?
    # body_rows, build_body_index.py kosarken gecerli olan BELGE SIRASINA gore
    # yazildi. rag_documents o gunden sonra yeniden uretildiyse ve sira
    # degistiyse, her govde YANLIS repoya baglanir - ve bu HATA VERMEZ,
    # sadece arama sonuclarini sessizce bozar. Bu yuzden bagimsiz bir
    # olcutle dogruluyoruz: govdesi olmasi gereken kume = clean_chars > 900.
    beklenen = set(np.where(docs["clean_chars"].values > HEAD_CHARS)[0].tolist())
    gercek = set(rows.tolist())
    if beklenen != gercek:
        raise SystemExit(
            f"ESLEME BOZUK: clean_chars>{HEAD_CHARS} olan {len(beklenen):,} satir "
            f"var ama body_rows {len(gercek):,} satir gosteriyor "
            f"(yalniz-beklenen {len(beklenen-gercek)}, yalniz-rows {len(gercek-beklenen)}).\n"
            f"rag_documents yeniden uretilmis ve sira degismis olabilir. "
            f"build_body_index.py'yi yeniden kostur.")
    if len(E) != len(rows):
        raise SystemExit(f"vektor {len(E):,} / esleme {len(rows):,} - uyusmuyor")
    if E.shape[1] != EMB_DIM:
        raise SystemExit(f"boyut {E.shape[1]} != beklenen {EMB_DIM}")
    print(f"  KAPI 1  esleme gecerli ({len(rows):,} repo, clean_chars>{HEAD_CHARS})")

    adlar = docs["full_name"].values[rows]

    # 🚪 KAPI 2 - bu repolarin hepsi `repos` icinde mi?
    cur.execute("SELECT count(*) FROM repos")
    n_repos = cur.fetchone()[0]
    cur.execute("SELECT full_name FROM repos")
    repos_adlari = {r[0] for r in cur.fetchall()}
    yabanci = set(adlar.tolist()) - repos_adlari
    if yabanci:
        raise SystemExit(
            f"HIZALAMA: {len(yabanci):,} govde reposu `repos` icinde YOK "
            f"(orn {sorted(yabanci)[:3]}).\n"
            f"`repos` {n_repos:,} satir. Once DS21_RAG.py kosmali.")
    print(f"  KAPI 2  {len(adlar):,} reponun hepsi `repos` icinde ({n_repos:,} satir)")

    if not args.apply:
        print(f"\n  kuru kosum. {TABLO} kurulacak ve {len(adlar):,} satir yazilacak.")
        print(f"  tahmini boyut: {len(adlar)*EMB_DIM*4/1e6:.0f} MB")
        print("  gercekten yapmak icin: --apply")
        return 0

    basl("Yaz")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLO} (
            full_name text PRIMARY KEY,
            embedding vector({EMB_DIM}) NOT NULL
        )
    """)
    cur.execute(f"TRUNCATE {TABLO}")
    conn.commit()

    t0 = time.time()
    veri = [(ad, np.asarray(E[i], dtype=np.float32).tolist())
            for i, ad in enumerate(adlar)]
    execute_values(cur, f"INSERT INTO {TABLO} (full_name, embedding) VALUES %s",
                   veri, page_size=200)
    cur.execute(f"ANALYZE {TABLO}")
    conn.commit()
    print(f"  {len(veri):,} satir yazildi  ({time.time()-t0:.0f} sn)")

    basl("Dogrulama")
    cur.execute(f"SELECT count(*), count(embedding) FROM {TABLO}")
    n, nv = cur.fetchone()
    print(f"  {TABLO:<12} satir {n:,} · vektor {nv:,}")
    cur.execute(f"SELECT count(*) FROM {TABLO} g LEFT JOIN repos r "
                f"USING (full_name) WHERE r.full_name IS NULL")
    oksuz = cur.fetchone()[0]
    print(f"  `repos`da karsiligi olmayan govde: {oksuz}")
    cur.execute("SELECT count(*) FROM repos")
    print(f"  repos EL DEGMEDI: {cur.fetchone()[0]:,} satir")
    if n != len(adlar) or oksuz:
        raise SystemExit("dogrulama BASARISIZ")
    print("\n  siradaki: yeni repo_ara SQL'ini Python'dan olc (n8n'e HENUZ dokunma)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
