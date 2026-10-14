# -*- coding: utf-8 -*-
###############################################################
# KURULMUS BIR RAG INDEKSINI SABIT SORGU SETIYLE OLC
###############################################################
# Girdi
#   outputs/eval_queries.parquet            SABIT sinav (make_eval_queries.py)
#   outputs/rag_documents<SUFFIX>.parquet   olculecek indeksin belgeleri
#   outputs/embeddings<SUFFIX>.npy          olculecek indeksin vektorleri
# Cikti
#   outputs/rag_sweep.csv                   her kosu bir satir (biriktirir)
#
# NEDEN AYRI BIR ARAC
# HEAD_CHARS / chunk gibi ayarlari karsilastirmak icin sinav SABIT kalmali.
# Olcum DS21_RAG.py'nin icinde kalsaydi, her ayar kendi kosusunda sorgu seti
# uretebilir (dosya yoksa) ve iki ayar farkli sinavdan gecmis olurdu.
#
# SORGU VEKTORLERI BIR KEZ HESAPLANIR ve outputs/eval_query_vectors.npy'ye
# yazilir. Sorgu metinleri ayara bagli DEGIL, o yuzden her tarama turunda
# yeniden gommek saf israf olurdu (360 sorgu ~ 20 sn, ama daha onemlisi:
# ayni metni iki kez gomup farkli vektor almak tarama sonucunu kirletirdi).
###############################################################
import argparse
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
import numpy as np
import pandas as pd

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(KOK, "outputs")
SORGU = os.path.join(OUT, "eval_queries.parquet")
SWEEP = os.path.join(OUT, "rag_sweep.csv")

EMB_MODEL = os.environ.get("DS21_EMB_MODEL", "bge-m3")
OLLAMA = "http://localhost:11434/api/embed"

# SORGU TALIMAT ONEKI - modele gore degisir, ve ATLANIRSA HAKSIZ KARSILASTIRMA
# olur. Qwen3-Embedding'in kendi model karti soyle diyor:
#   "omitting instructions on the query side reduces retrieval performance by
#    approximately 1-5%"
# Yani oneki koymadan Qwen3'u bge-m3 ile karsilastirmak, Qwen3'u kendi
# onerilen kullanimindan mahrum birakip "daha kotu" ilan etmek olurdu.
# BELGE tarafina onek KONMAZ (model karti: "No need to add instruction for
# retrieval documents") - o yuzden bu yalnizca burada, sorgu tarafinda.
GOREV = ("Given a developer's problem statement, retrieve the GitHub "
         "repository that solves it")
SORGU_ONEKI = {
    "qwen3-embedding": f"Instruct: {GOREV}\nQuery:",
}


def sorgu_metni(q):
    """Model ailesine gore sorguya onek ekler. bge-m3 icin onek yok."""
    for anahtar, onek in SORGU_ONEKI.items():
        if EMB_MODEL.startswith(anahtar):
            return f"{onek}{q}"
    return q


# SORGU VEKTORU ONBELLEGI MODEL BASINA AYRI. Tek bir dosya kullanmak,
# bge-m3'un sorgu vektorlerini Qwen3 indeksiyle carpmak demekti - iki farkli
# uzayin nokta carpimi anlamsiz sayilar uretir ve model "cok kotu" gorunurdu.
# Dosya adina modeli yazmak bu hatayi imkansiz kiliyor.
QVEC = os.path.join(OUT, f"eval_query_vectors_{EMB_MODEL.replace(':', '_')}.npy")


def basl(t):
    print("\n" + "#" * 18 + f" {t} " + "#" * 18)


# BATCH bir tercih degil, MODELE GORE degisen bir tavan - ve bu dosyada
# koruma yoktu, DS21_RAG.py'de vardi. Olculdu (2026-09-07, qwen3-0.6b):
#   256 sorgu (54 KB!) -> HTTP 400 "Post .../tokenize: dial tcp ... refused"
#   128 sorgu (27 KB)  -> HTTP 200
# Boyut sorunu DEGIL (54 KB kucuk); Ollama'nin ic tokenizer servisi bu
# modelde 256 girdide baglanti kuramiyor. Yani tavan hem KARAKTERE hem
# OGE SAYISINA bagli ve model degisince ikisi de degisiyor.
# Bu yuzden: kucuk basla + hata alirsan BOL. Bolme burada calisir cunku
# hata gecici (baglanti), bulasici degil.
AZAMI_OGE = 64
AZAMI_KARAKTER = 200_000


def gom(texts, timeout=900):
    import requests
    if not texts:
        return []
    try:
        r = requests.post(OLLAMA, json={"model": EMB_MODEL, "input": texts},
                          timeout=timeout)
        if r.status_code == 200:
            v = r.json()["embeddings"]
            if len(v) == len(texts):
                return v
            raise RuntimeError(f"{len(v)} vektor / {len(texts)} metin")
        hata = f"HTTP {r.status_code}: {r.text[:90]}"
    except Exception as exc:
        hata = str(exc)[:90]
    if len(texts) == 1:
        raise RuntimeError(f"tek metin gomulemedi: {hata}")
    orta = len(texts) // 2
    print(f"    {len(texts)} basarisiz ({hata}) - boluyorum")
    return gom(texts[:orta], timeout) + gom(texts[orta:], timeout)


def partiler(texts):
    """Hem oge sayisi hem karakter butcesine gore boler."""
    p, k = [], 0
    for t in texts:
        if p and (len(p) >= AZAMI_OGE or k + len(t) > AZAMI_KARAKTER):
            yield p
            p, k = [], 0
        p.append(t)
        k += len(t)
    if p:
        yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help="olculecek indeksin eki, orn _h2000")
    ap.add_argument("--label", default=None, help="rapor etiketi (varsayilan: suffix)")
    ap.add_argument("--requery", action="store_true",
                    help="sorgu vektorlerini yeniden hesapla")
    ap.add_argument("--body", action="store_true",
                    help="govde vektorleriyle max-pool (Secenek A)")
    args = ap.parse_args()
    etiket = args.label or (args.suffix or "base")

    doc_yol = os.path.join(OUT, f"rag_documents{args.suffix}.parquet")
    emb_yol = os.path.join(OUT, f"embeddings{args.suffix}.npy")
    for y in (SORGU, doc_yol, emb_yol):
        if not os.path.exists(y):
            raise SystemExit(f"eksik girdi: {y}")

    basl(f"Girdi  [{etiket}]")
    ev = pd.read_parquet(SORGU)
    docs = pd.read_parquet(doc_yol, columns=["full_name", "primary_chars"])
    E = np.load(emb_yol)
    print(f"  sorgu    {len(ev):,}   ({(ev['lang']=='en').sum()} en / {(ev['lang']=='tr').sum()} tr)")
    print(f"  belge    {len(docs):,}   ortalama {docs['primary_chars'].mean():,.0f} karakter")
    print(f"  vektor   {E.shape}")
    if len(docs) != len(E):
        raise SystemExit(f"belge/vektor uyusmazligi: {len(docs)} vs {len(E)}")

    # --- sorgu vektorleri: bir kez, sonra onbellek --------------------------
    if os.path.exists(QVEC) and not args.requery:
        QV = np.load(QVEC)
        if len(QV) != len(ev):
            raise SystemExit(f"onbellekteki sorgu vektoru {len(QV)}, sorgu {len(ev)}"
                             f" - eval_queries degisti, --requery ile yenile")
        print(f"  sorgu vektoru onbellekten: {QVEC}")
    else:
        onekli = [sorgu_metni(q) for q in ev["query"]]
        print(f"  {len(ev):,} sorgu gomuluyor  (model {EMB_MODEL})")
        if onekli[0] != ev["query"].iloc[0]:
            print(f"  talimat oneki UYGULANDI: {onekli[0][:70]}...")
        t0 = time.time()
        acc = []
        for grup in partiler(onekli):
            acc.extend(gom(grup))
        QV = np.asarray(acc, dtype=np.float32)
        np.save(QVEC, QV)
        print(f"  {time.time()-t0:.0f} sn -> {QVEC}")

    # --- recall -------------------------------------------------------------
    basl(f"recall  [{etiket}]")
    satir = {n: i for i, n in enumerate(docs["full_name"])}
    eksik = [f for f in ev["full_name"] if f not in satir]
    if eksik:
        raise SystemExit(f"{len(eksik)} sorgu hedefi indekste yok, orn {eksik[:3]}")
    truth = np.array([satir[f] for f in ev["full_name"]])

    # bge-m3 ciktilari L2-normalize; nokta carpim = kosinus benzerligi.
    SIMS = QV @ E.T

    # --- SECENEK A: govde vektorleriyle MAX-POOL ---------------------------
    # skor(repo) = max( sim(sorgu, bas), sim(sorgu, govde) )
    # ORTALAMA DEGIL. Ortalama alsaydik HEAD_CHARS=2000'in kaybettigi tuzaga
    # dusserdik: ilgisiz govde metni sinyali seyreltirdi. max ile govde
    # ilgisizse skoru DUSURMEZ, sadece katki vermez.
    # Govdesi olmayan repolar (README <= 900 kar.) dokunulmadan kalir -
    # onlar icin skor zaten yalnizca bas vektorudur.
    if args.body:
        b_emb = os.path.join(OUT, f"embeddings{args.suffix}_body.npy")
        b_row = os.path.join(OUT, f"body_rows{args.suffix}.npy")
        for y in (b_emb, b_row):
            if not os.path.exists(y):
                raise SystemExit(f"--body istendi ama eksik: {y}\n"
                                 f"  once: python build_body_index.py "
                                 f"--suffix {args.suffix} --model {EMB_MODEL}")
        EB = np.load(b_emb)
        rows = np.load(b_row)
        if len(EB) != len(rows):
            raise SystemExit(f"govde vektoru {len(EB)} / satir eslemesi {len(rows)}")
        if EB.shape[1] != E.shape[1]:
            raise SystemExit(f"boyut uyusmazligi: bas {E.shape[1]} govde {EB.shape[1]}")
        SB = QV @ EB.T                       # (sorgu x govde)
        onceki = SIMS[:, rows].copy()
        SIMS[:, rows] = np.maximum(onceki, SB)
        kazanan = int((SB > onceki).sum())
        print(f"  GOVDE MAX-POOL: {len(rows):,} repo · govde kazandi "
              f"{kazanan:,}/{onceki.size:,} hucrede (%{100*kazanan/onceki.size:.1f})")
    gold = SIMS[np.arange(len(ev)), truth]
    # Dogru reponun sirasi = kendisinden DAHA benzer kac belge var + 1.
    # argsort'tan hizli ve beraberlikte kayirmiyor.
    rank = (SIMS > gold[:, None]).sum(axis=1) + 1

    K = (1, 5, 10, 20)
    genel = {f"r@{k}": float((rank <= k).mean()) for k in K}
    print("  GENEL   " + "   ".join(f"r@{k} {genel[f'r@{k}']:.3f}" for k in K)
          + f"   medyan sira {int(np.median(rank))}")

    print("\n  banda gore:")
    for b in ["<100", "100-1K", "1K-10K", "10K+"]:
        m = (ev["band"] == b).values
        if m.sum():
            print(f"    {b:<8} n={m.sum():>3}   "
                  + "   ".join(f"r@{k} {(rank[m]<=k).mean():.3f}" for k in K))
    print("\n  dile gore:")
    for l in ["en", "tr"]:
        m = (ev["lang"] == l).values
        if m.sum():
            print(f"    {l:<8} n={m.sum():>3}   "
                  + "   ".join(f"r@{k} {(rank[m]<=k).mean():.3f}" for k in K))

    # --- DUYARLILIK: ad ortusmesi ------------------------------------------
    # Bazi sorgular reponun adiyla ortusuyor. Bunun SIZINTI mi yoksa sadece
    # ALAN KELIMESI mi oldugu kuralla ayrilamiyor - iki ayri regex denendi,
    # ikisi de yanlis pozitif dolu cikti ("federated learning", "license plate",
    # "chatbot" bir markanin degil, isin adi). Repo yaptigi isin adini
    # tasiyorsa ortusme kacinilmaz.
    # Karar kurala degil OLCUME birakiliyor: recall hem tam sette hem
    # isaretliler cikarilmis sette raporlanir. Fark kucukse konu kapanir;
    # buyukse skorun bir kismi ad eslesmesinden geliyor demektir.
    if "name_overlap" in ev.columns:
        temiz = (~ev["name_overlap"]).values
        print(f"\n  ad ortusmesi isaretli {int(ev['name_overlap'].sum())} sorgu:")
        print(f"    tam set    (n={len(ev)})  "
              + "   ".join(f"r@{k} {(rank<=k).mean():.3f}" for k in K))
        print(f"    temiz set  (n={temiz.sum()})  "
              + "   ".join(f"r@{k} {(rank[temiz]<=k).mean():.3f}" for k in K))
        d5 = (rank <= 5).mean() - (rank[temiz] <= 5).mean()
        print(f"    r@5 farki  {d5:+.3f}  ->  "
              + ("ihmal edilebilir" if abs(d5) < 0.02 else "DIKKAT: skorun bir kismi ad eslesmesinden"))

    # --- tesHIS -------------------------------------------------------------
    basl("Teshis")
    print(f"  r@20 - r@5 = {genel['r@20'] - genel['r@5']:.3f}   (LLM rerank denendi ve reddedildi: tools/DS21_Rerank.py)")
    print(f"  r@20 = {genel['r@20']:.3f}  ->  sorgularin %{100*(1-genel['r@20']):.0f}'inde"
          f" dogru repo ilk 20'de yok")

    # --- kaydet -------------------------------------------------------------
    kayit = {"an": time.strftime("%Y-%m-%d %H:%M"), "etiket": etiket,
             "suffix": args.suffix, "belge": len(docs),
             "ort_karakter": round(float(docs["primary_chars"].mean()), 1),
             "sorgu": len(ev), **{k: round(v, 4) for k, v in genel.items()},
             "medyan_sira": int(np.median(rank))}
    df = pd.DataFrame([kayit])
    if os.path.exists(SWEEP):
        df = pd.concat([pd.read_csv(SWEEP), df], ignore_index=True)
    df.to_csv(SWEEP, index=False)
    print(f"\n  {SWEEP}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
