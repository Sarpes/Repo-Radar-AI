# -*- coding: utf-8 -*-
###############################################################
# SECENEK A | "GOVDE" vektorleri: README[900:4000]
###############################################################
# NEDEN
# HEAD_CHARS taramasi 900 > 2000 cikardi. Sebep metnin MIKTARI degil, TEK
# VEKTORE SIKISTIRILMASI: tek vektor anlamin ortalamasidir, README uzadikca
# amac sinyali kurulum/API metniyle seyreliyor (olculdu: amac ifadeleri
# 0-900'de %41,3, 900-2000'de %20,8).
#
# Bu dosya ayni metni ikinci bir vektore koyar ve arama sirasinda MAX alinir:
#     skor(repo) = max( sim(sorgu, bas), sim(sorgu, govde) )
# Ortalama DEGIL. Govde ilgisizse skoru DUSURMEZ, sadece katki vermez -
# yani "daha fazla metin" bir risk olmaktan cikip opsiyona doner.
#
# KIMLIK SORUNU YOK: bas vektoru full_name + description + topics tasiyor.
# Govde vektoru tek basina sonuc uretmez, YALNIZCA kendi reposunun skorunu
# yukseltebilir. Sonuc listesi hala repo listesi, parca listesi degil.
#
# CIKTI
#   outputs/embeddings<SUFFIX>_body.npy    (n_govde x dim)
#   outputs/body_rows<SUFFIX>.npy          govde satiri -> belge satiri eslemesi
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
OLLAMA = "http://localhost:11434/api/embed"

BAS_SONU = 900      # doc_primary'nin README'den aldigi son karakter
GOVDE_SONU = 4000   # govdenin bittigi yer
AZAMI_OGE = 32
# 64 x ~2.760 = 176k patladi. Govde metinleri doc_primary'den 2,5x uzun
# ve daha yogun (kod, URL, tablo) - karakter basina token orani yuksek.
AZAMI_KARAKTER = 90_000


def basl(t):
    print("\n" + "#" * 18 + f" {t} " + "#" * 18)


# IKI FARKLI HATA SINIFI, IKI FARKLI KURTARMA - olculdu 2026-09-07:
#
#   "dial tcp ... refused"  (Ollama'nin ic tokenize servisi)
#       GECICI ve isteğe ozgu. Bolmek ISE YARAR, daha kucuk istek gecer.
#
#   "input length exceeds the context length"
#       BULASICI. Bir kez tetiklenince Ollama bir sure bozuk durumda kalir:
#       256 -> 128 -> 64 -> ... -> 2 -> tek belge, HEPSI ayni hatayi verir.
#       Bolmek ISE YARAMAZ. Ama AYNI batch dakikalar sonra sorunsuz geciyor
#       (dogrulandi: bge-m3 govde kosusu coktu, sonra ayni 64'luk batch
#       175 KB ile 4,3 sn'de gecti).
#       Dogru kurtarma BEKLEMEK, bolmek degil.
#
# Bu ayrimi yapmadan "retry var" demek yanlisti: kurtarma mekanizmasi
# dogru yazilmisti ama YANLIS HATA SINIFI icin.
BULASICI = "context length"
# 20 sn YETMEDI (olculdu: 5x20=100 sn bekledi, gecmedi; ayni batch
# birkac dakika sonra HTTP 200 verdi). Zehirlenme dakikalarca suruyor.
BEKLEME = 60
AZAMI_BEKLEME = 6


def gom(model, texts, timeout=900, bekleme_hakki=AZAMI_BEKLEME):
    import requests
    if not texts:
        return []
    try:
        r = requests.post(OLLAMA, json={"model": model, "input": texts},
                          timeout=timeout)
        if r.status_code == 200:
            v = r.json()["embeddings"]
            if len(v) == len(texts):
                return v
            raise RuntimeError(f"{len(v)} vektor / {len(texts)} metin")
        hata = f"HTTP {r.status_code}: {r.text[:80]}"
    except Exception as exc:
        hata = str(exc)[:80]

    if BULASICI in hata:
        if bekleme_hakki <= 0:
            raise RuntimeError(f"bulasici hata {AZAMI_BEKLEME} beklemeden sonra "
                               f"gecmedi: {hata}")
        print(f"    bulasici hata - {BEKLEME} sn bekleyip AYNI batch'i "
              f"({len(texts)}) yeniden deniyorum  [{bekleme_hakki} hak]")
        time.sleep(BEKLEME)
        return gom(model, texts, timeout, bekleme_hakki - 1)

    if len(texts) == 1:
        raise RuntimeError(f"tek metin gomulemedi: {hata}")
    orta = len(texts) // 2
    print(f"    {len(texts)} basarisiz ({hata}) - boluyorum")
    return (gom(model, texts[:orta], timeout, bekleme_hakki)
            + gom(model, texts[orta:], timeout, bekleme_hakki))


def partiler(texts):
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
    ap.add_argument("--suffix", required=True, help="belge dosyasinin eki, orn _qwen06")
    ap.add_argument("--model", default=os.environ.get("DS21_EMB_MODEL", "bge-m3"))
    args = ap.parse_args()

    doc_yol = os.path.join(OUT, f"rag_documents{args.suffix}.parquet")
    if not os.path.exists(doc_yol):
        raise SystemExit(f"eksik: {doc_yol}")
    emb_yol = os.path.join(OUT, f"embeddings{args.suffix}_body.npy")
    satir_yol = os.path.join(OUT, f"body_rows{args.suffix}.npy")

    basl("Govde belgeleri")
    docs = pd.read_parquet(doc_yol, columns=["full_name", "text_clean", "clean_chars"])
    var = (docs["clean_chars"] > BAS_SONU).values
    satirlar = np.where(var)[0]
    govdeler = [docs["text_clean"].iloc[i][BAS_SONU:GOVDE_SONU] for i in satirlar]
    uz = np.array([len(g) for g in govdeler])
    print(f"  toplam repo         : {len(docs):,}")
    print(f"  govdesi olan (>{BAS_SONU}) : {len(satirlar):,}  (%{100*len(satirlar)/len(docs):.1f})")
    print(f"  govdesi olmayan     : {len(docs)-len(satirlar):,}  -> tek vektorle kalir")
    print(f"  govde uzunlugu      : ortalama {uz.mean():,.0f}  medyan {np.median(uz):,.0f}"
          f"  max {uz.max():,}")
    print(f"  toplam metin        : {uz.sum()/1024**2:.1f} MB")

    basl(f"Gomme  ({args.model})")
    t0, acc = time.time(), []
    for grup in partiler(govdeler):
        acc.extend(gom(args.model, grup))
        if len(acc) % 3200 < AZAMI_OGE:
            h = len(acc) / (time.time() - t0)
            print(f"  {len(acc):>6,} / {len(govdeler):,}  {h:5.1f}/sn  "
                  f"eta {(len(govdeler)-len(acc))/h/60:4.1f} dk")
    E = np.asarray(acc, dtype=np.float32)
    print(f"  {E.shape}  {time.time()-t0:.0f} sn")

    np.save(emb_yol, E)
    np.save(satir_yol, satirlar)
    print(f"\n  {emb_yol}  ({os.path.getsize(emb_yol)/1024**2:.0f} MB)")
    print(f"  {satir_yol}")
    print(f"\n  olcmek icin: python eval_rag.py --suffix {args.suffix} "
          f"--body --label <etiket>")


if __name__ == "__main__":
    main()
