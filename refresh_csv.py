# -*- coding: utf-8 -*-
###############################################################
# FAZ 3 | CSV'yi YERINDE, KOLON BAZLI TAZELE
###############################################################
# 45 kolonun 13'u degisir, 32'si aynen kalir.
#
# NEDEN 32'si donuk - iki gerekce, ikincisi daha guclu:
#   1. Sizinti boslugu. T = anlik goruntu - 180 oldugu icin HER SEYI
#      tazelemek boslugu sonsuza kadar 180 gunde sabitler. Betimleyici
#      kolonlari dondurup yalnizca etiketi tazelemek T'yi ileri tasir,
#      metadata yerinde kalir: bosluk 143 gune duser ve arsiv biriktikce
#      kapanir.
#   2. STEP 3'un sizinti denetimi bagimsiz kalir. popularity_tier,
#      maintenance_status, project_maturity gibi kolonlar BASKASININ
#      urettigi kurallardir; onlari biz yeniden uretirsek denetim
#      "kendi yazdigimiz if'in bir if oldugunu kanitladik"a doner.
#
# METIN DUZEYINDE CALISIR, sayisal degil. CSV dtype=str okunur, yalnizca
# 13 kolonun HUCRELERI degistirilir, geri yazilir. Boylece 32 kolonun
# bayt-bayt ayni kaldigi VARSAYILMAZ, garanti edilir (ve assert edilir).
# Yazici sadakati olculdu: to_csv(lineterminator='\r\n') + sondaki CRLF
# kirpilinca orijinal dosyayi bayt-bayt yeniden uretiyor.
#
# TARIH KURALI - olculerek secildi (2026-09-06):
#   API tam zaman damgasi verir, CSV yalnizca gg/aa/yy tutar. Yas
#   kolonlarini tam zaman damgasiyla hesaplamak, CSV'den DOGRULANAMAYAN
#   bir sayi uretir. Gece yarisi (tarih) aritmetigi ise:
#     - kodun kendi age_at_T hesabiyla ayni,
#     - repo_age_days == age_at_T + 180 iddiasini %51,2'den %100'e cikarir
#       (DS21_Final_v2.py:601'deki "exactly" ifadesi su an YANLIS),
#     - herkes tarafindan dosyadan yeniden uretilebilir.
#   Bedeli olculdu: STEP 3.3'un karar agaci 1.000000 -> 0.999959 (24.664'te
#   1 satir). Bu yuzden STEP 3 denetimi snapshots/2026-07-31 uzerinde
#   kosmali - orada yedi bulgunun yedisi de bozulmadan kalir.
#
# KULLANIM
#   python refresh_csv.py              rapor verir, HICBIR SEY YAZMAZ
#   python refresh_csv.py --apply      CSV'yi yerinde gunceller
#   python refresh_csv.py --apply --min-coverage 0.5    esigi gevsetir
###############################################################

import argparse
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(PROJECT_ROOT, "Data", "github_ai_intelligence_2026.csv")
META_GLOB = r"C:\Miuul\ds21final\meta\meta_*.jsonl"
STATUS_PATH = os.path.join(PROJECT_ROOT, "outputs", "refresh_status.jsonl")
SNAP_DIR = os.path.join(PROJECT_ROOT, "snapshots")

# --- kolon politikasi -------------------------------------------------------
# API alan adi -> CSV kolon adi
TAZELE = {
    "pushed_at": "pushed_at",
    "updated_at": "updated_at",
    "stargazers_count": "stars",
    "forks_count": "forks",
    "open_issues_count": "open_issues",
    "size": "size_kb",
    "archived": "archived",
}
YENIDEN_HESAPLA = ["watchers", "repo_age_days", "days_since_last_push",
                   "stars_per_day", "popularity_tier", "maintenance_status"]

TIER_SINIR = [-1, 99, 999, 9999, 10 ** 12]
TIER_ETIKET = ["Low (<100 stars)", "Medium (100-1K)", "High (1K-10K)", "Mega (10K+ stars)"]
STAT_SINIR = [-1, 29, 89, 364, 10 ** 9]
STAT_ETIKET = ["Active", "Recent", "Stale", "Inactive"]


def header(text):
    print("\n" + "#" * 21 + f" {text} " + "#" * 21)


# --- bicimleyiciler: mevcut dosyadaki YAZILISLARA birebir uyar ---------------

def gg_aa_yy(iso):
    """API'nin ISO-8601'ini CSV'nin gg/aa/yy bicimine cevirir.

    Bu kirpma bilerek yapiliyor: CSV'nin sozlesmesi tarih, zaman damgasi
    degil. DS21_Final_v2.py alti ayri yerde format='%d/%m/%y' ile okuyor;
    baska bir bicim ya patlar ya da (satir 297, errors='coerce') sessizce
    NaT uretir.
    """
    return pd.Timestamp(iso).strftime("%d/%m/%y")


def bool_yaz(v):
    return "TRUE" if bool(v) else "FALSE"


def spd_yaz(v):
    """stars_per_day yazilisi: 4 basamaga yuvarla, sondaki .0'i at.

    Mevcut dosyada olculdu: 22.062 satir 4 basamak, 44 satir 0 basamak.
    Python'un repr'i tam olarak boyle davranir. Kural mevcut kolonu
    %100 yeniden uretiyor (verify_recompute_rules.py).
    """
    if pd.isna(v):
        return ""
    s = repr(round(float(v), 4))
    return s[:-2] if s.endswith(".0") else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="CSV'yi gercekten yaz")
    ap.add_argument("--min-coverage", type=float, default=0.99,
                    help="bu orandan az repo tazelendiyse yazma (varsayilan 0.99)")
    args = ap.parse_args()

    # ---------------------------------------------------------------- girdi
    header("Girdi")
    ham = open(CSV_PATH, "rb").read()
    d = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    print(f"{CSV_PATH}\n  {d.shape[0]:,} satir x {d.shape[1]} kolon")

    kolon_sirasi = list(d.columns)
    donuk = [c for c in kolon_sirasi
             if c not in TAZELE.values() and c not in YENIDEN_HESAPLA]
    print(f"  tazelenecek {len(TAZELE)} + yeniden hesaplanacak {len(YENIDEN_HESAPLA)}"
          f" = {len(TAZELE) + len(YENIDEN_HESAPLA)}")
    print(f"  dokunulmayacak: {len(donuk)}")
    assert len(TAZELE) + len(YENIDEN_HESAPLA) + len(donuk) == d.shape[1], \
        "kolon politikasi 45 kolonu kapsamiyor"
    donuk_onceki = d[donuk].copy()

    # ------------------------------------------------------ n8n ciktilarini oku
    header("n8n metadata ciktilari")
    dosyalar = sorted(glob.glob(META_GLOB))
    if not dosyalar:
        raise SystemExit(f"HIC meta dosyasi yok: {META_GLOB}")

    # Ayni repo birden fazla kademede olabilir (repos_test.csv, CSV'nin ilk
    # 20 satiri, meta1..5 ile ORTUSUR). En son cekileni kazanir.
    kayit = {}
    for yol in dosyalar:
        n = 0
        for satir in open(yol, encoding="utf-8"):
            if not satir.strip():
                continue
            try:
                o = json.loads(satir)
            except json.JSONDecodeError:
                continue
            # ANAHTAR: repo.full_name, satirdaki full_name DEGIL.
            # Satirdaki full_name n8n'de pairedItem ile geri kazanilmaya
            # calisildi ve 2026-09-07'de YANLIS HIZALANDIGI olculdu:
            # meta2'de 266 satirin adi baska bir reponun yanitina yapismisti
            # (kanit: 266'nin 243'unde HER IKI ad da CSV'de - gercek bir
            # yeniden adlandirmada imkansiz; ayrica A->B, B->C kayma zinciri).
            # repo.full_name ise API'nin KENDI kimlik beyani, eslestirmeye
            # dayanmiyor, her zaman dogru.
            fn = (o.get("repo") or {}).get("full_name") or o.get("full_name")
            if not fn:
                continue
            n += 1
            eski = kayit.get(fn)
            if eski is None or o.get("cekilme", "") >= eski.get("cekilme", ""):
                kayit[fn] = o
        print(f"  {os.path.basename(yol):<24} {n:>7,} satir")
    print(f"\n  benzersiz repo: {len(kayit):,}")

    var = d["full_name"].isin(kayit)
    kapsama = var.mean()
    print(f"  CSV ile eslesen: {int(var.sum()):,} / {len(d):,}  ({100*kapsama:.2f}%)")
    # YENIDEN ADLANDIRMA. GitHub eski adrese yonlendirme yapar ve yanitta YENI
    # full_name doner. Duzeltilmis workflow istenen adi pairedItem ile geri
    # kazanip anahtar olarak onu yaziyor, yani yeni satirlarda yetim OLMAMALI.
    # Duzeltme oncesi yazilmis satirlar yanit adini tasiyor -> yetim gorunur ve
    # o repolar tazelenmemis sayilir (kapsama esigi zaten yazmayi engeller).
    yetimler = sorted(set(kayit) - set(d["full_name"]))
    ad_degisen = sum(1 for o in kayit.values() if o.get("ad_degisti"))
    print(f"  CSV'de olmayan : {len(yetimler)}  (duzeltme oncesi satirlarda beklenir)")
    print(f"  ad_degisti isaretli: {ad_degisen}  (duzeltilmis workflow'dan)")
    if yetimler:
        print("    ornek:", ", ".join(yetimler[:5]))
        print("    -> bu repolar TAZELENMEMIS sayilir; kademe duzeltilmis")
        print("       workflow ile yeniden kosuldugunda dogru anahtarla yazilirlar.")
    for fn in yetimler:
        kayit.pop(fn, None)

    # ------------------------------------------------------------ tazele
    header("TAZELE - 7 kolon")
    idx = d.index[var]
    for api_alan, csv_kolon in TAZELE.items():
        degerler = []
        for i in idx:
            r = kayit[d.at[i, "full_name"]]["repo"]
            v = r.get(api_alan)
            if csv_kolon in ("pushed_at", "updated_at"):
                degerler.append(gg_aa_yy(v))
            elif csv_kolon == "archived":
                degerler.append(bool_yaz(v))
            else:
                degerler.append(str(int(v)))
        onceki = d.loc[idx, csv_kolon].copy()
        d.loc[idx, csv_kolon] = degerler
        degisen = int((onceki.values != np.array(degerler)).sum())
        print(f"  {csv_kolon:<14} <- {api_alan:<20} {degisen:>7,} satir degisti")

    # -------------------------------------------------- yeniden hesapla
    header("YENIDEN HESAPLA - 6 kolon")
    created = pd.to_datetime(d["created_at"], format="%d/%m/%y")
    pushed = pd.to_datetime(d["pushed_at"], format="%d/%m/%y")
    updated = pd.to_datetime(d["updated_at"], format="%d/%m/%y")
    SNAPSHOT = updated.max()
    print(f"  SNAPSHOT = max(updated_at) = {SNAPSHOT.date()}")
    # T is NOT set here: it is the model's contract (model.pkl cutoff_T). A local
    # SNAPSHOT-180 copy once printed a Wednesday T and seeded the week-boundary leak.

    yildiz = pd.to_numeric(d["stars"])
    yas = (SNAPSHOT - created).dt.days
    dslp = (SNAPSHOT - pushed).dt.days

    d["watchers"] = d["stars"]                                  # API tuhafligi
    d["repo_age_days"] = yas.astype(int).astype(str)
    d["days_since_last_push"] = dslp.astype(int).astype(str)
    d["stars_per_day"] = (yildiz / yas.replace(0, np.nan)).map(spd_yaz)
    d["popularity_tier"] = pd.cut(yildiz, TIER_SINIR, labels=TIER_ETIKET).astype(str)
    d["maintenance_status"] = pd.cut(dslp, STAT_SINIR, labels=STAT_ETIKET).astype(str)
    for c in YENIDEN_HESAPLA:
        print(f"  {c:<22} yeniden yazildi")

    # ---------------------------------------------------------- kontroller
    header("Cikis kontrolleri")
    ok = True

    def kontrol(ad, kosul, ayrinti=""):
        nonlocal ok
        ok &= bool(kosul)
        print(f"  {'OK  ' if kosul else 'HATA'} {ad}{('  ' + ayrinti) if ayrinti else ''}")

    kontrol("sekil 24.664 x 45", d.shape == (24664, 45), str(d.shape))
    kontrol("kolon adlari ve SIRASI ayni", list(d.columns) == kolon_sirasi)
    kontrol(f"donuk {len(donuk)} kolon degismedi", d[donuk].equals(donuk_onceki))
    kontrol("watchers == stars", (d["watchers"] == d["stars"]).all())
    yeni_tier = pd.cut(pd.to_numeric(d["stars"]), TIER_SINIR, labels=TIER_ETIKET).astype(str)
    kontrol("popularity_tier kurali", (d["popularity_tier"] == yeni_tier).all())
    try:
        for c in ("created_at", "updated_at", "pushed_at"):
            pd.to_datetime(d[c], format="%d/%m/%y")
        kontrol("tum tarihler %d/%m/%y ile okunuyor", True)
    except Exception as e:
        kontrol("tum tarihler %d/%m/%y ile okunuyor", False, str(e)[:60])

    if not ok:
        raise SystemExit("\nKONTROL BASARISIZ - hicbir sey yazilmadi.")

    # ------------------------------------------------------------ yaz
    header("Yazma")
    if kapsama < args.min_coverage:
        print(f"  kapsama {100*kapsama:.2f}% < esik {100*args.min_coverage:.0f}%")
        print("  -> YAZILMADI. Tum kademeler bitmeden yazmak, tazelenmemis")
        print("     satirlarin SNAPSHOT'a gore yeniden hesaplanmasina yol acar.")
        print("     Bilerek yazmak icin: --apply --min-coverage <daha dusuk>")
        return
    if not args.apply:
        print("  kuru kosu (--apply verilmedi) - hicbir sey yazilmadi")
        return

    metin = d.to_csv(index=False, encoding="utf-8", lineterminator="\r\n")
    bayt = metin.encode("utf-8")
    if bayt.endswith(b"\r\n"):
        bayt = bayt[:-2]        # orijinal dosya sondaki satir sonu olmadan bitiyor
    open(CSV_PATH, "wb").write(bayt)
    print(f"  {CSV_PATH}  ({len(bayt)/1024**2:.2f} MB, oncesi {len(ham)/1024**2:.2f} MB)")

    # SNAPSHOT'I YAZILMIS CSV'DEN YENIDEN OKUYARAK yaz, bellekteki metin
    # cercevesinden DEGIL. Iki sebep:
    #   1. TIP TUTARLILIGI. Calisma cercevesi dtype=str (32 kolonun bayt-bayt
    #      korundugunu garanti etmek icin). Onu dogrudan parquet'e yazmak 45
    #      kolonu da metin yapardi; Faz 0'daki 2026-07-31 snapshot'i ise
    #      cikarilmis tiplerle yazilmisti (int64 25 / object 17 / bool 2 /
    #      float64 1). Ayni seride iki farkli tip semasi, nokta-zamanda-dogru
    #      arsivi kullanilamaz hale getirir - olculdu: 45 kolonun 28'i farkli.
    #      Referans, zincirin gercekte okudugu sey: pd.read_csv varsayilan
    #      cikarimla (DS21_Final_v2.py:244 dtype vermiyor).
    #   2. BEDAVA GIDIS-DONUS KONTROLU. Yeni yazilan CSV'yi geri okumak,
    #      dosyanin gercekten ayristirilabildigini de kanitlar.
    geri = pd.read_csv(CSV_PATH)
    assert geri.shape == (24664, 45), f"yazilan CSV geri okunamadi: {geri.shape}"
    snap = os.path.join(SNAP_DIR, str(SNAPSHOT.date()))
    os.makedirs(snap, exist_ok=True)
    geri.to_parquet(os.path.join(snap, "metadata.parquet"), index=False, compression="zstd")
    print(f"  {snap}  metadata.parquet   (tarihli arsiv, cikarilmis tipler - Faz 6)")

    os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        for fn in d["full_name"]:
            o = kayit.get(fn)
            f.write(json.dumps({
                "full_name": fn,
                "tazelendi": o is not None,
                "cekilme": (o or {}).get("cekilme"),
                "etag": (o or {}).get("etag"),
            }, ensure_ascii=False) + "\n")
    print(f"  {STATUS_PATH}  ({len(d):,} satir, ETag'ler dahil)")


if __name__ == "__main__":
    main()
