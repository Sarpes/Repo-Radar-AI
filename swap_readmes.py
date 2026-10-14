# -*- coding: utf-8 -*-
###############################################################
# FAZ 2 son adimi | YENI README'leri devreye al
###############################################################
# n8n README'leri C:\Miuul\ds21final\readme\ altina yazdi.
# Bu betik onlari DOGRULAR, sonra Data/readme_raw/ ile takas eder.
#
# TAKAS SIRASI - onemli:
#   1. once DOGRULA (eksik/bos dosya var mi, ad semasi tutuyor mu)
#   2. eskiyi Data/readme_raw_snap<tarih>/ olarak YENIDEN ADLANDIR (silme)
#   3. yeniyi Data/readme_raw/ olarak TASI
# Boylece hicbir anda "README yok" durumu olusmuyor ve geri donus
# tek bir yeniden adlandirma.
#
# ⚠️ TASI, KOPYALAMA. DS21_Final_v2.py'nin README_DATE'i dosyalarin
# mtime'indan okunuyor. Kopyalama her dosyaya kopyalama zamanini basar
# ve o tarih "birisi en son ne zaman cp calistirdi"ya doner.
# shutil.move ayni birim icinde rename'e denk gelir, mtime korunur.
#
# KULLANIM
#   python swap_readmes.py            sadece dogrular, RAPOR verir
#   python swap_readmes.py --apply    takasi yapar
###############################################################
import argparse
import os
import shutil
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
import pandas as pd

KOK = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(KOK, "Data", "github_ai_intelligence_2026.csv")
ESKI = os.path.join(KOK, "Data", "readme_raw")
YENI = r"C:\Miuul\ds21final\readme"

# Cekim sirasinda 20 repoluk 'test' kademesi de ayni klasore yazdi;
# eksiklik hesabinda onlar da sayilir, ayrica ele alinmaz.
MIN_KAPSAMA = 0.90     # bu oranin altinda takas YAPILMAZ


def basl(t):
    print("\n" + "#" * 18 + f" {t} " + "#" * 18)


def guvenli_ad(full_name):
    """make_n8n_inputs.py ile AYNI kural - diskte dogrulanmisti (23.798/23.798)."""
    return full_name.replace("/", "__") + ".md"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="takasi gercekten yap")
    ap.add_argument("--min-coverage", type=float, default=MIN_KAPSAMA)
    args = ap.parse_args()

    basl("Girdi")
    if not os.path.isdir(YENI):
        raise SystemExit(f"yeni README klasoru yok: {YENI}")
    d = pd.read_csv(CSV, usecols=["full_name"])
    istenen = {guvenli_ad(fn) for fn in d["full_name"]}
    yeni_dosyalar = set(os.listdir(YENI))
    eski_dosyalar = set(os.listdir(ESKI)) if os.path.isdir(ESKI) else set()
    print(f"  CSV repo         : {len(d):,}")
    print(f"  ESKI  {ESKI}\n                     {len(eski_dosyalar):,} dosya")
    print(f"  YENI  {YENI}\n                     {len(yeni_dosyalar):,} dosya")

    basl("Dogrulama")
    ok = True

    def k(ad, kosul, detay=""):
        nonlocal ok
        ok &= bool(kosul)
        print(f"  {'OK  ' if kosul else 'HATA'} {ad:<40}{detay}")

    eslesen = istenen & yeni_dosyalar
    kapsama = len(eslesen) / len(istenen)
    k("kapsama >= esik", kapsama >= args.min_coverage,
      f"{100*kapsama:.2f}%  ({len(eslesen):,}/{len(istenen):,})")

    yabanci = yeni_dosyalar - istenen
    k("CSV disi dosya yok", not yabanci,
      f"{len(yabanci)} yabanci" + (f"  ornek {sorted(yabanci)[:2]}" if yabanci else ""))

    # BOS DOSYA. Ilk surumde "bos dosya yok" diye MUTLAK bir kapiydi ve
    # takasi engelledi. Incelendi: 4 dosya bos cikti ve AYNI 4 dosya 28.08
    # cekiminde de 0 bayt. Iki bagimsiz cekim, farkli gunler, ayni sonuc ->
    # bunlar GitHub'da gercekten bos README'ler, basarisiz cekim degil.
    # Zincir bunu zaten dogru ele aliyor: raw_chars > 0 -> has_readme False,
    # ve 'dokumantasyon yok' modelde bir SINYAL, bir eksiklik degil.
    # Bu yuzden kapi mutlak degil, TOLERANSLI: yalnizca boş dosya orani
    # anlamli olursa (cekimin bozuldugunu dusundurecek kadar) durdurur.
    bos = [f for f in eslesen if os.path.getsize(os.path.join(YENI, f)) == 0]
    bos_esik = max(25, int(0.001 * len(eslesen)))
    k("bos dosya orani makul", len(bos) <= bos_esik,
      f"{len(bos)} bos  (esik {bos_esik})")
    if bos:
        print(f"       ornek: {', '.join(sorted(bos)[:3])}")

    # eskisinden ciddi kayip var mi
    if eski_dosyalar:
        kayip = len(eski_dosyalar) - len(yeni_dosyalar)
        k("eskisine gore ciddi kayip yok", kayip <= 0.05 * len(eski_dosyalar),
          f"{kayip:+,} dosya")

    toplam_mb = sum(os.path.getsize(os.path.join(YENI, f))
                    for f in list(yeni_dosyalar)[:100000]) / 1024 ** 2
    print(f"       toplam boyut: {toplam_mb:,.0f} MB")
    en_yeni = max(os.path.getmtime(os.path.join(YENI, f))
                  for f in list(yeni_dosyalar)[:5000])
    print(f"       en yeni mtime: {pd.Timestamp(en_yeni, unit='s').date()}"
          f"   <- README_DATE bu olacak")

    if not ok:
        raise SystemExit("\nDOGRULAMA BASARISIZ - takas yapilmadi.")
    if not args.apply:
        print("\n  kuru kosu (--apply verilmedi) - takas yapilmadi")
        return

    basl("Takas")
    etiket = pd.Timestamp(max(os.path.getmtime(os.path.join(ESKI, f))
                              for f in list(eski_dosyalar)[:5000]),
                          unit="s").strftime("%m%d") if eski_dosyalar else "eski"
    arsiv = os.path.join(KOK, "Data", f"readme_raw_snap{etiket}")
    if os.path.exists(arsiv):
        raise SystemExit(f"arsiv adi zaten var: {arsiv}")
    os.rename(ESKI, arsiv)
    print(f"  eski -> {arsiv}  ({len(eski_dosyalar):,} dosya)")
    t0 = time.time()
    shutil.move(YENI, ESKI)
    print(f"  yeni -> {ESKI}  ({len(yeni_dosyalar):,} dosya, {time.time()-t0:.0f} sn)")
    os.makedirs(YENI, exist_ok=True)   # n8n bir dahaki koşuda buraya yazacak

    son = set(os.listdir(ESKI))
    print(f"\n  dogrulama: {len(son):,} dosya yerinde")
    en_yeni2 = max(os.path.getmtime(os.path.join(ESKI, f))
                   for f in list(son)[:5000])
    print(f"  README_DATE = {pd.Timestamp(en_yeni2, unit='s').date()}"
          f"  (mtime korundu: {'EVET' if en_yeni2 == en_yeni else 'HAYIR - kopyalanmis!'})")


if __name__ == "__main__":
    main()
