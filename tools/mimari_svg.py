# -*- coding: utf-8 -*-
###############################################################
# mimari_svg.py | ARCHITECTURE DIAGRAMS AS STANDALONE SVG FILES
###############################################################
# The panel draws four architecture diagrams (chapters 03, 05, 06, 07) as
# inline SVG, with numbers read from the artifacts at render time. This file
# calls the SAME functions in panel_v2/grafik.py with the SAME arguments the
# panel passes, and writes each diagram to figures/mimari/ so README.md can
# show it. One source of truth: change a diagram in grafik.py, rerun this.
#
# Standalone means: xmlns added, fixed width/height from the viewBox, and a
# paper-coloured background (tema.css --kagit), so the dark ink stays readable
# on GitHub's dark theme too.
#
# READ-ONLY over outputs/. Writes only figures/mimari/*.svg.
#
# USAGE
#   python tools/mimari_svg.py                  artifacts from this project
#   python tools/mimari_svg.py --kok <dizin>    artifacts from another checkout
###############################################################

import argparse
import logging
import os
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.getLogger("streamlit").setLevel(logging.ERROR)      # bare mode: no ScriptRunContext noise

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK / "panel_v2"))
import grafik as g      # noqa: E402
import veri             # noqa: E402

KAGIT = "#f4f6f4"
HEDEF = KOK / "figures" / "mimari"


def bagimsiz(html):
    """Inline panel SVG -> a file a browser or GitHub can open on its own."""
    m = re.search(r"<svg\b[^>]*>.*</svg>", html, re.S)
    if not m:
        raise SystemExit("cizimde <svg> bulunamadi")
    svg = m.group(0)
    acilis = re.match(r"<svg\b[^>]*>", svg).group(0)
    _, _, w, h = re.search(r'viewBox="([\d. ]+)"', acilis).group(1).split()
    yeni = re.sub(r'\swidth="[^"]*"', "", acilis)
    yeni = yeni.replace("<svg", f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"', 1)
    return yeni + f'<rect width="100%" height="100%" fill="{KAGIT}"/>' + svg[len(acilis):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kok", help="outputs/ ve Data/ baska bir proje kokunden okunsun")
    a = ap.parse_args()
    if a.kok:
        k = Path(a.kok)
        veri.ROOT, veri.OUT = k, k / "outputs"
        veri.CSV = k / "Data" / "github_ai_intelligence_2026.csv"
        veri.CSV_ORIJINAL = k / "snapshots" / "2026-07-31" / "github_ai_intelligence_2026.csv"

    J, hata = veri.sunum_olcumler()
    if J is None:
        raise SystemExit(hata)
    m, b, sw, OLC = veri.model_ozeti(), veri.blok_sayilari(), veri.rag_denemeleri(), veri.OLCUMLER
    GEN, RSK, RES = J["genisletme"], J["rag_sinav_kullanici"], J["rag_esik"]
    r5 = lambda k: sw.loc[k, "r@5"]                                         # noqa: E731

    # Same calls as panel_v2/app.py (chapters 03, 05, 06, 07).
    cizimler = {
        "veri_genisletme.svg": g.genisletme_mimarisi(GEN),
        "model_hatti.svg": g.model_mimarisi(m["n"], m["ozellik_n"], b, m["T"], m["H"], m["taban"], m["auc"],
                                            len(veri.sozlesme()["yasakli"])),
        "rag.svg": g.rag_mimarisi(GEN["csv_repo"], GEN["readme_repo"], OLC["govde_repo"][0], OLC["bas_karakter"][0],
                                  OLC["govde_bitis"][0], OLC["vektor_boyut"][0], RES["esik"], RSK["sorgu"],
                                  len(RES["negatif_sorgu"]), RES["negatif_kesilen"], r5("head900"),
                                  r5("qwen8b-YENI-bas+govde"), uretilen=veri.sinav_uretimi(RSK["sorgu"])["uretilen"]),
        "sohbet.svg": g.sohbet_mimarisi(esik=RES["esik"], repo_n=GEN["csv_repo"], govde_n=OLC["govde_repo"][0]),
    }
    os.makedirs(HEDEF, exist_ok=True)
    for ad, html in cizimler.items():
        (HEDEF / ad).write_text(bagimsiz(html), encoding="utf-8", newline="\n")
        print(f"  figures/mimari/{ad:22s} {(HEDEF / ad).stat().st_size / 1024:5.1f} KB")


if __name__ == "__main__":
    main()
