# -*- coding: utf-8 -*-
###############################################################
# figur_uret.py | PANEL FIGURES (light theme)
###############################################################
# WHAT IT DOES
#   Draws the presentation figures of tools/eda_figures.py again on the
#   panel's paper colour (tema.css --kagit), into figures/panel/, so a figure
#   has no visible edge on the page.
#
# WHY IT IMPORTS eda_figures INSTEAD OF COPYING IT
#   One source of truth for what each figure claims. The palette constants are
#   module globals, looked up when a figure function runs, so swapping them
#   after import restyles every figure without editing eda_figures.py.
#   Literal "white"/"black" colours written inside individual figures are
#   caught by a recolour pass just before saving.
#
# READ-ONLY over outputs/ and Data/. Writes only figures/panel/*.png.
#
# USAGE
#   python panel_v2\figur_uret.py         everything
#   python panel_v2\figur_uret.py --only F03 one figure (F03, R06 ...)
###############################################################

import argparse
import importlib.util
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.text import Text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
CSV = os.path.join(ROOT, "Data", "github_ai_intelligence_2026.csv")
FIG = os.path.join(ROOT, "figures", "panel")

###############################################################
# LIGHT PALETTE - matches panel_v2/tema.css (accent = eda_figures S1)
###############################################################
SURFACE = "#f4f6f4"
CARD = "#ffffff"
INK, INK2, INK3 = "#1b2420", "#4d5852", "#7b857f"
GRID = "#dde2dd"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
SEQ = ["#e3ecf7", "#c7dbf3", "#a9c8ee", "#7aaae6", "#4d8ddc", "#2a78d6", "#1c5cab"]

RC = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 11,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.labelcolor": INK2,
}


def kaydet(ad, iddia):
    """Same contract as eda_figures.kaydet: every figure states its claim."""
    os.makedirs(FIG, exist_ok=True)
    fig = plt.gcf()
    _yeniden_boya(fig)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, ad), dpi=140, bbox_inches="tight")
    plt.close("all")
    print(f"  {ad:<34} {iddia}")


###############################################################
# RECOLOUR PASS - literal light colours inside individual figures
###############################################################
_ESKI = {}          # eda_figures constant (rgba) -> panel colour


def _ayni(c, hedef):
    return np.allclose(to_rgba(c)[:3], to_rgba(hedef)[:3], atol=0.02)


def _yeni_renk(c, varsayilan=None):
    for eski, yeni in _ESKI.items():
        if _ayni(c, eski):
            return yeni
    return varsayilan


def _yeniden_boya(fig):
    fig.set_facecolor(SURFACE)
    for ax in fig.axes:
        if _ayni(ax.get_facecolor(), "#ffffff") or _yeni_renk(ax.get_facecolor()) == SURFACE:
            ax.set_facecolor(SURFACE)
    for t in fig.findobj(Text):
        yeni = _yeni_renk(t.get_color())
        if yeni:
            t.set_color(yeni)
        kutu = t.get_bbox_patch()
        if kutu is not None:
            if _ayni(kutu.get_facecolor(), "#ffffff") or _yeni_renk(kutu.get_facecolor()) in (SURFACE, CARD):
                kutu.set_facecolor(CARD)
            if _yeni_renk(kutu.get_edgecolor()) == GRID:
                kutu.set_edgecolor(GRID)
    for p in fig.findobj(matplotlib.patches.Patch):
        if p is fig.patch:
            continue
        fc = p.get_facecolor()
        if fc[3] > 0 and (_ayni(fc, "#ffffff") or _yeni_renk(fc) == SURFACE):
            p.set_facecolor(CARD)
        elif fc[3] > 0:
            yeni = _yeni_renk(fc)
            if yeni:
                p.set_facecolor(to_rgba(yeni, fc[3]))
        ec = p.get_edgecolor()
        if ec[3] > 0:
            yeni = _yeni_renk(ec)
            if yeni:
                p.set_edgecolor(yeni)
    for ln in fig.findobj(matplotlib.lines.Line2D):
        yeni = _yeni_renk(ln.get_color())
        if yeni:
            ln.set_color(yeni)


###############################################################
# 1) THE 24 PRESENTATION FIGURES, RESTYLED
###############################################################
def eda_modulu():
    yol = os.path.join(ROOT, "tools", "eda_figures.py")
    spec = importlib.util.spec_from_file_location("eda_figures", yol)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)          # module top level: palette + model contract only

    eski = {"SURFACE": m.SURFACE, "INK": m.INK, "INK2": m.INK2, "INK3": m.INK3,
            "GRID": m.GRID, "S1": m.S1, "S2": m.S2, "S3": m.S3,
            "GOOD": m.GOOD, "WARN": m.WARN, "CRIT": m.CRIT}
    yeni = {"SURFACE": SURFACE, "INK": INK, "INK2": INK2, "INK3": INK3, "GRID": GRID,
            "S1": S1, "S2": S2, "S3": S3, "GOOD": GOOD, "WARN": WARN, "CRIT": CRIT}
    _ESKI.clear()
    for ad, renk in eski.items():
        _ESKI[renk] = yeni[ad]
    _ESKI["#000000"] = INK
    for ad, renk in yeni.items():
        setattr(m, ad, renk)
    m.SEQ = SEQ
    m.kaydet = kaydet
    plt.rcParams.update(RC)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="tek figur: F03, R06 ...")
    a = ap.parse_args()

    m = eda_modulu()
    isler = dict(m.KAYIT)
    if a.only:
        isler = {a.only: isler[a.only]}
    print("#" * 63)
    print(f"# figur_uret.py  ->  {FIG}")
    print("#" * 63)
    hata = 0
    for ad, fn in isler.items():
        plt.rcParams.update(RC)
        try:
            fn()
        except SystemExit as e:
            hata += 1
            print(f"  {ad} ATLANDI: {e}")
        except Exception as e:                                     # noqa: BLE001
            hata += 1
            print(f"  {ad} HATA: {type(e).__name__}: {e}")
            plt.close("all")
    print(f"\n{len(isler) - hata}/{len(isler)} figur yazildi")
    return 1 if hata else 0


if __name__ == "__main__":
    sys.exit(main())
