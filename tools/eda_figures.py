# -*- coding: utf-8 -*-
###############################################################
# eda_figures.py | PRESENTATION FIGURE SET
###############################################################
# WHY A SEPARATE FILE
# -------------------
# DS21_Final_v2.py already writes 10 figures, but they are the STANDARD EDA set
# (distributions, missing map, correlation, SHAP). The findings that actually
# distinguish this project - the leakage audit, the time axis, the RAG sweep -
# are text-only. This file draws the ARGUMENTS, not more histograms.
#
# It is READ-ONLY over the chain's artifacts: it opens the CSV, the parquets and
# the .npy files and writes PNGs. It never touches DS21_Final_v2.py's outputs.
#
# OUTPUT GOES TO figures/sunum/  - a separate folder on purpose. The chain
# rewrites figures/*.png on every run; the presentation set must survive that.
#
# WHY PNG AND NOT LIVE CHARTS IN STREAMLIT
# ----------------------------------------
# The presentation IS the Streamlit app, and on 2026-09-09 this machine crashed
# mid-work. A PNG renders even if Python, the parquets or the database are
# unavailable on the day. Streamlit shows them with st.image(). Deterministic
# beats interactive when the demo cannot be repeated.
#
# COLOUR
# ------
# Palette from the dataviz reference instance, validated with its own script
# (categorical slots 1-3 pass all checks; the status trio does NOT pass CVD
# separation - red vs green measures dE 4.1 for deuteranopes). Therefore:
#   ⚠️ A BADGE COLOUR NEVER CARRIES MEANING ALONE. Every badge-coloured mark
#   ships with a visible label or percentage. This mirrors what the chatbot
#   already does (emoji + % + text).
#
# USAGE
#   python eda_figures.py              all available figures
#   python eda_figures.py --only F03   just one
#   python eda_figures.py --liste      what can be drawn right now, and why not
###############################################################

import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

###############################################################
# PATHS
###############################################################
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
CSV = os.path.join(ROOT, "Data", "github_ai_intelligence_2026.csv")
FIG = os.path.join(ROOT, "figures", "sunum")

###############################################################
# PALETTE  (dataviz reference instance - validated, do not eyeball)
###############################################################
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # categorical slots 1-3
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"  # status - label always
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a887f"
GRID = "#e6e5e0"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 11,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})

# Model contract - the label window.
#
# AUDIT 2026-09-11 → docs/AUDIT_DEFTERI.md#eda-figures-86: BU BLOK YALAN SOYLUYORDU.
def _sozlesme():
    yol = os.path.join(OUT, "model.pkl")
    if os.path.exists(yol):
        import joblib
        m = joblib.load(yol)
        return (pd.Timestamp(m["cutoff_T"]), int(m["horizon_days"]),
                float(m.get("oof_auc", float("nan"))), len(m["feature_cols"]),
                m.get("label_source"), m.get("label_wording"), m.get("run_id"))
    return pd.Timestamp("2026-03-08"), 182, float("nan"), 59, None, None, None


# AUDIT 2026-09-11 | Label NAMES come from model.pkl too (DS21_Final_v2.py
# LABEL_WORDING). A model.pkl written before that carries none -> figures that
# name the label refuse to draw instead of guessing "commit".
T, H, _, _, LABEL_SOURCE, LW, RUN_ID = _sozlesme()


def _etiket_kapisi():
    if not LW:
        raise SystemExit("model.pkl has no label_wording - rerun DS21_Final_v2.py. "
                         "Figures that name the label are not drawn.")


def kaydet(ad, iddia):
    """Save and print the CLAIM the figure makes. A figure without a claim is
    decoration; forcing the claim into the call site keeps the set honest."""
    os.makedirs(FIG, exist_ok=True)
    yol = os.path.join(FIG, ad)
    plt.tight_layout()
    plt.savefig(yol, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  {ad:<34} {iddia}")


def csv_oku(sutunlar=None):
    return pd.read_csv(CSV, usecols=sutunlar) if sutunlar else pd.read_csv(CSV)


def bin_ayir(x):
    return f"{x:,}".replace(",", ".")


###############################################################
# F01 | NOT A CHART - the numbers are the message
###############################################################
def f01():
    kutular = []
    d = csv_oku(["full_name"])
    kutular.append((bin_ayir(len(d)), "GitHub reposu"))
    kutular.append(("45", "ham sütun"))
    p = os.path.join(OUT, "commit_weeks.parquet")
    if os.path.exists(p):
        n = len(pd.read_parquet(p, columns=["week"]))
        kutular.append((bin_ayir(n), "commit-hafta kaydı"))
    rd = os.path.join(OUT, "rag_documents_qwen8b.parquet")
    if os.path.exists(rd):
        n = int(pd.read_parquet(rd, columns=["has_readme"])["has_readme"].sum())
        kutular.append((bin_ayir(n), "README belgesi"))
    e = os.path.join(OUT, "embeddings_qwen8b.npy")
    if os.path.exists(e):
        sh = np.load(e, mmap_mode="r").shape
        kutular.append((f"{sh[1]:,}".replace(",", "."), "boyutlu vektör"))
    m = os.path.join(OUT, "model.pkl")
    if os.path.exists(m):
        import joblib
        kutular.append((str(len(joblib.load(m)["feature_cols"])), "model özelliği"))

    n = len(kutular)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.6))
    for ax, (buyuk, kucuk) in zip(np.atleast_1d(axes), kutular):
        ax.axis("off")
        ax.add_patch(Rectangle((0.02, 0.08), 0.96, 0.84, transform=ax.transAxes,
                               facecolor="white", edgecolor=GRID, lw=1.2, zorder=0))
        ax.text(0.5, 0.60, buyuk, ha="center", va="center", fontsize=27,
                fontweight="bold", color=S1, transform=ax.transAxes)
        ax.text(0.5, 0.28, kucuk, ha="center", va="center", fontsize=11,
                color=INK2, transform=ax.transAxes)
    fig.suptitle("Veri kümesi tek bakışta", fontsize=15, fontweight="bold", y=1.04)
    kaydet("F01_sayilar.png", "olcek: bu bir oyuncak veri kumesi degil")


###############################################################
# F02 | The recommendable pool - filtering IS the product
###############################################################
def f02():
    d = csv_oku(["full_name", "stars", "archived", "license", "maintenance_status"])
    n0 = len(d)
    a = d[d["maintenance_status"].isin(["Active", "Recent"])]
    b = a[a["stars"] >= 100]
    c = b[~b["archived"].astype(str).str.lower().isin(["true", "1", "yes"])]
    lis = c[~c["license"].isin(["No License", "NOASSERTION"])]

    adim = [("Külliyatın tamamı", n0), ("Bakımı sürüyor", len(a)),
            ("≥ 100 yıldız", len(b)), ("Arşivlenmemiş", len(c)),
            ("Kullanılabilir lisans", len(lis))]
    etiket = [x[0] for x in adim][::-1]
    deger = [x[1] for x in adim][::-1]

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    renk = [SEQ[3]] * len(deger)
    renk[-1] = S1                      # the starting total, darkest
    renk[0] = GOOD                     # the survivors
    ax.barh(etiket, deger, color=renk, height=0.62, zorder=3)
    for i, v in enumerate(deger):
        ax.text(v + n0 * 0.012, i, f"{bin_ayir(v)}  (%{100*v/n0:.0f})",
                va="center", fontsize=11, color=INK)
    ax.set_xlim(0, n0 * 1.22)
    ax.set_xlabel("repo sayısı")
    ax.set_title("Önerilebilir havuz: 24.664 → 4.717")
    ax.grid(axis="y", visible=False)
    kaydet("F02_havuz.png",
           "kulliyatin %81'i olu/lisanssiz/terk -> FILTRELEME urunun kendisi")


###############################################################
# F03 | watchers == stars, on every single row
###############################################################
def f03():
    d = csv_oku(["stars", "watchers"])
    ayni = int((d["stars"] == d["watchers"]).sum())
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(d["stars"].clip(lower=1), d["watchers"].clip(lower=1),
               s=7, alpha=0.30, color=S1, edgecolors="none", zorder=3)
    lim = [1, max(d["stars"].max(), 1) * 1.3]
    ax.plot(lim, lim, color=CRIT, lw=2, ls="--", zorder=4, label="y = x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("stars"); ax.set_ylabel("watchers")
    ax.set_title("`watchers` bir sütun değil, `stars`ın kopyası")
    ax.legend(loc="upper left")
    ax.text(0.97, 0.06, f"{bin_ayir(ayni)} / {bin_ayir(len(d))} satır çizgi üzerinde\n"
                        f"eşleşme: %{100*ayni/len(d):.1f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=GRID))
    kaydet("F03_watchers_stars.png", "birebir kopya sutun -> modele girerse sizinti")


###############################################################
# F04 | maintenance_status is days_since_last_push, cut into bins
###############################################################
def f04():
    d = csv_oku(["maintenance_status", "days_since_last_push"])
    g = (d.groupby("maintenance_status")["days_since_last_push"]
           .agg(["min", "max", "count"]).sort_values("min"))
    renk = {"Active": GOOD, "Recent": S3, "Stale": WARN, "Inactive": CRIT}

    # THE CLAIM IS THE BOUNDARIES, NOT THE TAIL. On a linear axis the Inactive
    # range (365-4980) crushes the other three into slivers and the clean
    # partition - the whole point - becomes invisible. So the view is clipped
    # just past the last boundary and the tail is annotated instead of drawn.
    KESIM = 620
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for i, (ad, r) in enumerate(g.iterrows()):
        gorunen = min(r["max"], KESIM) - r["min"] + 1
        ax.barh(i, gorunen, left=r["min"], height=0.55,
                color=renk.get(ad, S1), zorder=3)
        if r["max"] > KESIM:
            ax.annotate("", xy=(KESIM + 55, i), xytext=(KESIM, i), zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=renk.get(ad, S1), lw=2))
        # LABEL IS MANDATORY: the status colours fail CVD separation, so the
        # range text - not the hue - carries the meaning.
        ax.text(KESIM + 90, i, f"{int(r['min'])} – {int(r['max'])} gün   "
                               f"({bin_ayir(int(r['count']))} repo)",
                va="center", fontsize=11, color=INK)
    for sinir in (30, 90, 365):
        ax.axvline(sinir, color=INK3, lw=1.1, ls=":", zorder=2)
    # The boundaries ARE the ticks - a separate label layer duplicated them and
    # "365" collided with the default "400".
    ax.set_xticks([0, 30, 90, 365, 620, 1000, 1400])
    ax.set_yticks(range(len(g))); ax.set_yticklabels(g.index)
    ax.set_xlabel("days_since_last_push   (görünüm 620 günde kesildi)")
    ax.set_xlim(-15, 1500)
    ax.set_title("`maintenance_status` bir gözlem değil, bir `if`")
    ax.grid(axis="y", visible=False)
    ortusme = bool((g["min"].values[1:] <= g["max"].values[:-1]).any())
    # Verdict goes BELOW the axes: inside, it collided with the shortest bar's
    # own label. Same fix as F17 - captions never fight the plot for space.
    fig.text(0.5, -0.06,
             "Kutular arasında örtüşme: " + ("VAR" if ortusme else "YOK — kusursuz bölme")
             + ".  Bu sütun bir gözlem değil, `days_since_last_push` üzerine yazılmış bir kural.",
             ha="center", va="top", fontsize=12, color=CRIT if ortusme else INK2)
    kaydet("F04_maintenance_bins.png",
           "etiketin kendisi -> hedefe %100 bagli, modele ASLA girmez")


###############################################################
# F05 | popularity_tier = f(stars), a step function
###############################################################
def f05():
    d = csv_oku(["popularity_tier", "stars"])
    g = d.groupby("popularity_tier")["stars"].agg(["min", "max", "count"]).sort_values("min")
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    for i, (ad, r) in enumerate(g.iterrows()):
        ax.barh(i, max(r["max"] - r["min"], 1), left=max(r["min"], 1),
                height=0.55, color=SEQ[min(i + 2, len(SEQ) - 1)], zorder=3)
        ax.text(r["max"] * 1.15, i, f"{bin_ayir(int(r['min']))} – {bin_ayir(int(r['max']))} yildiz   "
                                    f"({bin_ayir(int(r['count']))} repo)",
                va="center", fontsize=11, color=INK)
    for esik in (100, 1000, 10000):
        ax.axvline(esik, color=CRIT, lw=1.4, ls=":", zorder=4)
        ax.text(esik, len(g) - 0.05, f" {bin_ayir(esik)}", color=CRIT, fontsize=10, va="top")
    ax.set_xscale("log")
    ax.set_yticks(range(len(g))); ax.set_yticklabels(g.index)
    ax.set_xlabel("stars (log)")
    ax.set_xlim(1, g["max"].max() * 9)
    ax.set_title("`popularity_tier` = eşiklerle kesilmiş `stars`")
    ax.set_ylim(-0.6, len(g) + 0.1)          # a band above the top bar for the threshold labels
    ax.grid(axis="y", visible=False)
    kaydet("F05_popularity_tier.png",
           "%100 kural eslesmesi -> bu sutunu tahmin etmek bir if'i yeniden turetmek")


###############################################################
# F06 | The 14 banned columns, grouped by WHY
###############################################################
def f06():
    p = os.path.join(OUT, "model.pkl")
    if not os.path.exists(p):
        return print("  F06 atlandi: model.pkl yok")
    import joblib
    yasak = list(joblib.load(p)["banned_columns"])
    # Reasons taken from STEP 3's own audit sections, not invented here.
    neden = {
        "watchers": "3.1 kimlik", "stars_per_day": "3.1 aritmetik",
        "framework_stack": "3.1 kimlik",
        "popularity_tier": "3.2 kutulanmis", "maintenance_status": "3.2 kutulanmis",
        "project_maturity": "3.3 BIRLESIK kural",
        "days_since_last_push": "etiketin kendisi",
    }
    varsayilan = "tahmin gününden sonra ölçüldü"
    grup_renk = {"3.1 kimlik": S1, "3.1 aritmetik": S1, "3.2 kutulanmis": S2,
                 "3.3 BIRLESIK kural": CRIT, "etiketin kendisi": CRIT, varsayilan: INK3}
    satir = [(c, neden.get(c, varsayilan)) for c in yasak]
    satir.sort(key=lambda x: (x[1], x[0]))

    # Row pitch is derived from the row count, so the last row always ends above
    # the footer note (fixed 0.062 put row 14 on top of it).
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.axis("off")
    ax.text(0.0, 1.0, f"Modele ASLA girmeyen {len(yasak)} sütun", fontsize=15,
            fontweight="bold", va="top", transform=ax.transAxes)
    ust, alt = 0.89, 0.19
    adim = (ust - alt) / max(len(satir) - 1, 1)
    y = ust
    for ad, n in satir:
        ax.add_patch(Rectangle((0.0, y - 0.02), 0.016, 0.04, transform=ax.transAxes,
                               facecolor=grup_renk.get(n, INK3), edgecolor="none"))
        ax.text(0.035, y, ad, fontsize=12, va="center", transform=ax.transAxes,
                family="monospace", color=INK)
        ax.text(0.58, y, n, fontsize=11, va="center", transform=ax.transAxes, color=INK2)
        y -= adim
    ax.text(0.0, 0.0,
            "Sızıntı denetimi bir yorum değil, `model.pkl`'in içinde taşınıyor:\n"
            "modeli açan herkes bu listeyi de açıyor.",
            fontsize=11, color=INK2, va="bottom", transform=ax.transAxes)
    kaydet("F06_yasakli_sutunlar.png",
           "denetim sonucu MODELLE BIRLIKTE seyahat ediyor")


###############################################################
# F07 | The time axis - the single most misread thing
###############################################################
def f07():
    # The scoring day is read from scores_today.parquet, NOT taken from the clock:
    # a PNG drawn with today's date goes stale the day after it is committed, and
    # claims a scoring run that never happened on that day.
    yol = os.path.join(OUT, "scores_today.parquet")
    if not os.path.exists(yol):
        raise SystemExit("scores_today.parquet yok - score_today.py once kosulmali")
    skor = pd.Timestamp(pd.read_parquet(yol, columns=["cutoff"])["cutoff"].iloc[0])
    son = T + pd.Timedelta(days=H)
    bas = T - pd.Timedelta(days=420)
    fig, ax = plt.subplots(figsize=(11, 3.9))
    # The scoring band runs to skor+H and its label needs room past that.
    ax.set_xlim(bas, skor + pd.Timedelta(days=300))
    ax.set_ylim(0, 3.1)
    ax.axhline(1.0, color=INK3, lw=1.6, zorder=2)

    ax.add_patch(Rectangle((bas, 1.06), T - bas, 0.42, color=S1, alpha=0.85, zorder=3))
    ax.text(bas + (T - bas) / 2, 1.27, "ÖZELLİKLER  (yalnızca tahmin gününden önce)",
            ha="center", va="center", color="white", fontsize=11, fontweight="bold", zorder=4)

    ax.add_patch(Rectangle((T, 0.52), son - T, 0.42, color=S2, alpha=0.9, zorder=3))
    ax.text(T + (son - T) / 2, 0.73, f"ETİKET  ({H} gün)",
            ha="center", va="center", color="white", fontsize=11, fontweight="bold", zorder=4)

    ax.add_patch(Rectangle((skor, 1.62), pd.Timedelta(days=H), 0.42,
                           color=GOOD, alpha=0.85, zorder=3))
    # Label sits ABOVE the band, not inside it: the text is wider than the
    # span, so an inside label spilled onto the surface.
    ax.text(skor, 2.12, f"score_today.py  →  skor günü + {H} gün",
            ha="left", va="bottom", color=GOOD, fontsize=11, fontweight="bold", zorder=4)

    for x, ad, dy in ((T, f"tahmin günü = {T.date()}", 2.5), (son, f"+{H} gün = {son.date()}", 2.5),
                      (skor, f"skor günü = {skor.date()}", 2.85)):
        ax.axvline(x, color=INK, lw=1.3, ls="--", zorder=5)
        ax.text(x, dy, ad, ha="center", fontsize=10.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GRID))
    ax.set_yticks([])
    ax.set_title("Eğitim geçmişe bakar, tahmin geleceğe — aynı donmuş model")
    ax.grid(axis="y", visible=False)
    kaydet("F07_zaman_ekseni.png",
           "etiket penceresi KAPALI olmali (egitim), ACIK olmali (tahmin)")


###############################################################
# F15 | The RAG sweep - model decisions vs code decisions
###############################################################
def f15():
    p = os.path.join(OUT, "rag_sweep.csv")
    if not os.path.exists(p):
        return print("  F15 atlandi: rag_sweep.csv yok")
    d = pd.read_csv(p).drop_duplicates("etiket", keep="last").reset_index(drop=True)
    # Which lever moved: the model, or a decision in our code?
    kod = d["etiket"].str.contains("head|govde", case=False, regex=True)
    renk = [S2 if k else S1 for k in kod]

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    x = np.arange(len(d))
    ax.bar(x, d["r@5"], color=renk, width=0.6, zorder=3)
    for i, v in enumerate(d["r@5"]):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=11, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(d["etiket"], rotation=18, ha="right", fontsize=10)
    ax.set_ylim(0, max(d["r@5"]) * 1.22)
    ax.set_ylabel("recall@5")
    ax.set_title("Her satır bir karar — ve hepsi aynı sabit sınavla ölçüldü")
    h = [plt.Rectangle((0, 0), 1, 1, color=S1), plt.Rectangle((0, 0), 1, 1, color=S2)]
    ax.legend(h, ["model kararı", "kod kararı"], loc="upper left")
    ax.grid(axis="x", visible=False)
    kaydet("F15_rag_sweep.png",
           "4 model denendi 2'si elendi; 2 kod karari olculdu biri REDDEDILDI")


###############################################################
# F16 | recall@k - head alone vs head+body
###############################################################
def f16():
    p = os.path.join(OUT, "rag_sweep.csv")
    if not os.path.exists(p):
        return print("  F16 atlandi: rag_sweep.csv yok")
    d = pd.read_csv(p).drop_duplicates("etiket", keep="last")
    bas = d[d["etiket"].str.fullmatch("qwen3-8b|qwen8b-YENI")]
    gov = d[d["etiket"].str.contains("bas\\+govde") & (d["suffix"] == "_qwen8b")]
    if bas.empty or gov.empty:
        return print("  F16 atlandi: 8b bas/govde satirlari yok")
    ks = [1, 5, 10, 20]
    yb = [float(bas.iloc[-1][f"r@{k}"]) for k in ks]
    yg = [float(gov.iloc[-1][f"r@{k}"]) for k in ks]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(ks, yb, "-o", color=S1, lw=2.2, ms=9, zorder=3, label="yalnız baş (900 kar.)")
    ax.plot(ks, yg, "-o", color=S2, lw=2.2, ms=9, zorder=4, label="baş + gövde (max)")
    for k, a, b in zip(ks, yb, yg):
        ax.annotate(f"+{b-a:+.3f}".replace("++", "+"), (k, b), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=10, color=S2)
    ax.set_xticks(ks); ax.set_xlabel("k (ilk kaç sonuç)")
    ax.set_ylabel("recall@k")
    ax.set_title("Gövde vektörü sıralamanın TEPESİNİ iyileştiriyor")
    ax.legend(loc="lower right")
    # DUZELTME 2026-09-11 | eski iddia "r@10'da hafif odun" diyordu. Oranlari
    # HAM SAYIYA cevirince o "odun" 261 -> 259, yani IKI SORGU: gurultu.
    # k=1 +10 · k=5 +11 · k=10 -2 · k=20 +4 sorgu; n=354'te %95 GA +/-18 sorgu.
    # Yani govde hicbir k'da anlamli ZARAR vermiyor, ama kazanci da GA icinde.
    # Kucuk orneklemde ondalik gosterim gurultuye sahte kesinlik giydiriyor.
    kaydet("F16_recall_k.png",
           "k=1 +10 · k=5 +11 · k=10 -2 · k=20 +4 sorgu  ->  hepsi GA (+/-18) icinde")


###############################################################
# F17 | THE PRODUCT CLAIM, measured for the first time
###############################################################
def f17():
    ger = [os.path.join(OUT, f) for f in
           ("eval_queries.parquet", "eval_query_vectors_qwen3-embedding_8b.npy",
            "embeddings_qwen8b.npy", "rag_documents_qwen8b.parquet",
            "scores_today.parquet")]
    if not all(os.path.exists(x) for x in ger):
        return print("  F17 atlandi: gerekli artefaktlardan biri yok")

    q = pd.read_parquet(ger[0])
    QV = np.load(ger[1]).astype(np.float32)
    E = np.load(ger[2], mmap_mode="r")
    docs = pd.read_parquet(ger[3], columns=["full_name"])
    sk = pd.read_parquet(ger[4], columns=["full_name", "badge_today"])

    sims = QV @ np.asarray(E).T                       # (354, 24664)
    bp, br = (os.path.join(OUT, "embeddings_qwen8b_body.npy"),
              os.path.join(OUT, "body_rows_qwen8b.npy"))
    if os.path.exists(bp) and os.path.exists(br):
        B = np.load(bp, mmap_mode="r")
        rows = np.load(br)
        sb = QV @ np.asarray(B).T                     # (354, n_govde)
        tam = np.full_like(sims, -1.0)
        tam[:, rows] = sb
        sims = np.maximum(sims, tam)                  # max-pool, ortalama DEGIL

    ilk5 = np.argsort(-sims, axis=1)[:, :5]
    ad = docs["full_name"].values
    rozet = sk.set_index("full_name")["badge_today"]
    sec = pd.Series(ad[ilk5.ravel()]).map(rozet).fillna("VERI_YOK")

    # THE COMPARISON IS THE POINT. "60% of recommendations are RED" proves
    # nothing on its own - the corpus is 67% RED to begin with. What the two
    # bars together show is that semantic search barely moves that number:
    # retrieval cannot rank on a property it does not encode. That is exactly
    # the gap the badge fills.
    sira = ["GREEN", "AMBER", "RED", "VERI_YOK"]
    # NO EMOJI: DejaVu Sans has no glyphs for the badge circles and they render
    # as tofu boxes - which would break the very CVD mitigation they exist for.
    etiket = {"GREEN": "AKTİF", "AMBER": "BELİRSİZ", "RED": "TERK RİSKİ",
              "VERI_YOK": "VERİ YOK"}
    renk = {"GREEN": GOOD, "AMBER": WARN, "RED": CRIT, "VERI_YOK": INK3}

    kul = sk["badge_today"].value_counts().reindex(sira).fillna(0)
    one = sec.value_counts().reindex(sira).fillna(0)
    satirlar = [("Külliyatın tamamı", kul, len(sk)),
                (f"İlk-5 öneri ({len(q)} sorgu)", one, int(one.sum()))]

    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    for i, (baslik, say, top) in enumerate(satirlar):
        sol = 0.0
        for k in sira:
            v = float(say.get(k, 0))
            if v <= 0:
                continue
            pay = 100 * v / top
            # 2px surface gap between stacked fills (dataviz mark spec):
            # drawn as a surface-coloured edge rather than a real gap, so the
            # segments still sum to exactly 100.
            ax.barh(i, pay, left=sol, height=0.46, color=renk[k], zorder=3,
                    edgecolor=SURFACE, linewidth=1.6)
            # LABEL MANDATORY - badge colours fail CVD separation on their own.
            if pay > 6:
                ax.text(sol + pay / 2, i, f"{etiket[k]}\n%{pay:.1f}", ha="center",
                        va="center", fontsize=11.5, fontweight="bold",
                        color="white", zorder=4)
            sol += pay
        ax.text(-1.5, i, baslik, ha="right", va="center", fontsize=12, color=INK)

    r_kul = 100 * float(kul.get("RED", 0)) / len(sk)
    r_one = 100 * float(one.get("RED", 0)) / int(one.sum())
    ax.set_xlim(0, 100); ax.set_ylim(-0.55, 1.55)
    ax.set_yticks([]); ax.set_xlabel("pay (%)")
    # Title kept short on purpose: the long version overflowed the axes. The
    # reasoning goes BELOW the plot as a caption - an annotation box inside the
    # axes collided with the title, which no amount of shortening fixes.
    ax.set_title("Rozet, aramanın yapamadığını yapıyor", pad=14)
    ax.grid(visible=False)
    fig.text(0.5, -0.02,
             f"Külliyatın %{_virgul(r_kul, 1)}'i terk riskli; anlamsal arama bunu ancak "
             f"%{_virgul(r_one, 1)}'e indiriyor — {_virgul(r_kul - r_one, 1)} puan. "
             f"Arama, kodlamadığı bir özelliğe göre sıralayamaz.",
             ha="center", va="top", fontsize=12, color=INK2)
    kaydet("F17_urun_iddiasi.png",
           f"arama %{r_kul:.1f} -> %{r_one:.1f} ancak indiriyor -> ROZET aramanin yapamadigini yapiyor")


###############################################################
KAYIT = {"F01": f01, "F02": f02, "F03": f03, "F04": f04, "F05": f05,
         "F06": f06, "F07": f07, "F15": f15, "F16": f16, "F17": f17}




###############################################################
# R-SERIES | THE RAW DATASET
###############################################################
# These do NOT repeat the chain's own 8 raw-data figures (distributions, target
# balance, ai_category, language, missing map, correlation, owner grouping,
# readme corpus). They answer the questions those leave open.

def r01():
    """When was this corpus born? The AI boom is visible in one bar."""
    d = csv_oku(["created_year"])
    g = d["created_year"].value_counts().sort_index()
    g = g[g.index >= 2014]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    renk = [S2 if y == 2023 else SEQ[3] for y in g.index]
    ax.bar(g.index.astype(str), g.values, color=renk, width=0.72, zorder=3)
    for x, v in zip(range(len(g)), g.values):
        ax.text(x, v + 60, bin_ayir(int(v)), ha="center", fontsize=9.5, color=INK2)
    kat = g.get(2023, 0) / max(g.get(2022, 1), 1)
    ax.annotate(f"x{kat:.1f}", xy=(list(g.index).index(2023), g.get(2023)),
                xytext=(list(g.index).index(2023) - 1.4, g.max() * 0.92),
                fontsize=15, fontweight="bold", color=S2,
                arrowprops=dict(arrowstyle="-|>", color=S2, lw=2))
    ax.set_ylabel("repo sayısı")
    ax.set_xlabel("kuruluş yılı")
    ax.set_title("Külliyat 2023'te patladı")
    ax.set_ylim(0, g.max() * 1.16)
    ax.grid(axis="x", visible=False)
    fig.text(0.5, -0.04, "2026 yılı eksik: anlık görüntü 2026-07-31'de donduruldu.",
             ha="center", va="top", fontsize=11, color=INK2)
    kaydet("R01_kurulus_yili.png", "2022 -> 2023 iki kattan fazla artis")


def r02():
    """Star concentration. A mean is meaningless on this distribution."""
    d = csv_oku(["stars"]).sort_values("stars")
    s = d["stars"].values
    kum = np.cumsum(s) / s.sum() * 100
    x = np.arange(1, len(s) + 1) / len(s) * 100
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    ax.plot(x, kum, color=S1, lw=2.4, zorder=4)
    ax.plot([0, 100], [0, 100], color=INK3, lw=1.6, ls="--", zorder=3,
            label="eşit dağılım olsaydı")
    for pay, ad in ((50, "alttaki %50"), (90, "alttaki %90")):
        y = float(np.interp(pay, x, kum))
        ax.plot([pay, pay], [0, y], color=S2, lw=1.2, ls=":", zorder=3)
        ax.plot(pay, y, "o", color=S2, ms=9, zorder=5)
        ax.annotate(f"{ad}\ntoplam yıldızın %{y:.1f}'i", xy=(pay, y),
                    xytext=(pay - 44, min(y + 16, 88)), fontsize=11, color=INK,
                    arrowprops=dict(arrowstyle="-|>", color=S2, lw=1.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("repoların yüzdesi (az yıldızlıdan çoğa)")
    ax.set_ylabel("toplam yıldızın kümülatif yüzdesi")
    ax.set_title("Yıldızın %78,7'si repoların %10'unda")
    ax.legend(loc="upper left")
    med, ort = int(np.median(s)), s.mean()
    fig.text(0.5, -0.03,
             "medyan " + bin_ayir(med) + "  ortalama " + f"{ort:,.0f}".replace(",", ".")
             + "  en yüksek " + bin_ayir(int(s.max()))
             + "  —  bu dağılımda ORTALAMA yanıltıcıdır.",
             ha="center", va="top", fontsize=11, color=INK2)
    kaydet("R02_yildiz_yogunlasmasi.png", "guc yasasi: ortalama degil MEDYAN raporla")


def r03():
    """Licence composition, split by whether it is usable at work."""
    d = csv_oku(["license"])
    g = d["license"].value_counts()
    SORUNLU = {"No License", "NOASSERTION", "GPL-3.0", "AGPL-3.0", "GPL-2.0"}
    ilk = g.head(9)
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    renk = [S2 if k in SORUNLU else S1 for k in ilk.index]
    ax.barh(list(ilk.index)[::-1], list(ilk.values)[::-1],
            color=renk[::-1], height=0.66, zorder=3)
    for i, v in enumerate(list(ilk.values)[::-1]):
        ax.text(v + g.max() * 0.014, i, bin_ayir(int(v)), va="center",
                fontsize=11, color=INK)
    n = len(d)
    sorunlu = int(g[[k for k in g.index if k in SORUNLU]].sum())
    ax.set_xlim(0, g.max() * 1.16)
    ax.set_xlabel("repo sayısı")
    ax.set_title("Külliyatın %" + f"{100*sorunlu/n:.1f}" + "'i iş yerinde hukuken sorunlu")
    ax.grid(axis="y", visible=False)
    h = [plt.Rectangle((0, 0), 1, 1, color=S1), plt.Rectangle((0, 0), 1, 1, color=S2)]
    ax.legend(h, ["izin verici (MIT, Apache, BSD…)",
                  "lisanssız / belirsiz / copyleft"], loc="lower right")
    fig.text(0.5, -0.04,
             "Lisanssız " + bin_ayir(int(g.get("No License", 0)))
             + " + belirsiz " + bin_ayir(int(g.get("NOASSERTION", 0)))
             + " + GPL/AGPL " + bin_ayir(int(g.get("GPL-3.0", 0) + g.get("AGPL-3.0", 0)))
             + " = " + bin_ayir(sorunlu) + " repo.  Yıldızdan daha sert bir filtre.",
             ha="center", va="top", fontsize=11, color=INK2)
    kaydet("R03_lisans.png", "%35 hukuken kullanilamaz -> yildizdan ONEMLI filtre")


def r04():
    """The 17 technology flags - what this corpus is actually made of."""
    d = csv_oku(None)
    bay = [c for c in d.columns if c.startswith("uses_")]
    g = d[bay].sum().sort_values()
    ad = [c.replace("uses_", "") for c in g.index]
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    ax.barh(ad, g.values, color=SEQ[3], height=0.68, zorder=3)
    n = len(d)
    for i, v in enumerate(g.values):
        ax.text(v + g.max() * 0.014, i,
                bin_ayir(int(v)) + "  (%" + f"{100*v/n:.1f}" + ")",
                va="center", fontsize=10.5, color=INK)
    ax.set_xlim(0, g.max() * 1.26)
    ax.set_xlabel("repo sayısı")
    ax.set_title("17 teknoloji bayrağı — külliyat neyden oluşuyor")
    ax.grid(axis="y", visible=False)
    hic = int((d[bay].sum(axis=1) == 0).sum())
    fig.text(0.5, -0.03,
             "Hiçbir bayrağı olmayan repo: " + bin_ayir(hic)
             + " (%" + f"{100*hic/n:.1f}" + ").  Bayraklar metinden türetilmiş, kesin değil.",
             ha="center", va="top", fontsize=11, color=INK2)
    kaydet("R04_teknoloji_bayraklari.png", "kulliyatin bilesimi + bayrak kapsamasi")


def r05():
    """Column information value. Two columns carry none at all."""
    d = csv_oku(None)
    # uses_* flags are EXCLUDED: a binary flag having two values is not a
    # finding, and 17 of them buried the actual signal (fork/visibility) at the
    # bottom of the chart. What remains is the interesting set - two degenerate
    # columns, one real binary, and the three derived categoricals that the
    # leakage audit later condemns.
    benzersiz = d.drop(columns=[c for c in d.columns if c.startswith("uses_")]).nunique()
    z = benzersiz[benzersiz <= 4].sort_values()
    fig, ax = plt.subplots(figsize=(9.8, 4.4))
    renk = [CRIT if v <= 1 else (WARN if v == 2 else SEQ[3]) for v in z.values]
    ax.barh(list(z.index), list(z.values), color=renk, height=0.62, zorder=3)
    for i, (ad, v) in enumerate(z.items()):
        ek = "   <-- SIFIR BİLGİ" if v <= 1 else ""
        ax.text(v + 0.07, i, str(int(v)) + " değer" + ek, va="center",
                fontsize=11, color=CRIT if v <= 1 else INK)
    ax.set_xlim(0, 5.8)
    ax.set_xlabel("benzersiz değer sayısı")
    ax.set_title("En az bilgi taşıyan sütunlar")
    ax.grid(axis="y", visible=False)
    bos = d.isna().mean().sort_values(ascending=False).head(3)
    ozet = "  ".join(a + " %" + f"{100*v:.0f}" for a, v in bos.items() if v > 0)
    fig.text(0.5, -0.05,
             "`fork` ve `visibility` her satırda aynı değeri taşıyor — modele girseler "
             "bile hiçbir şey söylemezler.\nEn çok boş olan sütunlar: " + ozet,
             ha="center", va="top", fontsize=11, color=INK2)
    kaydet("R05_sutun_bilgi_degeri.png", "45 sutunun 2'si dejenere, 1'i yarisindan fazla bos")


def r06():
    """Is this corpus embeddable at all? The RAG design rests on this."""
    d = csv_oku(["description", "topics"])
    aciklama = d["description"].fillna("").astype(str).str.len()
    konu = d["topics"].fillna("").astype(str).apply(
        lambda t: 0 if not t.strip() else len(t.split()))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.4))

    a1.hist(aciklama.clip(upper=300), bins=40, color=S1, zorder=3)
    a1.axvline(aciklama.median(), color=S2, lw=2.2, zorder=4)
    a1.text(aciklama.median() + 8, a1.get_ylim()[1] * 0.88,
            "medyan " + str(int(aciklama.median())) + " karakter", color=S2, fontsize=11)
    a1.set_xlabel("description uzunluğu (karakter, 300'de kırpıldı)")
    a1.set_ylabel("repo sayısı")
    a1.set_title("Açıklama")
    a1.grid(axis="x", visible=False)

    a2.hist(konu.clip(upper=25), bins=25, color=S3, zorder=3)
    a2.axvline(konu.median(), color=S2, lw=2.2, zorder=4)
    a2.text(konu.median() + 0.6, a2.get_ylim()[1] * 0.88,
            "medyan " + str(int(konu.median())) + " etiket", color=S2, fontsize=11)
    a2.set_xlabel("topics etiket sayısı (25'te kırpıldı)")
    a2.set_title("Konu etiketleri")
    a2.grid(axis="x", visible=False)

    hic = int((konu == 0).sum())
    fig.suptitle("Külliyat gömülebilir mi? — RAG tasarımının dayanağı",
                 fontsize=15, fontweight="bold", y=1.03)
    fig.text(0.5, -0.04,
             "Hiç konu etiketi olmayan repo: " + bin_ayir(hic)
             + ".  Her repo en az bir metin sinyali taşıyor — `doc_primary` bu yüzden "
               "kimlik + açıklama + etiketle başlıyor.",
             ha="center", va="top", fontsize=11, color=INK2)
    kaydet("R06_metin_zenginligi.png", "her reponun metni var -> gomme yapilabilir")


KAYIT.update({"R01": r01, "R02": r02, "R03": r03,
              "R04": r04, "R05": r05, "R06": r06})



###############################################################
# F-SERIES 08-14 | THE MODEL BLOCK
###############################################################
# Bu yedi figur MODELIN ARGUMANLARINI cizer. Zincirin kendi 10 figuru
# (dagilimlar, kalibrasyon, SHAP beeswarm) STANDART settir; buradakiler
# juriye anlatilacak KARARLARI cizer.
#
# VERI KAYNAGI VE DURUSTLUK
# -------------------------
# AUDIT 2026-09-11 → docs/AUDIT_DEFTERI.md#eda-figures-747: Burada ABLASYON / KATMAN / KALIB / ESIK diye dort tablo
def _rapor():
    yol = os.path.join(OUT, "model_rapor.json")
    if not os.path.exists(yol):
        return None
    with open(yol, encoding="utf-8") as f:
        return json.load(f)


RAPOR = _rapor()


def _sozlesme_kapisi():
    """Figur, cizdigi sayilarla AYNI kosumun modelini mi anlatiyor?"""
    if RAPOR is None:
        raise SystemExit("model_rapor.json yok - DS21_Final_v2.py'yi kostur. "
                         "Rapor okuyan figurler cizilmedi.")
    if RAPOR.get("run_id") != RUN_ID:
        raise SystemExit(f"SOZLESME KAPISI: model_rapor.json run_id "
                         f"{RAPOR.get('run_id')!r} != model.pkl run_id {RUN_ID!r}. "
                         f"Iki dosya farkli kosumlardan - figur cizilmedi.")


def _virgul(x, nd=4, isaret=False):
    """0.0141 -> '0,0141' (isaret=True: '+0,0141'). Figur metinleri Turkce."""
    s = f"{x:+.{nd}f}" if isaret else f"{x:.{nd}f}"
    return s.replace(".", ",")


###############################################################
# F08 | ABLASYON MERDIVENI - her blok ne kazandiriyor
###############################################################
def f08():
    """Tek slaytta 'ozellik muhendisligi ne kazandirdi' sorusunun cevabi."""
    _sozlesme_kapisi()
    AD = {"ref": "yalnız cm_silence_weeks", "meta": "metadata",
          "readme": "+ README yapısı", "commit": "+ commit geçmişi",
          "push": "+ repo etkinliği"}
    A = {r["block"]: r for r in RAPOR["ablation"]}
    sira = ["ref", "meta", "readme", "commit", "push"]
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    adlar = [AD[b] for b in sira]
    auc = [A[b]["auc"] for b in sira]
    std = [A[b]["fold_std"] for b in sira]
    kullanim = [A[b]["used"] for b in sira]

    renk = []
    for k in kullanim:
        renk.append(INK3 if k is None else (S1 if k else GRID))
    y = np.arange(len(adlar))
    ax.barh(y, auc, xerr=std, color=renk, height=0.62, zorder=3,
            error_kw=dict(ecolor=INK2, lw=1.4, capsize=4))
    ax.set_yticks(y)
    ax.set_yticklabels(adlar)
    ax.invert_yaxis()
    # Iki metin sutununa yer birakiliyor: deger ve kazanc. Ilk surumde
    # egri oklar + legend deger etiketlerinin USTUNE bindi; dogrudan
    # etiketleme hem cakismayi cozuyor hem de gostergeyi gereksiz kiliyor.
    ax.set_xlim(0.75, 1.085)
    ax.set_xticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_xlabel("AUC  (GroupKFold-5, sahibe göre gruplu)")
    ax.set_title("Her blok ne kazandırıyor")
    ax.grid(axis="y", visible=False)

    for i, (v, s) in enumerate(zip(auc, std)):
        ax.text(v + s + 0.005, i, _virgul(v, 4), va="center", fontsize=11, color=INK)

    # Kazanc sutunu - sabit x'te, cakisma imkansiz
    KAZ_X = 1.012
    for i, etiket in ((3, f"commit {_virgul(auc[3]-auc[2], 4, isaret=True)}"),
                      (4, f"etkinlik {_virgul(auc[4]-auc[3], 4, isaret=True)}")):
        ax.text(KAZ_X, i, etiket, fontsize=11.5, color=S2,
                fontweight="bold", va="center")

    # Legend yerine DOGRUDAN etiket - bar uzerine yazi
    # Barin ORTASINA - barin ucuna gore konumlandirinca hata cubuguna deger.
    ax.text((0.75 + auc[2]) / 2, 2, "KULLANILMIYOR", ha="center", va="center",
            fontsize=10.5, color=INK2, fontweight="bold")
    ax.text((0.75 + auc[0]) / 2, 0, "referans", ha="center", va="center",
            fontsize=10.5, color="white", fontweight="bold")
    readme_kaz, commit_kaz, push_kaz = auc[2] - auc[1], auc[3] - auc[2], auc[4] - auc[3]
    fig.text(0.5, -0.04,
             f"README bloğu {_virgul(readme_kaz, isaret=True)} kazandırıyor ama etiket "
             f"penceresi içinde DEĞİŞİYOR (192× kirlilik, tek seferlik denetim) ve "
             f"commit'in üstüne yalnız {_virgul(RAPOR['readme_on_commit'], isaret=True)} "
             f"koyuyor - çıkarıldı.\nHata çubukları kat standart sapması; "
             f"etkinliğin {_virgul(push_kaz, isaret=True)} kazancı kendi std'sinin "
             f"({_virgul(std[4])}) {_virgul(push_kaz / std[4], 1)} katı.",
             ha="center", va="top", fontsize=11, color=INK2)
    kaydet("F08_ablasyon.png", f"commit {_virgul(commit_kaz, isaret=True)} · "
                               f"etkinlik {_virgul(push_kaz, isaret=True)} · README çıkarıldı")


###############################################################
# F09 | KATMANLI AUC - "model sadece sessizlige mi bakiyor?"
###############################################################
def f09():
    """Jurinin soracagi soru. Global AUC katkiyi GIZLER."""
    _sozlesme_kapisi()
    _etiket_kapisi()
    AD = {">2 years silent": "2+ yıl sessiz", "1-2 years silent": "1-2 yıl sessiz",
          "4-52 weeks silent": "4-52 hafta sessiz", "<4 weeks silent": "4 haftadan az"}
    S = {r["slice"]: r for r in RAPOR["slices"]}
    KATMAN = [(AD[k], S[k]["n"], S[k]["rate"], S[k]["silence_auc"], S[k]["model_auc"])
              for k in AD if k in S]
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    n = len(KATMAN)
    y = np.arange(n)
    g = 0.34
    sil = [k[3] for k in KATMAN]
    mod = [k[4] for k in KATMAN]
    ax.barh(y - g / 2, sil, height=g, color=GRID, zorder=3, label="yalnız sessizlik")
    ax.barh(y + g / 2, mod, height=g, color=S1, zorder=3, label="tam model")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{k[0]}\nn={bin_ayir(k[1])} · {LW['pos_tr']} %{_virgul(k[2]*100, 1)}"
                        for k in KATMAN])
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.02)
    ax.set_xlabel("AUC (dilim içinde)")
    ax.set_title("Model, sessizliğin ÜSTÜNE ne koyuyor")
    ax.grid(axis="y", visible=False)
    for i, (a, b) in enumerate(zip(sil, mod)):
        ax.text(b + 0.006, i + g / 2, _virgul(b, 3), va="center", fontsize=10.5, color=INK)
        ax.text(a + 0.006, i - g / 2, _virgul(a, 3), va="center", fontsize=10.5, color=INK2)
        ax.annotate(_virgul(b - a, 3, isaret=True), xy=(0.53, i), fontsize=12,
                    color=S2, fontweight="bold", va="center")
    # Ilk surumde legend son satirin deger etiketini kesiyordu; ust sag
    # bolge (ilk satirin gri barinin otesi) bostur, oraya alindi.
    ax.legend(loc="upper right")
    tum = S["ALL"]["model_auc"] - S["ALL"]["silence_auc"]
    kaz = [b - a for a, b in zip(sil, mod)]
    fig.text(0.5, -0.05,
             f"Butun veride model sessizligin uzerine yalnizca {_virgul(tum, 3, True)} "
             f"koyuyor - tek sayiyla bakinca 'model sessizlikten ibaret' gorunur.\n"
             f"AYNI SURE sessiz kalmis repolari karsilastirdiginizda katki "
             f"{_virgul(min(kaz), 3, True)} ile {_virgul(max(kaz), 3, True)} arasina "
             f"cikiyor. Ayrim burada doguyor.",
             ha="center", va="top", fontsize=11, color=INK2)
    kaydet("F09_katmanli_auc.png", f"global {_virgul(tum, 3, True)} ama dilim icinde "
                                   f"{_virgul(min(kaz), 3, True)}..{_virgul(max(kaz), 3, True)}")


###############################################################
# F10 | ROZET GECIS MATRISI - canli skorlama neden gerekli
###############################################################
def f10():
    """Verilerden HESAPLANIYOR, sabit degil - iki parquet karsilastiriliyor."""
    rs = pd.read_parquet(os.path.join(OUT, "repo_scores.parquet"))
    st = pd.read_parquet(os.path.join(OUT, "scores_today.parquet"))
    k = rs.merge(st, on="full_name", how="inner")
    k = k[k["badge_today"].notna() & (k["badge_today"] != "VERI_YOK")]
    # `badge` esigi metnin ICINDE tasiyor ('RED <0.30') - LIKE ile kirp.
    eski = k["badge"].astype(str).str.split().str[0]
    yeni = k["badge_today"].astype(str)
    sira = ["RED", "AMBER", "GREEN"]
    ad = {"RED": "KIRMIZI", "AMBER": "SARI", "GREEN": "YESIL"}
    M = pd.crosstab(eski, yeni).reindex(index=sira, columns=sira, fill_value=0)

    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    ax.imshow(M.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(3), [ad[s] for s in sira])
    ax.set_yticks(range(3), [ad[s] for s in sira])
    ax.set_xlabel("BUGUN")
    ax.set_ylabel("egitim penceresi (T)")
    degisen = int(M.values.sum() - np.trace(M.values))
    ax.set_title(f"Rozetlerin %{100*degisen/M.values.sum():.1f}'i degisti")
    ax.grid(False)
    enb = M.values.max()
    for i in range(3):
        for j in range(3):
            v = M.values[i, j]
            vurgu = (sira[i] == "GREEN" and sira[j] == "RED")
            ax.text(j, i, bin_ayir(int(v)), ha="center", va="center",
                    fontsize=15 if vurgu else 13,
                    fontweight="bold" if vurgu else "normal",
                    color=S2 if vurgu else ("white" if v > enb * 0.5 else INK))
    yesil_kirmizi = int(M.loc["GREEN", "RED"])
    fig.text(0.5, -0.03,
             f"YESIL -> KIRMIZI: {bin_ayir(yesil_kirmizi)} repo. Statik bir model "
             f"bunlari HALA 'aktif' diye gosterirdi -\nurunun onlemek icin var "
             f"oldugu hata tam olarak budur. Toplam {bin_ayir(degisen)} rozet yer degistirdi.",
             ha="center", va="top", fontsize=11, color=INK2)
    kaydet("F10_rozet_gecisi.png",
           f"yesil->kirmizi {yesil_kirmizi} repo · statik model onlari canli sanardi")


###############################################################
# F11 | KALIBRASYON - rozetler ESIK, siralama degil
###############################################################
def f11():
    """Neden CatBoost sevk edildi: AUC farki kucuk, KALIBRASYON farki 5 kat."""
    _sozlesme_kapisi()
    C = RAPOR["calibration"]
    KALIB = {"LightGBM (kalibresiz)": C["lgbm_uncalibrated"],
             "CatBoost (sevk edilen)": C["shipped"]}
    kat = C["lgbm_uncalibrated"] / C["shipped"]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    adlar = list(KALIB.keys())
    v = [KALIB[a] for a in adlar]
    renk = [GRID, S1]
    ax.barh(adlar, v, color=renk, height=0.5, zorder=3)
    for i, x in enumerate(v):
        ax.text(x + 0.0006, i, f"{x:.4f}", va="center", fontsize=12, color=INK)
    ax.set_xlim(0, max(v) * 1.25)
    ax.set_xlabel("ortalama |tahmin - gercek|  (10 desilde)")
    ax.set_title(f"Kalibrasyon hatasi - {_virgul(kat, 1)} kat fark")
    ax.grid(axis="y", visible=False)
    fig.text(0.5, -0.06,
             f"AUC'de iki model arasinda {_virgul(abs(C['auc_shipped'] - C['auc_lgbm']), 3)} "
             f"var; kalibrasyonda {_virgul(kat, 1)} KAT.\n"
             "Rozetler olasilik ESIKLERI (>=%60 · <%30), siralama degil - "
             "yani bu proje icin dogru olcut kalibrasyon.",
             ha="center", va="top", fontsize=11, color=INK2)
    kaydet("F11_kalibrasyon.png", f"CatBoost {_virgul(C['shipped'])} vs LightGBM "
                                  f"{_virgul(C['lgbm_uncalibrated'])} = {_virgul(kat, 1)} kat")


###############################################################
# F12 | ROZET ESIKLERI GERCEKTE NE ANLAMA GELIYOR
###############################################################
def f12():
    _sozlesme_kapisi()
    AD = {"RED <0.30": "KIRMIZI  <%30", "AMBER .30-.60": "SARI   %30-60",
          "GREEN >=0.60": "YESIL   >=%60"}
    ESIK = [(AD[b["badge"]], b["n"], b["rate"]) for b in RAPOR["badges"]]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    adlar = [e[0] for e in ESIK]
    oran = [e[2] for e in ESIK]
    n = [e[1] for e in ESIK]
    ax.bar(adlar, oran, color=[CRIT, WARN, GOOD], width=0.55, zorder=3)
    for i, (o, k) in enumerate(zip(oran, n)):
        ax.text(i, o + 0.02, f"%{o*100:.1f}", ha="center", fontsize=14,
                fontweight="bold", color=INK)
        ax.text(i, 0.03, f"n={bin_ayir(k)}", ha="center", fontsize=10.5, color="white")
    ax.set_ylim(0, 1.0)
    _etiket_kapisi()
    ax.set_ylabel(f"gercekte {LW['pos_tr']} oran")
    ax.set_title("Rozet ne vaat ediyor, gerceklik ne diyor")
    ax.grid(axis="x", visible=False)
    fig.text(0.5, -0.05,
             "Rozet rengi TEK BASINA anlam tasimaz - her yerde yuzde ve metin "
             "esligindedir (renk korlugu olcumu gerekcesiyle).\n"
             f"Esikler mutlak (BADGE_MODE=absolute): orneklem tabani "
             f"%{_virgul(RAPOR['base_rate'] * 100, 1)}'dan %25'e inseydi rozetlerin yalniz "
             f"%{_virgul(RAPOR['badge_moves_if_base_rate']['0.25'] * 100, 1)}'i yer degistirirdi.",
             ha="center", va="top", fontsize=11, color=INK2)
    kaydet("F12_rozet_esikleri.png",
           " · ".join(f"{e[0].split()[0].lower()} %{_virgul(e[2] * 100, 1)}" for e in ESIK)
           + f" {LW['pos_tr']}")


###############################################################
# F13 | HAFTA SINIRI KUSURU - muhendislik durustlugu slayti
###############################################################
def f13():
    """Veri degil KAVRAM cizimi. Bulunan, olculen ve kapatilan bir sizinti."""
    fig, ax = plt.subplots(figsize=(10.6, 4.4))
    ax.set_xlim(-0.6, 7.2)
    ax.set_ylim(-1.5, 2.2)
    ax.axis("off")
    gunler = ["PZR\n08", "PZT\n09", "SAL\n10", "ÇAR\n11", "PER\n12", "CUM\n13", "CMT\n14"]
    for i, g in enumerate(gunler):
        onceki = i < 3
        ax.add_patch(Rectangle((i, 0), 0.92, 1.0, facecolor=S1 if onceki else CRIT,
                               alpha=0.85, edgecolor=SURFACE, lw=2))
        ax.text(i + 0.46, 0.5, g, ha="center", va="center", color="white",
                fontsize=11, fontweight="bold")
    ax.annotate("T = 11 Mart\n(ÇARŞAMBA)", xy=(3.0, 1.05), xytext=(3.0, 1.95),
                ha="center", fontsize=12, fontweight="bold", color=INK,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
    ax.plot([3.0, 3.0], [-0.35, 1.0], color=INK, lw=2.5, ls="--", zorder=5)
    ax.text(1.4, -0.62, "T'den ÖNCE  ✓", ha="center", fontsize=12, color=S1,
            fontweight="bold")
    ax.text(5.2, -0.62, "T'den SONRA  ✗  4 gün", ha="center", fontsize=12,
            color=CRIT, fontweight="bold")
    ax.add_patch(Rectangle((0, -1.15), 6.92, 0.32, facecolor=INK3, alpha=0.25))
    ax.text(3.46, -0.99, "GitHub tek satır olarak verir:  week = 2026-03-08",
            ha="center", va="center", fontsize=11, color=INK)
    ax.set_title("Bulunan sızıntı: haftalık kova bir NOKTA değil ARALIK", pad=16)
    fig.text(0.5, -0.02,
             "`week < T` filtresi haftanın BAŞLANGICINA bakıyordu, T Çarşamba'ya "
             "düştüğü için 11-14 Mart özelliklere sızdı (2.740 repo).\n"
             "Kök sebep: T veriden türetiliyordu (SNAPSHOT-180) ve CSV tazelenince "
             "sessizce Çarşamba'ya kaydı.  Çözüm: T Pazar'a hizalandı, H=182 (26 tam hafta).",
             ha="center", va="top", fontsize=11, color=INK2)
    kaydet("F13_hafta_siniri.png", "4 günlük ileriye bakış · 2.740 repo · kapatıldı")


###############################################################
# F14 | ETIKET DOGRULAMASI - iki bagimsiz kaynak ayni seyi mi soyluyor
###############################################################
def f14():
    """outputs/label_compare.parquet'ten - DS21_Final_v2.py STEP 4.4'un tablosu.

    AUDIT 2026-09-11 | Eskisi features.parquet['target']'i "SEVK EDILEN: commit"
    diye ciziyordu; etiket C'ye gecilince o sutun C oldu ve figur C'yi yanlis
    adla pushed_at'e kiyasladi - hata vermeden. Artik etiketi ICAT etmiyor,
    v2'nin yazdigi tabloyu okuyor ve yalnizca gozlenebilir satirlari ciziyor.
    """
    _etiket_kapisi()
    k = pd.read_parquet(os.path.join(OUT, "label_compare.parquet"))
    sevk = LABEL_SOURCE
    digerleri = [c for c in k.columns if c not in ("full_name", sevk)]
    alt = "commit" if "commit" in digerleri else digerleri[0]
    k = k[k[alt].notna()]
    a, b = k[sevk].astype(int), k[alt].astype(int)
    M = pd.crosstab(b, a).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    uyum = (a == b).mean()
    AD = {"etiket_c": "etiket C (varsayılan dala ulaşan değişiklik)",
          "commit": "commit (/stats/contributors)",
          "pushed_at": "pushed_at > T"}

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    ax.imshow(M.values, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1], [LW["neg_tr"], LW["pos_tr"]])
    ax.set_yticks([0, 1], [f"{alt} YOK", f"{alt} VAR"])
    ax.set_xlabel(f"SEVK EDİLEN etiket:  {AD.get(sevk, sevk)}")
    ax.set_ylabel(f"bağımsız kaynak:  {AD.get(alt, alt)}")
    ax.set_title(f"İki kaynak %{_virgul(uyum*100, 2)} aynı şeyi söylüyor")
    ax.grid(False)
    enb = M.values.max()
    for i in range(2):
        for j in range(2):
            v = int(M.values[i, j])
            capraz = (i != j)
            ax.text(j, i, bin_ayir(v), ha="center", va="center",
                    fontsize=15 if capraz else 13,
                    fontweight="bold" if capraz else "normal",
                    color=S2 if capraz else ("white" if v > enb * 0.5 else INK))
    fig.text(0.5, -0.03,
             f"{bin_ayir(len(k))} repoda iki bağımsız kaynak aynı soruyu cevapladı "
             f"({bin_ayir(int(M.loc[0, 1]) + int(M.loc[1, 0]))} uyuşmazlık).\n"
             f"Karşılaştırma yalnızca {alt} etiketinin GÖZLENEBİLDİĞİ satırlarda; "
             f"gözlenemeyen repo negatif sayılmadı.\n"
             f"Ürün metni etiketi izler: '{LW['urun']}'.",
             ha="center", va="top", fontsize=11, color=INK2)
    kaydet("F14_etiket_dogrulamasi.png",
           f"{sevk} vs {alt} uyum %{uyum*100:.2f} · n={len(k)} · "
           f"uyusmazlik {int(M.loc[0, 1]) + int(M.loc[1, 0])}")


KAYIT.update({"F08": f08, "F09": f09, "F10": f10, "F11": f11,
              "F12": f12, "F13": f13, "F14": f14})

###############################################################
# F18 | OZELLIK ONEMI, BLOKLARA GORE - F08'in ic dokumu
###############################################################
# NEDEN ZINCIRDEKI BEESWARM'IN YERINE DEGIL, YANINA
# --------------------------------------------------
# figures/10_shap_beeswarm.png dagilimi gosterir: bir ozelligin degeri
# yukselince tahmin ne yone gidiyor. Bu figur BASKA bir soruyu cevapliyor:
# HANGI BLOK tasiyor? Renkler F08'in ablasyon merdiveniyle ayni hikayeyi
# anlatir - orada bloklarin TOPLU katkisi, burada ICINDEKI dagilim.
#
# SABIT SAYI YOK: SHAP model.pkl + features.parquet'ten HER KOSUDA yeniden
# hesaplaniyor (CatBoost'ta 22.270 x 59 icin ~2 sn). Bu oturumda elle yazilmis
# sabitlerin bayatladigi IKI vaka bulundu; hesaplanabilecek bir sey sabitlenmez.
def f18():
    import joblib
    from catboost import Pool

    m = joblib.load(os.path.join(OUT, "model.pkl"))
    mdl, feat = m["model"], m["feature_cols"]
    cat = [c for c in m.get("cat_cols", []) if c in feat]
    yasakli = set(m.get("banned_columns", []))

    f = pd.read_parquet(os.path.join(OUT, "features.parquet"))
    X = f[feat].copy()
    for c in cat:
        X[c] = X[c].astype(str)

    # 🚪 KAPI: sizinti denetimi modelle birlikte seyahat ediyor. Yasakli bir
    # sutun ozellik listesine sizmissa figur bunu GOSTERMEZ, CIZMEZ.
    sizan = yasakli & set(feat)
    if sizan:
        raise SystemExit(f"SIZINTI: yasakli sutun(lar) modelde: {sorted(sizan)}")

    sv = mdl.get_feature_importance(Pool(X, cat_features=cat), type="ShapValues")
    onem = pd.Series(np.abs(sv[:, :-1]).mean(axis=0), index=feat)

    def blok(ad):
        if ad.startswith("cm_"):
            return "commit geçmişi"
        if ad.startswith("pa_"):
            return "repo etkinliği"
        if ad.startswith("rm_"):
            return "README"
        return "metadata"

    RENK = {"commit geçmişi": S1, "repo etkinliği": S2, "metadata": INK3,
            "README": GRID}
    N = 18
    ilk = onem.sort_values(ascending=False).head(N)[::-1]

    fig, ax = plt.subplots(figsize=(10.0, 7.4))
    renkler = [RENK[blok(a)] for a in ilk.index]
    y = np.arange(len(ilk))
    ax.barh(y, ilk.values, color=renkler, height=0.68, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(ilk.index, fontsize=10.5)
    ax.set_xlim(0, ilk.max() * 1.20)
    ax.set_xlabel("ortalama |SHAP|  (tahmine katkı büyüklüğü)")
    ax.set_title(f"En etkili {N} özellik - hangi blok taşıyor")
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(ilk.values):
        ax.text(v + ilk.max() * 0.012, i, _virgul(v, 3), va="center",
                fontsize=10, color=INK)

    # Bar boylari asagi dogru KISALDIGI icin sag-alt bos - legend oraya.
    h = [Rectangle((0, 0), 1, 1, color=RENK[b])
         for b in ("commit geçmişi", "repo etkinliği", "metadata")]
    ax.legend(h, ["commit geçmişi", "repo etkinliği", "metadata"],
              loc="lower right", fontsize=11)

    pay = onem.groupby(onem.index.map(blok)).sum()
    pay = (pay / pay.sum() * 100).sort_values(ascending=False)
    ozet = "  ·  ".join(f"{b} %{_virgul(p, 1)}" for b, p in pay.items())
    pa_sira = int(onem.rank(ascending=False)["pa_events"]) if "pa_events" in onem else None
    fig.text(0.5, -0.025,
             f"Toplam |SHAP| payı:  {ozet}\n"
             f"İlk sırada `cm_silence_weeks` var ve bu BEKLENEN - o, yasaklı "
             f"`days_since_last_push`'un DÜRÜST, tahmin gününe göre hesaplanmış hâli.\n"
             f"Listede yasaklı {len(yasakli)} sütundan hiçbiri YOK; denetim model.pkl'in "
             f"içinde taşınıyor ve bu figür çizilmeden önce yeniden koşuyor.",
             ha="center", va="top", fontsize=11, color=INK2)
    kaydet("F18_ozellik_onemi.png",
           f"{pay.index[0]} %{pay.iloc[0]:.0f} · pa_events {pa_sira}. sırada · yasaklı sütun YOK")


KAYIT["F18"] = f18


###############################################################
# F19 | Confusion matrix at the green line + the three badges vs what happened
###############################################################
# Out-of-fold predictions of the shipped model (repo_scores.parquet), so every
# repo is scored by a model that never saw it. Badges are recomputed from the
# probability and checked against the stored badge column before drawing.
def _bin(x):
    """14797 -> '14.797' (Turkish thousands separator)."""
    return f"{int(x):,}".replace(",", ".")


def f19():
    yol = os.path.join(OUT, "repo_scores.parquet")
    if not os.path.exists(yol):
        raise SystemExit("repo_scores.parquet yok - DS21_Final_v2.py once kosulmali")
    s = pd.read_parquet(yol, columns=["p_active_180d", "badge", "target"])
    p, y = s["p_active_180d"].to_numpy(), s["target"].to_numpy().astype(int)
    rozet = np.where(p >= 0.60, "GREEN", np.where(p >= 0.30, "AMBER", "RED"))
    uyusma = (rozet == s["badge"].astype(str).str.split().str[0].to_numpy()).mean()
    if uyusma != 1.0:
        raise SystemExit(f"rozet esikleri kayitli rozetle uyusmuyor ({uyusma:.4f})")

    yesil = p >= 0.60
    m = np.array([[np.sum(~yesil & (y == 0)), np.sum(yesil & (y == 0))],
                  [np.sum(~yesil & (y == 1)), np.sum(yesil & (y == 1))]])
    kesinlik, duyarlilik = m[1, 1] / m[:, 1].sum(), m[1, 1] / m[1, :].sum()
    sira = ["RED", "AMBER", "GREEN"]
    r = np.array([[np.sum((rozet == k) & (y == 0)), np.sum((rozet == k) & (y == 1))] for k in sira])

    def hucre(ax, i, j, deger, pay, alt, vurgu):
        ax.add_patch(Rectangle((j, i), 1, 1, facecolor=SEQ[min(6, 1 + int(pay * 5.5))], edgecolor=SURFACE, lw=4))
        koyu = pay > 0.55
        ax.text(j + .5, i + .38, _bin(deger), ha="center", va="center", fontsize=22, fontweight="bold",
                color="white" if koyu else INK)
        ax.text(j + .5, i + .68, alt, ha="center", va="center", fontsize=11, linespacing=1.5,
                color="white" if koyu else INK2)
        if vurgu:
            ax.add_patch(Rectangle((j + .02, i + .02), .96, .96, fill=False, edgecolor=CRIT, lw=2.5))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1, 1.15], "wspace": .45})
    ad = [["doğru: güncellenmedi", "yanlış yeşil"], ["kaçırılan yeşil", "doğru yeşil"]]
    for i in range(2):
        for j in range(2):
            pay = m[i, j] / m[i].sum()
            hucre(a1, i, j, m[i, j], pay, f"{ad[i][j]}\nsatırın %{_virgul(pay * 100, 1)}", (i, j) == (0, 1))
    a1.set_yticks([.5, 1.5], ["gerçek:\ngüncellenmedi", "gerçek:\ngüncellendi"])
    a1.set_xticks([.5, 1.5], ["tahmin: güncellenmez\n(< 0,60)", "tahmin: güncellenir\n(≥ 0,60, yeşil)"])
    a1.set_title(f"Yeşil rozet çizgisinde (0,60)\nkesinlik %{_virgul(kesinlik * 100, 1)} · "
                 f"duyarlılık %{_virgul(duyarlilik * 100, 1)}")
    adlar = {"RED": ("Kırmızı  < %30", CRIT), "AMBER": ("Sarı  %30-60", "#a86b00"), "GREEN": ("Yeşil  ≥ %60", GOOD)}
    for i, k in enumerate(sira):
        for j in range(2):
            pay = r[i, j] / r[i].sum()
            hucre(a2, i, j, r[i, j], pay, f"rozetin %{_virgul(pay * 100, 1)}", False)
    a2.set_yticks([.5, 1.5, 2.5], [f"{adlar[k][0]}\n{_bin(r[i].sum())} repo" for i, k in enumerate(sira)])
    for t, k in zip(a2.get_yticklabels(), sira):
        t.set_color(adlar[k][1])
        t.set_fontweight("bold")
    a2.set_xticks([.5, 1.5], ["gerçek: güncellenmedi", "gerçek: güncellendi"])
    a2.set_title("Kullanıcının gördüğü üç rozet ve gerçekte ne oldu")
    for ax, n in ((a1, 2), (a2, 3)):
        ax.set_xlim(0, 2)
        ax.set_ylim(n, 0)
        ax.grid(False)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.text(0.5, 0.0, f"Katlar dışı tahminler · {_bin(len(s))} repo · GroupKFold(5, owner) · koşum {RUN_ID}",
             ha="center", fontsize=10.5, color=INK3)
    kaydet("F19_karisiklik_matrisi.png",
           f"0,60'ta yanlis yesil {m[0, 1]} · kacirilan {m[1, 0]} · kesinlik {kesinlik:.3f}")


KAYIT["F19"] = f19

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="tek bir figur, orn F03")
    ap.add_argument("--liste", action="store_true")
    a = ap.parse_args()
    if a.liste:
        for k in KAYIT:
            print(f"  {k}")
        return 0
    print("#" * 63)
    print(f"# eda_figures.py  ->  {FIG}")
    print("#" * 63)
    isler = {a.only: KAYIT[a.only]} if a.only else KAYIT
    for ad, fn in isler.items():
        try:
            fn()
        except Exception as e:                                   # noqa: BLE001
            print(f"  {ad} HATA: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
