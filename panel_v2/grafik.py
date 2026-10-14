# -*- coding: utf-8 -*-
###############################################################
# grafik.py | panel_v2 - interactive charts drawn from the artifacts
###############################################################
# Why native charts instead of the PNGs: several PNGs carry text typed at
# drawing time that has since gone stale (F02 "4.717", F07 "T+180", F16
# "inside the CI"). A chart built here from the data cannot say something
# the data does not.
#
# Palette = tema.css variables (light theme), so charts sit in the same visual language.
###############################################################

import pandas as pd
import plotly.graph_objects as go

ROZET_RENK = {"RED": "#d03b3b", "AMBER": "#d4960f", "GREEN": "#0ca30c", "VERI_YOK": "#8b948f"}
ROZET_AD = {"RED": "🔴 Kırmızı", "AMBER": "🟡 Sarı", "GREEN": "🟢 Yeşil", "VERI_YOK": "Veri yok"}   # HTML only
ROZET_DUZ = {"RED": "Kırmızı <%30", "AMBER": "Sarı %30-60", "GREEN": "Yeşil ≥%60", "VERI_YOK": "Veri yok"}  # inside charts: emoji render unevenly in SVG
BLUE, CYAN, TEAL, GREEN = "#2a78d6", "#0e7c86", "#0e9488", "#0ca30c"
MUTED, INK, GRID, CARD, SOLUK = "#7b857f", "#1b2420", "#dde2dd", "#ffffff", "#c9d0ca"

CONFIG = {"displayModeBar": False}


def _tr(x, nd=0):
    s = f"{x:,.{nd}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def _yerlesim(fig, h=380, legend=False, **kw):
    fig.update_layout(
        height=h, margin=dict(l=10, r=24, t=36, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Instrument Sans, Segoe UI, sans-serif", size=14, color=INK),
        hoverlabel=dict(bgcolor=CARD, font_color="#1b2420", bordercolor=GRID),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        **kw)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def _dikey_cizgi(fig, x, metin, renk, xanchor="center"):
    fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper", line=dict(color=renk, width=2, dash="dot"))
    fig.add_annotation(x=x, y=1.02, yref="paper", text=metin, showarrow=False,
                       font=dict(color=renk, size=13), yanchor="bottom", xanchor=xanchor)


###############################################################
# 01 · time axis
###############################################################
def zaman_ekseni(z, geri_gun=365):
    T, son, bugun, ps = z["T"], z["T_son"], z["bugun"], z["pencere_sonu"]
    gun = pd.Timedelta(days=geri_gun)
    satir = [
        ("Eğitim · feature'lar", T - gun, T, BLUE, "tahmin gününden önce"),
        ("Eğitim · target", T, son, GREEN, f"sonraki {z['H']} gün · sonuç biliniyor"),
        ("Canlı · feature'lar", bugun - gun, bugun, "#6fa3e3", "bugünkü kesimden önce"),
        ("Canlı · tahmin", bugun, ps, CYAN, "henüz yaşanmadı"),
    ]
    fig = go.Figure()
    for ad, bas, bit, renk, not_ in satir:
        fig.add_trace(go.Bar(
            y=[ad], x=[(bit - bas).total_seconds() * 1000], base=[bas.isoformat()], orientation="h",
            marker=dict(color=renk, opacity=0.85 if "tahmin" not in ad else 0.45,
                        line=dict(color=renk, width=1.5)),
            text=[not_], textposition="inside", insidetextanchor="middle",
            textfont=dict(color="#ffffff" if "tahmin" not in ad else "#1b2420", size=13),
            hovertemplate=f"<b>{ad}</b><br>{bas:%d.%m.%Y} → {bit:%d.%m.%Y}<extra></extra>"))
    # labels grow toward opposite sides of their lines so they never overlap
    _dikey_cizgi(fig, T.isoformat(), f"tahmin günü {T:%d.%m.%Y}", CYAN, xanchor="right")
    if abs((son - bugun).days) <= 7:          # label window closes where live scoring starts: one label
        _dikey_cizgi(fig, bugun.isoformat(), f"{z['H']} gün sonra = bugünkü kesim {bugun:%d.%m.%Y}", "#d4960f",
                     xanchor="left")
    else:
        _dikey_cizgi(fig, son.isoformat(), f"{z['H']} gün sonra", GREEN)
        _dikey_cizgi(fig, bugun.isoformat(), f"bugünkü kesim {bugun:%d.%m.%Y}", "#d4960f", xanchor="left")
    _yerlesim(fig, h=330, barmode="overlay", bargap=0.35)
    fig.update_xaxes(type="date", tickformat="%b %Y")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_layout(margin=dict(l=10, r=24, t=56, b=10))
    return fig


###############################################################
# 03 · ablation ladder
###############################################################
def ablasyon(abl):
    sira = [("ref", "Yalnız sessizlik süresi · referans", MUTED),
            ("meta", "Metadata", BLUE),
            ("readme", "+ README (modelde YOK)", SOLUK),
            ("commit", "+ Commit geçmişi", BLUE),
            ("push", "+ Repo etkinliği", GREEN)]
    ad = [s[1] for s in sira]
    auc = [abl[s[0]]["auc"] for s in sira]
    std = [abl[s[0]]["fold_std"] for s in sira]
    fig = go.Figure(go.Bar(
        y=ad, x=auc, orientation="h", marker_color=[s[2] for s in sira],
        error_x=dict(type="data", array=std, color="#4d5852", thickness=1.5, width=4),
        text=[_tr(a, 3) for a in auc], textposition="inside", insidetextanchor="end",
        textfont=dict(size=15, color=["#1b2420" if s[2] in (MUTED, SOLUK) else "#ffffff" for s in sira]),
        hovertemplate="%{y}<br>AUC %{x:.4f}<extra></extra>"))
    kazanc = {"+ Commit geçmişi": abl["commit"]["auc"] - abl["readme"]["auc"],
              "+ Repo etkinliği": abl["push"]["auc"] - abl["commit"]["auc"]}
    for y, k in kazanc.items():
        fig.add_annotation(x=0.985, y=y, text=f"<b>+{_tr(k, 3)}</b>", showarrow=False,
                           font=dict(color=CYAN, size=15), xanchor="right")
    _yerlesim(fig, h=360)
    fig.update_xaxes(range=[0.78, 0.99], title="ROC-AUC · aynı katlar, GroupKFold(owner)")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return fig


###############################################################
# 03 · label agreement
###############################################################
def etiket_uyumu(ct):
    z = ct.values
    metin = [[_tr(v) for v in r] for r in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=["etiket: güncellenmedi", "etiket: güncellendi"],
        y=["commit: yok", "commit: var"], text=metin, texttemplate="%{text}",
        textfont=dict(size=20, color="#1b2420"), showscale=False,
        colorscale=[[0, "#eef2ee"], [1, "#9cc0ea"]], hoverinfo="skip", xgap=4, ygap=4))
    _yerlesim(fig, h=300)
    fig.update_yaxes(autorange="reversed")
    return fig


###############################################################
# 04 · silence slices
###############################################################
DILIM_AD = {">2 years silent": "2+ yıl sessiz", "1-2 years silent": "1-2 yıl sessiz",
            "4-52 weeks silent": "4-52 hafta sessiz", "<4 weeks silent": "4 haftadan az"}


def dilimler(sl):
    d = {s["slice"]: s for s in sl}
    anahtar = [k for k in DILIM_AD if k in d]
    y = [f"<b>{DILIM_AD[k]}</b><br><span style='font-size:12px;color:#4d5852'>"
         f"{_tr(d[k]['n'])} repo · %{_tr(d[k]['rate'] * 100, 1)} güncellendi</span>" for k in anahtar]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=y, x=[d[k]["silence_auc"] for k in anahtar], orientation="h", name="Yalnız sessizlik süresi",
                         marker_color=MUTED, text=[_tr(d[k]["silence_auc"], 3) for k in anahtar],
                         textposition="inside", hovertemplate="sessizlik AUC %{x:.3f}<extra></extra>"))
    fig.add_trace(go.Bar(y=y, x=[d[k]["model_auc"] for k in anahtar], orientation="h", name="Tam model",
                         marker_color=BLUE, text=[_tr(d[k]["model_auc"], 3) for k in anahtar],
                         textposition="inside", textfont=dict(color="#ffffff"),
                         hovertemplate="model AUC %{x:.3f}<extra></extra>"))
    for yy, k in zip(y, anahtar):
        fig.add_annotation(x=0.97, y=yy, text=f"<b>+{_tr(d[k]['model_auc'] - d[k]['silence_auc'], 3)}</b>",
                           showarrow=False, font=dict(color=CYAN, size=16), xanchor="right")
    _yerlesim(fig, h=430, legend=True, barmode="group", bargap=0.28, bargroupgap=0.08)
    fig.update_xaxes(range=[0.5, 0.98], title="ROC-AUC · aynı süre sessiz kalmış repolar arasında")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return fig


def kalibrasyon(egri):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Mükemmel kalibrasyon",
                             line=dict(color=MUTED, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=egri["tahmin"], y=egri["gercek"], mode="lines+markers", name="CatBoost (sevk edilen)",
                             line=dict(color=CYAN, width=3), marker=dict(size=10, color=CYAN,
                                                                        line=dict(color="#f4f6f4", width=2)),
                             customdata=egri["n"],
                             hovertemplate="tahmin %{x:.1%} → gerçek %{y:.1%}<br>%{customdata} repo<extra></extra>"))
    _yerlesim(fig, h=400, legend=True)
    fig.update_xaxes(range=[0, 1], tickformat=".0%", title="modelin tahmini (10 kova)")
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title="gerçekte güncellenen oran")
    return fig


###############################################################
# 04 · badges
###############################################################
def rozet_gerceklesme(badges):
    kod = [b["badge"].split()[0] for b in badges]
    fig = go.Figure(go.Bar(
        x=[ROZET_DUZ[k] for k in kod], y=[b["rate"] for b in badges],
        marker_color=[ROZET_RENK[k] for k in kod],
        text=[f"<b>%{_tr(b['rate'] * 100, 1)}</b><br>{_tr(b['n'])} repo" for b in badges],
        textposition="outside", cliponaxis=False, textfont=dict(size=16),
        hovertemplate="%{x}<br>gerçekte güncellenen %{y:.1%}<extra></extra>"))
    for esik, ad in ((0.30, "eşik %30"), (0.60, "eşik %60")):
        fig.add_hline(y=esik, line=dict(color=MUTED, dash="dot"))
        fig.add_annotation(x=1, xref="paper", y=esik, text=ad, showarrow=False, xanchor="right",
                           yanchor="bottom", font=dict(color="#4d5852", size=12))
    _yerlesim(fig, h=400, bargap=0.45)
    fig.update_yaxes(range=[0, 1.12], tickformat=".0%", title="gerçekte güncellenen oran")
    return fig


###############################################################
# 04 · badge transition (why live scoring)
###############################################################
def rozet_gecisi(ct):
    sira = list(ct.index)
    z = ct.values
    metin = [[_tr(v) for v in r] for r in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"bugün: {ROZET_DUZ[k].split()[0]}" for k in sira], y=[f"Mart: {ROZET_DUZ[k].split()[0]}" for k in sira],
        text=metin, texttemplate="%{text}", textfont=dict(size=19, color="#1b2420"),
        colorscale=[[0, "#eef2ee"], [0.25, "#c7dbf3"], [1, "#2a78d6"]], showscale=False,
        hovertemplate="%{y} → %{x}<br>%{text} repo<extra></extra>", xgap=4, ygap=4))
    gi, ri = sira.index("GREEN"), sira.index("RED")
    fig.add_shape(type="rect", x0=ri - 0.5, x1=ri + 0.5, y0=gi - 0.5, y1=gi + 0.5,
                  line=dict(color=ROZET_RENK["RED"], width=3))
    _yerlesim(fig, h=380)
    fig.update_yaxes(autorange="reversed")
    return fig


###############################################################
# 05 · today's badges + pool funnel
###############################################################
def bugun_halka(sayim):
    sira = [k for k in ("GREEN", "AMBER", "RED", "VERI_YOK") if k in sayim.index]
    n = [int(sayim[k]) for k in sira]
    top = sum(n)
    sade = {"RED": "Kırmızı", "AMBER": "Sarı", "GREEN": "Yeşil", "VERI_YOK": "Veri yok"}
    fig = go.Figure(go.Pie(
        labels=[sade[k] for k in sira], values=n, hole=0.64, sort=False, direction="clockwise",
        marker=dict(colors=[ROZET_RENK[k] for k in sira], line=dict(color="#f4f6f4", width=3)),
        text=[f"%{_tr(100 * v / top, 0)}" if v / top >= 0.03 else "" for v in n], textinfo="text",
        textposition="inside", textfont=dict(size=16, color=["#ffffff" if k in ("RED", "GREEN") else "#1b2420" for k in sira]),
        hovertemplate="%{label}<br>%{value:,} repo<extra></extra>"))
    fig.add_annotation(text=f"<b>{_tr(top)}</b><br><span style='font-size:12px'>repo · bugün</span>",
                       showarrow=False, font=dict(size=22, color="#1b2420"))
    _yerlesim(fig, h=360, legend=True)
    fig.update_layout(legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"))
    return fig


def havuz(adimlar):
    n0 = adimlar[0][1]
    renk = [MUTED] + [BLUE] * (len(adimlar) - 2) + [GREEN]
    fig = go.Figure(go.Funnel(
        y=[a for a, _ in adimlar], x=[n for _, n in adimlar],
        text=[f"<b>{_tr(n)}</b> · %{_tr(100 * n / n0, 0)}" for _, n in adimlar], textinfo="text",
        # wide steps carry their label inside, narrow ones outside (an outside label on the widest bar is clipped)
        textposition=["inside" if n / n0 > 0.5 else "outside" for _, n in adimlar],
        textfont=dict(size=15, color=["#ffffff" if n / n0 > 0.5 else "#1b2420" for _, n in adimlar]),
        marker=dict(color=renk),
        connector=dict(line=dict(color=GRID, width=1), fillcolor="rgba(42,120,214,.12)"),
        hoverinfo="skip"))
    _yerlesim(fig, h=360)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


###############################################################
# 05 · retrieval measurements
###############################################################
RAG_AD = [("head900", "bge-m3", MUTED), ("qwen3-0.6b", "Qwen3 0.6B", MUTED),
          ("head2000", "bge-m3 · 2.000 karakter", SOLUK),
          ("qwen3-8b", "Qwen3 8B", BLUE),
          ("qwen8b-YENI-bas+govde", "8B + README body (canlı)", GREEN)]


def rag_modeller(sweep):
    # best model on top: sort by the measured score, not by list order; a reversed axis draws the first row on top
    sat = sorted(((ad, sweep.loc[k, "r@5"], renk) for k, ad, renk in RAG_AD if k in sweep.index),
                 key=lambda s: s[1], reverse=True)
    fig = go.Figure(go.Bar(
        y=[s[0] for s in sat], x=[s[1] for s in sat], orientation="h", marker_color=[s[2] for s in sat],
        text=[_tr(s[1], 3) for s in sat], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>recall@5 %{x:.3f}<extra></extra>"))
    _yerlesim(fig, h=320)
    fig.update_xaxes(range=[0.4, 0.72], title="recall@5 · doğru repo ilk 5'te (354 soru)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", autorange="reversed")
    return fig


def rag_recall(sweep):
    k = [1, 5, 10, 20]
    fig = go.Figure()
    for anahtar, ad, renk, konum in (("qwen8b-YENI", "Yalnız README head", MUTED, "bottom center"),
                                     ("qwen8b-YENI-bas+govde", "Head + body (canlı)", GREEN, "top center")):
        if anahtar in sweep.index:
            y = [sweep.loc[anahtar, f"r@{i}"] for i in k]
            fig.add_trace(go.Scatter(x=[f"ilk {i}" for i in k], y=y, mode="lines+markers+text", name=ad,
                                     line=dict(color=renk, width=3), marker=dict(size=10),
                                     text=[_tr(v, 3) for v in y], textposition=konum, textfont=dict(color=renk),
                                     hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    _yerlesim(fig, h=320, legend=True)
    fig.update_yaxes(range=[0.4, 0.86], title="doğru repo bu sonuçların içinde")
    return fig


###############################################################
# 04 · ablation ladder of the SHIPPED CatBoost (sunum_olcumler.json "merdiven", B34)
###############################################################
def merdiven(m):
    sira = [("metadata", "Yalnız metadata", MUTED),
            ("metadata_commit", "+ Commit geçmişi", BLUE),
            ("full", "+ Repo etkinliği (sevk edilen model)", GREEN)]
    ad = [s[1] for s in sira]
    auc = [m[k]["auc"] for k, _, _ in sira]
    std = [m[k]["fold_std"] for k, _, _ in sira]
    fig = go.Figure(go.Bar(
        y=ad, x=auc, orientation="h", marker_color=[s[2] for s in sira],
        error_x=dict(type="data", array=std, color="#4d5852", thickness=1.5, width=4),
        text=[f"<b>{_tr(a, 4)}</b>" for a in auc], textposition="inside", insidetextanchor="end",
        textfont=dict(size=16, color=["#1b2420", "#ffffff", "#ffffff"]),
        customdata=[m[k]["n_ozellik"] for k, _, _ in sira],
        hovertemplate="%{y}<br>AUC %{x:.4f} · %{customdata} özellik<extra></extra>"))
    for i in (1, 2):
        fig.add_annotation(x=0.992, y=ad[i], text=f"<b>+{_tr(auc[i] - auc[i - 1], 4)}</b>", showarrow=False,
                           font=dict(color=CYAN, size=16), xanchor="right")
    ref = m["sessizlik"]["auc"]
    fig.add_shape(type="line", x0=ref, x1=ref, y0=-0.5, y1=2.5, line=dict(color="#d4960f", width=2, dash="dot"))
    fig.add_annotation(x=ref, y=-0.62, text=f"yalnız “kaç haftadır sessiz” · {_tr(ref, 4)}", showarrow=False,
                       font=dict(color="#d4960f", size=13), xanchor="right", yanchor="bottom")
    _yerlesim(fig, h=340)
    fig.update_xaxes(range=[0.78, 1.0], title="ROC-AUC · CatBoost · aynı 5 kat, GroupKFold(owner) · README yok")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_layout(margin=dict(l=10, r=24, t=56, b=10))
    return fig


###############################################################
# 04 · pa_events page-censoring spike (B2)
###############################################################
def diken_gruplari(d):
    """The six groups of the spike chart - shared with the chapter text so the text quotes the chart's own numbers."""
    n, repo, poz = d["n"], d["repo"], d["pozitif"]
    out = []
    for etiket, lo, hi in (("90-99 olay", 90, 99), ("tam 100 olay", 100, 100), ("101-110 olay", 101, 110),
                           ("190-199 olay", 190, 199), ("tam 200 olay", 200, 200), ("201-210 olay", 201, 210)):
        sec = [i for i, x in enumerate(n) if lo <= x <= hi]
        r = sum(repo[i] for i in sec)
        out.append({"ad": etiket, "adet": r, "oran": sum(repo[i] * poz[i] for i in sec) / r if r else 0, "tam": lo == hi})
    return out


def pa_diken(d):
    """Six bars, one question: what share of the repos in this group was updated?
    The two exact page sizes (100, 200) sit next to their neighbours. Was a 180-bar histogram with a second
    y-axis for the rate - nobody could read it (Sarp, 14.09)."""
    G = diken_gruplari(d)
    ad, oran, adet = [x["ad"] for x in G], [x["oran"] for x in G], [x["adet"] for x in G]
    renk = ["#d03b3b" if x["tam"] else BLUE for x in G]
    fig = go.Figure(go.Bar(
        x=ad, y=oran, marker_color=renk,
        text=[f"<b>%{_tr(o * 100, 1)}</b><br>{_tr(a)} repo" for o, a in zip(oran, adet)],
        textposition="outside", cliponaxis=False, textfont=dict(size=15),
        hovertemplate="%{x}<br>güncellenen %{y:.1%}<extra></extra>"))
    taban = d["cerceve_pozitif"]
    fig.add_hline(y=taban, line=dict(color=MUTED, dash="dot"))
    # label left of the first bar group, above the line: on the right it hid behind the tallest bar
    fig.add_annotation(x=1.5, y=taban, text=f"bütün repolarda %{_tr(taban * 100, 1)}", showarrow=False,
                       xanchor="center", yanchor="bottom", font=dict(color="#4d5852", size=12), bgcolor="#f4f6f4")
    _yerlesim(fig, h=400, bargap=0.35)
    fig.update_yaxes(range=[0, 1.15], tickformat=".0%", title="gerçekte güncellenen oran")
    fig.update_xaxes(title="8 Mart öncesi sayılan etkinlik olayı (eski okuma yöntemi)")
    return fig


###############################################################
# 05 · threshold curve + confusion matrix
###############################################################
def esik_egrisi(e):
    eg = e["egri"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eg["esik"], y=eg["precision"], mode="lines", name="kesinlik (precision)",
                             line=dict(color=GREEN, width=3), hovertemplate="eşik %{x:.2f} · kesinlik %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=eg["esik"], y=eg["recall"], mode="lines", name="duyarlılık (recall)",
                             line=dict(color=BLUE, width=3), hovertemplate="eşik %{x:.2f} · duyarlılık %{y:.3f}<extra></extra>"))
    _dikey_cizgi(fig, e["f1_en_iyi"]["esik"], f"F1'e göre en iyi · {_tr(e['f1_en_iyi']['esik'], 2)}", MUTED)
    _dikey_cizgi(fig, e["sevk"]["esik"], f"yeşil rozet çizgisi · {_tr(e['sevk']['esik'], 2)}", "#0ca30c")
    _yerlesim(fig, h=380, legend=True)
    fig.update_xaxes(range=[0.05, 0.95], title="olasılık eşiği")
    fig.update_yaxes(range=[0, 1.02], title="oran")
    fig.update_layout(margin=dict(l=10, r=24, t=70, b=10), legend=dict(y=1.12))
    return fig


def karisiklik(s):
    z = [[s["dn"], s["yp"]], [s["yn"], s["dp"]]]
    metin = [[f"<b>{_tr(s['dn'])}</b><br>doğru: güncellenmedi", f"<b>{_tr(s['yp'])}</b><br>yanlış yeşil"],
             [f"<b>{_tr(s['yn'])}</b><br>eksik satılan", f"<b>{_tr(s['dp'])}</b><br>doğru: güncellendi"]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["tahmin: güncellenmez", "tahmin: güncellenir (yeşil)"], y=["gerçek: güncellenmedi", "gerçek: güncellendi"],
        text=metin, texttemplate="%{text}", textfont=dict(size=16, color="#1b2420"), showscale=False,
        colorscale=[[0, "#eef2ee"], [1, "#9cc0ea"]], hoverinfo="skip", xgap=4, ygap=4))
    fig.add_shape(type="rect", x0=0.5, x1=1.5, y0=-0.5, y1=0.5, line=dict(color="#d03b3b", width=3))
    _yerlesim(fig, h=330)
    fig.update_yaxes(autorange="reversed")
    return fig


###############################################################
# 05 · AUC by age at T · random vs grouped split
###############################################################
def yas_auc(bantlar):
    fig = go.Figure(go.Bar(
        x=[b["bant"] for b in bantlar], y=[b["auc"] for b in bantlar], marker_color=[MUTED, BLUE, BLUE, GREEN],
        text=[f"<b>{_tr(b['auc'], 3)}</b><br>{_tr(b['n'])} repo · %{_tr(b['pozitif'] * 100, 0)} güncellendi" for b in bantlar],
        textposition="outside", cliponaxis=False, textfont=dict(size=14),
        hovertemplate="%{x}<br>AUC %{y:.4f}<extra></extra>"))
    _yerlesim(fig, h=380, bargap=0.4)
    fig.update_yaxes(range=[0.8, 0.97], title="ROC-AUC (katlar dışı)")
    fig.update_xaxes(title="T anında repo yaşı")
    return fig


###############################################################
# 06 · star bands · similarity threshold
###############################################################
def yildiz_bandi(bantlar):
    x = [f"{b['bant']} yıldız<br><span style='font-size:12px;color:#4d5852'>{b['n']} soru</span>" for b in bantlar]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=[b["r5_bas"] for b in bantlar], name="yalnız README head", marker_color=MUTED,
                         text=[_tr(b["r5_bas"], 3) for b in bantlar], textposition="outside", cliponaxis=False))
    fig.add_trace(go.Bar(x=x, y=[b["r5_bas_govde"] for b in bantlar], name="head + body (canlı)", marker_color=GREEN,
                         text=[_tr(b["r5_bas_govde"], 3) for b in bantlar], textposition="outside", cliponaxis=False))
    _yerlesim(fig, h=380, legend=True, barmode="group", bargap=0.3)
    fig.update_yaxes(range=[0, 1], title="recall@5")
    return fig


def esik_dagilim(e):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=e["sinav_top1"], name=f"sınav soruları ({len(e['sinav_top1'])})", marker_color=BLUE,
                               opacity=0.75, xbins=dict(start=0.35, end=0.95, size=0.02),
                               hovertemplate="benzerlik %{x}<br>%{y} soru<extra></extra>"))
    fig.add_trace(go.Histogram(x=e["negatif_top1"], name=f"konu dışı sorular ({len(e['negatif_top1'])})",
                               marker_color="#d03b3b", opacity=0.9, xbins=dict(start=0.35, end=0.95, size=0.02),
                               hovertemplate="benzerlik %{x}<br>%{y} soru<extra></extra>"))
    _dikey_cizgi(fig, e["esik"], f"eşik {_tr(e['esik'], 2)} · altı “uygun repo yok”", "#d4960f")
    _yerlesim(fig, h=380, legend=True, barmode="overlay")
    fig.update_xaxes(title="en yakın reponun benzerliği")
    fig.update_yaxes(title="soru sayısı")
    fig.update_layout(margin=dict(l=10, r=24, t=70, b=10), legend=dict(y=1.12))
    return fig


###############################################################
# 07 · RepoRadarAI architecture (inline SVG)
###############################################################
# Every box is a real component, read from the live system on 14.09; node names as renamed on 18.09:
#   site/vercel.json (rewrites to the Tailscale Funnel host) · gecit.bat (funnel --https=443 -> 127.0.0.1:8787)
#   web/gecit.js (esZamanli 4, kuyrukMax 12, istekId, oturum dondurme, temizle)
#   n8n DS21chat0002: Sohbet -> RepoRadarAI Agent (qwen3:14b, memory window 2, maxIterations 15) -> Adlari Dogrula -> Alt Bilgi
#     tools: Repo_Ara (-> DS21search0001), Populer_Repolar, Risk, Istatistik   (README 10.1-10.2)
#   n8n/RepoRadarAI Agent.json in this repo is the 16.09 export and still carries the old names (README S6).
#   n8n DS21search0001: Sorguyu Hazirla -> Ollama Gomme (qwen3-embedding:8b) -> Vektoru Hazirla/pgvector Ara -> Return (ESIK)
def sohbet_mimarisi(esik=0.58, repo_n=24664, govde_n=21816, limit=10, es_zamanli=4, kuyruk=12, bellek_tur=2, adim=15):
    W, H = 1200, 1010
    tr = lambda n: f"{n:,}".replace(",", ".")
    p = []

    def bant(y, h, etiket, sag=None):
        # short label on the left (arrows drop into boxes just right of it); workflow ids go to the right edge
        p.append(f'<rect x="10" y="{y}" width="{W - 20}" height="{h}" rx="16" fill="rgba(42,120,214,0.04)" '
                 f'stroke="rgba(42,120,214,0.22)"/><text x="26" y="{y + 19}" class="e">{etiket}</text>')
        if sag:
            p.append(f'<text x="{W - 26}" y="{y + 19}" class="k" text-anchor="end">{sag}</text>')

    def kutu(x, y, w, h, ust, baslik, satirlar, renk):
        p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#ffffff" stroke="{renk}" stroke-width="1.6"/>'
                 f'<text x="{x + 14}" y="{y + 20}" class="u" fill="{renk}">{ust}</text>'
                 f'<text x="{x + 14}" y="{y + 40}" class="b">{baslik}</text>')
        for i, s in enumerate(satirlar):
            p.append(f'<text x="{x + 14}" y="{y + 60 + i * 17}" class="s">{s}</text>')

    def ok(yol, kesik=False, etiket=None, ex=0, ey=0):
        p.append(f'<path d="{yol}" fill="none" stroke="#8a938d" stroke-width="1.8"'
                 f'{" stroke-dasharray=\"6 5\"" if kesik else ""} marker-end="url(#ok)"/>')
        if etiket:
            p.append(f'<text x="{ex}" y="{ey}" class="k">{etiket}</text>')

    MAVI, CAMGOBEGI, YESIL, MOR, TURUNCU, GRI = "#2a78d6", "#0e7c86", "#0ca30c", "#7c5cc4", "#eb6834", "#8a938d"
    bant(10, 135, "KULLANICI")
    kutu(60, 40, 330, 90, "TARAYICI", "reporadarai.vercel.app", ["Soruyu gönderir, cevabı 3 sn'de bir sorar"], MAVI)
    bant(155, 120, "İNTERNET")
    kutu(60, 185, 330, 78, "VERCEL", "Yönlendirme", ["/sor · /durum · /saglik → tünel"], GRI)
    kutu(440, 185, 330, 78, "TAILSCALE FUNNEL", "Güvenli tünel (https)", ["İnternetten bu bilgisayara tek kapı"], GRI)
    bant(285, 130, "BU BİLGİSAYAR · GEÇİT")
    kutu(60, 315, 1080, 90, "GECIT.JS · 127.0.0.1:8787", "Dışarıya açık tek kapı",
         [f"Aynı anda {es_zamanli} soru, en fazla {kuyruk} bekleyen  ·  aynı istek iki kez çalışmaz"], CAMGOBEGI)
    p.append('<text x="74" y="392" class="s">Konu değişince yeni oturum açar  ·  cevaptaki HTML\'i siler, '
             'yalnız github.com bağlantılarını bırakır</text>')
    bant(425, 240, "n8n · SOHBET", sag="akış: DS21chat0002")
    kutu(30, 455, 190, 100, "1 · WEBHOOK", "Sohbet", ["Soruyu ve oturumu alır"], YESIL)
    kutu(250, 455, 330, 100, "2 · AI AGENT", "RepoRadarAI Agent · qwen3:14b",
         [f"Bellek: son {bellek_tur} tur · en fazla {adim} adım", "Araç seçer, aramayı İngilizce yazar"], YESIL)
    kutu(615, 455, 250, 100, "3 · POSTGRES", "Adları doğrula", ["Cevaptaki repolar arşivde mi?", "Soruyu query_log'a yazar"], YESIL)
    kutu(900, 455, 260, 100, "4 · KOD", "Kontrol ve alt bilgi", ["Kod bloğu → sabit cevap", "Arşivde olmayan repo → cevap yok"], YESIL)
    kutu(60, 580, 160, 72, "ARAÇ", "Repo_Ara", ["anlamsal arama"], MOR)
    kutu(232, 580, 160, 72, "ARAÇ", "Populer_Repolar", ["yıldıza göre · SQL"], MOR)
    kutu(404, 580, 160, 72, "ARAÇ", "Risk", ["kırmızı, olgun · SQL"], MOR)
    kutu(576, 580, 160, 72, "ARAÇ", "Istatistik", ["kategori, dil · SQL"], MOR)
    bant(675, 140, "n8n · ARAMA", sag="akış: DS21search0001 · Repo_Ara çağırınca çalışır")
    kutu(40, 705, 230, 95, "1 · KOD", "Sorguyu hazırla", ["Dil, rozet ve adet süzgeçleri"], MOR)
    kutu(300, 705, 250, 95, "2 · OLLAMA", "Sorguyu vektöre çevir", ["qwen3-embedding:8b · 4.096 sayı"], MOR)
    kutu(580, 705, 300, 95, "3 · PGVECTOR", "En yakın repoları bul",
         ["README head ve body: en yakını", f"En ilgili {limit} repo, en aktiften"], MOR)
    kutu(910, 705, 250, 95, "4 · KOD", f"Eşik {esik:.2f}".replace(".", ","), ["Altındaysa “uygun repo yok”", "Sonucu ajana döndürür"], MOR)
    bant(825, 175, "VERİ VE MODELLER")
    kutu(40, 855, 300, 130, "OLLAMA · BU BİLGİSAYARDA", "Dil modelleri", ["qwen3:14b → sohbet", "qwen3-embedding:8b → arama"], TURUNCU)
    kutu(370, 855, 470, 130, "POSTGRESQL + PGVECTOR", "repoquest veritabanı",
         [f"repos: {tr(repo_n)} repo · vektör · bugünkü rozet · yıldız", f"repo_govde: {tr(govde_n)} README body",
          "query_log: sorulan sorular"], TURUNCU)
    kutu(870, 855, 290, 130, "PYTHON · ÇEKİMDEN SONRA", "Veriyi hazırlayanlar",
         ["score_today.py → bugünkü rozetler", "DS21_RAG.py → vektörler"], TURUNCU)

    ok("M225,130 L225,183", etiket="istek / cevap", ex=233, ey=162)
    ok("M390,224 L438,224")
    ok("M605,263 L605,313")
    ok("M180,405 L180,453", etiket="soru", ex=188, ey=440)
    ok("M220,505 L248,505")
    ok("M580,505 L613,505")
    ok("M865,505 L898,505")
    ok("M1030,455 L1030,407", etiket="cevap", ex=1038, ey=436)
    ok("M140,580 L140,570 L288,570 L288,557", kesik=True)   # lands right of the agent->Ollama line at x=262
    ok("M312,580 L312,557", kesik=True)
    ok("M484,580 L484,557", kesik=True)
    ok("M656,580 L656,568 L560,568 L560,557", kesik=True, etiket="ajanın araçları", ex=750, ey=620)
    ok("M140,652 L140,703", etiket="çağırır", ex=148, ey=685)
    ok("M270,752 L298,752")
    ok("M550,752 L578,752")
    ok("M880,752 L908,752")
    ok("M425,800 L425,818 L260,818 L260,853")
    ok("M730,800 L730,853", etiket="SQL", ex=738, ey=836)
    ok("M262,555 L262,562 L18,562 L18,920 L38,920", kesik=True)   # below the Sohbet box, above the tools row
    ok("M870,920 L842,920", kesik=True)

    return (f'<div class="mimari-ipucu">Diyagramı yana kaydırarak inceleyebilirsin →</div><div class="mimari"><svg viewBox="0 0 {W} {H}" width="100%" role="img" '
            'aria-label="RepoRadarAI sohbet botu mimarisi: tarayıcı, Vercel, Tailscale, geçit, n8n sohbet ve arama akışları, '
            'PostgreSQL ve Ollama" style="max-width:1200px;display:block;margin:0.4rem 0 1.2rem">'
            '<defs><marker id="ok" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
            '<path d="M0,0 L10,5 L0,10 z" fill="#8a938d"/></marker>'
            '<style>.e{fill:#0e7c86;font:800 12px Instrument Sans,Segoe UI,sans-serif;letter-spacing:.12em}'
            '.u{font:800 11px Instrument Sans,Segoe UI,sans-serif;letter-spacing:.1em}'
            '.b{fill:#1b2420;font:700 15px Instrument Sans,Segoe UI,sans-serif}'
            '.s{fill:#3f4a44;font:400 12.5px Instrument Sans,Segoe UI,sans-serif}'
            '.k{fill:#4d5852;font:italic 400 11.5px Instrument Sans,Segoe UI,sans-serif}</style></defs>'
            + "".join(p) + "</svg></div>")


###############################################################
# 03 · 05 · 06 · architecture diagrams, same visual language as sohbet_mimarisi (07)
###############################################################
# Every box is a real component, read from the code on 14.09:
#   03  n8n/DS21tarihce0001.json (schedule 15 min, Commit Iste / Push Iste) · DS21meta0001.json (weekly,
#       GET /repos/{repo}) · DS21readme0001.json (monthly, raw README main/master/readme.md, Kayip Kaydet)
#       tarihce.py run_gates (REFUSED: a gate failed. Nothing was written.) · refresh_csv.py (13 columns, 32 kept)
#   05  DS21_Final_v2.py STEP 12.1 candidates (silence ref · LightGBM · LightGBM+isotonic · CatBoost),
#       12.1b selection rule, GroupKFold(5) by owner, C_KINDS label events · score_today.py reproduction gate,
#       Sunday cutoff, --write-db UPDATE repos
#   06  DS21_RAG.py build_primary_doc (full_name, description, topics, language, README[:HEAD_CHARS=900]) ·
#       tools/build_body_index.py + govde_yukle.py (README[900:4000], table repo_govde, score = max) ·
#       tools/make_eval_queries.py (GEN_MODEL qwen3:14b, fixed set) · outputs/sunum_olcumler.json rag_esik
def _cizim(W, H, aria, isaret):
    p = []

    def bant(y, h, etiket, sag=None):
        p.append(f'<rect x="10" y="{y}" width="{W - 20}" height="{h}" rx="16" fill="rgba(42,120,214,0.04)" '
                 f'stroke="rgba(42,120,214,0.22)"/><text x="26" y="{y + 19}" class="e">{etiket}</text>')
        if sag:
            p.append(f'<text x="{W - 26}" y="{y + 19}" class="k" text-anchor="end">{sag}</text>')

    def kutu(x, y, w, h, ust, baslik, satirlar, renk):
        p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#ffffff" stroke="{renk}" stroke-width="1.6"/>'
                 f'<text x="{x + 14}" y="{y + 20}" class="u" fill="{renk}">{ust}</text>'
                 f'<text x="{x + 14}" y="{y + 40}" class="b">{baslik}</text>')
        for i, s in enumerate(satirlar):
            p.append(f'<text x="{x + 14}" y="{y + 60 + i * 17}" class="s">{s}</text>')

    def ok(yol, kesik=False, etiket=None, ex=0, ey=0):
        p.append(f'<path d="{yol}" fill="none" stroke="#8a938d" stroke-width="1.8"'
                 f'{" stroke-dasharray=\"6 5\"" if kesik else ""} marker-end="url(#{isaret})"/>')
        if etiket:
            p.append(f'<text x="{ex}" y="{ey}" class="k">{etiket}</text>')

    def bitir():
        return (f'<div class="mimari-ipucu">Diyagramı yana kaydırarak inceleyebilirsin →</div><div class="mimari"><svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{aria}" '
                'style="max-width:1200px;display:block;margin:0.4rem 0 1.2rem">'
                f'<defs><marker id="{isaret}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
                '<path d="M0,0 L10,5 L0,10 z" fill="#8a938d"/></marker>'
                '<style>.e{fill:#0e7c86;font:800 12px Instrument Sans,Segoe UI,sans-serif;letter-spacing:.12em}'
                '.u{font:800 11px Instrument Sans,Segoe UI,sans-serif;letter-spacing:.1em}'
                '.b{fill:#1b2420;font:700 15px Instrument Sans,Segoe UI,sans-serif}'
                '.s{fill:#3f4a44;font:400 12.5px Instrument Sans,Segoe UI,sans-serif}'
                '.k{fill:#4d5852;font:italic 400 11.5px Instrument Sans,Segoe UI,sans-serif}</style></defs>'
                + "".join(p) + "</svg></div>")

    return bant, kutu, ok, bitir


_MAVI, _CAMGOBEGI, _YESIL, _MOR, _TURUNCU, _GRI = "#2a78d6", "#0e7c86", "#0ca30c", "#7c5cc4", "#eb6834", "#8a938d"
_tr0 = lambda n: f"{int(n):,}".replace(",", ".")


def genisletme_mimarisi(G):
    """03 · how the Kaggle snapshot was extended from GitHub."""
    bant, kutu, ok, bitir = _cizim(1200, 740, "Veri setinin genişletilmesi: Kaggle CSV ve GitHub API, n8n çekim akışları, "
                                              "Python kontrolü ve model dosyaları", "ok03")
    bant(10, 125, "KAYNAKLAR")
    kutu(40, 40, 340, 85, "KAGGLE CSV", "github_ai_intelligence_2026",
         [f"{_tr0(G['csv_repo'])} repo · {G['csv_sutun']} kolon", "31 Temmuz 2026 · tek anlık görüntü"], _MAVI)
    kutu(430, 40, 340, 85, "GITHUB REST API", "api.github.com", ["/stats/contributors · /activity", "/repos/{repo} → metadata"], _MAVI)
    kutu(820, 40, 340, 85, "GITHUB RAW", "raw.githubusercontent.com", ["README.md · main · master · readme.md"], _MAVI)

    bant(150, 150, "n8n · ÇEKİM")
    kutu(40, 180, 310, 105, "DS21meta0001 · HAFTADA BİR", "Metadata", ["Güncel repo bilgileri", "Default branch adı"], _YESIL)
    kutu(380, 180, 440, 105, "DS21tarihce0001 · HAFTADA BİR", "Commit geçmişi + repo etkinliği",
         ["Her repo için durum kaydı · hata → yeniden dene", "ETag: değişmeyen cevap 304, kotadan düşmez",
          "Önce her repoya bir kez, sonra derinleş"], _YESIL)
    kutu(850, 180, 310, 105, "DS21readme0001 · AYDA BİR", "README", ["Üç adresi sırayla dener", "Bulamadığını kayıp listesine yazar"], _YESIL)

    bant(315, 130, "PYTHON · KONTROL")
    kutu(40, 345, 530, 85, "REFRESH_CSV.PY", "CSV'yi yerinde tazeler",
         ["Metadata'dan 13 kolon güncellenir", "Kalan 32 kolonun değişmediği kontrol edilir"], _CAMGOBEGI)
    kutu(600, 345, 560, 85, "TARIHCE.PY", "Ham kayıt → model dosyaları",
         ["Kapılar: tekrar olay yok · sessizce kaybolan repo yok · tahmin günü", "modelle aynı. Biri geçmezse hiçbir şey yazılmaz."],
         _CAMGOBEGI)

    bant(460, 150, "ÇIKTILAR")
    kutu(40, 490, 265, 100, "DATA · CSV", f"{_tr0(G['metadata_eslesen'])} repo eşleşti", ["Tazelenmiş metadata"], _TURUNCU)
    kutu(325, 490, 265, 100, "COMMIT_WEEKS.PARQUET", f"{_tr0(G['commit_toplam'])} commit",
         [f"{_tr0(G['commit_repo'])} repo · hafta hafta", "Yalnız default branch"], _TURUNCU)
    kutu(610, 490, 265, 100, "PUSH_EVENTS.PARQUET", f"{_tr0(G['etkinlik_olay'])} olay",
         [f"{_tr0(G['etkinlik_repo'])} repo · olay olay", "Hangi dala, kim tarafından"], _TURUNCU)
    kutu(895, 490, 265, 100, "DATA · README_RAW", f"{_tr0(G['readme_repo'])} README", ["Modelde yok", "Yalnız aramada"], _TURUNCU)

    bant(625, 105, "NEREYE GİDİYOR")
    kutu(40, 655, 835, 58, "→ 4 · ÖZELLİK MÜHENDİSLİĞİ", "metadata · commit geçmişi · repo etkinliği", [], _MAVI)
    kutu(895, 655, 265, 58, "→ 6 · RAG", "README → arama vektörleri", [], _MOR)

    ok("M520,125 L520,142 L195,142 L195,178")
    ok("M680,125 L680,178")
    ok("M1005,125 L1005,178")
    ok("M40,100 L17,100 L17,387 L38,387", kesik=True)
    ok("M260,285 L260,343")
    ok("M600,285 L600,302 L880,302 L880,343")
    ok("M1160,240 L1176,240 L1176,470 L1027,470 L1027,488")
    ok("M230,430 L230,488")
    ok("M700,430 L700,450 L457,450 L457,488")
    ok("M742,430 L742,488")
    for x in (230, 457, 742):
        ok(f"M{x},590 L{x},653")
    ok("M1027,590 L1027,653")
    return bitir()


def model_mimarisi(n, ozellik_n, blok, T, H, taban, auc, yasak_n):
    """05 · from features to the badge in the database."""
    AY = ("Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık")
    t = pd.Timestamp(T)
    gun = f"{t.day} {AY[t.month - 1]}"
    bant, kutu, ok, bitir = _cizim(1200, 925, "Model zinciri: girdiler, feature ve target, GroupKFold ile model seçimi, "
                                              "çıktılar ve bugünkü skorlama", "ok05")
    bant(10, 125, "GİRDİLER · 3. BÖLÜMDEN")
    kutu(40, 40, 265, 85, "KAGGLE CSV", "Metadata", ["Yaş, açıklama, dil, lisans…", "Sonradan değişen sayılar yok"], _MAVI)
    kutu(325, 40, 265, 85, "COMMIT_WEEKS", "Commit geçmişi", ["Hafta hafta, default branch"], _MAVI)
    kutu(610, 40, 265, 85, "PUSH_EVENTS", "Repo etkinliği", ["Olay olay: push, PR merge…"], _MAVI)
    kutu(895, 40, 265, 85, "METADATA", "Default branch adı", ["Target'ı kurmak için"], _MAVI)

    bant(150, 165, "FEATURE · TARGET", sag=f"DS21_Final_v2.py · {gun} {t.year}")
    kutu(40, 180, 520, 115, "FEATURE", f"{ozellik_n} feature · yalnız tahmin gününden önce",
         [f"{blok['meta']} metadata · {blok['commit']} commit geçmişi · {blok['etkinlik']} repo etkinliği",
          f"Commit: {gun}'tan önce biten haftalar", f"Etkinlik: {gun}'tan önceki 51 hafta"], _MAVI)
    kutu(590, 180, 570, 115, "TARGET", f"Sonraki {H} günde default branch'e değişiklik ulaştı mı?",
         ["Sayılan: push · PR merge · merge queue · force push",
          f"{_tr0(n)} repo · %{taban * 100:.1f}".replace(".", ",") + " güncellendi"], _YESIL)

    bant(330, 110, "FEATURES.PARQUET")
    kutu(40, 355, 1120, 78, "LEAKAGE KAPILARI · BİRİ GEÇMEZSE EĞİTİM DURUR", f"{_tr0(n)} repo × {ozellik_n} feature + target",
         [f"{yasak_n} yasaklı kolon modele giremez  ·  feature tarihi tahmin gününü geçemez  ·  "
          "hiçbir feature yasaklı bir kolonla 0,85'ten fazla ilişkili değil"], _CAMGOBEGI)

    bant(455, 150, "MODEL SEÇİMİ", sag="GroupKFold(5) · aynı owner'ın repoları hep aynı katta")
    kutu(40, 485, 360, 105, "ADAYLAR", "Aynı 5 kat, aynı feature'lar",
         ["Sessizlik tek başına (referans)", "LightGBM · LightGBM + isotonic", "CatBoost"], _MOR)
    kutu(430, 485, 400, 105, "KURAL · SONUÇLARDAN ÖNCE YAZILDI", "En iyi AUC, sonra kalibrasyon",
         ["Bir fold-std içindekiler berabere sayılır", "Berabere olanlardan kalibrasyonu en iyisi",
          "Hâlâ berabereyse en basit model"], _MOR)
    kutu(860, 485, 300, 105, "KAZANAN", f"CatBoost · AUC {auc:.4f}".replace(".", ","),
         ["500 ağaç · derinlik 6 · tohum 42", "Ayar araması yok"], _YESIL)

    bant(620, 125, "ÇIKTILAR")
    kutu(40, 650, 353, 80, "MODEL.PKL", "Model + sözleşme", ["Feature listesi · yasaklı kolonlar · T · H"], _TURUNCU)
    kutu(423, 650, 353, 80, "REPO_SCORES.PARQUET", "Görmediği katta verilen skorlar", ["Paneldeki bütün metrikler bunlardan"], _TURUNCU)
    kutu(806, 650, 354, 80, "MODEL_RAPOR.JSON", "Rapor", ["Kalibrasyon · dilimler · rozetler"], _TURUNCU)

    bant(760, 155, "CANLI", sag="score_today.py · eğitim yok, aynı model")
    kutu(40, 790, 480, 105, "1 · KAPI", "Kopya kod aslıyla aynı mı?",
         ["Tahmin günüyle features.parquet'i birebir yeniden üretir", "Üretemezse durur"], _CAMGOBEGI)
    kutu(550, 790, 300, 105, "2 · BUGÜN", "Aynı model, bugünün verisi", ["Kesim: verinin bittiği son Pazar"], _CAMGOBEGI)
    kutu(880, 790, 280, 105, "3 · --WRITE-DB", "PostgreSQL · repos", ["Bugünkü olasılık ve rozet", "Kırmızı &lt;%30 · sarı · yeşil ≥%60"],
         _TURUNCU)

    ok("M250,125 L250,178")
    ok("M457,125 L457,178")
    ok("M742,125 L742,178")
    ok("M742,140 L520,140 L520,178", kesik=True)
    ok("M950,125 L950,178")
    ok("M300,295 L300,353")
    ok("M875,295 L875,353")
    ok("M300,433 L300,483")
    ok("M400,537 L428,537")
    ok("M830,537 L858,537")
    ok("M1010,590 L1010,648")
    ok("M1010,608 L600,608 L600,648")
    ok("M600,608 L300,608 L300,648")
    ok("M300,730 L300,788")
    ok("M520,842 L548,842")
    ok("M850,842 L878,842")
    return bitir()


def rag_mimarisi(repo_n, readme_n, govde_n, bas_karakter, govde_bitis, vektor_boyut, esik, soru_n, negatif_n, negatif_kesilen,
                 r5_ilk, r5_son, uretilen=None):
    """06 · how the search index is built and how its choices were measured."""
    O2 = lambda x, nd: f"{x:.{nd}f}".replace(".", ",")
    bant, kutu, ok, bitir = _cizim(1200, 895, "RAG: metin, belge ve vektör üretimi, pgvector tabloları, sorgu anı ve "
                                              "seçimlerin sınavı", "ok06")
    bant(10, 125, "METİN")
    kutu(40, 40, 540, 85, "KAGGLE CSV", "Kimlik ve açıklama", ["Repo adı · açıklama · topics · dil"], _MAVI)
    kutu(620, 40, 540, 85, "DATA · README_RAW", f"{_tr0(readme_n)} README", ["Web sayfası için yazılmış metin temizlenir"], _MAVI)

    bant(150, 150, "PYTHON · BELGELER")
    kutu(40, 180, 540, 105, "DS21_RAG.PY · README HEAD", "Her repo için bir belge",
         ["Ad + açıklama + topics + dil", f"+ README'nin ilk {_tr0(bas_karakter)} karakteri", f"{_tr0(repo_n)} repo"], _MOR)
    kutu(620, 180, 540, 105, "BUILD_BODY_INDEX.PY · README BODY", f"README'nin {_tr0(bas_karakter)}-{_tr0(govde_bitis)} arası",
         [f"README'si {_tr0(bas_karakter)} karakteri aşan repolar", f"{_tr0(govde_n)} repo"], _MOR)

    bant(315, 110, "OLLAMA · BU BİLGİSAYARDA")
    kutu(40, 345, 1120, 62, "EMBEDDING", f"qwen3-embedding:8b → her belge {_tr0(vektor_boyut)} sayılık bir vektör", [], _TURUNCU)

    bant(440, 130, "PGVECTOR", sag="PostgreSQL · repoquest veritabanı")
    kutu(40, 470, 540, 85, "REPOS", f"{_tr0(repo_n)} satır", ["Head vektörü · bugünkü rozet · yıldız · dil"], _TURUNCU)
    kutu(620, 470, 540, 85, "REPO_GOVDE · GOVDE_YUKLE.PY", f"{_tr0(govde_n)} satır", ["Body vektörü · ayrı tablo"], _TURUNCU)

    bant(585, 140, "SORGU ANINDA", sag="ayrıntısı 7. bölümde")
    kutu(40, 615, 260, 90, "1", "Soru → vektör", ["Ajan aramayı İngilizce yazar"], _CAMGOBEGI)
    kutu(326, 615, 260, 90, "2", "Benzerlik", ["max(head, body)", "Ortalama değil"], _CAMGOBEGI)
    kutu(612, 615, 260, 90, "3", f"Eşik {O2(esik, 2)}", ["Altındaysa “uygun repo yok”"], _CAMGOBEGI)
    kutu(898, 615, 262, 90, "4", "İlk 10 repo", ["En aktiften başlayarak"], _CAMGOBEGI)

    bant(740, 145, "SINAV · SEÇİMLERİN DAYANAĞI")
    uretilen = uretilen or soru_n
    kutu(40, 770, 360, 95, "MAKE_EVAL_QUERIES.PY", f"{_tr0(uretilen)} soru üretildi",
         ([f"{uretilen - soru_n} elendi: repo adını ele veren ya da boş"] if uretilen > soru_n else [])
         + [f"Sınavda {_tr0(soru_n)} soru · qwen3:14b yazdı"], _YESIL)
    kutu(430, 770, 360, 95, "MODEL SEÇİMİ", "recall@5: doğru repo ilk 5'te mi?",
         [f"bge-m3 {O2(r5_ilk, 3)} → Qwen3 8B + body {O2(r5_son, 3)}"], _YESIL)
    kutu(820, 770, 340, 95, "EŞİK SEÇİMİ", f"{negatif_n} konu dışı soru",
         [f"{O2(esik, 2)}: {negatif_kesilen}'ini keser", "Doğru cevabı silmez"], _YESIL)

    ok("M310,125 L310,178")
    ok("M890,125 L890,178")
    ok("M890,140 L500,140 L500,178")
    ok("M310,285 L310,343")
    ok("M890,285 L890,343")
    ok("M310,407 L310,468")
    ok("M890,407 L890,468")
    ok("M310,555 L310,572 L456,572 L456,613")
    ok("M890,555 L890,572 L456,572 L456,613")
    ok("M300,660 L324,660")
    ok("M586,660 L610,660")
    ok("M872,660 L896,660")
    ok("M400,817 L428,817")
    ok("M790,817 L818,817")
    ok("M850,768 L850,707", kesik=True, etiket="eşiği seçti", ex=858, ey=742)
    return bitir()

