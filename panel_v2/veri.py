# -*- coding: utf-8 -*-
###############################################################
# veri.py | panel_v2 - every number the panel shows comes from here
###############################################################
# RULE (kept from panel/): no number is typed into a page. Numbers read from
# an artifact at render time cannot go stale.
#
# ONE EXCEPTION, made explicit: OLCUMLER holds a handful of measurements the
# pipeline printed once but never wrote to a file the panel can read (e.g. the
# paired McNemar test). Each carries its source, so a reader can re-run it.
#
# READ-ONLY. Nothing here writes to disk or to the database.
###############################################################

import json
import pickle
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent          # panel_v2/ -> project root
OUT = ROOT / "outputs"
CSV = ROOT / "Data" / "github_ai_intelligence_2026.csv"
# The ORIGINAL Kaggle file (31.07 snapshot, sha256 87AE9EFF...). Data/ holds the September refresh:
# 13 columns re-fetched, so stars moved and the 100-star sampling cliff is diluted there (89 -> 205 repos at 100-104).
# The sampling limit (08) describes how the pool was drawn, so it reads this file.
CSV_ORIJINAL = ROOT / "snapshots" / "2026-07-31" / "github_ai_intelligence_2026.csv"
FIG_DIRS = [ROOT / "figures" / "panel", ROOT / "figures"]

# Measured once, not stored in a readable artifact. (value, source)
OLCUMLER = {
    "hafta_siniri_repo": (2740, "DS21_Final_v2.py AUDIT 2026-09-10 (:727)"),
    "hafta_siniri_gun": (4, "DS21_Final_v2.py AUDIT 2026-09-10 (:727)"),
    "owner_rastgele_auc": (0.8265, "figures/07_owner_grouping.png, koşum 2026-09-14 15:25 (STEP 9, yalnız metadata)"),
    "owner_grup_auc": (0.8170, "figures/07_owner_grouping.png, koşum 2026-09-14 15:25 (STEP 9, yalnız metadata)"),
    "readme_kat": (192, "DS21_Final_v2.py STEP 11c.5"),
    "bayrak_bedeli": (0.0010, "DS21_Final_v2.py AUDIT 2026-09-11 (:2382)"),
    "mcnemar_p": (0.0074, "tools/rag_sinav.py, 2026-09-11 · uyuşmayan çift 13'e 2"),
    "etiket_uyum_pushed": (None, "label_compare.parquet"),
    # descriptive metadata (description, topics, uses_*, ai_category) was frozen before the September refresh
    "metadata_dondurma": ("2026-07-31", "DS21_Final_v2.py:3215 METADATA_DATE"),
    "metadata_degisen_pay": (0.02, "DS21_Final_v2.py:3246-3247 model kartı · ~%2 · 38 gün"),
    "metadata_degisim_gun": (38, "DS21_Final_v2.py:3246-3247 model kartı"),
    "vektor_boyut": (4096, "qwen3-embedding:8b · outputs/embeddings_qwen8b.npy (24.664, 4096)"),
    "bas_karakter": (900, "DS21_RAG.py:150 HEAD_CHARS · tools/govde_yukle.py:57"),
    "govde_bitis": (4000, "tools/build_body_index.py:3 README[900:4000] · DS21_RAG.py:786 QUERY_SOURCE_CHARS"),
    "govde_repo": (21816, "outputs/embeddings_qwen8b_body.npy (21.816, 4096) · tools/govde_yukle.py:12"),
    "dal_sessizlik_auc": (0.9016, "sunum/kosum/logs/prob_sessizlik_sizinti.log:6 (varsayılan dala son olay sessizliği, tek başına)"),
    "kopya_esik": (0.85, "DS21_Final_v2.py STEP 11c.2 · logs/AB/v2_full.log:937-939 (hiçbir özellik aşmıyor)"),
    "main_master_disi_repo": (37, "DS21_Final_v2.py:2075-2079 (etiket çalışmasında develop/dev/ts kullanan repo)"),
    # B32 · live observation, 13.09: an ASCII-typed Turkish question was translated off-topic by the agent
    "ascii_soru": ("yun atki orgusunu ogreten bir uygulama", "sunum/kosum/DS_BULGULAR.md B32 · n8n çalışma 62962"),
    "ascii_sorgu": ("teaching newbies how to code", "n8n çalışma 62962 (repo_ara girdisi)"),
    "ascii_benzerlik": (0.66, "n8n çalışma 62963 (en yüksek benzerlik 0,6617)"),
    "turkce_sorgu": ("knitting tutorial for beginners", "n8n çalışma 62966 · en yüksek benzerlik 0,4958 → kesildi"),
}


@st.cache_data(show_spinner=False)
def yildiz_ucurumu():
    """Repos per one-star unit just below and just above 100 stars (quota-sampled pool, v2 STEP 5.4).
    Original snapshot: the cliff is a trace of how the pool was drawn; the refreshed CSV blurs it (B35 correction)."""
    s = csv_orijinal()["stars"]
    alt = int(s.between(95, 99).sum()) / 5
    ust = int(s.between(100, 104).sum()) / 5
    return {"alt": alt, "ust": ust, "kat": alt / ust if ust else None}


def fig(ad):
    """Dark panel figure first, the chain's own figure as fallback."""
    for d in FIG_DIRS:
        p = d / ad
        if p.exists():
            return p
    return None


# ---------- Turkish number formatting ----------
def sayi(x):
    return f"{int(round(x)):,}".replace(",", ".")


def ondalik(x, nd=4):
    return f"{x:.{nd}f}".replace(".", ",")


def yuzde(x, nd=1):
    return "%" + f"{x * 100:.{nd}f}".replace(".", ",")


def rozet_kodu(etiket):
    """'GREEN >=0.60' -> 'GREEN'."""
    return str(etiket).split()[0]


# ---------- loaders ----------
@st.cache_data(show_spinner=False)
def csv():
    return pd.read_csv(CSV)


@st.cache_data(show_spinner=False)
def sinav_uretimi(kalan):
    """Generated vs kept evaluation questions. tools/make_eval_queries.py samples (N_EN + N_TR) per star band and
    drops questions that leak the repo name or come out empty; the drop log was not kept, so both reasons are named."""
    import ast as _ast
    agac = _ast.parse((ROOT / "tools" / "make_eval_queries.py").read_text(encoding="utf-8"))
    sabit = {}
    for dugum in agac.body:
        if isinstance(dugum, _ast.Assign):
            for hedef in dugum.targets:
                if isinstance(hedef, _ast.Tuple) and isinstance(dugum.value, _ast.Tuple):
                    for h, v in zip(hedef.elts, dugum.value.elts):
                        sabit[h.id] = _ast.literal_eval(v)
                elif isinstance(hedef, _ast.Name) and hedef.id == "BANTLAR":
                    sabit["BANTLAR"] = dugum.value.elts
    uretilen = (sabit["N_EN"] + sabit["N_TR"]) * len(sabit["BANTLAR"])
    return {"uretilen": uretilen, "kalan": kalan, "elenen": uretilen - kalan}


@st.cache_data(show_spinner=False)
def csv_orijinal():
    return pd.read_csv(CSV_ORIJINAL)


@st.cache_data(show_spinner=False)
def features():
    return pd.read_parquet(OUT / "features.parquet")


@st.cache_data(show_spinner=False)
def rapor():
    return json.loads((OUT / "model_rapor.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def skorlar_bugun():
    return pd.read_parquet(OUT / "scores_today.parquet")


@st.cache_data(show_spinner=False)
def oof_skorlar():
    return pd.read_parquet(OUT / "repo_scores.parquet")


@st.cache_data(show_spinner=False)
def sozlesme():
    """The contract model.pkl carries - without the model object itself."""
    with open(OUT / "model.pkl", "rb") as f:
        m = pickle.load(f)
    return {"yasakli": list(m["banned_columns"]), "ozellik": list(m["feature_cols"]),
            "etiket": m.get("label_source")}


@st.cache_data(show_spinner=False)
def readme_sayisi():
    try:
        d = pd.read_parquet(OUT / "rag_documents_qwen8b.parquet", columns=["has_readme"])
        return int(d["has_readme"].sum())
    except Exception:
        return None


# ---------- derived facts ----------
@st.cache_data(show_spinner=False)
def veri_ozeti():
    d = csv()
    f = features()
    oc = f["owner"].value_counts()
    return {
        "repo": len(d),
        "sutun": d.shape[1],
        "owner_tum": d["owner"].nunique(),
        "yildiz_10_alti": int((d["stars"] < 10).sum()),
        "yildiz_medyan": float(d["stars"].median()),
        "cerceve": len(f),
        "owner_cerceve": len(oc),
        "owner_coklu": int((oc > 1).sum()),
        "owner_en_buyuk": (oc.index[0], int(oc.iloc[0])),
        "commit_hafta": pq.ParquetFile(OUT / "commit_weeks.parquet").metadata.num_rows,
        "push_olay": pq.ParquetFile(OUT / "push_events.parquet").metadata.num_rows,
        "readme": readme_sayisi(),
        "ozellik": [c for c in f.columns if c not in ("full_name", "owner", "target")],
    }


@st.cache_data(show_spinner=False)
def model_ozeti():
    r = rapor()
    s = oof_skorlar()
    y, p = s["target"], s["p_active_180d"]
    abl = {a["block"]: a for a in r["ablation"]}
    return {
        "auc": r["calibration"]["auc_shipped"],
        "auc_kontrol": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "kalib": r["calibration"]["shipped"],
        "kalib_lgbm": r["calibration"]["lgbm_uncalibrated"],
        "taban": r["base_rate"],
        "n": r["n_rows"],
        "ozellik_n": r["n_features"],
        "T": r["cutoff_T"],
        "H": r["horizon_days"],
        "run_id": r["run_id"],
        "ablasyon": abl,
        "readme_on_commit": r["readme_on_commit"],
        "dilimler": r["slices"],
        "rozetler": r["badges"],
        "onsel": r["badge_moves_if_base_rate"],
    }


def blok_sayilari():
    ad = veri_ozeti()["ozellik"]
    cm = [c for c in ad if c.startswith("cm_")]
    pa = [c for c in ad if c.startswith("pa_")]
    return {"meta": len(ad) - len(cm) - len(pa), "commit": len(cm), "etkinlik": len(pa)}


@st.cache_data(show_spinner=False)
def zaman():
    r = rapor()
    s = skorlar_bugun()
    T = pd.Timestamp(r["cutoff_T"])
    return {"T": T, "H": r["horizon_days"], "T_son": T + pd.Timedelta(days=r["horizon_days"]),
            "bugun": pd.Timestamp(s["cutoff"].max()), "pencere_sonu": pd.Timestamp(s["window_end"].max())}


@st.cache_data(show_spinner=False)
def rozet_dagilimi_bugun():
    return skorlar_bugun()["badge_today"].value_counts()


@st.cache_data(show_spinner=False)
def kolay_ve_zor_gruplar():
    """cap_audit rows: the all-negative 'no activity' group and the revived group."""
    g = {}
    for row in rapor()["cap_audit"]:
        if "(yok)" in row["group"]:
            g["sessiz"] = row
        elif "starts after T" in row["group"]:
            g["dirilen"] = row
    return g


@st.cache_data(show_spinner=False)
def kalibrasyon_egrisi(kova=10):
    s = oof_skorlar()
    q = pd.qcut(s["p_active_180d"].rank(method="first"), kova, labels=False)
    return (s.groupby(q).agg(tahmin=("p_active_180d", "mean"), gercek=("target", "mean"), n=("target", "size"))
            .reset_index(drop=True))


@st.cache_data(show_spinner=False)
def rozet_gecisi():
    """Badge in the training window (OOF, at T) -> badge today. VERI_YOK excluded."""
    eski = oof_skorlar()[["full_name", "badge"]].assign(eski=lambda d: d["badge"].map(rozet_kodu))
    yeni = skorlar_bugun()[["full_name", "badge_today"]]
    j = eski.merge(yeni, on="full_name")
    j = j[j["badge_today"] != "VERI_YOK"]
    sira = ["RED", "AMBER", "GREEN"]
    ct = pd.crosstab(j["eski"], j["badge_today"]).reindex(index=sira, columns=sira, fill_value=0)
    return ct


@st.cache_data(show_spinner=False)
def havuz_hunisi():
    d = csv()
    a = d[d["maintenance_status"].isin(["Active", "Recent"])]
    b = a[a["stars"] >= 100]
    c = b[~b["archived"].astype(str).str.lower().isin(["true", "1", "yes"])]
    e = c[~c["license"].isin(["No License", "NOASSERTION"])]
    return [("Külliyatın tamamı", len(d)), ("Bakımı sürüyor (bugün)", len(a)),
            ("≥ 100 yıldız", len(b)), ("Arşivlenmemiş", len(c)), ("Lisansı net", len(e))]


@st.cache_data(show_spinner=False)
def etiket_uyumu():
    d = pd.read_parquet(OUT / "label_compare.parquet")
    g = d[d["commit"].notna()]
    ct = pd.crosstab(g["commit"].astype(int), g["etiket_c"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    return {"tablo": ct, "uyum": float((g["commit"].astype(int) == g["etiket_c"]).mean()), "n": len(g)}


@st.cache_data(show_spinner=False)
def rag_denemeleri():
    return pd.read_csv(OUT / "rag_sweep.csv").groupby("etiket").tail(1).set_index("etiket")


# ---------- presentation measurements (tools/sunum_olcumler.py) ----------
# One file, stamped with the model_rapor run_id. A file from another model run
# is refused rather than shown: its numbers belong to a different model.
@st.cache_data(show_spinner=False)
def sunum_olcumler():
    yol = OUT / "sunum_olcumler.json"
    if not yol.exists():
        return None, "outputs/sunum_olcumler.json yok · çalıştır: python tools/sunum_olcumler.py"
    j = json.loads(yol.read_text(encoding="utf-8"))
    if j.get("run_id") != rapor()["run_id"]:
        return None, (f"sunum_olcumler.json başka bir model koşumuna ait ({j.get('run_id')} ≠ {rapor()['run_id']}) · "
                      "yeniden çalıştır: python tools/sunum_olcumler.py")
    return j, None
