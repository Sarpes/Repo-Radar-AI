# -*- coding: utf-8 -*-
###############################################################
# STEP 17 | CANLI SKORLAMA - "bugunden itibaren H gun"
###############################################################
# NE SORUNUN CEVABI DEGISIYOR
#   repo_scores.parquet  ->  P(degisiklik, (T, T+H])   KAPANMIS, egitim penceresi
#   bu script            ->  P(degisiklik, [BUGUN, BUGUN+H])   ileriye bakan tek hal
# T, H ve pa_* penceresi model.pkl sozlesmesinden okunur; burada sabit yok.
#
# EGITIM YOK. Ayni model.pkl, farkli girdi (kesme = bugun).
# SIZINTI YOK - yapisal olarak: [bugun, bugun+H] henuz olmadi.
#
# 🚪 YENIDEN URETIM KAPISI - bu scriptin en onemli parcasi
# Ozellik ureticileri DS21_Final_v2.py'den KOPYALANDI. Kopya ile orijinal
# arasinda sessiz bir fark olursa tahminler cop olur ve HICBIR HATA VERMEZ.
# Bu yuzden script once cutoff = CUTOFF_T ile calisir ve features.parquet'i
# BIREBIR yeniden uretip uretmedigini olcer. Uretemezse durur.
#
# KULLANIM
#   python score_today.py              kapi + bugunun skorlari, rapor
#   python score_today.py --write-db   pgvector'daki repos tablosunu da gunceller
###############################################################

import argparse
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(ROOT, "Data", "github_ai_intelligence_2026.csv")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.pkl")
COMMIT_PATH = os.path.join(OUTPUT_DIR, "commit_weeks.parquet")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "features.parquet")
FETCH_LOG_PATH = os.path.join(OUTPUT_DIR, "fetch_log.jsonl")
# AUDIT 2026-09-08 - must mirror DS21_Final_v2.py exactly. HTTP 204 means the
# repo has no commits (real signal); the rest are our own collection failures.
EMPTY_STATUS = {"error:204", "empty"}
FAIL_STATUS = {"not_found", "timeout_202", "error:ConnectionError",
               "error:fetch"}   # error:fetch = tarihce.py, our fetch failed
OUT_PATH = os.path.join(OUTPUT_DIR, "scores_today.parquet")

PG = dict(host="localhost", port=5433, dbname="repoquest",
          user="ds21", password=os.environ.get("DS21_PG_PASSWORD", ""))

ASOF_CATEGORICAL = ["language", "license", "ai_category"]


def header(t):
    print("\n" + "#" * 21 + f" {t} " + "#" * 21)


# ------------------------------------------------------------------
# DS21_Final_v2.py:199-225 ile BIREBIR AYNI olmali. Kapi bunu kanitlar.
# ------------------------------------------------------------------
def build_asof_features(frame, cutoff, flags):
    created_local = pd.to_datetime(frame["created_at"], format="%d/%m/%y")
    X = pd.DataFrame(index=frame.index)
    X["age_at_T"] = (cutoff - created_local).dt.days
    X["topics_n"] = frame["topics"].fillna("").str.split().str.len()
    X["desc_len"] = frame["description"].fillna("").str.len()
    X["name_len"] = frame["name"].str.len()
    X["has_homepage"] = frame["homepage"].notna().astype(int)
    # owner_n, VERILEN dilim icinde sayar. Egitimde dilim = T'de var olanlar;
    # cikarimda = bugun var olan her sey. Tanim ayni, kapsam farkli.
    X["owner_n"] = frame["owner"].map(frame["owner"].value_counts())
    for c in flags:
        X[c] = frame[c].values
    for c in ASOF_CATEGORICAL:
        X[c] = frame[c].astype("category")
    return X


# ------------------------------------------------------------------
# DS21_Final_v2.py:1538-1618 ile BIREBIR AYNI olmali.
# ------------------------------------------------------------------
def build_commit_features(cutoff):
    cw = pd.read_parquet(COMMIT_PATH)
    cw["full_name"] = cw["full_name"].astype(str)
    cw = cw[cw["week"] < cutoff].sort_values(["full_name", "week"])

    g = cw.groupby("full_name", sort=False)
    F = pd.DataFrame(index=g.size().index)

    F["cm_active_weeks"] = g.size()
    F["cm_total"] = g["commits"].sum()
    F["cm_mean_per_active_week"] = F["cm_total"] / F["cm_active_weeks"]
    F["cm_peak_week"] = g["commits"].max()

    for w in (4, 13, 26, 52):
        lo = cutoff - pd.Timedelta(weeks=w)
        F[f"cm_{w}w"] = (cw[cw["week"] >= lo].groupby("full_name")["commits"]
                         .sum().reindex(F.index).fillna(0).astype(int))

    last_week = g["week"].max()
    first_week = g["week"].min()
    F["cm_silence_weeks"] = ((cutoff - last_week).dt.days // 7).astype(int)
    F["cm_span_weeks"] = ((last_week - first_week).dt.days // 7).astype(int)
    F["cm_weeks_since_first"] = ((cutoff - first_week).dt.days // 7).astype(int)

    gap_weeks = (cw.groupby("full_name")["week"].diff().dt.days // 7) - 1
    gap_weeks = gap_weeks.clip(lower=0)
    gg = gap_weeks.groupby(cw["full_name"])
    F["cm_max_gap"] = gg.max().reindex(F.index).fillna(0)
    F["cm_mean_gap"] = gg.mean().reindex(F.index).fillna(0)

    F["cm_accel_4w"] = F["cm_4w"] / (F["cm_13w"] - F["cm_4w"] + 1)
    F["cm_recent_share"] = F["cm_13w"] / F["cm_total"].clip(lower=1)

    F["cm_contrib_max"] = g["contributors"].max()
    F["cm_contrib_mean"] = g["contributors"].mean()
    F["cm_solo_week_share"] = (cw["contributors"] == 1).groupby(cw["full_name"]).mean()

    peak_idx = g["commits"].idxmax()
    peak_week = cw.loc[peak_idx, "week"].values
    F["cm_weeks_since_peak"] = ((cutoff - pd.Series(peak_week, index=F.index))
                                .dt.days // 7).astype(int)

    assert cw["week"].max() < cutoff, "commit ozellikleri cutoff'u gecti"
    return F


def build_fetch_flags(names):
    """Split the commit-block NaN by CAUSE. Mirrors DS21_Final_v2.py."""
    import json
    n = len(names)
    if not os.path.exists(FETCH_LOG_PATH):
        print(f"UYARI: {FETCH_LOG_PATH} yok -> fetch bayraklari sifir")
        return pd.DataFrame({"cm_empty_repo": [0] * n, "cm_fetch_failed": [0] * n})
    durum = {}
    with open(FETCH_LOG_PATH, encoding="utf-8") as f:
        for satir in f:
            try:
                r = json.loads(satir)
            except ValueError:
                continue
            if r.get("full_name"):
                durum[r["full_name"]] = r.get("status")
    t = pd.Series(list(names)).map(durum)
    return pd.DataFrame({"cm_empty_repo": t.isin(EMPTY_STATUS).astype("int8").values,
                         "cm_fetch_failed": t.isin(FAIL_STATUS).astype("int8").values})


# ------------------------------------------------------------------
# pa_* | BILEREK IKINCI BIR UYGULAMA (K4)
# Bu fonksiyonlar DS21_Final_v2.py'dekilerin KOPYASI DEGIL. KAPI'nin butun
# gucu iki BAGIMSIZ uygulamanin ayni sayiyi uretmesinden geliyor; kopyalasaydik
# ayni hatayi iki kez yapar ve kapi hicbir sey yakalamazdi. 2026-09-08'de bu
# ayrim 27 satirlik gercek bir kaymayi yakalamisti.
# Ayni sonuc, FARKLI pandas yollari: value_counts(...) vs groupby().size(),
# assign+groupby ortalama vs boolean-seri gruplama.
PUSH_PATH = os.path.join(OUTPUT_DIR, "push_events.parquet")
DEFAULT_REFS = {"refs/heads/main", "refs/heads/master"}
CHURN_KINDS = {"branch_creation", "branch_deletion"}


def build_push_features(cutoff, weeks):
    """weeks = model.pkl["pa_window_weeks"]: pa_* sabit pencerede (DS_BULGULAR B2)."""
    if not os.path.exists(PUSH_PATH):
        return None
    pe = pd.read_parquet(PUSH_PATH)
    pe = pe.loc[pe["ts"].between(cutoff - pd.DateOffset(weeks=weeks), cutoff, inclusive="left")]
    if pe.empty:
        return None

    # farkli yol: value_counts(cok sutunlu) -> seviye-0 toplami / maksimumu
    cift = pe.value_counts(["full_name", "actor"])
    toplam = cift.groupby(level=0).sum()
    F = pd.DataFrame(index=toplam.index)
    F["pa_events"] = toplam
    F["pa_actors_distinct"] = cift.groupby(level=0).size()
    F["pa_actor_top_share"] = cift.groupby(level=0).max() / toplam

    # farkli yol: once sutun olarak ekle, sonra o sutunun ortalamasini al
    yardim = pe.assign(
        _def=pe["ref"].isin(DEFAULT_REFS),
        _pr=pe["kind"] == "pr_merge",
        _fp=pe["kind"] == "force_push",
        _ch=pe["kind"].isin(CHURN_KINDS),
    )
    gy = yardim.groupby("full_name")
    F["pa_default_ref_share"] = gy["_def"].mean()
    F["pa_refs_distinct"] = gy["ref"].nunique()
    F["pa_pr_merge_share"] = gy["_pr"].mean()
    F["pa_force_push_share"] = gy["_fp"].mean()
    F["pa_branch_churn"] = gy["_ch"].mean()

    assert pe["ts"].max() < cutoff, "push ozellikleri cutoff'u gecti"
    return F


def build_X(df, cutoff, flags, feat_cols, pa_weeks):
    """Modelin istedigi cerceve: metadata + commit + pa_*. Sutun kumesini model.pkl
    belirler; README ve cekim-durumu bayraklari modelde yok, uretilmez."""
    X = build_asof_features(df, cutoff, flags)

    cm = build_commit_features(cutoff)
    X = X.join(cm.reindex(df["full_name"].values).set_index(X.index), how="left")

    pa = build_push_features(cutoff, pa_weeks)
    if pa is not None:
        X = X.join(pa.reindex(df["full_name"].values).set_index(X.index), how="left")

    eksik = [c for c in feat_cols if c not in X.columns]
    assert not eksik, f"eksik ozellik: {eksik}"
    return X[feat_cols]


def predict(model_blob, X):
    Xc = X.copy()
    for c in model_blob["cat_cols"]:
        Xc[c] = Xc[c].astype(str)
    return model_blob["model"].predict_proba(Xc.fillna(-999))[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-db", action="store_true",
                    help="pgvector repos tablosunu da guncelle")
    args = ap.parse_args()

    header("Girdi")
    m = joblib.load(MODEL_PATH)
    feat = m["feature_cols"]
    CUTOFF_T = pd.Timestamp(m["cutoff_T"])
    H = m["horizon_days"]
    print(f"model.pkl : {m['kind']}")
    print(f"  egitim T = {CUTOFF_T.date()} · ufuk {H} gun · {len(feat)} ozellik")
    print(f"  egitimdeki out-of-fold AUC = {m['oof_auc']:.4f}")

    df = pd.read_csv(CSV_PATH)
    flags = [c for c in df.columns if c.startswith("uses_")]
    print(f"{CSV_PATH}\n  {len(df):,} repo")

    # ---------------------------------------------------------- KAPI
    header("KAPI | ozellik ureticileri egitimdekiyle ayni mi?")
    print("cutoff = CUTOFF_T ile features.parquet birebir yeniden uretilmeli.")
    created = pd.to_datetime(df["created_at"], format="%d/%m/%y")
    # AUDIT 2026-09-08 → docs/AUDIT_DEFTERI.md#score-today-244: features.parquet artik cekimi BASARISIZ olan repolari
    # AUDIT 2026-09-11 → docs/AUDIT_DEFTERI.md#score-today-249: The gate REBUILT the training frame with its own copy of
    _bayrak = build_fetch_flags(df["full_name"].values)
    _gozlenmedi = _bayrak["cm_fetch_failed"].astype(bool).values   # live VERI_YOK, below
    ref = pd.read_parquet(FEATURES_PATH)
    uygun = df[df["full_name"].isin(ref["full_name"])].reset_index(drop=True)
    assert len(uygun) == len(ref) == ref["full_name"].nunique(), (
        f"frame mismatch: CSV matches {len(uygun):,} rows, features.parquet has {len(ref):,}")
    print(f"  egitim cercevesi features.parquet'ten: {len(uygun):,} repo "
          f"(T oncesi olusturulan {int((created < CUTOFF_T).sum()):,})")
    X_kapi = build_X(uygun, CUTOFF_T, flags, feat, m["pa_window_weeks"])

    ref_X = ref.set_index("full_name").reindex(uygun["full_name"])[feat]
    X_kapi.index = uygun["full_name"]

    print(f"  satir: uretilen {len(X_kapi):,}  referans {len(ref_X):,}")
    ok = True
    for c in feat:
        a, b = X_kapi[c], ref_X[c]
        if str(a.dtype) == "category" or a.dtype == object:
            esit = (a.astype(str).values == b.astype(str).values)
        else:
            esit = np.isclose(a.astype(float).values, b.astype(float).values,
                              rtol=1e-9, atol=1e-9, equal_nan=True)
        if not esit.all():
            ok = False
            print(f"  FARK  {c:<24} {int((~esit).sum()):,} satir")
    if not ok:
        raise SystemExit("\nKAPI GECILEMEDI - ozellik ureticileri farkli. "
                         "Bugunun skorlari HESAPLANMADI.")
    print(f"  OK  {len(feat)} ozelligin hepsi birebir ayni")

    # Ayni modeli ayni ozelliklerle calistirinca repo_scores'a benzemeli.
    p_kapi = predict(m, X_kapi.reset_index(drop=True))
    rs = pd.read_parquet(os.path.join(OUTPUT_DIR, "repo_scores.parquet"))
    rs = rs.set_index("full_name").reindex(uygun["full_name"])
    from scipy.stats import spearmanr
    rho = spearmanr(p_kapi, rs["p_active_180d"].values).statistic
    print(f"  refit-model vs out-of-fold skorlar: Spearman {rho:.4f}")
    print("  (1.0 beklenmiyor: repo_scores OUT-OF-FOLD, model.pkl her seyle yeniden egitildi)")

    # ------------------------------------------------------- BUGUN
    header(f"BUGUN | P(degisiklik, [bugun, bugun+{H}])")
    # AUDIT 2026-09-10 → docs/AUDIT_DEFTERI.md#score-today-298: canli cutoff da PAZAR'a hizalanmali.
    _bugun_ham = pd.Timestamp.today().normalize()
    BUGUN = _bugun_ham - pd.Timedelta(days=(_bugun_ham.dayofweek + 1) % 7)
    # AUDIT 2026-09-13 | cutoff, verinin BITTIGI gunun Pazar'ini da gecemez.
    # Cekim 10.09'da bitti; 13.09 Pazar'inda cutoff 09-13 olunca yarim dolu
    # 09-06 haftasi "tam" sayildi ve ayni veriyle 474 rozet degisti (DS_BULGULAR B1).
    _veri_son = pd.read_parquet(PUSH_PATH, columns=["ts"])["ts"].max().normalize()
    _veri_pazar = _veri_son - pd.Timedelta(days=(_veri_son.dayofweek + 1) % 7)
    if _veri_pazar < BUGUN:
        print(f"veri {_veri_son.date()}'de bitiyor -> cutoff {BUGUN.date()} yerine {_veri_pazar.date()}")
        BUGUN = _veri_pazar
    assert BUGUN.dayofweek == 6, "canli cutoff PAZAR'a oturmali"
    if BUGUN != _bugun_ham:
        print(f"cutoff {_bugun_ham.date()} ({_bugun_ham.day_name()}) -> "
              f"{BUGUN.date()} PAZAR'a hizalandi "
              f"({(_bugun_ham - BUGUN).days} gun geri)")
    print(f"cutoff = {BUGUN.date()}  ->  pencere [{BUGUN.date()} , "
          f"{(BUGUN + pd.Timedelta(days=H)).date()}]")

    cw_son = pd.read_parquet(COMMIT_PATH, columns=["week"])["week"].max()
    gecikme = (BUGUN - cw_son).days
    print(f"commit verisi son hafta: {cw_son.date()}  ({gecikme} gun geride)")
    if gecikme > 21:
        print("  UYARI: commit verisi 3 haftadan eski. cm_silence_weeks BASKIN")
        print("         ozellik (onem 18.60) - once tazelenmeli.")

    X_bugun = build_X(df, BUGUN, flags, feat, m["pa_window_weeks"])   # TUM repolar, T filtresi YOK
    p_bugun = predict(m, X_bugun)

    out = pd.DataFrame({
        "full_name": df["full_name"],
        "p_active_180d_today": p_bugun,
        "cutoff": BUGUN.date().isoformat(),
        "window_end": (BUGUN + pd.Timedelta(days=H)).date().isoformat(),
    })
    # Cekimi basarisiz repolar: commit ozellikleri NaN ve bu bir OLCUM eksigi,
    # reponun ozelligi degil. Dusuk bir skor gostermek "bu proje olmus" demek
    # olurdu; dogru cevap "bilmiyoruz".
    out.loc[_gozlenmedi, "p_active_180d_today"] = np.nan
    print(f"skorlanamayan (cekim basarisiz): {int(_gozlenmedi.sum()):,}")

    out["badge_today"] = pd.cut(out["p_active_180d_today"], [-.01, .30, .60, 1.01],
                                labels=["RED", "AMBER", "GREEN"]).astype(str)
    # pd.cut NaN'i "nan" dizgesine cevirir; acik bir deger daha durust ve
    # asagi akista (SQL, panel) filtrelenebilir.
    out.loc[out["p_active_180d_today"].isna(), "badge_today"] = "VERI_YOK"

    print(f"\nskorlanan repo: {len(out):,}  (repo_scores'ta {len(rs):,} vardi)")
    print(out["badge_today"].value_counts().to_string())

    # ---------------------------------------------- eski ile karsilastir
    header("Eski pencere ile fark")
    k = out.set_index("full_name").join(rs[["p_active_180d", "badge"]], how="inner")
    k = k.dropna(subset=["p_active_180d"])
    eski_band = k["badge"].str.split().str[0]
    degisen = (k["badge_today"] != eski_band)
    print(f"her iki pencerede de skorlanan: {len(k):,}")
    print(f"  ROZETI DEGISEN : {int(degisen.sum()):,}  (%{100*degisen.mean():.2f})")
    print(f"  |P| ortalama fark: {(k['p_active_180d_today'] - k['p_active_180d']).abs().mean():.4f}")
    print("\n  gecis matrisi (eski -> bugun):")
    print(pd.crosstab(eski_band, k["badge_today"]).to_string())

    out.to_parquet(OUT_PATH, index=False, compression="zstd")
    print(f"\nyazildi: {OUT_PATH}")

    if args.write_db:
        header("pgvector guncelleniyor")
        import psycopg2
        from psycopg2.extras import execute_batch
        conn = psycopg2.connect(**PG)
        cur = conn.cursor()
        cur.execute("ALTER TABLE repos ADD COLUMN IF NOT EXISTS p_active_today real")
        cur.execute("ALTER TABLE repos ADD COLUMN IF NOT EXISTS badge_today text")
        cur.execute("ALTER TABLE repos ADD COLUMN IF NOT EXISTS scored_at date")
        execute_batch(cur,
                      "UPDATE repos SET p_active_today=%s, badge_today=%s, scored_at=%s "
                      "WHERE full_name=%s",
                      # NaN yerine NULL: Postgres NaN'i gecerli bir `real`
                      # kabul eder ve DESC siralamada EN BASA koyar, ustelik
                      # IS NOT NULL onu elemez. "Bilmiyoruz" ancak NULL ile
                      # dogru ifade edilir.
                      [(None if pd.isna(r.p_active_180d_today)
                        else float(r.p_active_180d_today),
                        r.badge_today, BUGUN.date(), r.full_name)
                       for r in out.itertuples()], page_size=1000)
        conn.commit()
        cur.execute("SELECT count(*), count(p_active_today), "
                    "count(*) FILTER (WHERE p_active_today = 'NaN'::real) "
                    "FROM repos")
        _t, _s, _n = cur.fetchone()
        print(f"  satir {_t:,} · skoru olan {_s:,} · NULL {_t - _s:,} · NaN {_n:,}")
        assert _n == 0, "pgvector'de NaN kaldi - DESC siralamada basa gecer"
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
