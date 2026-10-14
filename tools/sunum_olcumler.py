# -*- coding: utf-8 -*-
###############################################################
# sunum_olcumler.py | every number panel_v2 shows that no artifact holds yet
###############################################################
# The panel rule is "no number typed into a page". Some slide numbers were only
# ever printed to a log or measured by a one-off probe (sunum/kosum/prob_*.py).
# This script re-measures them from the artifacts and writes ONE file:
#     outputs/sunum_olcumler.json
# Every entry carries its source. Entries that cannot be re-measured today
# (audit history: a leak that was closed days ago no longer exists in the data)
# live under "tarihce" with the ledger/log they come from.
#
# READ-ONLY on the pipeline: reads outputs/, Data/, the embeddings; writes only
# the json above. Needs Ollama (qwen3-embedding:8b) for the off-topic queries.
# The json is stamped with model_rapor.json run_id; panel_v2 refuses a stale file.
#
# RUN   python tools/sunum_olcumler.py            (~10 min, CatBoost CV runs)
#       python tools/sunum_olcumler.py --hizli    (skips the CV sections, keeps them from the old json)
###############################################################

import argparse
import json
import pickle
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
HEDEF = OUT / "sunum_olcumler.json"
T = pd.Timestamp("2026-03-08")


def log(s):
    print(s, flush=True)


rapor = json.loads((OUT / "model_rapor.json").read_text(encoding="utf-8"))
with open(OUT / "model.pkl", "rb") as fh:
    MP = pickle.load(fh)
assert pd.Timestamp(MP["cutoff_T"]) == T, f"model T {MP['cutoff_T']} != {T.date()}"
FEAT, CAT = list(MP["feature_cols"]), list(MP["cat_cols"])
CM = [c for c in FEAT if c.startswith("cm_")]
PA = [c for c in FEAT if c.startswith("pa_")]
META = [c for c in FEAT if c not in CM + PA]
f = pd.read_parquet(OUT / "features.parquet")
y, groups = f["target"], f["owner"]
sc = pd.read_parquet(OUT / "repo_scores.parquet").set_index("full_name").reindex(f["full_name"])
assert sc["target"].values.tolist() == y.tolist(), "repo_scores ve features satir sirasi uyusmuyor"
p_oof = sc["p_active_180d"].to_numpy()
J = {"run_id": rapor["run_id"], "uretildi": time.strftime("%Y-%m-%d %H:%M"), "kaynak_betik": "tools/sunum_olcumler.py"}


def catboost_cv(cols, splitter, grp=None):
    Xc = f[cols].copy()
    cat = [c for c in CAT if c in cols]
    for c in cat:
        Xc[c] = Xc[c].astype(str)
    Xc = Xc.fillna(-999)                      # same as DS21_Final_v2.py:2486-2487
    o, folds = np.zeros(len(Xc)), []
    it = splitter.split(Xc, y, grp) if grp is not None else splitter.split(Xc, y)
    for tr, te in it:
        m = CatBoostClassifier(iterations=500, learning_rate=.05, depth=6, cat_features=cat, verbose=0, random_seed=42)
        m.fit(Xc.iloc[tr], y.iloc[tr])
        o[te] = m.predict_proba(Xc.iloc[te])[:, 1]
        folds.append(roc_auc_score(y.iloc[te], o[te]))
    return {"auc": round(roc_auc_score(y, o), 4), "fold_std": round(float(np.std(folds)), 4),
            "pr_auc": round(average_precision_score(y, o), 4), "n_ozellik": len(cols)}


ap = argparse.ArgumentParser()
ap.add_argument("--hizli", action="store_true")
args = ap.parse_args()
ESKI = json.loads(HEDEF.read_text(encoding="utf-8")) if HEDEF.exists() else {}

###############################################################
# 1 · data expansion (03)
###############################################################
cw = pd.read_parquet(OUT / "commit_weeks.parquet", columns=["full_name", "week", "commits"])
pe_n = pd.read_parquet(OUT / "push_events.parquet", columns=["full_name", "event_id", "ts", "kind", "ref", "actor"])
docs = pd.read_parquet(OUT / "rag_documents_qwen8b.parquet", columns=["full_name", "has_readme"])
csv = pd.read_csv(ROOT / "Data" / "github_ai_intelligence_2026.csv", usecols=["full_name", "owner", "name", "stars"])
_ham = (OUT / "E2_refresh_kuru.log").read_bytes()   # PowerShell *> writes UTF-16LE, BOM not guaranteed
refresh_log = _ham.decode("utf-16-le" if _ham[1:2] == b"\x00" or _ham[:2] == b"\xff\xfe" else "utf-8", errors="replace").lstrip("﻿")
esles = [l for l in refresh_log.splitlines() if "CSV ile eslesen" in l][0]
esles_n = int(esles.split(":")[1].split("/")[0].strip().replace(",", ""))


def varsayilan_dallar():
    """full_name -> 'refs/heads/<branch>'. Same rule as DS21_Final_v2.py:848-896 (meta + ek_meta, latest fetch wins)."""
    d = {}

    def keep(ad, dal, an):
        if ad not in d or an >= d[ad][1]:
            d[ad] = (dal, an)

    for yol in sorted(Path(r"C:\Miuul\ds21final\meta").glob("meta_*.jsonl")):
        for s in open(yol, encoding="utf-8"):
            if not s.strip():
                continue
            try:
                r = json.loads(s)
            except ValueError:
                continue
            ad = (r.get("repo") or {}).get("full_name") or r.get("full_name")
            db = (r.get("repo") or {}).get("default_branch")
            if ad and db:
                keep(ad, db, str(r.get("cekilme", "")))
    for s in open(OUT / "ek_meta.jsonl", encoding="utf-8"):
        if not s.strip():
            continue
        r = json.loads(s)
        db = (r.get("repo") or {}).get("default_branch")
        if r.get("status") == 200 and r.get("created_match") and db:
            keep(r["istenen_full_name"], db, str(r.get("cekilme", "")))
    return {k: "refs/heads/" + v[0] for k, v in d.items()}


def sayim(s):
    return {str(k): int(v) for k, v in s.value_counts().items()}


# commit and event rows count different units: a commit_weeks row is one repo-WEEK, a push_events row is one EVENT
vd = pe_n["ref"] == pe_n["full_name"].astype(str).map(varsayilan_dallar())
TAVAN = 40 * 100                              # n8n/build_tarihce_workflow.py:75 SAYFA_TAVANI x 100 events per page
eski = cw["week"] < "2008-04-01"              # GitHub opened in April 2008; older weeks are imported history
fl = pd.read_json(OUT / "fetch_log.jsonl", lines=True)
cerceve = set(f["full_name"])
yok_cer = f[f["full_name"].isin(set(fl.loc[fl["push_status"] == "yok", "full_name"]))]
# README fetch status, same rule as sunum/kosum/prob_cekim_gelmeyen.py
disk = {p.name for p in (ROOT / "Data" / "readme_raw").iterdir()}
rm_eksik = csv.loc[~(csv["owner"] + "__" + csv["name"] + ".md").isin(disk), "full_name"]
mi = pd.DataFrame([json.loads(x) for x in open(r"C:\Miuul\ds21final\logs\misses.jsonl", encoding="utf-8") if x.strip()])
zaman_asimi = mi.assign(t=mi["hata"].astype(str).str.contains("timeout")).groupby("full_name")["t"].any()
rm_404 = int(rm_eksik.map(zaman_asimi).eq(False).sum())

J["genisletme"] = {
    "csv_repo": len(csv), "csv_sutun": 45,
    "commit_hafta_satir": len(cw), "commit_repo": int(cw["full_name"].nunique()),
    "commit_toplam": int(cw["commits"].sum()),
    "commit_2008_oncesi_repo": int(cw.loc[eski, "full_name"].nunique()),
    "commit_2008_oncesi_commit": int(cw.loc[eski, "commits"].sum()),
    "commit_durum": sayim(fl["status"]),
    "commit_eksik_cercevede": int(fl.loc[fl["status"] != "ok", "full_name"].isin(cerceve).sum()),
    "etkinlik_olay": len(pe_n), "etkinlik_repo": int(pe_n["full_name"].nunique()),
    "etkinlik_varsayilan_dal": int(vd.sum()), "etkinlik_varsayilan_dal_pay": round(float(vd.mean()), 4),
    "etkinlik_dal_ac_sil_pay": round(float(pe_n["kind"].isin(["branch_creation", "branch_deletion"]).mean()), 4),
    "etkinlik_bot_pay": round(float(pe_n["actor"].astype(str).str.contains("[bot]", regex=False).mean()), 4),
    "etkinlik_tekrar": int(pe_n["event_id"].duplicated().sum()),
    "etkinlik_tavan": TAVAN, "etkinlik_tavan_repo": int((pe_n.groupby("full_name").size() >= TAVAN).sum()),
    "etkinlik_ilk": str(pd.Timestamp(pe_n["ts"].min()).date()), "etkinlik_son": str(pd.Timestamp(pe_n["ts"].max()).date()),
    "etkinlik_durum": sayim(fl["push_status"]),
    "etkinlik_yok_cercevede": len(yok_cer), "etkinlik_yok_cercevede_pozitif": round(float(yok_cer["target"].mean()), 4),
    "readme_repo": int(docs["has_readme"].sum()),
    "readme_yok_404": rm_404, "readme_hata": int(len(rm_eksik) - rm_404),
    "metadata_eslesen": esles_n,
    "cekim": {"tur_dakika": 15, "deneme_commit": 5, "deneme_etkinlik": 3, "kapi": 11,
              "_kaynak": "n8n/build_tarihce_workflow.py:76 (15 dk tur), :197 (commit deneme<5), :223 (etkinlik deneme<3) · tarihce.py 11 kapi (logs tarihce_kuru.log:73-83)"},
    "_kaynak": "outputs/commit_weeks · push_events · fetch_log · rag_documents_qwen8b; varsayilan dal = v2:848-896 kurali; "
               "README 404/hata = prob_cekim_gelmeyen.py kurali (Data/readme_raw + misses.jsonl); metadata: outputs/E2_refresh_kuru.log",
}
# 04 · where cm_silence_weeks comes from: honest as-of-T value from commit history vs the banned September CSV column
_csv_ham = pd.read_csv(ROOT / "Data" / "github_ai_intelligence_2026.csv",
                       usecols=["full_name", "days_since_last_push", "pushed_at"]).set_index("full_name")
_fr = f.set_index("full_name")
J["sessizlik_ornek"] = []
for _r in ("pezzolabs/pezzo", "microsoft/markitdown", "TaskingAI/TaskingAI"):
    _son = cw.loc[(cw["full_name"] == _r) & (cw["week"] < T), "week"].max()
    _o = {"repo": _r, "son_commit_haftasi": str(pd.Timestamp(_son).date()), "gun": int((T - _son).days),
          "hafta": int((T - _son).days // 7), "cm_silence_weeks": int(_fr.loc[_r, "cm_silence_weeks"]),
          "csv_gun": int(_csv_ham.loc[_r, "days_since_last_push"]), "csv_pushed_at": str(_csv_ham.loc[_r, "pushed_at"]),
          "hedef": int(_fr.loc[_r, "target"]),
          "_kaynak": "commit_weeks.parquet (week < T) · features.parquet · CSV days_since_last_push (YASAKLI, Eylul olcumu)"}
    assert _o["hafta"] == _o["cm_silence_weeks"], f"{_r}: elle hesap {_o['hafta']} != features {_o['cm_silence_weeks']}"
    J["sessizlik_ornek"].append(_o)
log(f"[1] sessizlik ornek {[(x['repo'], x['hafta'], x['csv_gun']) for x in J['sessizlik_ornek']]}")
del cw, pe_n, vd
log(f"[1] genisletme {J['genisletme']}")

###############################################################
# 2 · ablation ladder of the SHIPPED model (04) - B34
###############################################################
if args.hizli and "merdiven" in ESKI:
    J["merdiven"] = ESKI["merdiven"]
else:
    g5 = GroupKFold(n_splits=5)
    r = {"full": catboost_cv(FEAT, g5, groups)}
    assert abs(r["full"]["auc"] - round(MP["oof_auc"], 4)) < 1e-4, f"sevk AUC yeniden uretilemedi {r['full']}"
    for k, cols in (("sessizlik", ["cm_silence_weeks"]), ("metadata", META), ("commit", CM), ("etkinlik", PA),
                    ("metadata_commit", META + CM), ("commit_etkinlik", CM + PA)):
        r[k] = catboost_cv(cols, g5, groups)
        log(f"[2] {k} {r[k]}")
    r["_kaynak"] = "CatBoost(iterations=500, lr=.05, depth=6, seed 42) · GroupKFold(5, owner) · README yok · FULL sevk OOF AUC'yi birebir uretir"
    J["merdiven"] = r

###############################################################
# 3 · pa_events censoring spike, OLD definition (04) - B2
###############################################################
pe = pd.read_parquet(OUT / "push_events.parquet", columns=["full_name", "ts"])
eski_pa = pe[pe["ts"] < T].groupby("full_name").size()
del pe
d = pd.DataFrame({"n": eski_pa.reindex(f["full_name"]).to_numpy(), "y": y.to_numpy()}).dropna()
d["n"] = d["n"].astype(int)
bins = d[(d["n"] >= 60) & (d["n"] <= 240)].groupby("n")["y"].agg(["size", "mean"]).reset_index()
J["pa_diken"] = {
    "n": bins["n"].tolist(), "repo": bins["size"].tolist(), "pozitif": [round(x, 4) for x in bins["mean"]],
    "tam_100_repo": int((d["n"] == 100).sum()), "tam_100_pozitif": round(float(d.loc[d["n"] == 100, "y"].mean()), 4),
    "tam_200_repo": int((d["n"] == 200).sum()), "tam_200_pozitif": round(float(d.loc[d["n"] == 200, "y"].mean()), 4),
    "cerceve_pozitif": round(float(y.mean()), 4),
    "_kaynak": "ESKI tanim: push_events ts < T, pencere yok (B2 oncesi pa_events) · cerceve features.parquet",
}
log(f"[3] 100: {J['pa_diken']['tam_100_repo']} repo %{100 * J['pa_diken']['tam_100_pozitif']:.1f}")

###############################################################
# 4 · model metrics (05)
###############################################################
taban = float(y.mean())
J["dogruluk_tuzagi"] = {"hep_sifir_dogruluk": round(1 - taban, 4), "taban": round(taban, 4),
                        "auc": round(roc_auc_score(y, p_oof), 4), "pr_auc": round(average_precision_score(y, p_oof), 4),
                        "brier": round(float(np.mean((p_oof - y) ** 2)), 4),
                        "_kaynak": "outputs/repo_scores.parquet (OOF)"}
esikler = np.round(np.arange(0.05, 0.96, 0.01), 2)
f1s = [f1_score(y, p_oof >= t) for t in esikler]
t_f1 = float(esikler[int(np.argmax(f1s))])


def matris(t):
    pr = p_oof >= t
    return {"esik": t, "precision": round(precision_score(y, pr), 4), "recall": round(recall_score(y, pr), 4),
            "f1": round(f1_score(y, pr), 4), "dogruluk": round(float((pr == y).mean()), 4),
            "dn": int(((~pr) & (y == 0)).sum()), "yp": int((pr & (y == 0)).sum()),
            "yn": int(((~pr) & (y == 1)).sum()), "dp": int((pr & (y == 1)).sum())}


J["esik"] = {"f1_en_iyi": matris(t_f1), "sevk": matris(0.60), "kirmizi_sari": matris(0.30),
             "egri": {"esik": esikler.tolist(), "precision": [round(precision_score(y, p_oof >= t, zero_division=0), 4) for t in esikler],
                      "recall": [round(recall_score(y, p_oof >= t), 4) for t in esikler]},
             "_kaynak": "outputs/repo_scores.parquet (OOF)"}
yas = f["age_at_T"].to_numpy()
bantlar = [("<6 ay", 0, 182), ("6-12 ay", 182, 365), ("1-3 yıl", 365, 1095), ("3+ yıl", 1095, 10 ** 6)]
J["yas_auc"] = [{"bant": ad, "n": int(((yas >= a) & (yas < b)).sum()),
                 "pozitif": round(float(y[(yas >= a) & (yas < b)].mean()), 4),
                 "auc": round(roc_auc_score(y[(yas >= a) & (yas < b)], p_oof[(yas >= a) & (yas < b)]), 4)}
                for ad, a, b in bantlar]
log(f"[4] esik F1 {t_f1} · yas {J['yas_auc']}")

# owner fate + random vs grouped split with the shipped model
oc = f["owner"].value_counts()
sub = f[f["owner"].isin(oc[oc >= 3].index)]
po = sub.groupby("owner")["target"].mean()
oybirligi = float(((po == 0) | (po == 1)).mean())
rng = np.random.default_rng(42)
karisik = []
for _ in range(20):
    s2 = sub.assign(target=rng.permutation(sub["target"].to_numpy()))
    q = s2.groupby("owner")["target"].mean()
    karisik.append(float(((q == 0) | (q == 1)).mean()))
grup = {"owner_3plus": int(len(po)), "oybirligi": round(oybirligi, 4), "bagimsiz_beklenen": round(float(np.mean(karisik)), 4)}
if args.hizli and "grup" in ESKI and "rastgele" in ESKI["grup"]:
    grup["rastgele"], grup["gruplu"] = ESKI["grup"]["rastgele"], ESKI["grup"]["gruplu"]
else:
    grup["rastgele"] = catboost_cv(FEAT, KFold(n_splits=5, shuffle=True, random_state=42))
    grup["gruplu"] = J["merdiven"]["full"]
grup["_kaynak"] = "features.parquet · oybirligi: >=3 repolu owner, 20 karistirma (v2 STEP 9.1 ile ayni) · AUC: sevk CatBoost, KFold(shuffle, 42) vs GroupKFold(owner)"
J["grup"] = grup
log(f"[4] grup {grup}")

###############################################################
# 5 · RAG (06)
###############################################################
PREFIX = "Instruct: Given a developer's problem statement, retrieve the GitHub repository that solves it\nQuery:"
docs_n = docs["full_name"]
E = np.asarray(np.load(OUT / "embeddings_qwen8b.npy", mmap_mode="r"), dtype=np.float32)
EB = np.asarray(np.load(OUT / "embeddings_qwen8b_body.npy", mmap_mode="r"), dtype=np.float32)
rows = np.load(OUT / "body_rows_qwen8b.npy")
ev = pd.read_parquet(OUT / "eval_queries.parquet")
QV = np.load(OUT / "eval_query_vectors_qwen3-embedding_8b.npy").astype(np.float32)
pos = {n: i for i, n in enumerate(docs_n)}
truth = np.array([pos[x] for x in ev["full_name"]])
S_bas = QV @ E.T
S_bg = S_bas.copy()
S_bg[:, rows] = np.maximum(S_bg[:, rows], QV @ EB.T)


def gold_rank(S):
    g = S[np.arange(len(ev)), truth]
    return (S > g[:, None]).sum(axis=1) + 1


rk_bas, rk_bg = gold_rank(S_bas), gold_rank(S_bg)
J["rag_yildiz_bandi"] = [{"bant": b, "n": int((ev["band"] == b).sum()),
                          "r5_bas": round(float((rk_bas[(ev["band"] == b).values] <= 5).mean()), 4),
                          "r5_bas_govde": round(float((rk_bg[(ev["band"] == b).values] <= 5).mean()), 4)}
                         for b in ["<100", "100-1K", "1K-10K", "10K+"]]

# live ordering (same rule as tools/rag_canli_sira.py)
st = pd.read_parquet(OUT / "scores_today.parquet").set_index("full_name").reindex(docs_n)
p_bugun = st["p_active_180d_today"].to_numpy()
veri_yok = (st["badge_today"] == "VERI_YOK").to_numpy()
S = S_bg.copy()
S[:, veri_yok] = -np.inf
top10 = np.argsort(-S, axis=1)[:, :10]
sim_r, live_r = np.full(len(ev), 99), np.full(len(ev), 99)
for qi in range(len(ev)):
    h = np.where(top10[qi] == truth[qi])[0]
    if len(h):
        sim_r[qi] = h[0] + 1
        order = top10[qi][np.argsort(-np.nan_to_num(p_bugun[top10[qi]], nan=-1.0), kind="stable")]
        live_r[qi] = np.where(order == truth[qi])[0][0] + 1
J["rag_sinav_kullanici"] = {
    "benzerlik_ilk5": round(float((sim_r <= 5).mean()), 4), "listede_ilk10": round(float((sim_r <= 10).mean()), 4),
    "canli_ilk5": round(float((live_r <= 5).mean()), 4), "canli_birinci": round(float((live_r == 1).mean()), 4),
    "disari_itilen": int(((sim_r <= 5) & (live_r > 5)).sum()), "iceri_cekilen": int(((sim_r > 5) & (sim_r <= 10) & (live_r <= 5)).sum()),
    "sorgu": len(ev), "_kaynak": "bas+govde · VERI_YOK atildi · ilk 10 benzerlik, sonra p_active_180d_today ile dizildi (n8n DS21search0001)"}
log(f"[5] sinav/kullanici {J['rag_sinav_kullanici']}")

i_m = pos["microsoft/markitdown"]
kom = E @ E[i_m]
kom[i_m] = -1
en = np.argsort(-kom)[:5]
J["rag_komsu"] = {"repo": "microsoft/markitdown", "komsular": [{"repo": docs_n.iloc[i], "benzerlik": round(float(kom[i]), 3)} for i in en],
                  "_kaynak": "embeddings_qwen8b.npy (bas vektoru) kosinus"}

NEGATIF = ["Where is the nearest coffee shop that is open late?", "Recipe for a traditional Turkish lentil soup",
           "How do I change a flat tire on a bicycle?", "Best hiking trails near Munich in autumn",
           "What is the capital of Australia?", "Tips for improving my sleep schedule",
           "How to knit a wool scarf for beginners", "Football match results from last weekend",
           "Cheap flights from Istanbul to Berlin", "How to repair a leaking kitchen faucet",
           "Write a poem about the sea", "Symptoms of the common cold in children"]   # = sunum/kosum/prob_negatif_kontrol.py
body = json.dumps({"model": "qwen3-embedding:8b", "input": [PREFIX + t for t in NEGATIF], "options": {"num_gpu": 0}}).encode()
req = urllib.request.Request("http://127.0.0.1:11434/api/embed", body, {"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as rsp:
    NV = np.array(json.loads(rsp.read())["embeddings"], dtype=np.float32)
NV /= np.linalg.norm(NV, axis=1, keepdims=True)
S_neg = NV @ E.T
S_neg[:, rows] = np.maximum(S_neg[:, rows], NV @ EB.T)
top1_neg = S_neg.max(axis=1)
top1_ev = S_bg.max(axis=1)
ESIK = 0.58                                   # live value in n8n DS21search0001 Return node (D2)
J["rag_esik"] = {
    "esik": ESIK, "sinav_top1": [round(float(x), 4) for x in top1_ev], "negatif_top1": [round(float(x), 4) for x in top1_neg],
    "negatif_sorgu": NEGATIF,
    "negatif_kesilen": int((top1_neg < ESIK).sum()), "sinav_kesilen": int((top1_ev < ESIK).sum()),
    "sinav_kesilen_dogru_ilk10": int(((top1_ev < ESIK) & (rk_bg <= 10)).sum()),
    "tablo": [{"esik": t, "negatif": int((top1_neg < t).sum()), "sinav": int((top1_ev < t).sum()),
               "sinav_dogru_ilk10": int(((top1_ev < t) & (rk_bg <= 10)).sum())} for t in (0.55, 0.58, 0.60, 0.62, 0.65, 0.70)],
    "canli_test": [{"soru": "Mutfak musluğundaki sızıntıyı adım adım tamir etmeyi anlatan bir araç", "sorgu": "how to fix leaky faucet step by step",
                    "en_iyi": 0.4002, "sonuc": "kesildi", "calisma": 62961},
                   {"soru": "Yeni başlayanlara yün atkı örgüsünü öğreten bir uygulama", "sorgu": "knitting tutorial for beginners",
                    "en_iyi": 0.4958, "sonuc": "kesildi", "calisma": 62966},
                   {"soru": "PDF dosyalarını markdown formatına çeviren bir araç", "sorgu": "convert pdf to markdown",
                    "en_iyi": 0.8133, "sonuc": "10 madde", "calisma": 62957}],
    "_kaynak": "bas+govde max · negatif 12 sorgu Ollama ile gomuldu (num_gpu 0, ayni talimat) · canli_test: n8n calisma kayitlari 13.09"}
log(f"[5] esik negatif {J['rag_esik']['negatif_kesilen']}/12 · sinav {J['rag_esik']['sinav_kesilen']}/{len(ev)}")

###############################################################
# 6 · audit history (04 timeline) - cannot be re-measured, source given
###############################################################
_pd = J["pa_diken"]                            # same measurement as the spike chart, so the card and the chart agree (T1)
_pd_repo = f"{_pd['tam_100_repo']:,}".replace(",", ".")
_pd_oran = "%" + f"{100 * _pd['tam_100_pozitif']:.1f}".replace(".", ",")
J["tarihce"] = {
    "sizinti_zaman_cizelgesi": [
        {"tarih": "2026-09-10", "ad": "Hafta sınırı",
         "ne": "Tahmin günü ilk başta bir Çarşamba'ya denk geliyordu. GitHub commit'leri haftalık topladığı için o haftanın "
               "kaydı tahmin gününden sonraki 4 günü de içeriyordu (2.740 repo).",
         "bedel": "Model 4 gün ileriyi görebiliyordu.",
         "karar": "Tahmin gününü bir Pazar'a çektim ve süreyi 182 güne (26 hafta) sabitledim. Kontrol artık haftanın bitiş gününe bakıyor.",
         "kaynak": "docs/AUDIT_DEFTERI.md#ds21-final-v2-727"},
        {"tarih": "2026-09-11", "ad": "Çekim bayrakları",
         "ne": "Altı özellik, Eylül'de yaptığım veri çekiminin sonucunu taşıyordu. “Çekim başarılı ama tahmin gününden önce "
               "commit yok” grubundaki 172 reponun %98,8'i güncellenmişti.",
         "bedel": "Bu özellikleri çıkarınca AUC 0,0010 düştü.",
         "karar": "Altı özelliği modelden çıkardım ve koda bunu denetleyen bir kontrol ekledim.",
         "kaynak": "docs/AUDIT_DEFTERI.md#ds21-final-v2-2270 · #ds21-final-v2-2321"},
        {"tarih": "2026-09-11", "ad": "Liste tavanı",
         "ne": "En yoğun 234 repoda okuyabildiğim kayıt tahmin gününden sonraki olaylarla doldu; öncesine hiç ulaşamadım.",
         "bedel": "Model bu boşluğu “çok aktif repo” diye okuyup tahmini yukarı itiyordu (+0,580 log-odds).",
         "karar": "Geçmişini göremediğim bu repoları eğitimden çıkardım.",
         "kaynak": "docs/AUDIT_DEFTERI.md#ds21-final-v2-1098"},
        {"tarih": "2026-09-13", "ad": "Sayfa tavanı",
         "ne": f"“Tahmin gününden önce tam 100 olay” aslında “tahmin gününden sonra hiç olay yok” demekti: {_pd_repo} repo, "
               f"yalnızca {_pd_oran}'ü güncellenmiş.",
         "bedel": "AUC 0,9515'ten 0,9380'e indi.",
         "karar": "Etkinlik özelliklerini artık tahmin gününden önceki sabit 51 haftadan hesaplıyorum.",
         "kaynak": "docs/AUDIT_DEFTERI.md#ds21-final-v2-2137 · sunum/kosum/DS_BULGULAR.md B2 · repo sayisi pa_diken ile ayni olcum"},
    ],
    "readme_kat": {"deger": 192, "kaynak": "docs/AUDIT_DEFTERI.md#ds21-final-v2-2281"},
    "mcnemar_p": {"deger": 0.0074, "kaynak": "tools/rag_sinav.py · logs/E/E5_rag_sinav.log"},
}

HEDEF.write_text(json.dumps(J, ensure_ascii=False, indent=1), encoding="utf-8")
log(f"\nyazildi: {HEDEF}  (run_id {J['run_id']})")
