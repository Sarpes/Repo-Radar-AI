# -*- coding: utf-8 -*-
###############################################################
# app.py | DS21 FINAL - SUNUM PANELI v2
###############################################################
# The presentation panel. Structure:
#   - 8 chapters, no time budget on screen
#   - every chapter opens with one "Teknoloji" line
#   - every chapter ends with a long, collapsible speaker text
#   - no number typed into a page: artifacts (veri.py) or outputs/sunum_olcumler.json
#     (tools/sunum_olcumler.py, stamped with the model run_id)
#
# Design language: tema.css (light, one accent shared with the figures).
# RUN: panel_v2.bat   (or: cd panel_v2 && python -m streamlit run app.py)
###############################################################

from pathlib import Path

import pandas as pd
import streamlit as st

import grafik as g
import veri

_simge = veri.fig("logo_reporadarai_64.png")   # browser tab icon: same mark as the sidebar, "◈" only as a fallback
st.set_page_config(page_title="RepoRadar · DS21 Final", page_icon=str(_simge) if _simge else "◈",
                   layout="wide", initial_sidebar_state="auto")

st.markdown("<style>" + (Path(__file__).parent / "tema.css").read_text(encoding="utf-8") + "</style>",
            unsafe_allow_html=True)

o = veri.veri_ozeti()
m = veri.model_ozeti()
b = veri.blok_sayilari()
S, O, Y = veri.sayi, veri.ondalik, veri.yuzde

# Turkish case suffix after a formatted number ("%88,4" -> "%88,4'ü"). Chosen by how the LAST digit is read,
# so it stays right when the number changes. Valid for numbers whose last digit is a unit digit (Y() always has one decimal).
EKLER = {
    "u":    dict(zip("0123456789", ["'ı", "'i", "'si", "'ü", "'ü", "'i", "'sı", "'si", "'i", "'u"])),
    "unu":  dict(zip("0123456789", ["'ını", "'ini", "'sini", "'ünü", "'ünü", "'ini", "'sını", "'sini", "'ini", "'unu"])),
    "unda": dict(zip("0123456789", ["'ında", "'inde", "'sinde", "'ünde", "'ünde", "'inde", "'sında", "'sinde", "'inde", "'unda"])),
    "e":    dict(zip("0123456789", ["'a", "'e", "'ye", "'e", "'e", "'e", "'ya", "'ye", "'e", "'a"])),
}


def ekle(s, tur):
    return s + EKLER[tur][[c for c in s if c.isdigit()][-1]]
J, J_HATA = veri.sunum_olcumler()
if J is None:
    st.error(f"**Sunum ölçümleri okunamadı.** {J_HATA}")
    st.stop()


###############################################################
# HELPERS
###############################################################
def html(s):
    st.markdown(s, unsafe_allow_html=True)


def kutu(sinif, icerik):
    html(f'<div class="{sinif}">{icerik}</div>')


def baslik(ust, metin):
    html(f'<div class="figure-header"><div class="figure-eyebrow">{ust}</div>'
         f'<div class="figure-title">{metin}</div></div>')


def serit(neden, gordu, karar, basliklar=("NEDEN BAKTIM", "NE GÖRDÜM", "KARARIM")):
    h1, h2, h3 = basliklar
    html(f'<div class="serit3"><div><span class="h">{h1}</span>{neden}</div>'
         f'<div><span class="h">{h2}</span>{gordu}</div><div><span class="h">{h3}</span>{karar}</div></div>')


def kpi(ogeler):
    """Row of number cards. st.metric draws an arrow next to every delta, which reads as 'increase'."""
    hucre = "".join(f'<div class="kpi"><div class="kpi-e">{e}</div><div class="kpi-d">{d}</div>'
                    + (f'<div class="kpi-a">{a}</div>' if a else "") + "</div>" for e, d, a in ogeler)
    html(f'<div class="kpi-grid" style="grid-template-columns:repeat({len(ogeler)},minmax(0,1fr))">{hucre}</div>')


def resim(ad):
    p = veri.fig(ad)
    if p:
        st.image(str(p), width="stretch")
    else:
        st.info(f"`{ad}` bulunamadı. Figürler için: `python panel_v2/figur_uret.py`")


def grafik(fig):
    st.plotly_chart(fig, width="stretch", config=g.CONFIG)


def buyuk_sayi(etiket, deger, aciklama):
    html(f'<div class="buyuk-sayi"><div class="etiket">{etiket}</div><div class="deger">{deger}</div>'
         f'<div class="aciklama">{aciklama}</div></div>')


def gecis(baslik_, metin):
    # The whole box links to the next chapter (?bolum=N is already read at startup). A link, not st.button:
    # the new chapter opens at the top of the page, the browser back button works, and it behaves inside the site iframe.
    # The <a> IS the box and its parts are <span>s: Streamlit renders HTML inside a markdown <p>, and a browser
    # splits an <a> that wraps <div>s there (measured 15.09: box not inside the link, only the text was clickable).
    sonraki = aktif + 2
    ic = (f'<span class="transition-label">SONRAKİ AŞAMA</span><span class="transition-title">{baslik_}</span>'
          f'<span class="transition-text">{metin}</span>')
    if sonraki <= len(BOLUMLER):
        html(f'<a class="transition-box transition-link" href="?bolum={sonraki}" target="_self">{ic}'
             f'<span class="transition-devam">Devam et →</span></a>')
    else:
        html(f'<div class="transition-box">{ic}</div>')


def teknoloji(metin):
    if metin:
        html(f'<div class="teknoloji"><span class="et">TEKNOLOJİ</span>{metin}</div>')




def bolum_basi(no, ad, tek):
    html(serit_html + "</div>")
    html(f'<div class="kicker">Bölüm {no}</div>')
    st.header(ad)
    teknoloji(tek)


###############################################################
# SIDEBAR
###############################################################
BOLUMLER = [
    ("01  Problem & Veri", "Problem", None),
    ("02  Keşifçi Veri Analizi", "EDA", "Python · pandas · matplotlib · Kaggle CSV (Eylül 2026 güncellemesi)"),
    ("03  Veri Setinin Genişletilmesi", "Genişletme", "GitHub REST API · n8n · Python"),
    ("04  Özellik Mühendisliği", "Özellik", f"Python · parquet · {m['ozellik_n']} feature"),
    ("05  Tahmin Modeli", "Model", "CatBoost · GroupKFold · SHAP"),
    ("06  RAG", "RAG", "PostgreSQL + pgvector · Ollama qwen3-embedding:8b"),
    ("07  RepoRadarAI", "RepoRadarAI", "n8n AI Agent · Ollama qwen3:14b · SQL · Vercel → Tailscale → gecit.js"),
    ("08  İlerideki Adımlar", "İleride", None),
]
etiketler = [ad for ad, _, _ in BOLUMLER]

_logo = veri.fig("logo_reporadarai.svg")   # original mark (magnifier + radar); emoji only as a fallback
st.sidebar.markdown(
    f'<div class="side-brand"><div class="side-brand-mark">{_logo.read_text(encoding="utf-8") if _logo else "📡"}</div><div>'
    '<div class="side-brand-title">RepoRadarAI</div></div></div>'
    '<div class="side-nav-kicker">SUNUM AKIŞI</div>', unsafe_allow_html=True)
# ?bolum=4 opens chapter 4 directly (a link per chapter; also lets a headless browser check each page)
_q = st.query_params.get("bolum", "")
_ilk = int(_q) - 1 if _q.isdigit() and 1 <= int(_q) <= len(etiketler) else 0
sayfa = st.sidebar.radio("Sunum akışı", etiketler, index=_ilk, label_visibility="collapsed")
aktif = etiketler.index(sayfa)
st.sidebar.markdown(
    '<div class="timeline-note"><div class="timeline-note-label">MODEL</div>'
    f'<div class="timeline-note-title">CatBoost · {m["ozellik_n"]} feature · AUC {O(m["auc"])}</div>'
    f'<div class="timeline-note-text">Tahmin günü {pd.Timestamp(m["T"]):%d.%m.%Y} · {m["H"]} günlük pencere · etiket: default branch\'e ulaşan değişiklik<br>'
    f'koşum {m["run_id"]} · ölçümler {J["uretildi"]}</div></div>', unsafe_allow_html=True)

serit_html = '<div class="journey">'
for i, (_, kisa, _) in enumerate(BOLUMLER):
    if i:
        serit_html += '<div class="journey-line"></div>'
    serit_html += f'<div class="journey-step{" active" if i <= aktif else ""}">0{i + 1} <span>{kisa}</span></div>'

GEN = J["genisletme"]
ESIK = J["esik"]
RSK = J["rag_sinav_kullanici"]
RES = J["rag_esik"]


###############################################################
# 01 PROBLEM & DATA
###############################################################
def bolum_01():
    html("""
    <div class="hero">
      <h1>RepoRadarAI</h1>
      <p>Teknoloji meraklıları ve kendini geliştirmeye yeni başlayanlar, her gün GitHub'daki yapay zeka araçları
      arasından seçim yapmaya çalışıyor. Yıldız sayısı bir aracın ne kadar popüler olduğunu söylüyor, ama önümüzdeki
      aylarda desteklenmeye devam edip etmeyeceğini söylemiyor. RepoRadarAI, 24.664 yapay zeka reposu arasından işinize
      en uygun olanları buluyor ve seçeceğiniz reponun geleceğini tahmin ediyor.</p>
    </div>""")
    html(serit_html + "</div>")
    kutu("callout", f"<b>Veri seti</b> · Kaggle · {S(o['repo'])} repo × {o['sutun']} sütun · tek anlık görüntü")
    gecis("Veriyi tanımak: Keşifçi Veri Analizi", "Bu 45 sütun bize ne anlatıyor, neyi anlatamıyor?")



###############################################################
# 02 EDA
###############################################################
# Figures, not live charts: they come from the model chain (STEP 9, figures/)
# and tools/eda_figures.py (figures/panel/, redrawn by panel_v2/figur_uret.py).
# Every number in the text is read from an artifact or the CSV, never typed.
def iki_resim(sol, sag):
    c1, c2 = st.columns(2)
    with c1:
        resim(sol)
    with c2:
        resim(sag)


def bolum_02():
    bolum_basi(2, "Keşifçi Veri Analizi", BOLUMLER[1][2])
    d = veri.csv()
    soz = veri.sozlesme()
    yeni_pay = float((d["created_year"] >= 2023).mean())
    tek_degerli = [c for c in d.columns if d[c].nunique(dropna=False) == 1]
    homepage_bos = float(d["homepage"].isna().mean())
    rastgele, grup = veri.OLCUMLER["owner_rastgele_auc"][0], veri.OLCUMLER["owner_grup_auc"][0]
    st.caption(f"{S(o['repo'])} repo × {o['sutun']} sütun. Önce veri setini, sonra sütunları, en son da modele neyin "
               "giremeyeceğini inceledim.")
    t = st.tabs(["🧭  Veri seti", "🧬  Sütunlar", "🚧  Kopya ve kural sütunlar", "👥  Owner", "🎯  Hedef ve zaman"])

    with t[0]:
        baslik("VERİ SETİ", "Nasıl bir topluluk?")
        iki_resim("R01_kurulus_yili.png", "R02_yildiz_yogunlasmasi.png")
        serit(f"Modelden önce elimdeki {S(o['repo'])} reponun nasıl bir topluluk olduğunu görmek istedim.",
              f"Repoların {ekle(Y(yeni_pay), 'u')} 2023 ve sonrasında kurulmuş. Yıldızlar ise birkaç dev repoda "
              "toplanıyor: ortalama, tipik bir repoyu hiç anlatmıyor.",
              "Yıldızı hep medyanla özetledim. Yıldız sayısı modele hiç girmedi, çünkü tahmin gününden sonra "
              "ölçülmüştü (3. sekme).")

    with t[1]:
        baslik("SÜTUNLAR", f"{o['sutun']} sütunun her biri bilgi taşıyor mu?")
        resim("R05_sutun_bilgi_degeri.png")
        iki_resim("R03_lisans.png", "R04_teknoloji_bayraklari.png")
        tek = " ve ".join(f"<code>{c}</code>" for c in tek_degerli)
        tutulan = ("Boş olmayı da bir bilgi olarak tuttum: <code>has_homepage</code> modelin "
                   f"{S(len(soz['ozellik']))} özelliğinden biri."
                   if "has_homepage" in soz["ozellik"] else "")
        serit(f"{o['sutun']} sütunun her biri gerçekten bilgi taşıyor mu, ona baktım.",
              f"{tek} her satırda aynı değeri taşıyor. <code>homepage</code> sütununun {ekle(Y(homepage_bos), 'u')} boş.",
              f"Bilgi taşımayan {S(len(tek_degerli))} sütunu çıkardım. {tutulan}")

    with t[2]:
        baslik("KOPYA VE KURAL SÜTUNLAR", "Bu sütunlar gözlem mi, yoksa bir kuralın sonucu mu?")
        iki_resim("F03_watchers_stars.png", "F04_maintenance_bins.png")
        iki_resim("F05_popularity_tier.png", "F06_yasakli_sutunlar.png")
        serit("Bazı sütunlar fazla iyi görünüyordu. Her birinin bağımsız bir gözlem mi, yoksa başka bir sütundan "
              "türetilmiş bir kural mı olduğunu test ettim.",
              "<code>watchers</code> yıldızla birebir aynı. <code>maintenance_status</code> ve "
              "<code>popularity_tier</code> tek bir sayısal sütunun aralıklara bölünmüş hâli.",
              f"Bu sütunları ve tahmin gününden sonra ölçülmüş olanları, toplam {S(len(soz['yasakli']))} sütunu "
              "modelden çıkardım. Liste <code>model.pkl</code>'in içinde saklanıyor.")

    with t[3]:
        baslik("OWNER", "Aynı owner'ın repoları bağımsız örnek değil")
        resim("07_owner_grouping.png")
        serit("Aynı owner'ın birden çok reposu varsa, bunlar birbirinden bağımsız örnekler sayılmaz.",
              f"Rastgele bölme aynı owner'ın repolarını hem eğitime hem teste koyuyor ve skoru şişiriyor "
              f"({O(rastgele)} → {O(grup)}).",
              "Doğrulamayı owner'a göre gruplu yaptım (GroupKFold).")
        html('<div class="kucuk-not">Bu iki sayı, yalnız metadata kullanan hızlı bir teşhis modelinden geliyor. '
             'Sevk edilen modelin sonuçları 5. bölümde.</div>')

    with t[4]:
        baslik("HEDEF VE ZAMAN", "“Gelecek” ne demek?")
        resim("F07_zaman_ekseni.png")
        serit("Bir reponun geleceğini tahmin etmek için önce “gelecek”in ne demek olduğunu tanımlamam gerekiyordu.",
              "Kaggle tablosu tek bir günün kaydı. Bir reponun sonradan ne yaptığını göremiyor.",
              f"Hedef: tahmin gününden sonraki {m['H']} gün içinde default branch'e bir değişiklik ulaşması. "
              "Özellikler yalnız tahmin gününden öncesini görüyor.")

    gecis("Veri Setinin Genişletilmesi", "Tek günlük bu kaydı repoların geçmişiyle nasıl genişlettim?")


###############################################################
# 03 DATA EXPANSION
###############################################################
def bolum_03():
    bolum_basi(3, "Veri Setinin Genişletilmesi", BOLUMLER[2][2])
    pay = GEN["etkinlik_varsayilan_dal_pay"]
    kesir = ("üçte biri" if 0.30 <= pay < 0.37 else "yarısı" if 0.45 <= pay < 0.55
             else "dörtte biri" if 0.22 <= pay < 0.28 else f"{Y(pay)} kadarı")
    baslik("NE EKLEDİM", "Kaggle tablosunu GitHub'dan nasıl genişlettim?")
    html(g.genisletme_mimarisi(GEN))
    serit("Kaggle tablosu tek bir günün kaydı. Bir reponun sonraki aylarda ne yaptığını görmek için geçmişine bakmam gerekiyordu.",
          "İki kaynak farklı şeyler sayıyor. Commit geçmişi, default branch'e giren commit'leri haftalık olarak topluyor. "
          f"Etkinlik kaydı ise her olayı tek tek tutuyor ve bu olayların yalnızca {kesir} default branch'e gidiyor.",
          "Modelin ana kaynağı commit geçmişi oldu. Etkinlik kaydını etiketi kurmak için kullandım, çünkü bir değişikliğin "
          "hangi dala gittiğini yalnızca o gösteriyor.")
    gecis("Özellik Mühendisliği", "Bu geçmişten, tahmin gününde bilinebilecek hangi sayıları çıkardım?")

    G, CK = GEN, GEN["cekim"]
    DC, DE = G["commit_durum"], G["etkinlik_durum"]
    AY = ("Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık")
    ay_yil = lambda s: f"{AY[pd.Timestamp(s).month - 1]} {pd.Timestamp(s).year}"
    binde = 1000 * G["commit_2008_oncesi_commit"] / G["commit_toplam"]
    binde_metin = "binde biri" if round(binde) <= 1 else f"binde {round(binde)} kadarı"
    egitim_metin = ("Bu repolar eğitime alınmadı, çünkü commit bilgisi olmayan bir repo model tarafından “sessiz” diye okunurdu."
                    if G["commit_eksik_cercevede"] == 0 else
                    f"⚠️ Bunların {G['commit_eksik_cercevede']} tanesi eğitim çerçevesinde duruyor.")
    milyon = O(G["etkinlik_olay"] / 1e6, 1)
    tekrar = "sıfır" if G["etkinlik_tekrar"] == 0 else S(G["etkinlik_tekrar"])
    yok_metin = ("hiçbiri güncellenmemiş" if G["etkinlik_yok_cercevede_pozitif"] == 0
                 else f"yalnızca {Y(G['etkinlik_yok_cercevede_pozitif'])} kadarı güncellenmiş")


###############################################################
# 04 FEATURE ENGINEERING
###############################################################
def bolum_04():
    bolum_basi(4, "Özellik Mühendisliği", BOLUMLER[3][2])
    AY = ("Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık")
    T_ = pd.Timestamp(m["T"])
    GUN_AY = f"{T_.day} {AY[T_.month - 1]}"
    kutu("callout", f"<b>Tek kural:</b> Tahmin günü olarak {GUN_AY} {T_.year}'yı seçtim. "
                    "Model yalnızca o güne kadar bilinen bilgileri görüyor.")

    ad = o["ozellik"]
    BAYRAK_AD = {"uses_agents": "ajan", "uses_mcp": "MCP", "uses_anthropic": "Anthropic"}
    sira = list(BAYRAK_AD)
    bayraklar = [BAYRAK_AD.get(c, c.replace("uses_", ""))
                 for c in sorted((c for c in ad if c.startswith("uses_")), key=lambda c: sira.index(c) if c in sira else 99)]
    yasak_n = len(veri.sozlesme()["yasakli"])
    t = st.tabs(["🧱  Feature'lar", "🚧  Leakage", "💯  Sayfa tavanı"])
    with t[0]:
        c1, c2, c3 = st.columns(3)
        kart = [("M", f"Metadata · {b['meta']}", "Kaggle CSV",
                 "Reponun yaşı, açıklama ve ad uzunluğu, konu etiketi sayısı, ana sayfası olup olmadığı, owner'ın kaç "
                 "reposu olduğu, dili, lisansı"
                 + (f", kategorisi ve {len(bayraklar)} teknoloji bayrağı ({', '.join(bayraklar)}).</p>" if bayraklar else " ve kategorisi.</p>")
                 + f"<p><b>Leakage kontrolü:</b> Yıldız, fork, issue sayısı ve son push tarihi gibi Eylül'de ölçülen {yasak_n} "
                 "sütunu modele almadım."),
                ("C", f"Commit geçmişi · {b['commit']}", f"Commit istatistiği · yalnızca {GUN_AY}'tan önceki haftalar",
                 "Reponun <b>kaç haftadır sessiz</b> olduğu, son 13, 26 ve 52 haftadaki commit sayısı, commit'ler arasındaki "
                 "en uzun ara, hızlanma ve tek kişinin çalıştığı haftaların oranı."),
                ("E", f"Repo etkinliği · {b['etkinlik']}", f"Etkinlik kaydı · {GUN_AY}'tan önceki 51 hafta",
                 "Değişikliği <b>kimin</b> yaptığı (en aktif kişinin payı), <b>nereye</b> gittiği (main ya da master dalına "
                 "giden pay) ve <b>ne tür</b> bir değişiklik olduğu (PR birleştirme, force push, dal açma ve silme).")]
        for c, (ik, bas, kaynak, metin) in zip((c1, c2, c3), kart):
            with c:
                html(f'<div class="story"><div class="story-icon">{ik}</div><h4>{bas}</h4>'
                     f'<p class="small-muted">{kaynak}</p><p>{metin}</p></div>')

        # Feature dictionary. Definitions are written from the builder code (DS21_Final_v2.py lines in the headers);
        # WHICH features are listed comes from the model itself, so a feature without a definition shows up as missing.
        TANIM = {
            "metadata": ("Metadata", "Kaggle CSV · DS21_Final_v2.py:217-226", [
                ("age_at_T", "Reponun yaşı", "Tahmin günü − oluşturulma tarihi, gün olarak"),
                ("topics_n", "Konu etiketi sayısı", "Konu etiketleri alanındaki etiket sayısı"),
                ("desc_len", "Açıklama uzunluğu", "Açıklamanın karakter sayısı"),
                ("name_len", "Ad uzunluğu", "Repo adının karakter sayısı"),
                ("has_homepage", "Ana sayfası var mı", "Ana sayfa alanı doluysa 1, boşsa 0"),
                ("owner_n", "Owner'ın repo sayısı", "Aynı owner'ın, tahmin gününde var olan repolarının sayısı"),
                ("uses_agents", "Ajan projesi mi", "Kaggle veri setinde hazır gelen 0/1 işaret"),
                ("uses_mcp", "MCP kullanıyor mu", "Kaggle veri setinde hazır gelen 0/1 işaret"),
                ("uses_anthropic", "Anthropic kullanıyor mu", "Kaggle veri setinde hazır gelen 0/1 işaret"),
                ("language", "Ana programlama dili", "CSV'deki değer, olduğu gibi"),
                ("license", "Lisans", "CSV'deki değer, olduğu gibi"),
                ("ai_category", "Yapay zekâ kategorisi", "Veri setindeki kategori (ör. Agent, RAG, LLM)"),
            ]),
            "commit": ("Commit geçmişi", "Commit istatistiği, yalnızca tahmin gününden önce biten haftalar · DS21_Final_v2.py:1941-1999", [
                ("cm_silence_weeks", "Kaç haftadır sessiz", "Tahmin günü ile son commit haftası arasındaki hafta sayısı"),
                ("cm_13w", "Son 13 haftadaki commit", "Tahmin gününden önceki son 13 haftanın commit toplamı"),
                ("cm_26w", "Son 26 haftadaki commit", "Tahmin gününden önceki son 26 haftanın commit toplamı"),
                ("cm_52w", "Son 52 haftadaki commit", "Tahmin gününden önceki son 52 haftanın commit toplamı"),
                ("cm_active_weeks", "Aktif hafta sayısı", "En az bir commit olan hafta sayısı"),
                ("cm_total", "Toplam commit", "Bütün haftaların commit toplamı"),
                ("cm_mean_per_active_week", "Aktif hafta başına commit", "Toplam commit ÷ aktif hafta sayısı"),
                ("cm_peak_week", "En yoğun hafta", "Tek bir haftadaki en yüksek commit sayısı"),
                ("cm_span_weeks", "Geliştirme süresi", "İlk ve son commit haftası arasındaki hafta sayısı"),
                ("cm_max_gap", "En uzun ara", "Commit atılan iki hafta arasındaki en uzun boşluk (hafta)"),
                ("cm_mean_gap", "Ortalama ara", "Commit atılan iki hafta arasındaki boşlukların ortalaması"),
                ("cm_accel_4w", "Hızlanma", "Son 4 haftadaki commit ÷ (önceki 9 haftadaki commit + 1)"),
                ("cm_solo_week_share", "Tek kişilik haftalar", "Tek kişinin commit attığı haftaların aktif haftalara oranı"),
                ("cm_weeks_since_peak", "Zirveden beri geçen süre", "Tahmin günü ile en yoğun hafta arasındaki hafta sayısı"),
            ]),
            "etkinlik": ("Repo etkinliği", "Etkinlik kaydı, tahmin gününden önceki 51 hafta · DS21_Final_v2.py:2050-2089", [
                ("pa_events", "Olay sayısı", "51 haftalık penceredeki olay sayısı"),
                ("pa_actor_top_share", "En aktif kişinin payı", "En çok olay üreten kişinin olayları ÷ toplam olay"),
                ("pa_default_ref_share", "Ana dala giden pay", "main ya da master dalına giden olaylar ÷ toplam olay"),
                ("pa_refs_distinct", "Dal çeşitliliği", "Olayların gittiği farklı dal sayısı"),
                ("pa_pr_merge_share", "PR birleştirme payı", "PR birleştirme olayları ÷ toplam olay"),
                ("pa_force_push_share", "Force push payı", "Force push olayları ÷ toplam olay"),
                ("pa_branch_churn", "Dal açma/silme payı", "Dal açma ve dal silme olayları ÷ toplam olay"),
            ]),
        }
        BOS_NOT = {"commit": "Commit geçmişi olmayan repoda bu değerlerin hepsi boş kalır.",
                   "etkinlik": "Pencerede hiç olayı olmayan repoda bu değerlerin hepsi boş kalır; model boşluğu “sessiz” diye okur."}
        tanimli = {k for _, _, satirlar in TANIM.values() for k, _, _ in satirlar}
        with st.expander(f"📋 Modeldeki {len(ad)} feature: ne anlama geliyor, nasıl hesaplanıyor?", expanded=False):
            html("<p>Bir feature'ın modele girmesi için üç şart koydum: tahmin gününde bilinebilmeli, sonucu gizlice "
                 "taşımamalı ve modele gerçekten yeni bir bilgi katmalı.</p>")
            for anahtar, (bas, kaynak, satirlar) in TANIM.items():
                satirlar = [s for s in satirlar if s[0] in ad]
                html(f'<div class="figure-header"><div class="figure-eyebrow">{kaynak}</div>'
                     f'<div class="figure-title">{bas} · {len(satirlar)}</div></div>'
                     '<table class="tablo"><tr><th>FEATURE</th><th>ANLAMI</th><th>NASIL HESAPLANIYOR</th></tr>'
                     + "".join(f"<tr><td><code>{k}</code></td><td>{a}</td><td>{h}</td></tr>" for k, a, h in satirlar) + "</table>"
                     + (f'<div class="kucuk-not">{BOS_NOT[anahtar]}</div>' if anahtar in BOS_NOT else ""))
            eksik = [c for c in ad if c not in tanimli]
            if eksik:
                st.warning(f"Tanımı yazılmamış feature: {', '.join(eksik)}")

    with t[1]:
        baslik("LEAKAGE", "Dört leakage kanalı buldum, hepsini kapattım")
        olaylar = J["tarihce"]["sizinti_zaman_cizelgesi"]
        html('<div class="cizelge">' + "".join(
            f'<div class="olay"><div class="ad">{x["ad"]}</div>'
            f'<div class="ne">{x["ne"]}</div><div class="bedel">Etkisi: {x["bedel"]}</div>'
            f'<div class="karar">Ne yaptım: {x["karar"]}</div></div>' for x in olaylar) + "</div>")
        html('<div class="kucuk-not">Ayrıntılar: <code>README.md</code>, 3.5 Zamansal sızıntı</div>')

    with t[2]:
        PD = J["pa_diken"]
        DG = g.diken_gruplari(PD)
        komsu_gruplar = [x["oran"] for x in DG if not x["tam"]]
        baslik("SAYFA TAVANI", "“Tam 100 olay” aslında “hiç yeni olay yok” demekti")
        grafik(g.pa_diken(PD))
        serit("Etkinlik kaydını en yeni olaydan geriye doğru, 100'erli sayfalar hâlinde okuyordum.",
              f"Tam 100 olayı olan {S(PD['tam_100_repo'])} reponun yalnızca {Y(PD['tam_100_pozitif'])}'ü güncellenmiş. "
              f"Komşu gruplarda bu oran %{round(100 * min(komsu_gruplar))} ile %{round(100 * max(komsu_gruplar))} arasında. "
              f"Aynı düşüş tam 200 olayda da var ({S(PD['tam_200_repo'])} repo).",
              "Bu sayı, repoda tahmin gününden sonra hiç olay olmadığını ele veriyordu. Etkinlik feature'larını artık sabit "
              "51 haftadan hesaplıyorum.")

    gecis("Tahmin Modeli", "Model ne kadar iyi, olasılıklarına ne kadar güvenebiliriz?")

    tarih_tr = lambda s, fmt=None: (lambda t: f"{t.day} {AY[t.month - 1]} {t.year}")(pd.to_datetime(s, format=fmt))
    PE = next(x for x in J["sessizlik_ornek"] if x["repo"] == "pezzolabs/pezzo")
    komsu = [PD["pozitif"][PD["n"].index(k)] for k in (99, 101) if k in PD["n"]]
    komsu_oran = f"%{round(100 * sum(komsu) / len(komsu))}" if komsu else "?"


###############################################################
# 05 MODELLING
###############################################################
def bolum_05():
    bolum_basi(5, "Tahmin Modeli", BOLUMLER[4][2])
    grp = veri.kolay_ve_zor_gruplar()
    DT = J["dogruluk_tuzagi"]
    GR = J["grup"]
    kutu("callout", f"<b>CatBoost</b> · {S(m['n'])} repo · {m['ozellik_n']} feature · GroupKFold")
    RZ = {veri.rozet_kodu(r["badge"]): r for r in m["rozetler"]}
    EV = ESIK["sevk"]
    F1 = ESIK["f1_en_iyi"]
    yb = J["yas_auc"]
    t = st.tabs(["🔀  Pipeline", "🕒  Zaman", "👥  Bölme", "🎯  Doğruluk", "📊  Önem", "🔍  Sessizlik", "📏  Kalibrasyon",
                 "🚦  Rozetler", "🧮  Matris", "⚖️  Eşik", "🌱  Yaş"])
    with t[0]:
        baslik("MODEL NASIL KURULDU", "Feature'lardan bugünkü rozete")
        html(g.model_mimarisi(m["n"], m["ozellik_n"], b, m["T"], m["H"], m["taban"], m["auc"], len(veri.sozlesme()["yasakli"])))
    with t[1]:
        z = veri.zaman()
        baslik("MODEL ZAMANDA NASIL İLERLİYOR", "Eğitimde geçmişe, canlıda geleceğe bakıyor")
        grafik(g.zaman_ekseni(z))
        serit("Leakage'ın en sık kaynağı, feature ile target'ın zamanını karıştırmak.",
              f"Eğitimde feature'lar yalnızca {z['T']:%d.%m.%Y} tarihinden önceki veriden geliyor, target ise sonraki "
              f"{z['H']} günden. Bu dönemin sonucu artık biliniyor. Canlıda aynı model {z['bugun']:%d.%m.%Y} tarihinden "
              f"sonraki {z['H']} güne bakıyor ve o dönem henüz yaşanmadı.",
              f"Tahmin gününü bir Pazar'a koydum ve süreyi {z['H']} gün, yani {z['H'] // 7} tam hafta seçtim. Commit "
              "verisi haftalık geldiği için yarım bir hafta, geleceği feature'lara sızdırırdı.")
    with t[2]:
        kutu("callout", "<b>Neden GroupKFold?</b> Aynı owner'ın repoları birbirine benziyor. Rastgele bölmede model, testteki "
                        "repoyu eğitimdeki kardeşinden tanır ve skor şişer.<br>"
                        "<b>Neyi çözdü?</b> Bir owner'ın bütün repoları aynı katta. Test skoru, modelin hiç görmediği owner'larda ölçülüyor.")
        serit("Aynı owner'ın repoları birbirine benziyor mu?",
              f"Benziyor. Repoları rastgele bölünce AUC {O(GR['rastgele']['auc'])}, owner'a göre bölünce {O(GR['gruplu']['auc'])} çıkıyor.",
              "Bir owner'ın repolarını hep aynı gruba koydum; aynı owner hiçbir zaman hem eğitimde hem testte olmuyor. Ekrandaki "
              "sayı bu dürüst bölmeden.")
    with t[3]:
        kpi([("ROC-AUC", O(DT["auc"]), "sıralama kalitesi (şans 0,5)"),
             ("PR-AUC", O(DT["pr_auc"]), f"şans düzeyi {O(DT['taban'], 3)}")])
        serit("Repoların çoğu güncellenmiyor, yani iki grup eşit büyüklükte değil.",
              f"Her repoya “güncellenmez” diyen ve hiçbir şey öğrenmeyen bir kural bile {Y(DT['hep_sifir_dogruluk'])} doğru çıkıyor.",
              f"Bu yüzden doğruluk oranı vermiyorum. Sıralamayı ROC-AUC ve PR-AUC ile, olasılıkların kalitesini Brier "
              f"({O(DT['brier'])}; modelin verdiği olasılık ile gerçek sonuç, yani 1 = güncellendi, 0 = güncellenmedi, "
              "arasındaki farkın karesinin ortalaması; 0 kusursuz demek) ve kalibrasyonla ölçüyorum.")
    with t[4]:
        baslik("MODEL NEYE BAKIYOR", "Model kararını en çok commit geçmişine dayandırıyor; listede yasaklı sütun yok")
        resim("F18_ozellik_onemi.png")
    with t[5]:
        grafik(g.dilimler(m["dilimler"]))
        serit("Tek başına en güçlü bilgi “kaç haftadır sessiz”. Model bunun üstüne ne katıyor?",
              "Aynı süredir sessiz olan repolar arasında sessizlik artık pek bir şey söylemiyor. Model ise her grupta belirgin "
              "şekilde daha iyi ayırıyor.",
              "Modelin değerini toplam skorla değil, bu gruplardaki farkla gösteriyorum.")
        if "sessiz" in grp:
            s = grp["sessiz"]
            yok_metin = "hiçbiri güncellenmedi" if s["actual"] == 0 else f"yalnızca {Y(s['actual'])} kadarı güncellendi"
            kutu("warning", f"<b>İşin kolay kısmını baştan söyleyeyim:</b> {S(s['n'])} repoda 2023'ten beri hiç hareket yok "
                            f"ve {yok_metin}. Bunları ayırmak zor değil.")
    with t[6]:
        c1, c2 = st.columns([3, 2])
        with c1:
            grafik(g.kalibrasyon(veri.kalibrasyon_egrisi()))
        with c2:
            buyuk_sayi("KALİBRASYON HATASI", O(m["kalib"], 5),
                       "Repoları tahmine göre 10 gruba ayırdım; her grupta tahmin ile gerçekleşen oran arasındaki ortalama "
                       "fark. Üstüne ayrıca bir düzeltme uygulamadım.")
        serit("Rozetler olasılığa göre veriliyor. Model “%70” dediği repoların yalnızca yarısında haklı çıksaydı, rozetler "
              "kullanıcıyı yanıltırdı.",
              "Noktalar kesik çizginin üstünde duruyor: model “%70” dediğinde gerçekten yaklaşık %70'i güncelleniyor.",
              "Olasılığı olduğu gibi rozete çeviriyorum; arada bir düzeltmeye gerek kalmadı.")
    with t[7]:
        kpi([(g.ROZET_AD[veri.rozet_kodu(rz["badge"])], f"{S(rz['n'])} repo", f"gerçekte güncellenen {Y(rz['rate'])}")
             for rz in m["rozetler"]])
        grafik(g.rozet_gerceklesme(m["rozetler"]))
        serit("Kullanıcıya “yeşil” dediğimde bunun gerçekte ne anlama geldiğini görmek istedim.",
              f"Yeşil dediğim her 10 repodan yaklaşık {round(10 * RZ['GREEN']['rate'])} tanesi gerçekten güncellendi. "
              f"Kırmızılarda bu oran {Y(RZ['RED']['rate'])}.",
              "%30 ve %60 eşiklerini modeli eğitmeden önce belirledim; sonuçlara bakıp ayarlamadım.")
    with t[8]:
        resim("F19_karisiklik_matrisi.png")
        serit("Rozetin verdiği söz tutuluyor mu? Her rozette kaç repo gerçekten güncellendi?",
              f"Yeşil dediğim {S(RZ['GREEN']['n'])} repodan {S(EV['dp'])} tanesi güncellendi, {S(EV['yp'])} tanesi "
              f"güncellenmedi. Kırmızıların yalnızca {Y(RZ['RED']['rate'])} kadarı güncellendi.",
              f"Yeşil rozeti %60 olasılıktan başlattım. Bunun bedeli, gerçekten güncellenen {S(EV['yn'])} reponun yeşil yerine "
              "sarı ya da kırmızı görünmesi.")
    with t[9]:
        grafik(g.esik_egrisi(ESIK))
        serit("Yeşil rozeti %60'tan başlatmayı eğitimden önce kararlaştırdım. Bu grafik o kararın bedelini gösteriyor.",
              f"%60 eşiğinde yeşil dediklerimin {ekle(Y(EV['precision']), 'u')} doğru çıkıyor. F1'e göre en iyi eşik "
              f"%{round(100 * F1['esik'])}; orada bu oran {ekle(Y(F1['precision']), 'e')} düşüyor. Yanlış yeşil {S(EV['yp'])} "
              f"repo, kaçırılan yeşil {S(EV['yn'])} repo.",
              "Yanlış bir yeşil, kaçırılan bir yeşilden daha pahalı: terk edilmiş bir aracı önermek, iyi bir aracı sarı "
              "göstermekten daha çok zarar verir.")
    with t[10]:
        grafik(g.yas_auc(yb))
        serit("Model her yaştaki repoda aynı güvenle mi çalışıyor?",
              f"6 aydan genç repolarda AUC {O(yb[0]['auc'])}, 3 yaşından büyüklerde {O(yb[-1]['auc'])}.",
              "Genç repoların commit geçmişi kısa, bu yüzden onlar için tahmin daha az güvenilir.")
    gecis("RAG", "Modelin nerede güçlü, nerede zayıf olduğunu ve her repoya hangi puanları verdiğini biliyoruz. "
                 "Şimdi bunları bir ürüne çevirelim.")

    dl = {s["slice"]: s for s in m["dilimler"]}
    dilim_satir = "".join(
        f"| {g.DILIM_AD[k]} | {S(dl[k]['n'])} | {O(dl[k]['silence_auc'], 3)} | {O(dl[k]['model_auc'], 3)} |" + chr(10)
        for k in g.DILIM_AD if k in dl)
    yas_satir = "".join(f"| {x['bant']} | {S(x['n'])} | {Y(x['pozitif'], 0)} | {O(x['auc'])} |" + chr(10) for x in yb)
    rozet_satir = "".join(f"| {g.ROZET_DUZ[veri.rozet_kodu(r['badge'])]} | {S(r['n'])} | {Y(r['rate'])} |" + chr(10)
                          for r in m["rozetler"])


###############################################################
# 06 RAG
###############################################################
def bolum_06():
    bolum_basi(6, "RAG", BOLUMLER[5][2])
    sw = veri.rag_denemeleri()
    KOM = J["rag_komsu"]
    yb = J["rag_yildiz_bandi"]
    OLC = veri.OLCUMLER
    r5 = lambda k: sw.loc[k, "r@5"]
    esik = O(RES["esik"], 2)
    t = st.tabs(["🔀  Pipeline", "🧠  Model", "🚫  Konu dışı", "🧭  Komşular"])
    with t[0]:
        baslik("ARAMA NASIL KURULDU", "README'den vektöre, vektörden cevaba")
        html(g.rag_mimarisi(GEN["csv_repo"], GEN["readme_repo"], OLC["govde_repo"][0], OLC["bas_karakter"][0],
                            OLC["govde_bitis"][0], OLC["vektor_boyut"][0], RES["esik"], RSK["sorgu"],
                            len(RES["negatif_sorgu"]), RES["negatif_kesilen"], r5("head900"), r5("qwen8b-YENI-bas+govde"),
                            uretilen=veri.sinav_uretimi(RSK["sorgu"])["uretilen"]))
    with t[1]:
        c1, c2 = st.columns(2)
        with c1:
            baslik("HANGİ MODEL", "recall@5: doğru repo ilk 5 sonuçta mı?")
            grafik(g.rag_modeller(sw))
        with c2:
            baslik("README BODY", "Yalnız README head mi, head + body mi?")
            grafik(g.rag_recall(sw))
        su = veri.sinav_uretimi(RSK["sorgu"])
        serit(f"Hangi model ve README'nin hangi kısmı doğru repoyu en üste getiriyor? Bunu ölçmek için {su['uretilen']} "
              f"soru ürettim, repo adını ele veren ya da boş kalan {su['elenen']} tanesini eledim ve {su['kalan']} soruyla "
              "sınav yaptım.",
              f"En büyük sıçramayı büyük model (8B) getirdi: {O(r5('qwen3-8b'), 3)}. README body'yi ayrı bir vektör olarak "
              f"eklemek bunu {O(r5('qwen8b-YENI-bas+govde'), 3)}'ye çıkardı.",
              "Canlı sistem 8B modelini kullanıyor ve her repo için head ile body'den hangisi soruya daha yakınsa onu alıyor.")
    with t[2]:
        grafik(g.esik_dagilim(RES))
        c1, c2 = st.columns(2)
        with c1:
            html('<table class="tablo"><tr><th>EŞİK</th><th>KESİLEN KONU DIŞI SORU</th><th>KESİLEN SINAV SORUSU</th>'
                 '<th>KESİLİP DE DOĞRU REPOSU İLK 10\'DA OLAN</th></tr>'
                 + "".join(f'<tr><td>{O(r["esik"], 2)}</td><td>{r["negatif"]} / {len(RES["negatif_top1"])}</td>'
                           f'<td>{r["sinav"]} / {len(RES["sinav_top1"])}</td><td>{r["sinav_dogru_ilk10"]}</td></tr>'
                           for r in RES["tablo"]) + "</table>")
        with c2:
            html('<table class="tablo"><tr><th>CANLI SİSTEMDE DENEME (13.09)</th><th>EN İYİ BENZERLİK</th><th>SONUÇ</th></tr>'
                 + "".join(f'<tr><td>{x["soru"]}</td><td>{O(x["en_iyi"])}</td><td>{x["sonuc"]}</td></tr>'
                           for x in RES["canli_test"]) + "</table>")
        serit("Vektör araması, konu tamamen alakasız olsa bile her soruya en yakın 10 repoyu getirir.",
              f"Konu dışı sorularda en yakın reponun benzerliği belirgin şekilde düşük. {esik} eşiği "
              f"{len(RES['negatif_top1'])} konu dışı sorudan {ekle(str(RES['negatif_kesilen']), 'unu')}, "
              f"{len(RES['sinav_top1'])} sınav sorusundan yalnızca {ekle(str(RES['sinav_kesilen']), 'unu')} kesiyor.",
              f"Eşik canlı sistemde çalışıyor: benzerlik {esik} altında kalırsa asistan “uygun repo bulamadım” diyor.")
    with t[3]:
        c1, c2 = st.columns(2)
        with c1:
            html(f'<table class="tablo"><tr><th>{KOM["repo"]} · EN YAKIN KOMŞULAR</th><th>BENZERLİK</th></tr>'
                 + "".join(f'<tr><td>{x["repo"]}</td><td>{O(x["benzerlik"], 3)}</td></tr>' for x in KOM["komsular"]) + "</table>")
        with c2:
            grafik(g.yildiz_bandi(yb))
        serit("Arama anlama göre mi çalışıyor, her repoda aynı derecede iyi mi?",
              f"Belgeleri markdown'a çeviren bir aracın en yakın komşuları da belge ve PDF dönüştürücüleri. 100 yıldızın "
              f"altındaki repolarda doğru repo {Y(yb[0]['r5_bas_govde'], 0)} ilk 5'te, 10 binin üstünde {Y(yb[-1]['r5_bas_govde'], 0)}.",
              "Ünlü repoların çok yakın benzeri var. Sınav tek doğru cevap kabul ettiği için bu grupta skor düşük görünüyor; "
              "kullanıcı için benzer bir alternatif çoğu zaman aynı işi görür.")
    gecis("RepoRadarAI", "Aynı sistemi şimdi canlı deneyelim.")

    SW_AD = {"head900": "bge-m3 (README'nin ilk 900 karakteri)", "head2000": "bge-m3 (README'nin ilk 2.000 karakteri)",
             "qwen3-0.6b": "Qwen3 0.6B", "qwen3-bas+govde": "Qwen3 0.6B + README body",
             "qwen3-8b": "Qwen3 8B", "qwen8b-YENI-bas+govde": "Qwen3 8B + README body (canlı)"}
    sat_sw = "".join(f"| {SW_AD[k]} | {O(r5(k), 3)} |" + chr(10) for k in SW_AD if k in sw.index)
    esik_satir = "".join(f"| {O(r['esik'], 2)} | {r['negatif']} / {len(RES['negatif_top1'])} | {r['sinav']} / "
                         f"{len(RES['sinav_top1'])} | {r['sinav_dogru_ilk10']} |" + chr(10) for r in RES["tablo"])
    ucak = dict(zip(RES["negatif_sorgu"], RES["negatif_top1"])).get("Cheap flights from Istanbul to Berlin")
    CT = {x["sorgu"]: x for x in RES["canli_test"]}
    dogru_kesilen = RES["sinav_kesilen_dogru_ilk10"]
    kesilen_metin = ("**bu sorulardan hiçbirinde doğru repo zaten ilk 10'da değildi.** Yani eşik kullanıcının görebileceği tek "
                     "bir doğru cevabı bile silmiyor." if dogru_kesilen == 0 else
                     f"bunlardan {dogru_kesilen} tanesinde doğru repo ilk 10'daydı; eşik o kadar doğru cevabı kullanıcıdan saklıyor.")


###############################################################
# 07 LIVE DEMO
###############################################################
def bolum_07():
    # Same header as bolum_basi, split in two columns so the QR sits beside the title (strip stays full width).
    html(serit_html + "</div>")
    sol, sag = st.columns([3, 2], vertical_alignment="center")
    with sol:
        html('<div class="kicker">Bölüm 7</div>')
        st.header("RepoRadarAI")
        teknoloji(BOLUMLER[6][2])
        baslik("SİSTEMİN MİMARİSİ", "Bir soru, cevaba dönüşene kadar nerelerden geçiyor?")
    with sag:
        # QR is generated once (segno) and verified by decoding it; the panel only shows the file.
        qr = veri.fig("QR_reporadarai.svg")
        if qr:
            html(f'<div class="qr-kutu"><div class="qr-kod">{qr.read_text(encoding="utf-8")}</div>'
                 '<div class="qr-b">reporadarai.vercel.app</div></div>')
    html(g.sohbet_mimarisi(esik=RES["esik"], repo_n=GEN["csv_repo"], govde_n=veri.OLCUMLER["govde_repo"][0]))
    serit("<b>Satırı model yazmıyor.</b> Link, rozet, yüzde ve yıldız SQL'de hazırlanıyor; model kopyalıyor, sayı uyduramıyor.",
          "<b>Sıra aktifliğe göre.</b> Hangi 10 repo: anlamsal benzerlik. Hangi sırayla: bugünkü aktiflik yüzdesi.",
          f"<b>Kod kapıları.</b> Benzerlik {O(RES['esik'], 2)} altındaysa “uygun repo yok”; arşivde olmayan repo adı ya da "
          "uydurma kurulum komutu geçen cevap gösterilmiyor.", basliklar=("SAYILAR", "SIRALAMA", "KAPILAR"))
    gecis("İlerideki Adımlar", "Bu sistemi bundan sonra nasıl büyüteceğim?")


###############################################################
# 08 NEXT STEPS
###############################################################
def bolum_08():
    bolum_basi(8, "İlerideki Adımlar", BOLUMLER[7][2])
    z = veri.zaman()
    adimlar = [
        ("1", "Repo kapsamını genişletmek",
         f"Bugün {S(GEN['csv_repo'])} yapay zekâ reposuyla çalışıyorum. Havuzu büyütmek, daha az bilinen ve yeni açılan "
         "repolar için de tahmin üretmemi sağlar."),
        ("2", "Repo etkinlik verisini derinleştirmek",
         "Etkinlik kaydını her repo için sınırlı sayıda sayfa okuyarak çektim ve en yoğun repolar bu sınıra takıldı. "
         "Daha derin bir geçmiş, bu repoları da eğitime almamı ve etkinlik feature'larını güçlendirmemi sağlar."),
        ("3", "Bot ve insan ayrımını modele eklemek",
         "Etkinlik kaydı, değişikliği kimin yaptığını da gösteriyor. Bot ve insan payını ayıran feature'lar, "
         "otomasyonun çalışmaya devam etmesi ile projenin gerçekten bakım görmesi arasındaki farkı modele öğretir."),
        ("4", "Web ve mobil uygulama olarak yayınlamak",
         "Bugün sistem tek bir bilgisayarda çalışıyor ve aynı anda 4 soruyu cevaplıyor. Donanım kısıtından kurtulduğumda, "
         "veritabanı yeterince genişlediğinde ve tahminlerim gerçek sonuçlarla doğrulandığında uygulamayı yayınlamayı planlıyorum. "
         f"Bugünkü tahminlerin ilk gerçek sınavı {z['pencere_sonu']:%d.%m.%Y} tarihinde."),
    ]
    for satir in (adimlar[:2], adimlar[2:]):
        for c, (no, bas, metin) in zip(st.columns(2), satir):
            with c:
                html(f'<div class="story"><div class="story-icon">{no}</div><h4>{bas}</h4><p>{metin}</p></div>')
    # web + mobile concept designs side by side, fixed width so they stay modest
    for c, ad in zip(st.columns(2, vertical_alignment="center"), ("ileride_web.jpeg", "ileride_mobil.jpeg")):
        p = veri.fig(ad)
        if p:
            with c:
                st.image(str(p), width=440)


[bolum_01, bolum_02, bolum_03, bolum_04, bolum_05, bolum_06, bolum_07, bolum_08][aktif]()
