@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM ===========================================================
REM  haftalik.bat  ^|  HAFTALIK TAZELEME - Python tarafi
REM ===========================================================
REM  n8n haftalik iki akis kosturur:
REM    DS21meta0001    pazartesi 02:00 - CSV metadata (7 kolon)
REM    DS21tarihce0001 15 dk'da bir  - commit+push tazeleme (7 gun esigi)
REM  Bu betik ikisinin ciktisini Python tarafinda ISLER.
REM  ⚠️ Her ikisi de SUNUMA (15.09) KADAR PASIF.
REM
REM  NEDEN TEK DOSYA
REM  Sira BILGISI koda gomulu olmali, dokumana degil. Ozellikle
REM  tarihce.py -> score_today.py sirasi: ilki kapida kalirsa
REM  ikincisi HIC kosmamali, yoksa yeni veri + eski skor karisimi
REM  olusur ve bunu kimse fark etmez.
REM
REM  NE YAPMAZ
REM  * modeli YENIDEN EGITMEZ (model.pkl donuk, sozlesmesiyle okunur)
REM  * DS21_RAG.py cagirmaz -> TRUNCATE yok -> chatbot HIC kesilmez
REM    (README tazelemesi AYLIK ve AYRI bir zincir; orada TRUNCATE var)
REM
REM  KULLANIM
REM    haftalik.bat            normal kosu
REM    haftalik.bat --kuru     tarihce.py kuru kosu, HICBIR SEY yazilmaz
REM ===========================================================

set "KOK=%~dp0"
set "PY=%USERPROFILE%\anaconda3\python.exe"
set "LOG=%KOK%outputs\haftalik.log"
set "KURU="
if /I "%~1"=="--kuru" set "KURU=1"

call :ZAMAN
echo. >> "%LOG%"
echo ============================================================ >> "%LOG%"
echo [%ZA%] HAFTALIK TAZELEME BASLADI %KURU% >> "%LOG%"
echo ============================================================ >> "%LOG%"
echo [%ZA%] haftalik tazeleme basladi

REM ---------------------------------------------------------------
REM  0 - ON KONTROL: pgvector ayakta mi?
REM      Ayakta degilse score_today --write-db sonunda patlardi,
REM      ama o ana kadar tarihce.py bosa kosmus olurdu.
REM ---------------------------------------------------------------
docker exec ds21-pgvector pg_isready -U ds21 -d repoquest >nul 2>&1
if errorlevel 1 (
  set "HATA=pgvector yanit vermiyor - 'docker start ds21-pgvector' deneyin"
  goto :HATA
)
call :ZAMAN & echo [%ZA%]   on kontrol: pgvector ayakta >> "%LOG%"

REM ---------------------------------------------------------------
REM  0b - ON KONTROL: cekim SU AN kosuyor mu?
REM       _kilit.json 20 dakikadan taze ise bir tur devam ediyordur.
REM       tarihce.py'in gerileme kapisi veri kaybini zaten onler,
REM       ama yarim turun ortasinda birlestirmenin anlami yok.
REM ---------------------------------------------------------------
"%PY%" -c "import os,sys,time; y=r'C:\Miuul\ds21final\tarihce\_kilit.json'; sys.exit(1 if os.path.exists(y) and time.time()-os.path.getmtime(y)<1200 else 0)"
if errorlevel 1 (
  set "HATA=cekim su an kosuyor (_kilit.json taze) - tur bitince tekrar deneyin"
  goto :HATA
)
call :ZAMAN & echo [%ZA%]   on kontrol: cekim kosmuyor >> "%LOG%"

REM ---------------------------------------------------------------
REM  0c - refresh_csv.py : n8n'in cektigi metadata'yi CSV'ye isle
REM       AUDIT 2026-09-11 | zincire EKLENDI. DS21meta0001 artik haftalik
REM       zamanli (pazartesi 02:00); bu adim onun ciktisini CSV'ye uygular.
REM
REM       NE TAZELENIR: 7 kolon API'den (pushed_at, updated_at, stars, forks,
REM       open_issues, size_kb, archived) + 6 turetilmis yeniden hesaplanir.
REM
REM       🔑 BUNLARIN HICBIRI MODELE GIRMEZ - olculdu: 13 kolonun 9'u YASAKLI,
REM       2'si etiket/cerceve, 2'si kullanilmiyor. Model 59 ozelligini cm_*/pa_*
REM       gecmisinden aliyor. Tazelemenin model ciktisina etkisi olculdu:
REM       6.922 repoda medyan fark 0,00000, rozet degisen 12 repo (%%0,17).
REM
REM       🚨 AMA: SNAPSHOT = max(updated_at) ve CUTOFF_T = SNAPSHOT - 182.
REM       Yani tazeleme T'YI OYNATABILIR. Bu zincir modeli YENIDEN EGITMEDIGI
REM       icin sorun degil - score_today, model.pkl'in DONUK cutoff_T'sini
REM       okur. Ama DS21_Final_v2.py bir dahaki kosuldugunda T degisebilir;
REM       oradaki uc assert (T pazar · H%%7 · pencere sonu pazar) hizalama
REM       bozulursa ZINCIRI DURDURUR, sessizce sizdirmaz.
REM       Kok neden kaydi: 07.09 tazelemesi T'yi pazardan carsambaya kaydirdi
REM       ve 4 gunluk ileriye-bakis sizintisi dogurdu (2.740 repo).
REM
REM       Kapi kalirsa zincir DURUR: bozuk CSV ile devam etmenin anlami yok.
REM ---------------------------------------------------------------
call :ZAMAN & echo [%ZA%] 0c/4 refresh_csv.py >> "%LOG%"
echo [%ZA%] 0c/4 refresh_csv.py
if defined KURU (
  "%PY%" "%KOK%refresh_csv.py" >> "%LOG%" 2>&1
) else (
  "%PY%" "%KOK%refresh_csv.py" --apply >> "%LOG%" 2>&1
)
if errorlevel 1 (
  set "HATA=refresh_csv.py basarisiz (kapi kaldi) - CSV DEGISMEDI, zincir durdu"
  goto :HATA
)

REM ---------------------------------------------------------------
REM  1 - tarihce.py : JSONL -> parquet + fetch_log
REM      11 kapinin HEPSI gecmeden --apply yazmaz. Kapi kalirsa
REM      exit 1 doner ve burada duruyoruz.
REM ---------------------------------------------------------------
call :ZAMAN & echo [%ZA%] 1/4 tarihce.py >> "%LOG%"
echo [%ZA%] 1/4 tarihce.py
if defined KURU (
  "%PY%" "%KOK%tarihce.py" >> "%LOG%" 2>&1
) else (
  "%PY%" "%KOK%tarihce.py" --apply >> "%LOG%" 2>&1
)
if errorlevel 1 (
  set "HATA=tarihce.py basarisiz (kapi kaldi ya da hata) - LOG'a bakin"
  goto :HATA
)

if defined KURU (
  call :ZAMAN & echo [%ZA%] KURU KOSU - score_today ve assert atlandi >> "%LOG%"
  echo [%ZA%] kuru kosu bitti, hicbir sey yazilmadi
  goto :SON
)

REM ---------------------------------------------------------------
REM  2 - score_today.py --write-db
REM      Modeli EGITMEZ. model.pkl'i acar, sozlesmesini okur,
REM      ozellikleri BUGUN kesimiyle yeniden uretir, tahmin eder.
REM      Icindeki KAPI: cutoff=CUTOFF_T ile features.parquet birebir
REM      yeniden uretilemezse skorlamayi REDDEDER.
REM      pgvector'e UPDATE yazar - TRUNCATE YOK, kor pencere YOK.
REM ---------------------------------------------------------------
call :ZAMAN & echo [%ZA%] 2/4 score_today.py --write-db >> "%LOG%"
echo [%ZA%] 2/4 score_today.py --write-db
"%PY%" "%KOK%score_today.py" --write-db >> "%LOG%" 2>&1
if errorlevel 1 (
  set "HATA=score_today.py basarisiz - skorlar YAZILMADI, LOG'a bakin"
  goto :HATA
)

REM ---------------------------------------------------------------
REM  3 - ASSERT : skorlar gercekten indi mi?
REM      badge_today NULL kalirsa arama SQL'indeki
REM         WHERE badge_today <> 'VERI_YOK'
REM      uc degerli mantik yuzunden HER SATIRI eler ve chatbot
REM      HATA VERMEDEN sifir sonuc dondurur. Sessiz ariza.
REM      Bu yuzden assert betigin parcasi, dokumanin degil.
REM ---------------------------------------------------------------
call :ZAMAN & echo [%ZA%] 3/4 assert: badge_today NULL sayaci >> "%LOG%"
echo [%ZA%] 3/4 assert
"%PY%" -c "import os,psycopg2,sys; c=psycopg2.connect(host='localhost',port=5433,user='ds21',password=os.environ.get('DS21_PG_PASSWORD',''),dbname='repoquest'); k=c.cursor(); k.execute(\"SELECT count(*) FILTER (WHERE badge_today IS NULL), count(*), max(scored_at) FROM repos\"); n,t,d=k.fetchone(); print(f'  NULL={n}  toplam={t}  scored_at={d}'); sys.exit(1 if n>0 or t==0 else 0)" >> "%LOG%" 2>&1
if errorlevel 1 (
  set "HATA=ASSERT KALDI - badge_today NULL var, chatbot SESSIZCE bos cevap verir"
  goto :HATA
)

:SON
call :ZAMAN
echo [%ZA%] TAMAM - haftalik tazeleme bitti >> "%LOG%"
echo [%ZA%] TAMAM
endlocal & exit /b 0

:HATA
call :ZAMAN
echo [%ZA%] *** HATA: %HATA% >> "%LOG%"
echo [%ZA%] *** HATA: %HATA%
echo [%ZA%] *** log: %LOG%
endlocal & exit /b 1

:ZAMAN
for /f "tokens=1-6 delims=/:. " %%a in ("%date% %time%") do set "ZA=%%c-%%b-%%a %%d:%%e"
exit /b 0
