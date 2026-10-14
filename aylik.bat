@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM ===========================================================
REM  aylik.bat  ^|  AYLIK README TAZELEME - Python tarafi
REM ===========================================================
REM  n8n DS21readme0001 ayin 1'inde 03:00'te kulliyatin TAMAMINI
REM  C:\Miuul\ds21final\readme\ altina ceker. Bu betik ondan
REM  SONRAKI zinciri kosturur.
REM
REM  ⚠️ haftalik.bat'tan FARKI: BU ZINCIR TRUNCATE PENCERESI ACAR.
REM  DS21_RAG.py `repos` tablosuna TRUNCATE atip 12 sutun yazar;
REM  `badge_today` NULL kalir ve arama SQL'indeki
REM      WHERE badge_today <> 'VERI_YOK'
REM  uc degerli mantik yuzunden HER SATIRI eler -> chatbot HATA
REM  VERMEDEN sifir sonuc dondurur. Pencere, DS21_RAG basladigi
REM  andan score_today bitene kadar ACIKTIR.
REM  Bu yuzden score_today KOSULSUZ kosar: DS21_RAG cokse bile
REM  pencereyi kapatmayi DENER. (Bu oturumda `&&` zinciri kirilip
REM  pencere iki kez acik kaldi - ders koda yazildi.)
REM
REM  KULLANIM
REM    aylik.bat            normal kosu
REM    aylik.bat --kuru     swap dogrulamasi, HICBIR SEY yazilmaz
REM ===========================================================

set "KOK=%~dp0"
set "PY=%USERPROFILE%\anaconda3\python.exe"
set "LOG=%KOK%outputs\aylik.log"
set "HAZIRLIK=C:\Miuul\ds21final\readme"
set "KURU="
if /I "%~1"=="--kuru" set "KURU=1"

REM Gomme modeli - YOKSA bge-m3'e duser ve 4096-vs-1024 ile patlar.
set "DS21_EMB_MODEL=qwen3-embedding:8b"
set "DS21_EMB_DIM=4096"
set "DS21_SUFFIX=_qwen8b"

call :ZAMAN
echo. >> "%LOG%"
echo ============================================================ >> "%LOG%"
echo [%ZA%] AYLIK README TAZELEME BASLADI %KURU% >> "%LOG%"
echo ============================================================ >> "%LOG%"
echo [%ZA%] aylik readme tazeleme basladi

REM ---------------------------------------------------------------
REM  0 - ON KONTROL: pgvector ayakta mi?
REM ---------------------------------------------------------------
docker exec ds21-pgvector pg_isready -U ds21 -d repoquest >nul 2>&1
if errorlevel 1 (
  set "HATA=pgvector yanit vermiyor - 'docker start ds21-pgvector' deneyin"
  goto :HATA
)

REM ---------------------------------------------------------------
REM  0b - ON KONTROL: n8n cekimi gercekten yapti mi?
REM       Hazirlik klasoru bos ya da cok az dosya varsa zincire
REM       girmenin anlami yok - swap eskiyi yeniyle DEGISTIRIR ve
REM       eksik bir kume ile degistirmek VERI KAYBIDIR.
REM       Esik 20.000: kulliyat 24.664, README'si olan ~23.875.
REM ---------------------------------------------------------------
"%PY%" -c "import os,sys; d=r'%HAZIRLIK%'; n=len([f for f in os.listdir(d) if f.endswith('.md')]) if os.path.isdir(d) else 0; print(f'  hazirlik klasoru: {n:,} .md'); sys.exit(0 if n>=20000 else 1)" >> "%LOG%" 2>&1
if errorlevel 1 (
  set "HATA=hazirlik klasorunde 20.000'den az README var - n8n turu bitmemis ya da basarisiz. SWAP YAPILMADI."
  goto :HATA
)
call :ZAMAN & echo [%ZA%]   on kontrol: README'ler yerinde >> "%LOG%"

REM ---------------------------------------------------------------
REM  1 - swap_readmes.py : DOGRULA sonra TAKAS
REM      Eskiyi Data/readme_raw_snap<tarih>/ olarak yeniden adlandirir,
REM      yeniyi Data/readme_raw/ yapar. Hicbir anda "README yok" olmaz.
REM ---------------------------------------------------------------
call :ZAMAN & echo [%ZA%] 1/4 swap_readmes.py >> "%LOG%"
echo [%ZA%] 1/4 swap_readmes.py
if defined KURU (
  "%PY%" "%KOK%swap_readmes.py" >> "%LOG%" 2>&1
  call :ZAMAN & echo [%ZA%] KURU KOSU - zincirin geri kalani atlandi >> "%LOG%"
  echo [%ZA%] kuru kosu bitti
  goto :SON
)
"%PY%" "%KOK%swap_readmes.py" --apply >> "%LOG%" 2>&1
if errorlevel 1 (
  set "HATA=swap_readmes.py basarisiz - README'ler DEGISMEDI, eski set duruyor"
  goto :HATA
)

REM ---------------------------------------------------------------
REM  2 - DS21_RAG.py : yeni metin -> gomme -> pgvector
REM      Gomme ONBELLEKLI: yalniz METNI DEGISEN repolar yeniden
REM      gomulur. Degismeyen README icin hash ayni -> atlanir.
REM      🚨 BU ADIM TRUNCATE PENCERESINI ACAR.
REM ---------------------------------------------------------------
call :ZAMAN & echo [%ZA%] 2/4 DS21_RAG.py  *** TRUNCATE PENCERESI ACILIYOR *** >> "%LOG%"
echo [%ZA%] 2/4 DS21_RAG.py  (pencere aciliyor)
"%PY%" "%KOK%DS21_RAG.py" >> "%LOG%" 2>&1
set "RAG_KODU=%ERRORLEVEL%"
if not "%RAG_KODU%"=="0" (
  echo [%ZA%]   UYARI: DS21_RAG.py cikis kodu %RAG_KODU% - score_today YINE DE kosacak >> "%LOG%"
  echo [%ZA%]   UYARI: DS21_RAG.py hata verdi, pencereyi kapatmayi deniyorum
)

REM ---------------------------------------------------------------
REM  3 - score_today.py --write-db  : KOSULSUZ
REM      DS21_RAG cokse bile kosar. Sebep: TRUNCATE + INSERT commit
REM      edilmis olabilir ve pencere ACIK kalmis olabilir; onu
REM      kapatmak, RAG'in basarisindan ONCELIKLIDIR.
REM ---------------------------------------------------------------
call :ZAMAN & echo [%ZA%] 3/4 score_today.py --write-db  (KOSULSUZ) >> "%LOG%"
echo [%ZA%] 3/4 score_today.py --write-db
"%PY%" "%KOK%score_today.py" --write-db >> "%LOG%" 2>&1
set "SKOR_KODU=%ERRORLEVEL%"

REM ---------------------------------------------------------------
REM  4 - ASSERT : pencere gercekten kapandi mi?
REM ---------------------------------------------------------------
call :ZAMAN & echo [%ZA%] 4/4 assert: badge_today NULL sayaci >> "%LOG%"
echo [%ZA%] 4/4 assert
"%PY%" -c "import os,psycopg2,sys; c=psycopg2.connect(host='localhost',port=5433,user='ds21',password=os.environ.get('DS21_PG_PASSWORD',''),dbname='repoquest'); k=c.cursor(); k.execute(\"SELECT count(*) FILTER (WHERE badge_today IS NULL), count(*), max(scored_at) FROM repos\"); n,t,d=k.fetchone(); print(f'  NULL={n}  toplam={t}  scored_at={d}'); sys.exit(1 if n>0 or t==0 else 0)" >> "%LOG%" 2>&1
if errorlevel 1 (
  set "HATA=ASSERT KALDI - badge_today NULL var, CHATBOT SESSIZCE BOS CEVAP VERIYOR. Elle: python score_today.py --write-db"
  goto :HATA
)

REM RAG ya da skorlama hata verdiyse pencere kapali olsa bile bildir.
if not "%RAG_KODU%"=="0" (
  set "HATA=pencere kapandi AMA DS21_RAG.py hata verdi (kod %RAG_KODU%) - gomme eksik olabilir"
  goto :HATA
)
if not "%SKOR_KODU%"=="0" (
  set "HATA=pencere kapandi AMA score_today.py hata verdi (kod %SKOR_KODU%)"
  goto :HATA
)

:SON
call :ZAMAN
echo [%ZA%] TAMAM - aylik readme tazeleme bitti >> "%LOG%"
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
