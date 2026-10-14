// DS21 GECIT - dis dunya ile n8n arasindaki DAR UC.
//
// NEDEN VAR: Tailscale Funnel bir PORTU yayinlar, bir YOLU degil. n8n'in 5678
// portunu acmak ayni konteynerdeki DigiSchafak, Janus ve VC akislarinin
// webhook'larini da yayinlar. Bu dosya disariya bakan tek yuzey olur; n8n
// 127.0.0.1'de kapali kalir.
//
// OLCULEN SAYILAR (11.09, RTX 4060 Ti, qwen3:14b, OLLAMA_NUM_PARALLEL=4):
//   tek istek                    42 sn
//   4 es zamanli        54-61 sn, toplam 61 sn
//   8 es zamanli   51-58 | 95-99 sn, toplam 99 sn
//   -> es zamanlilik 4'te doyuyor; 5. istek beklemeli, kuyruga girmeli.
// Bekleme tahmini bu olcumden turetiliyor, tahminden degil.
//
// Bagimlilik YOK - `node gecit.js` ile calisir.

'use strict';
const http = require('http');
const crypto = require('crypto');

const AYAR = {
  port:        Number(process.env.GECIT_PORT || 8787),
  arayuz:      process.env.GECIT_HOST || '127.0.0.1',
  n8n:         process.env.GECIT_N8N || 'http://127.0.0.1:5678/webhook/ds21chat0002hook/chat',
  kod:         process.env.GECIT_KOD || '',      // bos = kapi yok
  esZamanli:   Number(process.env.GECIT_ES_ZAMANLI || 4),    // = OLLAMA_NUM_PARALLEL
  kuyrukMax:   Number(process.env.GECIT_KUYRUK || 12),
  // Ziyaretci basina TOPLAM soru. 0 = SINIRSIZ (Sarp'in karari 11.09: "istedikleri
  // kadar devam edebilsinler"). Yerine kisi basina AYNI ANDA bir soru kurali
  // kondu: kimseyi durdurmaz ama bir kisinin kuyrugu tek basina doldurmasini
  // engeller - 4 yuvali tek GPU'da kuyrugun tek gercek kiti bu.
  soruSiniri:  Number(process.env.GECIT_SORU_SINIRI || 0),
  aktifSiniri: Number(process.env.GECIT_AKTIF_SINIRI || 1),
  // Olculen dalga suresi. 6 soruluk kuyruk testinde 1. dalga 50-60 sn, 2. dalga
  // 65 sn surdu. 65 secildi ki tahmin HER ZAMAN gercegin biraz ustunde kalsin -
  // erken gelen cevap iyi surpriz, gec gelen sikayet.
  bekleme:     Number(process.env.GECIT_BEKLEME_SN || 65),
  zamanAsimi:  Number(process.env.GECIT_ZAMAN_ASIMI_SN || 180) * 1000,
  // Sayfa bir cevabi 30 dk bekliyor (index.html EN_FAZLA). Is kaydi ondan UZUN
  // yasamali: temizlik kuyruktaki isi de siler, silinen soru sessizce duser.
  isOmru:      Number(process.env.GECIT_IS_OMRU_SN || 2100) * 1000,
  soruMax:     Number(process.env.GECIT_SORU_MAX || 300),    // karakter
  kodDeneme:   Number(process.env.GECIT_KOD_DENEME || 8),    // hatali kod hakki
  koken:       process.env.GECIT_KOKEN || '*',               // CORS
};

// --------------------------------------------------------------- DURUM
const isler = new Map();      // id -> {durum, soru, cevap, hata, sira, giris, bitis}
const kuyruk = [];            // bekleyen is id'leri
const ziyaretciSayaci = new Map();  // parmakizi -> kac soru sordu
const yanlisKod = new Map();        // parmakizi -> kac kez hatali kod denedi
const aktifSayaci = new Map();      // parmakizi -> su an kacta kac isi var

// istekId -> {isId, t}. Sayfa bir soruyu yeniden gonderdiginde AYNI ise
// baglanmasi icin. Olculdu (12.09): Vercel'in uc sunucusu alti istekte bir
// `DNS_HOSTNAME_EMPTY` ile dusuyor ve sayfa "Sunucuya ulasilamadi" gosteriyor.
// O vakada istek gecide HIC ulasmiyor (sayac kaniti: toplam artmadi), ama
// yeniden gondermeyi GUVENLI kilmak icin varsayima dayanamayiz: cevabin
// kayboldugu ama isin kuyruga girdigi bir durumda korumasiz tekrar, ayni
// soruyu IKI KEZ kosturur ve ikincisi "onceki sorun hala sirada" hatasina
// duser. Bu harita o sinifi kapatiyor - tekrar, ilk isin durumunu dondurur.
const istekKimlikleri = new Map();
let calisan = 0;

const sayac = { toplam: 0, bitti: 0, hata: 0, reddedilen: 0, zamanAsimi: 0 };

// --------------------------------------------------------------- YARDIMCI
function id() { return crypto.randomBytes(8).toString('hex'); }

function parmakizi(req) {
  // Funnel arkasinda gercek IP X-Forwarded-For'da gelir. Kimlik degil, yalnizca
  // hiz siniri icin kullaniliyor; yanlis olmasi kimseyi disari atmaz, yalnizca
  // iki kisiyi ayni kovaya koyar.
  const xff = String(req.headers['x-forwarded-for'] || '').split(',')[0].trim();
  return xff || req.socket.remoteAddress || 'bilinmiyor';
}

// Cevap istemciye markdown olarak gidiyor ve sayfada render edilecek.
// Model urettigi icin HTML gomulmesi ihtimali sifir degil (istem enjeksiyonu).
// Kural: HTML etiketi YOK, ve yalnizca github.com'a giden baglantilar kalir.
function temizle(metin) {
  let s = String(metin == null ? '' : metin);
  s = s.replace(/<[^>]*>/g, '');                      // HTML etiketi
  s = s.replace(/!\[[^\]]*\]\([^)]*\)/g, '');         // gorsel gomme
  s = s.replace(/\[([^\]]*)\]\(([^)]+)\)/g, (t, ad, url) => {
    return /^https:\/\/github\.com\/[A-Za-z0-9._/-]+$/.test(url.trim()) ? `[${ad}](${url.trim()})` : ad;
  });
  s = s.replace(/(javascript|data|vbscript):/gi, '');
  return s.trim();
}

// sira = 0 ise is ZATEN kosuyor -> tek dalga. sira >= 1 ise kuyrukta, onunde
// hem kosanlar hem kendinden onceki bekleyenler var.
//
// OLCULEN KUSUR (11.09): burasi once kosan isi de "onunde" sayiyordu, o yuzden
// 4. istek kosmaya baslamisken 110 sn diyordu ve 55 sn'de bitiyordu. Kapi:
// kosan isin tahmini TEK dalgadir, bu bir varsayim degil tanim.
function tahminiSaniye(sira) {
  if (!sira) return AYAR.bekleme;
  const onunde = sira - 1 + calisan;
  return Math.max(AYAR.bekleme, Math.ceil((onunde + 1) / AYAR.esZamanli) * AYAR.bekleme);
}

function json(res, kod, govde) {
  const g = Buffer.from(JSON.stringify(govde), 'utf8');
  res.writeHead(kod, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': g.length,
    'Access-Control-Allow-Origin': AYAR.koken,
    'Access-Control-Allow-Headers': 'Content-Type',
    'Cache-Control': 'no-store',
  });
  res.end(g);
}

function govdeOku(req) {
  return new Promise((coz, at) => {
    let veri = '', asti = false;
    req.on('data', (p) => {
      veri += p;
      if (veri.length > 8192) { asti = true; req.destroy(); }   // govde tavani
    });
    req.on('end', () => asti ? at(new Error('govde cok buyuk')) : coz(veri));
    req.on('error', at);
  });
}

// ------------------------------------------------------- OTURUM DONDURME
// NEDEN VAR: n8n'in Bellek dugumu yalnizca metni degil, repo_ara'nin DONDURDUGU
// 20 satirlik ham JSON'u da saklar. Olculdu (12.09): bellekte boyle bir yuk
// varken model yeni bir ihtiyac icin 3 turun 3'unde de HIC arama yapmadi ve
// eski listeyi uc basliga bolup uydurma yetenek tarifleri yazdi. Ayni istek
// bos bellekle sorulunca 2-4 arama yapti. Yani kusur modelin ayristirmasinda
// degil, bastirici baglamda. Istemle cozulmedi (dort deneme), bu yuzden
// KODDA cozuyoruz: mesaj acikca YENI BIR IHTIYAC tarif ediyorsa o turdan
// itibaren yeni bir oturum kimligi kullaniriz, boylece eski yuk baglamdan cikar.
const oturumEki = new Map();   // taban kimlik -> kacinci konu

// Turkce harfleri sadelestirir; ziyaretci "bunlardan" da yazar "bunlardan" da.
function sadelestir(t) {
  return String(t).toLowerCase()
    .replace(/[ıİ]/g, 'i').replace(/[şŞ]/g, 's').replace(/[ğĞ]/g, 'g')
    .replace(/[üÜ]/g, 'u').replace(/[öÖ]/g, 'o').replace(/[çÇ]/g, 'c');
}

// Onceki listeye gonderme yapan sozcukler. Biri bile varsa DONDURMEYIZ:
// yanlis dondurme, devam sorusunu baglamsiz birakir - sunumda en kotu hata.
const GONDERME = /\b(bunlar|bunlardan|bunlarin|bunu|bunun|sunu|sunlar|hangisi|hangileri|ilkini|ilki|birincisi|ikincisi|ucuncusu|sonuncusu|yukaridaki|ustteki|listedeki|onlarin|onlari|peki|ayni|digerleri)\b/;

// Yeni bir is tarif eden kaliplar. Dondurme icin BUNLARDAN biri sart.
const IHTIYAC = /(istiyorum|istiyoruz|lazim|ariyorum|ariyoruz|ihtiyacim|ihtiyacimiz|onerir misin|bul bana|kurmak|calistirmak|cevirmek|dokmek|bir arac|bir akis|bir kutuphane|bir model)/;

function etkinOturum(taban, soru) {
  const t = sadelestir(soru);
  const yeniKonu = IHTIYAC.test(t) && !GONDERME.test(t);
  if (yeniKonu) oturumEki.set(taban, (oturumEki.get(taban) || 0) + 1);
  // Sinirsiz buyumesin: eski tabanlari at.
  if (oturumEki.size > 500) {
    for (const k of oturumEki.keys()) { oturumEki.delete(k); if (oturumEki.size <= 400) break; }
  }
  const ek = oturumEki.get(taban) || 0;
  return ek ? (taban + '-k' + ek) : taban;
}

// --------------------------------------------------------------- N8N CAGRISI
function n8neSor(is) {
  return new Promise((coz) => {
    const govde = Buffer.from(JSON.stringify({
      action: 'sendMessage',
      // OTURUM: sayfa kendi kimligini gonderir, boylece n8n'deki Bellek dugumu
      // devreye girer ve "bunlardan hangisi Python?" gibi devam sorulari
      // calisir. Once her soru icin YENI kimlik uretiliyordu; bellek dugumu
      // kurulu oldugu halde HIC kullanilmiyordu.
      sessionId: is.oturum || ('gecit-' + is.id),
      chatInput: is.soru,
    }), 'utf8');

    const u = new URL(AYAR.n8n);
    const istek = http.request({
      hostname: u.hostname, port: u.port, path: u.pathname, method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': govde.length },
      timeout: AYAR.zamanAsimi,
    }, (yanit) => {
      let v = '';
      yanit.on('data', (p) => { v += p; });
      yanit.on('end', () => {
        if (yanit.statusCode !== 200) return coz({ ok: false, hata: 'n8n HTTP ' + yanit.statusCode });
        try {
          const j = JSON.parse(v);
          const cevap = temizle(j.output);
          if (!cevap) return coz({ ok: false, hata: 'bos cevap' });
          coz({ ok: true, cevap });
        } catch (e) { coz({ ok: false, hata: 'cozumlenemeyen yanit' }); }
      });
    });
    istek.on('timeout', () => { istek.destroy(); sayac.zamanAsimi++; coz({ ok: false, hata: 'zaman asimi' }); });
    istek.on('error', (e) => coz({ ok: false, hata: 'baglanti: ' + e.code }));
    istek.write(govde);
    istek.end();
  });
}

// --------------------------------------------------------------- KUYRUK
function kuyruguYurut() {
  while (calisan < AYAR.esZamanli && kuyruk.length) {
    const isId = kuyruk.shift();
    const is = isler.get(isId);
    if (!is || is.durum !== 'bekliyor') continue;
    is.durum = 'calisiyor';
    is.baslangic = Date.now();
    calisan++;
    n8neSor(is).then((s) => {
      if (s.ok) { is.durum = 'bitti'; is.cevap = s.cevap; sayac.bitti++; }
      else { is.durum = 'hata'; is.hata = s.hata; sayac.hata++; }
      is.bitis = Date.now();
      calisan--;
      // Aktif sayaci is BITTIGINDE duser - kuyruga girdiginde degil. Boylece
      // "ayni anda bir soru" kurali gercekten ayni ani olcer.
      const k = (aktifSayaci.get(is.pi) || 1) - 1;
      if (k > 0) aktifSayaci.set(is.pi, k); else aktifSayaci.delete(is.pi);
      kuyruguYurut();
    });
  }
}

function siraNo(isId) {
  const i = kuyruk.indexOf(isId);
  return i < 0 ? 0 : i + 1;
}

// eski isleri temizle - bellek sinirsiz buyumesin
setInterval(() => {
  const simdi = Date.now();
  for (const [k, v] of isler) if (simdi - v.giris > AYAR.isOmru) isler.delete(k);
}, 60000).unref();

// --------------------------------------------------------------- SUNUCU
const sunucu = http.createServer(async (req, res) => {
  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': AYAR.koken,
      'Access-Control-Allow-Headers': 'Content-Type',
      'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
    });
    return res.end();
  }

  const yol = new URL(req.url, 'http://x').pathname;

  if (yol === '/saglik') {
    return json(res, 200, {
      ok: true, calisan, kuyruk: kuyruk.length,
      esZamanli: AYAR.esZamanli, kuyrukMax: AYAR.kuyrukMax, sayac,
    });
  }

  if (yol === '/sor' && req.method === 'POST') {
    let g;
    try { g = JSON.parse(await govdeOku(req)); }
    catch (e) { sayac.reddedilen++; return json(res, 400, { hata: 'govde okunamadi' }); }

    // ---- AYNI ISTEGIN TEKRARI (12.09) ----------------------------------
    // Sayfa gecici bir aksaklikta soruyu bir kez yeniden gonderiyor ve ayni
    // `istekId` ile geliyor. O kimlik zaten bir ise baglandiysa YENI IS
    // ACMIYORUZ, var olanin durumunu donduruyoruz: hiz sayaclari artmaz,
    // "ayni anda bir soru" kurali tetiklenmez, kullanici tek cevap alir.
    const istekId = String(g.istekId || '').replace(/[^A-Za-z0-9_-]/g, '').slice(0, 40);
    if (istekId && istekKimlikleri.has(istekId)) {
      const eski = istekKimlikleri.get(istekId).isId;
      if (isler.has(eski)) {
        const s0 = siraNo(eski);
        return json(res, 202, {
          id: eski,
          durum: isler.get(eski).durum,
          sira: s0,
          onunde: (s0 ? s0 - 1 : 0) + calisan,
          tahminiSaniye: tahminiSaniye(s0),
          tekrar: true,
        });
      }
      istekKimlikleri.delete(istekId);
    }

    // ERISIM KODU ISTEGE BAGLI.
    // GECIT_KOD bos ise kapi YOKTUR - Sarp'in karari 11.09: "koda gerek yok".
    // Kod tanimliysa yanlis denemeler SAYILIR: ilk surumde kod kontrolu
    // ziyaretci sayacindan once geliyordu ve yanlis kod hicbir sayaci
    // artirmiyordu, yani kod sinirsiz denenebiliyordu.
    //
    // Kod olmadan da uc korumasiz degil: kisi basina AYNI ANDA bir soru
    // kurali, cevabi beklemeden yenisini gondermeyi imkansiz kilar - yani
    // kisi basina ~40-60 sn'de bir soru. Kuyruk tavani da 12'de duruyor.
    if (AYAR.kod) {
      const pi0 = parmakizi(req);
      const yanlis = yanlisKod.get(pi0) || 0;
      if (yanlis >= AYAR.kodDeneme) {
        sayac.reddedilen++;
        return json(res, 429, { hata: 'cok fazla hatali kod denemesi' });
      }
      if (String(g.kod || '') !== AYAR.kod) {
        yanlisKod.set(pi0, yanlis + 1);
        sayac.reddedilen++;
        return json(res, 403, { hata: 'erisim kodu gecersiz', kalanDeneme: AYAR.kodDeneme - (yanlis + 1) });
      }
    }
    const soru = String(g.soru || '').trim();
    if (!soru) { sayac.reddedilen++; return json(res, 400, { hata: 'soru bos' }); }
    if (soru.length > AYAR.soruMax) {
      sayac.reddedilen++;
      return json(res, 400, { hata: `soru en fazla ${AYAR.soruMax} karakter` });
    }

    const pi = parmakizi(req);
    const sorulan = ziyaretciSayaci.get(pi) || 0;
    if (AYAR.soruSiniri > 0 && sorulan >= AYAR.soruSiniri) {
      sayac.reddedilen++;
      return json(res, 429, { hata: `bu baglantidan en fazla ${AYAR.soruSiniri} soru sorulabilir` });
    }
    if ((aktifSayaci.get(pi) || 0) >= AYAR.aktifSiniri) {
      sayac.reddedilen++;
      return json(res, 429, {
        hata: 'onceki sorun hala siradaysa yenisini gonderemezsin',
        sebep: 'aktif',
      });
    }
    if (kuyruk.length >= AYAR.kuyrukMax) {
      sayac.reddedilen++;
      return json(res, 503, {
        hata: 'su an sira dolu', kuyruk: kuyruk.length,
        oneri: 'hazir ornek cevaplara goz at, birazdan tekrar dene',
      });
    }

    // Oturum kimligi ziyaretciden gelir ama DOGRUDAN n8n'e gecmez: dar bir
    // karakter kumesine ve uzunluga sikistirilir. Kimlik dogrulama araci
    // DEGIL - yalnizca ayni tarayicinin turlarini birbirine baglar.
    const oturumTabani = String(g.oturum || '').replace(/[^A-Za-z0-9_-]/g, '').slice(0, 40);
    const oturum = oturumTabani ? etkinOturum(oturumTabani, soru) : '';

    const isId = id();
    isler.set(isId, { id: isId, durum: 'bekliyor', soru, giris: Date.now(), pi,
                      oturum: oturum || null });
    ziyaretciSayaci.set(pi, sorulan + 1);
    aktifSayaci.set(pi, (aktifSayaci.get(pi) || 0) + 1);
    kuyruk.push(isId);
    sayac.toplam++;

    // Tekrar haritasi: en fazla 500 kayit ve 15 dakika. Sinirsiz birakilirsa
    // uzun sunumda bellekte birikir; is kayitlari da zaten suresi doluyor.
    if (istekId) {
      istekKimlikleri.set(istekId, { isId, t: Date.now() });
      if (istekKimlikleri.size > 500) {
        const esik = Date.now() - 15 * 60 * 1000;
        for (const [k, v] of istekKimlikleri) if (v.t < esik) istekKimlikleri.delete(k);
      }
    }

    kuyruguYurut();

    const s = siraNo(isId);
    return json(res, 202, {
      id: isId,
      durum: isler.get(isId).durum,
      sira: s,
      onunde: (s ? s - 1 : 0) + calisan,
      tahminiSaniye: tahminiSaniye(s),
      // Sinirsizken (soruSiniri=0) bu alan anlamsiz - eksi sayi dondurmek
      // yerine hic gondermiyoruz.
      kalanHakkin: AYAR.soruSiniri > 0 ? AYAR.soruSiniri - (sorulan + 1) : null,
    });
  }

  if (yol.startsWith('/durum/')) {
    const is = isler.get(yol.slice(7));
    if (!is) return json(res, 404, { hata: 'is bulunamadi ya da suresi doldu' });
    const s = siraNo(is.id);
    return json(res, 200, {
      id: is.id, durum: is.durum,
      sira: s, onunde: (s ? s - 1 : 0) + (is.durum === 'bekliyor' ? calisan : 0),
      tahminiSaniye: is.durum === 'bekliyor' ? tahminiSaniye(s)
                   : is.durum === 'calisiyor' ? AYAR.bekleme : 0,
      gecenSaniye: Math.round((Date.now() - is.giris) / 1000),
      cevap: is.durum === 'bitti' ? is.cevap : undefined,
      hata: is.durum === 'hata' ? is.hata : undefined,
    });
  }

  // --- SAYFA -------------------------------------------------------------
  // Gecit sayfayi da servis ediyor. Sebep: Funnel zaten bu portu yayinliyor,
  // yani sayfa icin AYRI bir tunel ya da barindirma gerekmiyor. Vercel sonra
  // gelecek ve orada durmaya devam edecek; burasi laptop acikken calisan,
  // her zaman en guncel surum.
  // YALNIZ index.html: dizin gezinme yok, yol birlestirme yok, bu yuzden
  // ".." ile disari cikilamaz.
  if (yol === '/' || yol === '/index.html') {
    const fs = require('fs');
    const p = require('path').join(__dirname, '..', 'site', 'index.html');
    return fs.readFile(p, (e, icerik) => {
      if (e) return json(res, 404, { hata: 'sayfa bulunamadi' });
      res.writeHead(200, {
        'Content-Type': 'text/html; charset=utf-8',
        'Content-Length': icerik.length,
        'Cache-Control': 'no-cache',
      });
      res.end(icerik);
    });
  }

  return json(res, 404, { hata: 'yol yok' });
});

sunucu.listen(AYAR.port, AYAR.arayuz, () => {
  console.log(`GECIT ayakta  http://${AYAR.arayuz}:${AYAR.port}`);
  console.log(`  n8n         ${AYAR.n8n}`);
  console.log(`  erisim kodu ${AYAR.kod ? 'VAR' : 'YOK - uc herkese acik'}`);
  console.log(`  es zamanli  ${AYAR.esZamanli}   kuyruk ${AYAR.kuyrukMax}   ` +
              `soru/ziyaretci ${AYAR.soruSiniri || 'sinirsiz'}   ayni anda ${AYAR.aktifSiniri}`);
  console.log(`  bekleme     ${AYAR.bekleme} sn/dalga (olculen)   zaman asimi ${AYAR.zamanAsimi / 1000} sn`);
});
