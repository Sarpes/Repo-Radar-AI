###############################################################
# DS21 - FINAL PROJECT  |  RAG SIDE
###############################################################
# DS21_Final_v2.py answers "will this repo still be developed?".
# This file answers "which repo fits the problem this person described?".
#
# They meet in the panel: retrieval ranks by relevance, the model re-ranks
# by abandonment risk, and every result carries a badge.
#
# WHY A SEPARATE FILE
#   The model chain is finished and verified. This half has different inputs
#   (README text, not features), a different runtime (Ollama, not CatBoost)
#   and a different failure mode. Bolting it onto a 2,700-line file would
#   make both harder to defend.
#
# INPUTS   Data/github_ai_intelligence_2026.csv   (identity + description + topics)
#          Data/readme_raw/                       (23,798 files, read-only)
# OUTPUT   outputs/rag_documents.parquet
#
# ARCHITECTURE (decided 2026-09-06): retrieval lives in n8n, backed by a
# pgvector container. Python prepares the text and fills the index once;
# n8n does the query path (Ollama embed -> PGVector -> AI Agent -> log).
###############################################################

###############################################################
# CONTENTS
###############################################################
#   STEP 14.1  README text extraction and cleaning   <- this round
#   STEP 14.2  Embedding generation (bge-m3)         blocked: ollama pull bge-m3
#   STEP 14.3  Load into pgvector                    blocked: container + psycopg2
#   STEP 15    Retrieval evaluation (recall@k)       blocked: needs 14.2
###############################################################

import json
import os
import re
import sys
import time

# line_buffering=True matters here, it is not decoration: STEP 14.2 runs for
# ~11 minutes, and without it Python buffers stdout whenever output is piped
# or redirected. The progress lines then appear only at the very end, which
# makes a long job indistinguishable from a hung one.
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(PROJECT_ROOT, "Data", "github_ai_intelligence_2026.csv")
README_DIR = os.path.join(PROJECT_ROOT, "Data", "readme_raw")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# --- SWEEP KANCALARI ---------------------------------------------------------
# Bu ucu de ortam degiskeninden gelir, cunku HEAD_CHARS'i taramak istiyoruz ve
# belge kurma mantigini IKINCI BIR YERE KOPYALAMAK en kotu secenek olurdu:
# kopya kacinilmaz olarak asildan ayrisir ve iki farkli "doc_primary" tanimiyla
# olculen sayilar karsilastirilamaz hale gelir. Tek uygulama, disaridan ayar.
#
#   DS21_HEAD_CHARS   README'nin doc_primary'ye giren karakter sayisi
#   DS21_SUFFIX       cikti dosyalarina eklenen ek (tarama kosulari birbirini ezmesin)
#   DS21_SKIP_PGVECTOR  '1' ise pgvector yuklemesi atlanir (tarama sirasinda gereksiz:
#                       recall numpy'da QV @ E.T ile olculuyor, veritabanina girmeden)
SUFFIX = os.environ.get("DS21_SUFFIX", "")
SKIP_PGVECTOR = os.environ.get("DS21_SKIP_PGVECTOR", "") == "1"

DOC_PATH = os.path.join(OUTPUT_DIR, f"rag_documents{SUFFIX}.parquet")


def header(text):
    print("\n" + "#" * 21 + f" {text} " + "#" * 21)


###############################################################
# STEP 14.1: README TEXT EXTRACTION AND CLEANING
###############################################################
# A README is written for a human reading a rendered web page. An embedding
# model reads the raw markdown, so it sees things a human never does: HTML
# wrappers, shields.io badge URLs, image links, code fences. On some repos
# that noise is most of the file - langchain's first 200 characters are
# nothing but <div>, <picture> and <source> tags.
#
# Cleaning is therefore not cosmetic. Every character of noise dilutes the
# vector, and a vector is an AVERAGE: the more boilerplate goes in, the more
# every repo looks like every other repo.
#
# WHAT WE REMOVE, AND WHY EACH ONE
#   code fences   a pip-install snippet does not describe what a tool DOES
#   badges        shields.io URLs are pure noise, identical across repos
#   images        alt text is usually "logo" or empty
#   HTML tags     layout, never meaning
#   link URLs     the anchor TEXT is meaningful, the URL is not -> keep text
#   heading marks "## Features" and "Features" mean the same to an embedder
#
# Measured on a 1,500-repo sample: 61% of characters survive (p10 0.36,
# p50 0.62, p90 0.84). So roughly 4 characters in 10 were never worth
# embedding.

RE_FENCE = re.compile(r"```[\s\S]*?```")
RE_BADGE = re.compile(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)")   # [![x](img)](link)
RE_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
RE_HTML = re.compile(r"<[^>]{1,200}>")        # bounded: never eat a whole file
RE_ALERT = re.compile(r">\s*\[!\w+\]")        # GitHub's > [!IMPORTANT] syntax
RE_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
RE_HEADMK = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
RE_RULE = re.compile(r"^\s*([-*_]\s*){3,}$", re.MULTILINE)
RE_WS = re.compile(r"[ \t]+")
# Control characters, keeping \t \n \r. Three READMEs are UTF-16 files read as
# UTF-8, which interleaves NUL bytes through the text (2,535 / 6,115 / 7,418 of
# them). Parquet stores NUL happily; Postgres text columns reject it outright,
# so without this the pgvector load dies on row ~9,000 with a ValueError that
# says nothing about which repo caused it. 11 more repos carry other control
# characters. None of them is meaningful text in any of the 14 files affected.
RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
# Blank-line collapse must tolerate lines that contain only spaces. A plain
# \n{3,} misses them, and stripped HTML leaves exactly that: runs of "\n \n \n".
RE_BLANK = re.compile(r"\n\s*\n(\s*\n)+")


def clean_markdown(text):
    """Markdown -> plain prose, keeping meaning and dropping presentation."""
    text = RE_CTRL.sub("", text)          # first: binary noise breaks everything after
    for pattern in (RE_FENCE, RE_BADGE, RE_IMAGE, RE_HTML, RE_ALERT):
        text = pattern.sub(" ", text)
    text = RE_LINK.sub(r"\1", text)      # [LangChain docs](url) -> LangChain docs
    text = RE_HEADMK.sub("", text)
    text = RE_RULE.sub(" ", text)
    text = text.replace("`", "").replace("|", " ")
    text = RE_WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return RE_BLANK.sub("\n\n", text).strip()


# STEP 10.3 named three hazards for a naive chunker. Two are handled here.
CAP_CHARS = 30_000    # hazard 1: one 9.9 MB README must not dominate the index
MIN_CHARS = 120       # hazard 2: empty and near-empty files make useless vectors

# HEAD_CHARS - how much README goes into the one-vector-per-repo doc.
# 900 was the original value and it captures only ~21.6% of the median README
# (4,016 chars) while using 314 of bge-m3's 8,192 available tokens. It is a
# candidate for tuning, not a law, so it is overridable: sweep_head_chars.py
# sets DS21_HEAD_CHARS and measures each value against the FIXED query set in
# eval_queries.parquet.
# Raising it does NOT add vectors (still one per repo) - the only cost is
# embedding time. The risk is the opposite of obvious: a single vector covering
# more text can get BLURRIER and retrieve worse. That is exactly why we measure.
HEAD_CHARS = int(os.environ.get("DS21_HEAD_CHARS", "900"))

# A fourth hazard, found only by looking at the output: `description` is a
# free-text CSV field with no length limit of its own. Aryia-Behroziuan/neurons
# has a 15,573-character description - somebody pasted an essay into it - which
# made its primary document 14x the median and would have produced a vector
# describing that essay rather than the repo.
# 400 is not a guess: GitHub's own UI caps descriptions at 350, this corpus has
# p99 = 335 and p99.9 = 404, and only 26 repos (0.11%) exceed 400 at all.
DESC_CHARS = 400


def build_primary_doc(full_name, description, topics, language, clean_text):
    """The document that becomes ONE vector for this repo.

    Fields are passed explicitly rather than as a row object, so this is
    built in the SAME pass that reads the file. There is no second lookup
    and therefore no way for the text and the metadata to drift apart.

    Hazard 3 from STEP 10.3: a chunk lifted from the middle of a README
    carries no repo identity. So identity goes FIRST, always.

    Order matters. Metadata leads because it is the most reliable
    description we have - coqui-ai/TTS opens its README with a news feed,
    but its CSV description says "a deep learning toolkit for text-to-speech".
    The README head then supplies detail the metadata cannot.

    ai_category is deliberately EXCLUDED. It is a derived single label that
    TF-IDF reproduces at 89.2% accuracy, i.e. it is wrong often enough to
    matter, and it is wrong on exactly the famous repos a demo will use
    (langchain is filed under RAG). `topics` carries the same information,
    is author-supplied, and is multi-label. Category stays a UI facet.
    """
    parts = [full_name]
    if pd.notna(description) and str(description).strip():
        parts.append(str(description).strip()[:DESC_CHARS])
    if pd.notna(topics) and str(topics).strip():
        parts.append("Topics: " + str(topics).strip())
    if pd.notna(language):
        parts.append("Language: " + str(language))
    if clean_text:
        parts.append(clean_text[:HEAD_CHARS])
    return "\n".join(parts)


header("STEP 14.1 | README text extraction")
t0 = time.time()

# Exactly four fields reach the document: full_name, description, topics,
# language. `owner`/`name` only build the README filename. ai_category is NOT
# read - see build_primary_doc for why it is kept out of the embedded text.
df = pd.read_csv(CSV_PATH, usecols=["full_name", "owner", "name",
                                    "description", "topics", "language"])
print(f"repos in CSV       : {len(df):,}")
print(f"README directory   : {README_DIR}")

rows = []
missing = truncated = 0
for row in df.itertuples(index=False):
    path = os.path.join(README_DIR, f"{row.owner}__{row.name}.md")
    raw = ""
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
    else:
        missing += 1
    clean = clean_markdown(raw) if raw.strip() else ""
    if len(clean) > CAP_CHARS:
        clean = clean[:CAP_CHARS]
        truncated += 1
    # Built here, in the same iteration that produced `clean`, from the same
    # `row`. Nothing is joined afterwards, so nothing can misalign.
    rows.append((row.full_name, len(raw), clean,
                 build_primary_doc(row.full_name, row.description, row.topics,
                                   row.language, clean)))

docs = pd.DataFrame(rows, columns=["full_name", "raw_chars", "text_clean", "doc_primary"])
docs["clean_chars"] = docs["text_clean"].str.len()
docs["has_readme"] = docs["raw_chars"] > 0
# Guard: too short to carry meaning -> keep the row; its primary doc still
# works, because metadata carries it.
docs["chunkable"] = docs["clean_chars"] >= MIN_CHARS
docs["primary_chars"] = docs["doc_primary"].str.len()

assert len(docs) == len(df), "row count changed - the loop dropped something"
assert docs["full_name"].is_unique, "duplicate full_name in the document table"

# Every part of the primary document is now capped, so its length is bounded.
# This guard exists because the description overrun was invisible until the
# max was printed - a cap that is never checked is a cap that silently lapses.
DOC_BOUND = len("Topics: ") + 500 + DESC_CHARS + HEAD_CHARS + 250   # + name, language
worst = docs["doc_primary"].str.len().max()
assert worst <= DOC_BOUND, (
    f"primary document reached {worst:,} chars, bound is {DOC_BOUND:,} - "
    f"some field is uncapped: {docs.loc[docs['doc_primary'].str.len().idxmax(), 'full_name']}")
print(f"\nguard: longest primary document {worst:,} chars (bound {DOC_BOUND:,}) - PASSED")

print(f"\nread in {time.time() - t0:.0f} s")
print(f"  with a README file : {int(docs['has_readme'].sum()):,} "
      f"({docs['has_readme'].mean() * 100:.2f}%)")
print(f"  missing            : {missing:,}   -> kept, they still get a primary doc")
print(f"  truncated at {CAP_CHARS:,}  : {truncated:,}")
print(f"  below {MIN_CHARS} chars clean : {int((~docs['chunkable']).sum()):,}"
      f"   -> too short to carry meaning; metadata carries their primary doc")

##########################
# 14.1.1  How much noise was actually removed?
##########################
header("STEP 14.1.1 | What cleaning removed")
kept = docs.loc[docs["raw_chars"] > 0, "clean_chars"] / docs.loc[docs["raw_chars"] > 0, "raw_chars"]
print(f"{'':16}{'raw':>12}{'clean':>12}{'kept':>8}")
for q in (.10, .25, .50, .75, .90):
    print(f"{f'p{int(q * 100)}':<16}{docs['raw_chars'].quantile(q):>12,.0f}"
          f"{docs['clean_chars'].quantile(q):>12,.0f}{kept.quantile(q):>8.2f}")
print(f"{'total':<16}{docs['raw_chars'].sum() / 1e6:>11,.1f}M{docs['clean_chars'].sum() / 1e6:>11,.1f}M"
      f"{docs['clean_chars'].sum() / docs['raw_chars'].sum():>8.2f}")
print(f"\n-> {(1 - docs['clean_chars'].sum() / docs['raw_chars'].sum()) * 100:.0f}% of the corpus was "
      f"presentation, not meaning. That is what we are NOT embedding.")

##########################
# 14.1.2  Primary document - the thing that becomes one vector
##########################
header("STEP 14.1.2 | Primary document")
p = docs["primary_chars"]
print(f"length  p10 {p.quantile(.10):,.0f} | p50 {p.quantile(.50):,.0f} | "
      f"p90 {p.quantile(.90):,.0f} | max {p.max():,.0f} characters")
print(f"        ~{p.median() / 3.5:,.0f} tokens at the median")

print("\nEXAMPLE - the primary document is what the user's query is matched against:")
for fn in ["coqui-ai/TTS", "microsoft/markitdown"]:
    hit = docs[docs["full_name"] == fn]
    if hit.empty:
        continue
    print(f"\n  --- {fn} ---")
    for line in hit.iloc[0]["doc_primary"].split("\n")[:4]:
        print(f"    {line[:88]}")

##########################
# 14.1.3  Save
##########################
header("STEP 14.1.3 | Saving")
os.makedirs(OUTPUT_DIR, exist_ok=True)
docs.to_parquet(DOC_PATH, index=False, compression="zstd")
print(f"{DOC_PATH}")
print(f"  {len(docs):,} rows x {docs.shape[1]} columns  "
      f"({os.path.getsize(DOC_PATH) / 1024**2:.2f} MB)")
print(f"\n  doc_primary   -> ONE vector per repo   "
      f"(the primary index, {len(docs):,} vectors)")
print("  text_clean    -> source of eval queries and of the body index (tools/build_body_index.py)")
print(f"  chunkable     -> clean text >= {MIN_CHARS} chars")

print("\nSTEP 14.1 complete.")


###############################################################
# STEP 14.2: EMBEDDINGS
###############################################################
# Every primary document becomes one point in 1024-dimensional space, where
# distance means "difference in meaning". The user's problem statement is
# placed in the same space; its nearest neighbours are the answer.
#
# MEASURED ON THIS MACHINE, 2026-09-06 - none of these are assumptions:
#   bge-m3 runs 100% on GPU (1.08 GB model, 4060 Ti has room)
#   output is 1024-dim and ALREADY L2-NORMALISED (norm = 1.0000), so cosine
#     similarity is a plain dot product and we never normalise again
#   context window 8192 tokens; our documents are ~323 tokens at the median,
#     so nothing is truncated
#
#   THROUGHPUT DEPENDS ALMOST ENTIRELY ON BATCH SIZE:
#     batch  16 ->  1.3 docs/s   (dominated by model warm-up)
#     batch  64 -> 15.4 docs/s
#     batch 128 -> 25.9 docs/s
#     batch 256 -> 37.6 docs/s   ~11 minutes for the whole corpus
#     batch 512 -> CRASHED the Ollama runner ("dial tcp ...: No connection")
#
# 512 documents x ~323 tokens is ~165k tokens in one request and the runner
# does not survive it. So 256 is not a preference, it is the measured
# ceiling, and the code halves the batch and retries rather than trusting it.

EMB_PATH = os.path.join(OUTPUT_DIR, f"embeddings{SUFFIX}.npy")

# Model de taranabilir olmali. Ayni sabit sinav (eval_queries.parquet) iki
# farkli gomme modelini de olcebilir - MTEB sirasi genel amaclidir, bizim
# gorevimiz cok ozel (gelistiricinin problem cumlesi -> GitHub reposu, iki
# dilli), o yuzden siralamayi kanit degil TAHMIN sayiyoruz.
# ⚠️ EMB_DIM modele gore degisir: bge-m3 ve qwen3-embedding:0.6b 1024,
#    qwen3-embedding:4b 2560, :8b 4096. Boyut degisirse pgvector semasi da
#    degisir - o yuzden ilk karsilastirma 1024'te kalan modellerle yapiliyor,
#    boylece TEK DEGISKEN model kimligi oluyor.
EMB_MODEL = os.environ.get("DS21_EMB_MODEL", "bge-m3")
EMB_DIM = int(os.environ.get("DS21_EMB_DIM", "1024"))
# Olculmus tavan, AMA MODELE VE BELGE UZUNLUGUNA GORE DEGISIR - bu yuzden
# artik sabit degil. bge-m3 + kisa doc_primary icin 256 calisiyor;
# qwen3-embedding:8b icin 64 olculdu (5,76 belge/sn). Buyuk batch
# "input length exceeds the context length" tetikliyor ve o hata
# BULASICI: sonrasinda kucuk istekler de reddediliyor, dakikalarca.
# Bu yuzden tavani zorlamak yerine guvenli tarafta kaliyoruz.
BATCH = int(os.environ.get("DS21_BATCH", "256"))
BATCH_KARAKTER = int(os.environ.get("DS21_BATCH_CHARS", "350000"))
OLLAMA = "http://localhost:11434/api/embed"


def embed_batch(texts, timeout=900):
    """Embed a list of texts. On failure, split and retry - never give up on
    a whole batch because one document in it was awkward."""
    import requests
    try:
        r = requests.post(OLLAMA, json={"model": EMB_MODEL, "input": texts},
                          timeout=timeout)
        if r.status_code == 200:
            vecs = r.json()["embeddings"]
            if len(vecs) == len(texts):
                return vecs
            raise RuntimeError(f"asked for {len(texts)} vectors, got {len(vecs)}")
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
    except Exception as exc:
        if len(texts) == 1:
            raise RuntimeError(f"single document failed: {exc}")
        mid = len(texts) // 2
        print(f"    batch of {len(texts)} failed ({str(exc)[:70]}) - splitting")
        time.sleep(3)     # let the runner come back up
        return embed_batch(texts[:mid], timeout) + embed_batch(texts[mid:], timeout)


header("STEP 14.2 | Embeddings")

# The cache is keyed on the CONTENT of the documents, not just their count.
# A shape check alone is a trap: changing one cleaning rule leaves the shape
# identical while every document differs, and the cache would then serve
# vectors of text that no longer exists - silently, and forever.
import hashlib

DOC_HASH_PATH = EMB_PATH + ".sha256"
# Hashed per row, then over the row hashes - not by joining the documents
# into one string. A join needs a separator that cannot occur in the text,
# and choosing one is exactly the kind of small decision that turns into a
# bug: the first attempt used a raw NUL as the separator, which Python
# refuses to even parse in source.
doc_hash = hashlib.sha256(
    pd.util.hash_pandas_object(docs["doc_primary"], index=False).values.tobytes()
).hexdigest()

E = None
if os.path.exists(EMB_PATH) and os.path.exists(DOC_HASH_PATH):
    cached = np.load(EMB_PATH)
    stored = open(DOC_HASH_PATH).read().strip()
    if cached.shape != (len(docs), EMB_DIM):
        print(f"cache is {cached.shape}, expected {(len(docs), EMB_DIM)} - rebuilding")
    elif stored != doc_hash:
        print("documents CHANGED since these vectors were made - rebuilding")
        print(f"  cached {stored[:16]}...  current {doc_hash[:16]}...")
    else:
        E = cached
        print(f"cache hit: {EMB_PATH}  {E.shape}  (documents unchanged)")
elif os.path.exists(EMB_PATH):
    print("vectors exist but no document hash beside them - rebuilding to be sure")

def parti_yap(texts, azami_belge=BATCH, azami_karakter=BATCH_KARAKTER):
    """Gruplari BELGE SAYISINA VE KARAKTER BUTCESINE gore boler.

    Sabit BATCH=256 yalnizca HEAD_CHARS=900 icin dogruydu; oradaki belgeler
    ~1.100 karakter, yani istek basina ~280k karakter. HEAD_CHARS=3000 ile
    ayni 256 belge 749k karakter yapiyor ve Ollama HTTP 400 donuyor:
        {"error":"the input length exceeds the context length"}
    (2026-09-07'de olculdu ve yeniden uretildi: 256 -> 400, 128 -> 200.)

    Daha kotusu, hata KALICI GORUNUYORDU: embed_batch bolerek yeniden dener,
    ama o an 128, 64 ... 2 ve tek belge de ayni hatayi verdi. Ayni belge
    dakikalar sonra tek basina SORUNSUZ gecti - yani Ollama istek reddedildikten
    sonra bir sure bozuk durumda kaliyor. Bolerek kurtarma bu yuzden ise
    yaramadi; cozum hataya hic dusmemek.

    Degismez olan sey belge SAYISI degil, istek basina TOPLAM METIN.
    350k karakter, calistigi olculen 128 x 2.700 = 346k'nin hemen altinda.
    """
    parti, karakter = [], 0
    for t in texts:
        if parti and (len(parti) >= azami_belge or karakter + len(t) > azami_karakter):
            yield parti
            parti, karakter = [], 0
        parti.append(t)
        karakter += len(t)
    if parti:
        yield parti


if E is None:
    texts = docs["doc_primary"].tolist()
    print(f"embedding {len(texts):,} documents with {EMB_MODEL}, "
          f"batch <= {BATCH} docs / 350k chars")
    print(f"estimated ~{len(texts) / 37.6 / 60:.0f} minutes at the measured rate\n")
    out, t0 = [], time.time()
    for grup in parti_yap(texts):
        out.extend(embed_batch(grup))
        done = len(out)
        rate = done / (time.time() - t0)
        print(f"  {done:>6,} / {len(texts):,}  ({done / len(texts) * 100:5.1f}%)  "
              f"{rate:5.1f} docs/s  eta {(len(texts) - done) / rate / 60:4.1f} min")
    E = np.asarray(out, dtype=np.float32)
    np.save(EMB_PATH, E)
    with open(DOC_HASH_PATH, "w") as fh:
        fh.write(doc_hash)      # written only on success, so a crash never
                                # leaves a hash claiming vectors that are wrong
    print(f"\ndone in {(time.time() - t0) / 60:.1f} minutes")

# Guards. A wrong shape here is silent everywhere downstream: pgvector would
# reject it, but a subtly wrong ROW ORDER would not - it would just return
# the wrong repos forever. Row i of E must be row i of docs, by construction.
assert E.shape == (len(docs), EMB_DIM), f"shape {E.shape}, expected {(len(docs), EMB_DIM)}"
norms = np.linalg.norm(E, axis=1)
print(f"\n{EMB_PATH}  ({os.path.getsize(EMB_PATH) / 1024**2:.0f} MB)")
print(f"  shape        {E.shape}")
print(f"  norms        min {norms.min():.4f}  max {norms.max():.4f}"
      f"   -> already unit length, cosine == dot product")
assert np.allclose(norms, 1.0, atol=1e-3), "vectors are not unit length"

##########################
# 14.2.1  Does the space actually mean anything?
##########################
# A shape check proves the file is well formed, not that it is USEFUL. The
# honest test is whether semantic neighbours come out semantic - so we ask
# the index a question it was never told the answer to.

header("STEP 14.2.1 | Sanity - are the neighbours meaningful?")
name_to_row = {n: i for i, n in enumerate(docs["full_name"])}
for probe in ["coqui-ai/TTS", "microsoft/markitdown"]:
    if probe not in name_to_row:
        continue
    sims = E @ E[name_to_row[probe]]          # dot product == cosine here
    top = np.argsort(-sims)[1:6]              # [0] is the repo itself
    print(f"\n  nearest to {probe}:")
    for j in top:
        print(f"    {sims[j]:.3f}  {docs['full_name'].iloc[j]}")

print("\nSTEP 14.2 complete.")


###############################################################
# STEP 14.3: LOAD INTO PGVECTOR
###############################################################
# The vectors exist as a 96 MB numpy file. That is enough for Python, but
# the query path runs in n8n, and n8n cannot read numpy or parquet - it can
# read Postgres natively. So the index moves into a database.
#
# CONTAINER (created 2026-09-06, deliberately separate):
#   ds21-pgvector · pgvector/pgvector:pg17 · port 5433 · network n8n_default
#   n8n reaches it as ds21-pgvector:5432, by service name, no IP needed.
#   ⛔ NOT installed into digischafak-postgres - that is Sarp's live job
#      system and an extension install is a change to it.
#
# ONE TABLE, NOT TWO. The model's output (badge, p_active, is_mature) lives
# in the same row as the vector, so a search returns the repo AND its badge
# in a single query. n8n never has to join two sources, which is exactly the
# kind of glue that breaks in a demo.

header("STEP 14.3 | Load into pgvector")

# TARAMA MODU BURADA BITER. DS21_SKIP_PGVECTOR=1 iken bu dosyanin isi
# "belgeleri kur + gom" ile sinirlidir; pgvector yuklemesi de, STEP 15
# degerlendirmesi de calismaz.
#
# Neden pgvector atlanir: recall QV @ E.T ile numpy'da olculuyor, veritabani
# o olcume hic girmiyor. pgvector demo/panel/n8n yolu icin var ve tarama
# bittikten sonra YALNIZCA KAZANAN ayarla bir kez doldurulur. Ara ayarlari
# yazmak 24.664 satiri bosuna uc kez yazip silmek olurdu - ve arada bir cokme
# olursa tabloda hangi ayarin vektorlerinin durdugu belirsiz kalirdi.
#
# Neden STEP 15 de calismaz: tarama sirasinda gereken tek sey recall@k, ama
# STEP 15 ayni zamanda agirlik taramasi yapiyor ve sorgu seti uretebiliyor -
# ikisi de ayar basina tekrarlanmamali. Olcum ayri bir araca tasindi:
#   python tools/eval_rag.py --suffix _h2000
# O arac SABIT eval_queries.parquet'i kullanir, yani ayarlar ayni sinavdan
# gecer. Tam kosu (tarama disi) her zamanki gibi bu dosyadan devam eder.
if SKIP_PGVECTOR:
    print("  DS21_SKIP_PGVECTOR=1 -> pgvector ve STEP 15 atlandi (tarama modu)")
    print(f"  cikti: {DOC_PATH}")
    print(f"         {EMB_PATH}")
    print(f"  olcmek icin: python tools/eval_rag.py --suffix '{SUFFIX}'")
    raise SystemExit(0)

import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

PG = dict(host="localhost", port=5433, user="ds21",
          password=os.environ.get("DS21_PG_PASSWORD", ""), dbname="repoquest")

# The scores frame is SHORTER than the document frame (counts printed below).
# That is not a bug - repos created after the cutoff T cannot have a label,
# so they were never in the modelling frame. They must still be searchable;
# they simply have no maintenance badge yet. A LEFT join says exactly that,
# and NULL is the honest value. Dropping them would hide the newest tools,
# which is the opposite of what a discovery product should do.
scores = pd.read_parquet(os.path.join(OUTPUT_DIR, "repo_scores.parquet"))
facets = pd.read_csv(CSV_PATH, usecols=["full_name", "language", "ai_category",
                                        "topics", "stars"])

load = (docs[["full_name", "doc_primary"]]
        .merge(scores[["full_name", "p_active_180d", "p_active_ranked",
                       "badge", "is_mature"]], on="full_name", how="left")
        .merge(facets, on="full_name", how="left"))
assert len(load) == len(docs), "the join changed the row count"
assert load["full_name"].is_unique, "the join duplicated rows"

unscored = load["badge"].isna().sum()
print(f"rows to load     : {len(load):,}")
print(f"with a badge     : {len(load) - unscored:,}")
print(f"WITHOUT a badge  : {unscored:,}  -> created after T, no label possible.")
print("                    Searchable, but the UI must say 'too new to score',")
print("                    never show them as red.")

conn = psycopg2.connect(**PG)
register_vector(conn)          # lets psycopg2 send numpy arrays as vectors
cur = conn.cursor()

# SEMAYI BU DOSYA KURAR. Onceden tablo elle yaratilmisti ve burada sadece
# TRUNCATE vardi; 2026-09-07'de gomme parametrelerini yeniden karara baglamak
# icin tablo YIKILDI ve bu adim "relation repos does not exist" ile patladi.
#
# Ayrica vektor boyutu MODELE BAGLI ve pgvector'de vector(N) SABIT bir tip:
#   bge-m3 / qwen3-embedding:0.6b -> 1024
#   qwen3-embedding:8b            -> 4096
# Model degisince eski tabloya yazmak "expected N dimensions, not M" verir.
# O yuzden mevcut boyutu okuyup UYUSMUYORSA tabloyu yeniden kuruyoruz -
# sessizce yanlis tipte tabloya yazmaktansa acikca yeniden kurmak dogru.
cur.execute("""
    SELECT a.atttypmod FROM pg_attribute a
    JOIN pg_class c ON c.oid = a.attrelid
    WHERE c.relname = 'repos' AND a.attname = 'embedding'
""")
mevcut = cur.fetchone()
mevcut_dim = mevcut[0] if mevcut else None
if mevcut_dim is not None and mevcut_dim != EMB_DIM:
    print(f"  tablo vector({mevcut_dim}) ama model {EMB_DIM} uretiyor -> yeniden kuruluyor")
    cur.execute("DROP TABLE repos CASCADE")
    mevcut = None

cur.execute(f"""
    CREATE TABLE IF NOT EXISTS repos (
        full_name       text PRIMARY KEY,
        doc_primary     text NOT NULL,
        embedding       vector({EMB_DIM}),
        p_active_180d   real,
        p_active_ranked real,
        badge           text,
        is_mature       boolean,
        language        text,
        ai_category     text,
        topics          text,
        stars           integer,
        -- metadata: LangChain'in VectorStorePGVector dugumu icin. O dugum
        -- id/vector/content/metadata dortlusunu bekliyor ve model ciktilarini
        -- TEK bir jsonb'de ariyor. Ayri kolonlari (badge, stars, ...) ham SQL
        -- icin tutuyoruz; bu kolon ayni veriyi ajan tarafina aciyor.
        -- Iki tuketici, tek tablo - kopyalama maliyeti birkac MB.
        metadata        jsonb
    )
""")
cur.execute("""
    CREATE TABLE IF NOT EXISTS query_log (
        id         bigserial PRIMARY KEY,
        asked_at   timestamptz NOT NULL DEFAULT now(),
        query      text,
        route      text,
        top_names  text[],
        latency_ms integer
    )
""")
conn.commit()
print(f"  sema hazir: repos.embedding = vector({EMB_DIM})")

cur.execute("TRUNCATE repos")  # this step is a full rebuild, never a patch

t0 = time.time()
rows = [
    (r.full_name, r.doc_primary, E[i],
     None if pd.isna(r.p_active_180d) else float(r.p_active_180d),
     None if pd.isna(r.p_active_ranked) else float(r.p_active_ranked),
     None if pd.isna(r.badge) else r.badge,
     None if pd.isna(r.is_mature) else bool(r.is_mature),
     None if pd.isna(r.language) else r.language,
     r.ai_category,
     None if pd.isna(r.topics) else r.topics,
     None if pd.isna(r.stars) else int(r.stars),
     json.dumps({
         "badge": None if pd.isna(r.badge) else r.badge,
         "stars": None if pd.isna(r.stars) else int(r.stars),
         "p_active_180d": None if pd.isna(r.p_active_180d) else round(float(r.p_active_180d), 4),
         "is_mature": None if pd.isna(r.is_mature) else bool(r.is_mature),
         "language": None if pd.isna(r.language) else r.language,
         "ai_category": r.ai_category,
     }, ensure_ascii=False))
    for i, r in enumerate(load.itertuples(index=False))
]
execute_values(cur, """
    INSERT INTO repos (full_name, doc_primary, embedding, p_active_180d,
                       p_active_ranked, badge, is_mature, language,
                       ai_category, topics, stars, metadata)
    VALUES %s
""", rows, page_size=500)
conn.commit()
print(f"\ninserted in {time.time() - t0:.0f} s")

# The HNSW index is built AFTER the data is in. Building it on an empty table
# means every insert maintains it incrementally, which is slower overall.
#
# AUDIT 2026-09-11 → docs/AUDIT_DEFTERI.md#ds21-rag-656: BU BLOK HER KOSUDA COKUYORDU ve bugun bir arizaya yol acti.
HNSW_MAX_DIM = 2000
print("indexes...")
t0 = time.time()
if EMB_DIM <= HNSW_MAX_DIM:
    cur.execute("CREATE INDEX IF NOT EXISTS repos_embedding_hnsw "
                "ON repos USING hnsw (embedding vector_cosine_ops)")
    print(f"  HNSW built ({EMB_DIM} <= {HNSW_MAX_DIM})")
else:
    print(f"  HNSW SKIPPED: {EMB_DIM} dims > pgvector's {HNSW_MAX_DIM} limit.")
    print(f"  Vector search is a SEQUENTIAL SCAN. Measured warm: ~340 ms on")
    print(f"  24,664 rows = 3% of the query budget, so this is a stated")
    print(f"  design limit, not a defect. Do not 'fix' it by lowering EMB_DIM")
    print(f"  without re-running eval_rag.py - 8b/4096 earns +10 r@5 points.")
cur.execute("CREATE INDEX IF NOT EXISTS repos_badge_idx ON repos (badge)")
cur.execute("ANALYZE repos")
conn.commit()
print(f"  done in {time.time() - t0:.0f} s")

##########################
# 14.3.1  Prove the whole path works from SQL
##########################
# Everything so far was verified in Python. But the query path is n8n -> SQL,
# and that is a different path with different failure modes. So the check has
# to run the way n8n will run it: one statement, vector distance, badge in
# the same row.

header("STEP 14.3.1 | End-to-end check, the way n8n will do it")
cur.execute("SELECT count(*), count(embedding), count(badge) FROM repos")
n, n_vec, n_badge = cur.fetchone()
print(f"rows {n:,} · with vector {n_vec:,} · with badge {n_badge:,}")
assert n == len(docs) and n_vec == n, "rows or vectors missing after load"

QUERY = "I need a tool that converts PDF documents into markdown for a RAG pipeline"
import requests
qv = np.asarray(requests.post(OLLAMA, json={"model": EMB_MODEL, "input": QUERY},
                              timeout=180).json()["embeddings"][0], dtype=np.float32)

print(f"\nquery: \"{QUERY}\"\n")
cur.execute("""
    SELECT full_name, 1 - (embedding <=> %s) AS similarity,
           badge, is_mature, p_active_ranked, stars
    FROM repos
    ORDER BY embedding <=> %s
    LIMIT 8
""", (qv, qv))
print(f"  {'repo':<40}{'sim':>7}{'badge':>16}{'mature':>8}{'stars':>9}")
for full_name, sim, badge_v, mature, _, stars in cur.fetchall():
    print(f"  {full_name:<40}{sim:>7.3f}{str(badge_v)[:14]:>16}"
          f"{('yes' if mature else '-'):>8}{stars or 0:>9,}")

cur.close()
conn.close()
print("\nSTEP 14.3 complete. The index is live and n8n can query it.")


###############################################################
# STEP 15: RETRIEVAL EVALUATION
###############################################################
# The model half has AUC, PR-AUC, Brier and a calibration curve. The
# retrieval half, which decides what the user actually sees, has had NO
# metric at all. That gap is the weakest part of the project, and this step
# closes it.
#
# THE OBVIOUS EVALUATION DOES NOT WORK. The plan said: use each repo's
# `description` as a query and check whether its own document comes back.
# But `description` is LINE 2 OF THAT DOCUMENT. We would be searching for a
# string with itself, recall@1 would be near 1.00, and it would mean nothing.
#
# Worse, it would give the WRONG ANSWER to the weight question: if relevance
# is already perfect, any other term can only hurt, and the sweep would
# conclude "use relevance alone" - the exact ranking we just disproved.
#
# So queries are GENERATED by the local LLM (qwen3:14b) from the document,
# under instructions to state the developer's PROBLEM and never name the
# tool. The query text is therefore not in the index. This also puts the
# required local-LLM component to work on something real rather than a demo.
#
# THE TEST SET MUST NOT MOVE WITH THE INDEX (fixed 2026-09-07).
# Queries used to be generated from `doc_primary` - the very text we index.
# That is fine for ONE configuration, but it makes tuning impossible: raise
# HEAD_CHARS and the index changes AND the LLM sees different text, so it
# writes different questions. Both sides of the comparison move and the
# recall numbers of two settings cannot be compared at all.
#
# Queries are now generated from `text_clean` - the FULL cleaned README,
# independent of any indexing parameter. HEAD_CHARS, chunk size and chunk
# overlap can now be swept against a FIXED test set.
#
# Consequence to keep in mind when reading the scores: the query author sees
# more of the README than the index stores. That makes the benchmark HARDER
# than the old one and the absolute numbers are NOT comparable to the
# 0.408 / 0.581 / 0.646 measured before this change. Within the new set,
# settings are comparable to each other - which is the point.

QUERY_PATH = os.path.join(OUTPUT_DIR, "eval_queries.parquet")
GEN_MODEL = "qwen3:14b"
N_PER_BAND_EN = 60
N_PER_BAND_TR = 30
STAR_BANDS = [(0, 100, "<100"), (100, 1_000, "100-1K"),
              (1_000, 10_000, "1K-10K"), (10_000, 10**9, "10K+")]

SYS_EN = (
    "A developer has a PROBLEM and is searching for a tool. You are given the tool. "
    "Write the one sentence that developer would type BEFORE they knew this tool existed.\n"
    "RULES:\n"
    "- Start with \"I need\", \"I want\", \"How do I\" or \"Looking for a way to\".\n"
    "- Describe the TASK they must accomplish, never the tool or its features.\n"
    "- Never name the tool, its author, or its repository.\n"
    "- 8 to 20 words. Plain sentence. No markdown. Output the sentence only.")
SYS_TR = (
    "A developer has a PROBLEM and is searching for a tool. You are given the tool. "
    "Write, IN TURKISH, the one sentence that developer would type BEFORE they knew "
    "this tool existed.\n"
    "RULES:\n"
    "- Write in natural Turkish, as a developer would actually type it.\n"
    "- Describe the TASK they must accomplish, never the tool or its features.\n"
    "- Never name the tool, its author, or its repository.\n"
    "- 8 to 20 words. Plain sentence. No markdown. Output the sentence only.")


# How much README the QUERY AUTHOR sees. This is not a tuning knob - it is
# part of the benchmark definition and must stay ABOVE every HEAD_CHARS we
# intend to test. It used to be 700, which is BELOW the 900 the index stored:
# the query author saw strictly less than the index, so a query could never
# require text that only a larger HEAD_CHARS would capture. That silently
# rigged the benchmark in favour of small HEAD_CHARS.
# 4,000 is chosen as ~the corpus median README (4,016 chars): enough for the
# model to understand what the tool is for, and comfortably above the 3,000
# ceiling of the settings we plan to sweep.
QUERY_SOURCE_CHARS = 4000


def generate_query(doc, system):
    import requests
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": GEN_MODEL, "system": system,
        "prompt": doc[:QUERY_SOURCE_CHARS], "stream": False,
        "options": {"temperature": 0.8, "num_predict": 60}, "think": False},
        timeout=300)
    return r.json().get("response", "").strip().replace("\n", " ")


header("STEP 15.1 | Building the evaluation set")

if os.path.exists(QUERY_PATH):
    ev = pd.read_parquet(QUERY_PATH)
    print(f"cache hit: {QUERY_PATH}  ({len(ev):,} queries)")
else:
    # Stratified by popularity, ON PURPOSE. A uniform sample would be almost
    # entirely small repos, because that is what the corpus is - and then a
    # popularity term in the ranking would look useless. Weighting towards
    # famous repos would make it look essential. Either sampling choice
    # ANSWERS THE QUESTION BEFORE MEASURING IT, so we stratify and report
    # every band separately instead of hiding the trade-off in one number.
    # `text_clean`, NOT `doc_primary` - see the header note. The source must be
    # independent of every indexing parameter or the test set moves with the
    # thing being tested. Repos with no README are excluded from the POOL only:
    # a query cannot be written from an empty document. They stay in the index
    # and can still be retrieved (and wrongly retrieved), so the metric is not
    # made easier - only the question source is restricted to repos that have
    # something to ask about.
    pool = (docs.loc[docs["clean_chars"] >= MIN_CHARS, ["full_name", "text_clean"]]
            .merge(pd.read_csv(CSV_PATH, usecols=["full_name", "stars"]), on="full_name"))
    print(f"query pool: {len(pool):,} repos with >= {MIN_CHARS} chars of README")
    picks = []
    for lo, hi, band in STAR_BANDS:
        sub = pool[(pool["stars"] >= lo) & (pool["stars"] < hi)]
        n = min(N_PER_BAND_EN + N_PER_BAND_TR, len(sub))
        take = sub.sample(n, random_state=42)
        for i, r in enumerate(take.itertuples(index=False)):
            picks.append((r.full_name, r.text_clean, r.stars, band,
                          "en" if i < N_PER_BAND_EN else "tr"))
    print(f"generating {len(picks):,} queries with {GEN_MODEL}")
    print(f"  ~{len(picks) * 3.9 / 60:.0f} minutes at the measured 3.9 s/query\n")

    out, t0 = [], time.time()
    for k, (fn, doc, stars, band, lang) in enumerate(picks, 1):
        q = generate_query(doc, SYS_EN if lang == "en" else SYS_TR)
        out.append((fn, q, stars, band, lang))
        if k % 40 == 0 or k == len(picks):
            rate = k / (time.time() - t0)
            print(f"  {k:>4} / {len(picks)}  {rate:4.2f} q/s  "
                  f"eta {(len(picks) - k) / rate / 60:4.1f} min")
    ev = pd.DataFrame(out, columns=["full_name", "query", "stars", "band", "lang"])
    ev.to_parquet(QUERY_PATH, index=False)
    print(f"\ngenerated in {(time.time() - t0) / 60:.1f} minutes")

# A generated query that leaked the repo's own name is not a query, it is the
# answer. Drop those rather than let them flatter the score.
leaked = [fn.split("/")[-1].lower() in q.lower()
          for fn, q in zip(ev["full_name"], ev["query"])]
print(f"\nqueries that leaked the repo name : {sum(leaked)}  -> dropped")
ev = ev[~np.array(leaked)].reset_index(drop=True)
print(f"usable queries                    : {len(ev):,}")
print(ev.groupby(["band", "lang"], observed=True).size().unstack(fill_value=0).to_string())
print("\nexamples:")
for _, r in ev.groupby("lang", observed=True).head(2).iterrows():
    print(f"  [{r['lang']}] {r['full_name']:<34} {r['query'][:80]}")


##########################
# 15.2  Score every ranking formula on the same queries
##########################
# The formula is a FUNCTION with weights, not a hardcoded expression, so the
# sweep below can move the weights without touching the ranking logic.

def rank_score(sim, p_active, log_stars, w_rel, w_act, w_pop):
    """Blend the three things a recommendation has to balance:
    relevance (does it match?), P(active) (will it survive?),
    adoption (has anyone found it useful?)."""
    return w_rel * sim + w_act * p_active + w_pop * log_stars


header("STEP 15.2 | recall@k")

meta_all = (docs[["full_name"]]
            .merge(pd.read_parquet(os.path.join(OUTPUT_DIR, "repo_scores.parquet"))
                   [["full_name", "p_active_ranked"]], on="full_name", how="left")
            .merge(pd.read_csv(CSV_PATH, usecols=["full_name", "stars"]),
                   on="full_name", how="left"))
# 0.35 for the 3,003 unscored repos: the corpus base rate, so a repo with no
# label is neither rewarded nor punished for it.
P_ACT = np.nan_to_num(meta_all["p_active_ranked"].values, nan=0.35).astype(np.float32)
LOG_STARS = np.log1p(np.nan_to_num(meta_all["stars"].values, nan=0.0)).astype(np.float32)
LOG_STARS /= LOG_STARS.max()
row_of = {n: i for i, n in enumerate(docs["full_name"])}

print(f"embedding {len(ev):,} queries...")
QV = []
for i in range(0, len(ev), BATCH):
    QV.extend(embed_batch(ev["query"].iloc[i:i + BATCH].tolist()))
QV = np.asarray(QV, dtype=np.float32)
SIMS = QV @ E.T                      # (queries x 24,664)
truth = np.array([row_of[f] for f in ev["full_name"]])

CANDIDATES = {
    "relevance only":            (1.0, 0.0, 0.0),
    "0.7 rel + 0.3 act (plan)":  (0.7, 0.3, 0.0),
    "0.6 / 0.2 / 0.2 stars":     (0.6, 0.2, 0.2),
    "0.5 / 0.2 / 0.3 stars":     (0.5, 0.2, 0.3),
    "0.6 / 0.1 / 0.3 stars":     (0.6, 0.1, 0.3),
    "0.8 / 0.1 / 0.1 stars":     (0.8, 0.1, 0.1),
}
K_POOL = 50      # re-ranking only reorders the top-50 relevance candidates


def recall_at(weights, mask=None):
    w_rel, w_act, w_pop = weights
    idx = np.arange(len(ev)) if mask is None else np.where(mask)[0]
    hits = {1: 0, 5: 0, 10: 0}
    for q in idx:
        pool_i = np.argpartition(-SIMS[q], K_POOL)[:K_POOL]
        sc = rank_score(SIMS[q][pool_i], P_ACT[pool_i], LOG_STARS[pool_i],
                        w_rel, w_act, w_pop)
        ranked = pool_i[np.argsort(-sc)]
        pos = np.where(ranked == truth[q])[0]
        if len(pos):
            for k in hits:
                if pos[0] < k:
                    hits[k] += 1
    return {k: v / max(len(idx), 1) for k, v in hits.items()}


print(f"\n{'formula':<28}{'recall@1':>10}{'recall@5':>10}{'recall@10':>11}")
print("-" * 59)
results = {}
for name, w in CANDIDATES.items():
    r = recall_at(w)
    results[name] = r
    print(f"{name:<28}{r[1]:>10.3f}{r[5]:>10.3f}{r[10]:>11.3f}")

##########################
# 15.3  Per band - where does the popularity term help, and where does it hurt?
##########################
header("STEP 15.3 | recall@5 by popularity band and language")
print(f"{'formula':<28}" + "".join(f"{b:>10}" for _, _, b in STAR_BANDS)
      + f"{'EN':>8}{'TR':>8}")
print("-" * 84)
for name, w in CANDIDATES.items():
    cells = [recall_at(w, ev["band"] == b)[5] for _, _, b in STAR_BANDS]
    en = recall_at(w, ev["lang"] == "en")[5]
    tr = recall_at(w, ev["lang"] == "tr")[5]
    print(f"{name:<28}" + "".join(f"{c:>10.3f}" for c in cells) + f"{en:>8.3f}{tr:>8.3f}")
print("\n-> If a formula wins overall by sacrificing the <100 band, that is a")
print("   product decision about who we serve, not a free improvement.")

best_name = max(results, key=lambda n: results[n][5])
print(f"\nBEST BY recall@5: {best_name}  {CANDIDATES[best_name]}")
print("\nSTEP 15 complete.")
