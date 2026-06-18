"""
main.py — EcoMind FastAPI Application  v2.0
Run:  uvicorn main:app --reload --host 0.0.0.0 --port 8000
Open: http://localhost:8000
"""
from __future__ import annotations
import io, logging, re
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.data_loader import (
    load_twitter, load_youtube, load_instagram,
    sentiment_counts, absa_counts, absa_counts_yt,
    emotion_counts, monthly_trend, aspect_emotion_matrix,
    pmi_scores, strategic_recommendations,
)
from app.nlp_engine    import BERTAnalyser, URLScraper, BatchAnalyser
from app.pdf_report    import generate_report
from app.social_scraper import SocialScraper

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR    = BASE_DIR / "static"

app = FastAPI(title="EcoMind", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

logger.info("Loading datasets ...")
TW = load_twitter()
YT = load_youtube()
IG = load_instagram()
logger.info("Datasets ready: TW=%d  YT=%d  IG=%d", len(TW), len(YT), len(IG))

_TW_SENT  = sentiment_counts(TW)
_YT_SENT  = sentiment_counts(YT, "sentiment")
_IG_SENT  = sentiment_counts(IG)
_PLATFORM = {"Twitter": _TW_SENT, "YouTube": _YT_SENT, "Instagram": _IG_SENT}
_ABSA_TW  = absa_counts(TW)
_ABSA_YT  = absa_counts_yt(YT)
_EMOTIONS = emotion_counts(TW)
_TREND    = monthly_trend(TW)
_PMI      = pmi_scores()
_RECS     = strategic_recommendations(_PLATFORM, _ABSA_TW)
_HEATMAP  = aspect_emotion_matrix(TW)

_BERT   = BERTAnalyser()
_SCRAPE = URLScraper()
_BATCH  = BatchAnalyser()
_SOCIAL = SocialScraper()


# ════════════════════════════════════════════════════════════
#  HTML PAGES
# ════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/analyzer", response_class=HTMLResponse)
async def analyzer(request: Request):
    return templates.TemplateResponse("analyzer.html", {"request": request})

@app.get("/social", response_class=HTMLResponse)
async def social_page(request: Request):
    return templates.TemplateResponse("social.html", {"request": request})


# ════════════════════════════════════════════════════════════
#  DATASET STATS API
# ════════════════════════════════════════════════════════════

@app.get("/api/overview")
async def api_overview():
    return JSONResponse({
        "total_posts":        len(TW) + len(YT) + len(IG),
        "twitter_total":      len(TW),
        "youtube_total":      len(YT),
        "instagram_total":    len(IG),
        "twitter_neg_pct":    round(_TW_SENT["negative"] / max(_TW_SENT["total"], 1) * 100, 1),
        "twitter_pos_pct":    round(_TW_SENT["positive"] / max(_TW_SENT["total"], 1) * 100, 1),
        "youtube_pos_pct":    round(_YT_SENT["positive"] / max(_YT_SENT["total"], 1) * 100, 1),
        "dominant_emotion":   max(_EMOTIONS, key=_EMOTIONS.get) if _EMOTIONS else "Anticipation",
        "top_aspect":         max(_ABSA_TW, key=lambda k: _ABSA_TW[k]["total"]) if _ABSA_TW else "Climate Change",
        "pmi_top_positive":   max(_PMI, key=_PMI.get),
        "pmi_top_negative":   min(_PMI, key=_PMI.get),
        "sentiment_accuracy": 88,
        "emotion_accuracy":   85,
    })

@app.get("/api/platform")
async def api_platform():
    result = {}
    for plat, stat in _PLATFORM.items():
        t = max(stat["total"], 1)
        result[plat] = {**stat,
            "pos_pct": round(stat["positive"] / t * 100, 1),
            "neg_pct": round(stat["negative"] / t * 100, 1),
            "neu_pct": round(stat["neutral"]  / t * 100, 1),
        }
    return JSONResponse(result)

@app.get("/api/absa")
async def api_absa():
    return JSONResponse({"twitter": _ABSA_TW, "youtube": _ABSA_YT})

@app.get("/api/emotions")
async def api_emotions():
    return JSONResponse({"distribution": _EMOTIONS, "heatmap": _HEATMAP})

@app.get("/api/trends")
async def api_trends():
    return JSONResponse(_TREND)

@app.get("/api/pmi")
async def api_pmi():
    return JSONResponse(_PMI)

@app.get("/api/recommendations")
async def api_recommendations():
    return JSONResponse(_RECS)


# ════════════════════════════════════════════════════════════
#  LIVE ANALYSIS API
# ════════════════════════════════════════════════════════════

@app.post("/api/analyse/text")
async def analyse_text(payload: dict):
    text = payload.get("text", "").strip()
    if not text:         raise HTTPException(422, "text is required")
    if len(text) > 5000: raise HTTPException(422, "text too long (max 5000 chars)")
    return JSONResponse(_BERT.analyse(text))


@app.post("/api/analyse/url")
async def analyse_url(payload: dict):
    url = payload.get("url", "").strip()
    if not url: raise HTTPException(422, "url is required")
    scraped = _SCRAPE.fetch(url)
    if scraped["error"]: raise HTTPException(400, scraped["error"])
    result = _BERT.analyse(scraped["text"][:2000])
    return JSONResponse({
        "url":          scraped["url"],
        "title":        scraped["title"],
        "word_count":   scraped["word_count"],
        "text_preview": scraped["text"][:400],
        **result,
    })


@app.post("/api/analyse/batch")
async def analyse_batch(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(422, "Only .csv files accepted")
    data = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(data), encoding="latin1")
    except Exception as e:
        raise HTTPException(400, f"CSV parse error: {e}")
    if df.empty:
        raise HTTPException(400, "Uploaded CSV is empty")
    return JSONResponse(_BATCH.run(df))


# ════════════════════════════════════════════════════════════
#  SOCIAL SCRAPER — shared NLP helper
# ════════════════════════════════════════════════════════════

def _run_nlp(items: list[dict]) -> dict:
    texts    = [i.get("text", "") for i in items if i.get("text")]
    analyses = _BERT.analyse_batch(texts[:200])
    sent     = {"positive": 0, "neutral": 0, "negative": 0}
    emo, asp = {}, {}
    pmi_sum  = 0.0
    labelled = []

    for item, a in zip(items[:200], analyses):
        sent[a["sentiment"]] = sent.get(a["sentiment"], 0) + 1
        emo[a["emotion"]]    = emo.get(a["emotion"],    0) + 1
        asp[a["aspect"]]     = asp.get(a["aspect"],     0) + 1
        pmi_sum             += a["pmi_score"]
        labelled.append({
            "text":       item.get("text", "")[:120],
            "author":     item.get("author", ""),
            "likes":      item.get("likes", 0),
            "sentiment":  a["sentiment"],
            "emotion":    a["emotion"],
            "aspect":     a["aspect"],
            "pmi_score":  a["pmi_score"],
            "bert_score": a["bert_score"],
        })

    n = max(len(analyses), 1)
    return {
        "sentiment_dist": sent,
        "emotion_dist":   emo,
        "aspect_dist":    asp,
        "avg_pmi":        round(pmi_sum / n, 4),
        "labelled_items": labelled[:20],
    }


# ════════════════════════════════════════════════════════════
#  SOCIAL SCRAPER — YouTube
#  FREE — YouTube Data API v3 (10,000 units/day free quota)
#  Get key: https://console.cloud.google.com
# ════════════════════════════════════════════════════════════

@app.post("/api/scrape/youtube")
async def scrape_youtube(payload: dict):
    """
    Body: {
        "video_url":   "https://youtube.com/watch?v=...",
        "api_key":     "YOUR_YOUTUBE_DATA_API_v3_KEY",
        "max_results": 100
    }
    """
    api_key   = payload.get("api_key",   "").strip()
    video_url = payload.get("video_url", "").strip()
    max_n     = int(payload.get("max_results", 100))

    if not video_url: raise HTTPException(422, "video_url is required")
    if not api_key:   raise HTTPException(422, "api_key is required (free YouTube Data API v3 key)")

    result = _SOCIAL.fetch_youtube(video_url, api_key, max_n)
    if result["error"]: raise HTTPException(400, result["error"])
    if result["items"]:
        result["nlp"] = _run_nlp(result["items"])
    return JSONResponse(result)


# ════════════════════════════════════════════════════════════
#  SOCIAL SCRAPER — Instagram
#  100% FREE — Instagram public oEmbed API
#  No API key needed · No login · Works on any PUBLIC post/reel/tv
# ════════════════════════════════════════════════════════════

@app.post("/api/scrape/instagram")
async def scrape_instagram(payload: dict):
    """
    Analyse a PUBLIC Instagram post / reel — FREE, no API key needed.

    Body: { "post_url": "https://www.instagram.com/p/SHORTCODE/" }

    Returns: author, caption, thumbnail + full NLP analysis on the caption.
    Note: caption-only. Comments require the paid Meta Graph API.
    """
    post_url = payload.get("post_url", "").strip()
    if not post_url:
        raise HTTPException(422, "post_url is required")

    result = _SOCIAL.fetch_instagram(post_url)
    if result["error"]:
        raise HTTPException(400, result["error"])
    if result["items"]:
        result["nlp"] = _run_nlp(result["items"])
    return JSONResponse(result)


# ════════════════════════════════════════════════════════════
#  PDF REPORTS
# ════════════════════════════════════════════════════════════

@app.get("/api/report/pdf")
async def pdf_report():
    pdf = generate_report(
        platform_stats=_PLATFORM, absa_tw=_ABSA_TW,
        emotions_tw=_EMOTIONS,    pmi_scores=_PMI,
        recommendations=_RECS,    monthly_trend=_TREND,
    )
    return Response(content=pdf, media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="ecomind_report.pdf"'})


@app.post("/api/report/batch-pdf")
async def batch_pdf(payload: dict):
    ps  = payload.get("platform_stats", _PLATFORM)
    ab  = payload.get("absa",           _ABSA_TW)
    em  = payload.get("emotions",       _EMOTIONS)
    pdf = generate_report(
        platform_stats=ps, absa_tw=ab, emotions_tw=em,
        pmi_scores=_PMI,
        recommendations=strategic_recommendations(ps, ab),
    )
    return Response(content=pdf, media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="ecomind_batch_report.pdf"'})


# ════════════════════════════════════════════════════════════
#  HEALTH CHECK
# ════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status":    "ok",
        "twitter":   len(TW),
        "youtube":   len(YT),
        "instagram": len(IG),
    }


# ════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn, webbrowser
    webbrowser.open("http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# ════════════════════════════════════════════════════════════
#  ELECTION / POLITICS MODULE
# ════════════════════════════════════════════════════════════

from app.election_engine import ElectionAnalyser

_ELECTION = ElectionAnalyser()

@app.get("/election", response_class=HTMLResponse)
async def election_page(request: Request):
    return templates.TemplateResponse("election.html", {"request": request})

@app.post("/api/election/analyse")
async def election_analyse(payload: dict):
    """
    Analyse any election/political text.
    Body: { "text": "...", "election": "India 2024" }
    """
    text     = payload.get("text", "").strip()
    election = payload.get("election", "General").strip()
    if not text:         raise HTTPException(422, "text is required")
    if len(text) > 5000: raise HTTPException(422, "text too long (max 5000 chars)")
    return JSONResponse(_ELECTION.analyse(text, election))

@app.post("/api/election/scrape/instagram")
async def election_ig_scrape(payload: dict):
    """
    Scrape a public Instagram post and run election-focused NLP.
    Body: { "post_url": "https://www.instagram.com/p/CODE/", "election": "India 2024" }
    """
    post_url = payload.get("post_url", "").strip()
    election = payload.get("election", "General").strip()
    if not post_url: raise HTTPException(422, "post_url is required")

    result = _SOCIAL.fetch_instagram(post_url)
    if result["error"]: raise HTTPException(400, result["error"])

    caption = result.get("title", "")
    nlp     = _ELECTION.analyse(caption[:2000], election)
    result["election_nlp"] = nlp
    return JSONResponse(result)

@app.post("/api/election/scrape/youtube")
async def election_yt_scrape(payload: dict):
    """
    Scrape YouTube comments and run election-focused NLP.
    Body: { "video_url": "...", "api_key": "...", "election": "India 2024", "max_results": 100 }
    """
    api_key   = payload.get("api_key",   "").strip()
    video_url = payload.get("video_url", "").strip()
    election  = payload.get("election",  "General").strip()
    max_n     = int(payload.get("max_results", 100))

    if not video_url: raise HTTPException(422, "video_url is required")
    if not api_key:   raise HTTPException(422, "api_key is required")

    result = _SOCIAL.fetch_youtube(video_url, api_key, max_n)
    if result["error"]: raise HTTPException(400, result["error"])
    if result["items"]:
        result["election_nlp"] = _ELECTION.analyse_batch(result["items"], election)
    return JSONResponse(result)

@app.get("/api/election/overview")
async def election_overview():
    """Returns pre-computed election sentiment overview stats."""
    return JSONResponse(_ELECTION.overview())
