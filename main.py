"""
LeadsAI — AI_Backend (Worker Service)
Hosts heavy computational services: RAG, Gemini, CV, STT, TTS.
Exposes internal endpoints for the Main Backend (Orchestrator).
"""
import base64
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cv_service import BehaviorAnalyzer, CVService
from db import close_pool, get_pool, init_db, tenant_conn
from gemini_client import GeminiClient
from models import BehaviorEvent
from rag_engine import RAGEngine
from stt_service import STTService
from tts_service import TTSService
from scraper import fetch_html, extract_body_text

import asyncio
import ipaddress
import socket
from urllib.parse import urlparse

load_dotenv(override=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
GEMINI_MODEL_GENERATION = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ── Security constants ─────────────────────────────────────────────────────────
MAX_AUDIO_BYTES  = 10 * 1024 * 1024   # 10 MB decoded audio limit for STT
MAX_INGEST_BYTES = 10 * 1024 * 1024   # 10 MB file upload limit for RAG ingest
_BLOCKED_PORTS   = {22, 3306, 5432, 6379, 27017}


# ── Internal auth dependency ───────────────────────────────────────────────────
async def require_internal_token(x_internal_token: str = Header(None)):
    """Validate that the caller provides the shared internal token."""
    expected = os.getenv("AI_INTERNAL_TOKEN")
    if not expected or x_internal_token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def validate_url_safe(url: str) -> str:
    """Reject private/loopback/reserved IPs and non-HTTP schemes (SSRF protection)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(400, "Only http/https URLs allowed")
    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(400, "Invalid URL")
    if parsed.port and parsed.port in _BLOCKED_PORTS:
        raise HTTPException(400, "Blocked port")
    try:
        for res in socket.getaddrinfo(hostname, None):
            ip = ipaddress.ip_address(res[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                raise HTTPException(400, "Cannot crawl private/internal addresses")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "Could not resolve hostname")
    return url
# ═════════════════════════════════════════════════════════════════════════════
# LIFESPAN
# ═════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 AI_Backend starting (Heavy worker service)...")
    await init_db(reset=False, seed=False)
    
    app.state.rag = RAGEngine(
        chroma_mode=os.getenv("CHROMA_MODE", "persistent"),
        chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_data"),
        chroma_host=os.getenv("CHROMA_HOST", "localhost"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8001")),
    )
    app.state.gemini = GeminiClient()
    app.state.stt = STTService()
    app.state.tts = TTSService()
    app.state.cv = CVService()
    app.state.behavior = BehaviorAnalyzer()
    app.state.rag.set_gemini(app.state.gemini)

    # Warm up BM25 keyword index from existing ChromaDB data
    try:
        await _warmup_bm25(app.state.rag)
    except Exception as e:
        logger.warning(f"⚠️ BM25 warmup failed (non-fatal): {e}")

    # Launch periodic vector refresh background task
    refresh_task = asyncio.create_task(_periodic_vector_refresh(app))

    yield
    refresh_task.cancel()
    logger.info("💤 AI_Backend shutting down...")
    await close_pool()

app = FastAPI(
    title="LeadsAI-AI_Backend",
    version="3.1.0",
    description="Heavy worker service for AI computations",
    lifespan=lifespan,
)

# Deny all browser CORS (internal service — defense-in-depth)
app.add_middleware(CORSMiddleware, allow_origins=[], allow_methods=[], allow_headers=[])

# ═════════════════════════════════════════════════════════════════════════════
# INTERNAL MODELS
# ═════════════════════════════════════════════════════════════════════════════
class RAGRetrieveReq(BaseModel):
    query: str
    tenant_id: str
    top_k: int = 5


class GeminiChatReq(BaseModel):
    message: str
    history: List[Dict[str, Any]]
    context_chunks: List[Dict[str, Any]]
    tenant_config: Dict[str, Any]
    language: str = "en"

class IntentAnalyzeReq(BaseModel):
    history: List[Dict[str, Any]]
    tenant_config: Dict[str, Any]

class BehaviorProcessReq(BaseModel):
    event: BehaviorEvent
    tenant_id: str

class STTProcessReq(BaseModel):
    audio_b64: str
    session_id: str
    language: str = "en"

class TTSSynthesizeReq(BaseModel):
    text: str
    language: str = "en"
    session_id: Optional[str] = None

# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "ok", "service": "AI_Backend"}


# ═════════════════════════════════════════════════════════════════════════════
# INTERNAL KB ENDPOINTS (structured KB rebuild + single-page scrape)
# ═════════════════════════════════════════════════════════════════════════════

class RebuildReq(BaseModel):
    tenant_id: str
    chunks: List[Dict[str, Any]]

class SingleScrapeReq(BaseModel):
    url: str

class EnrichProductReq(BaseModel):
    text: str
    url: str


@app.post("/internal/rag/rebuild", dependencies=[Depends(require_internal_token)])
async def rag_rebuild(req: RebuildReq):
    """Flush structured KB chunks (company/product/QA) and re-ingest. Preserves file-upload chunks."""
    result = await app.state.rag.rebuild_structured(
        tenant_id=req.tenant_id, chunks=req.chunks
    )
    return {"status": "rebuilt", **result}


@app.post("/internal/scrape/single", dependencies=[Depends(require_internal_token)])
async def scrape_single_page(req: SingleScrapeReq):
    """Fetch one URL, extract clean body text — for product enrichment form pre-fill."""
    
    url = req.url.strip()
    if not url.startswith("http"):
        url = "https://" + url
    validate_url_safe(url)
    try:
        final_url, html = await fetch_html(url)
        extracted = extract_body_text(html, final_url)
        return {
            "url": final_url,
            "title": extracted.get("title", ""),
            "meta_description": extracted.get("meta_description", ""),
            "text": extracted.get("text", ""),
        }
    except Exception as e:
        logger.error(f"Single page scrape failed for {url}: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to fetch page: {e}")


@app.post("/internal/gemini/enrich-product", dependencies=[Depends(require_internal_token)])
async def enrich_product(req: EnrichProductReq):
    """
    Use Gemini to extract structured product fields from raw page text.
    Returns: {name, category, sub_category, description, pricing, min_order_qty, image_url}
    """
    prompt = f"""You are a product data extractor. Given the following text scraped from a product page, extract structured product information.

Source URL: {req.url}

Page text:
{req.text[:8000]}

Extract the following fields as a JSON object. Use empty string for fields you cannot determine. Do NOT invent data — only extract what is present:
{{
  "name": "Product/service name",
  "category": "Product category",
  "sub_category": "Product sub-category",
  "description": "Brief product description (2-3 sentences)",
  "pricing": "Price or price range if mentioned",
  "min_order_qty": "Minimum order quantity if mentioned",
  "image_url": "Direct URL to the main product image if found in the page"
}}

Respond ONLY with the JSON object, no other text."""

    import json as _json
    try:
        result = await app.state.gemini.generate_text(prompt)
        # Try to parse JSON from the response
        text = result.strip()
        # Handle markdown code blocks
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
            text = text.strip()
        parsed = _json.loads(text)
        return parsed
    except _json.JSONDecodeError:
        logger.warning(f"Gemini returned non-JSON for product enrichment: {result[:200]}")
        return {"name": "", "category": "", "sub_category": "", "description": "", "pricing": "", "min_order_qty": "", "image_url": ""}
    except Exception as e:
        logger.error(f"Product enrichment failed: {e}")
        raise HTTPException(status_code=502, detail=f"Enrichment failed: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# PERIODIC VECTOR REFRESH (background task)
# ═════════════════════════════════════════════════════════════════════════════

async def _warmup_bm25(rag: RAGEngine):
    """Populate BM25 keyword index from all existing ChromaDB tenant collections on startup."""
    vs = rag.vector_store
    if vs._client is None:
        logger.info("⚠️ No ChromaDB client — skipping BM25 warmup")
        return

    try:
        collections = vs._client.list_collections()
        loaded = 0
        for col in collections:
            tenant_id = col.name  # safe name = tenant_id
            chunks = vs.get_all_chunks(tenant_id)
            if chunks:
                rag.bm25.index(tenant_id, chunks)
                loaded += 1
                logger.info(f"  📚 BM25 warmed: {tenant_id} ({len(chunks)} chunks)")
        logger.info(f"✅ BM25 warmup complete: {loaded} tenant(s) indexed")
    except Exception as e:
        logger.error(f"BM25 warmup error: {e}")


async def _periodic_vector_refresh(app_instance: FastAPI):
    """
    Background loop that runs inside AI_Backend's lifespan.
    Wakes every hour; at the configured time, rebuilds structured KB vectors for all active tenants.
    """
    import hashlib
    logger.info("🔄 Periodic vector refresh scheduler started")

    while True:
        try:
            await asyncio.sleep(3600)  # Check every hour

            pool = await get_pool()
            async with pool.acquire() as conn:
                interval_row = await conn.fetchval(
                    "SELECT value FROM platform_settings WHERE key='vector_refresh_interval_hours'"
                )
                time_row = await conn.fetchval(
                    "SELECT value FROM platform_settings WHERE key='vector_refresh_time'"
                )

            interval_hours = int(interval_row or 24)
            refresh_time = time_row or "00:00"

            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            target_hour, target_min = map(int, refresh_time.split(":"))

            # Only run if current hour matches target
            if now.hour != target_hour or now.minute > target_min + 30:
                continue

            logger.info(f"🔄 Starting periodic vector refresh for all tenants...")

            async with pool.acquire() as conn:
                tenants = await conn.fetch(
                    "SELECT id FROM tenants WHERE status='active'"
                )

            for tenant_row in tenants:
                tid = tenant_row["id"]
                try:
                    chunks = await _build_structured_chunks(tid)
                    await app_instance.state.rag.rebuild_structured(tid, chunks)
                    logger.info(f"  ✅ Refreshed tenant {tid}: {len(chunks)} chunks")
                except Exception as e:
                    logger.error(f"  ❌ Refresh failed for tenant {tid}: {e}")

            logger.info(f"🔄 Periodic refresh cycle complete ({len(tenants)} tenants)")

        except asyncio.CancelledError:
            logger.info("🔄 Periodic vector refresh cancelled")
            break
        except Exception as e:
            logger.error(f"Periodic refresh error: {e}")
            await asyncio.sleep(60)  # Brief backoff on unexpected errors


async def _build_structured_chunks(tenant_id: str) -> List[Dict[str, Any]]:
    """Build all structured KB chunks for a tenant from DB data."""
    import hashlib
    chunks = []

    async with tenant_conn(tenant_id) as conn:
        # 1. Company data → one chunk per section
        company_rows = await conn.fetch(
            "SELECT section, field_key, field_value FROM kb_company_data "
            "WHERE tenant_id=$1 ORDER BY section, display_order", tenant_id
        )
        sections: Dict[str, List[str]] = {}
        for row in company_rows:
            sec = row["section"]
            if sec not in sections:
                sections[sec] = []
            if row["field_value"].strip():
                sections[sec].append(f"{row['field_key'].replace('_', ' ').title()}: {row['field_value']}")

        for sec_name, lines in sections.items():
            if lines:
                text = f"{sec_name.replace('_', ' ').title()}\n" + "\n".join(lines)
                chunk_id = f"kb_company_{hashlib.md5(f'{tenant_id}:{sec_name}'.encode()).hexdigest()[:12]}"
                chunks.append({
                    "id": chunk_id,
                    "text": text,
                    "source": f"Company Data - {sec_name.replace('_', ' ').title()}",
                    "chunk_index": 0,
                    "word_count": len(text.split()),
                })

        # 2. Products → one chunk per product
        products = await conn.fetch(
            "SELECT * FROM kb_products WHERE tenant_id=$1 ORDER BY category, name", tenant_id
        )
        for prod in products:
            parts = [f"Product: {prod['name']}"]
            if prod["category"]:
                parts.append(f"Category: {prod['category']}")
            if prod["sub_category"]:
                parts.append(f"Sub-category: {prod['sub_category']}")
            if prod["description"]:
                parts.append(f"Description: {prod['description']}")
            if prod["pricing"]:
                parts.append(f"Pricing: {prod['pricing']}")
            if prod["min_order_qty"]:
                parts.append(f"Minimum Order Quantity: {prod['min_order_qty']}")
            text = "\n".join(parts)
            chunk_id = f"kb_product_{prod['id']}"
            chunks.append({
                "id": chunk_id,
                "text": text,
                "source": f"Product - {prod['name']}",
                "chunk_index": 0,
                "word_count": len(text.split()),
            })

        # 3. Custom Q/A → one chunk per Q/A pair
        qas = await conn.fetch(
            "SELECT id, question, answer FROM knowledge_qa WHERE tenant_id=$1", tenant_id
        )
        for qa in qas:
            text = f"Question: {qa['question']}\nAnswer: {qa['answer']}"
            chunk_id = f"qa_{qa['id']}"
            chunks.append({
                "id": chunk_id,
                "text": text,
                "source": "Custom Q/A",
                "chunk_index": 0,
                "word_count": len(text.split()),
            })

    return chunks


@app.get("/v1/system")
async def system_info():
    return {
        "status":  "ok",
        "service": "AI_Backend",
        "stt":     getattr(app.state.stt,  "model_info",  {}),
        "tts":     getattr(app.state.tts,  "engine_info", {}),
        "cv":      getattr(app.state.cv,   "engine_info", {}),
        "rag":     {"chroma_mode": os.getenv("CHROMA_MODE", "persistent")},
    }

@app.post("/internal/rag/retrieve", dependencies=[Depends(require_internal_token)])
async def rag_retrieve(req: RAGRetrieveReq):
    chunks = await app.state.rag.retrieve(query=req.query, tenant_id=req.tenant_id, top_k=req.top_k)
    return {"chunks": chunks}

@app.post("/internal/rag/ingest", dependencies=[Depends(require_internal_token)])
async def rag_ingest(tenant_id: str = Form(...), file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > MAX_INGEST_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_INGEST_BYTES // (1024*1024)} MB)")
    job_id = await app.state.rag.ingest(
        content=content, filename=file.filename,
        content_type=file.content_type, tenant_id=tenant_id
    )
    return {"job_id": job_id}

@app.get("/internal/rag/ingest/{job_id}", dependencies=[Depends(require_internal_token)])
async def rag_job_status(job_id: str):
    return await app.state.rag.get_job_status(job_id)

@app.get("/internal/rag/stats", dependencies=[Depends(require_internal_token)])
async def rag_stats(tenant_id: str):
    count = app.state.rag.chunk_count(tenant_id)
    return {"count": count}

@app.get("/internal/rag/chunks", dependencies=[Depends(require_internal_token)])
async def rag_all_chunks(tenant_id: str):
    chunks = app.state.rag.vector_store.get_all_chunks(tenant_id)
    return {"chunks": chunks}


class QAIngestReq(BaseModel):
    qa_id: str
    question: str
    answer: str
    tenant_id: str


@app.post("/internal/rag/ingest-qa", dependencies=[Depends(require_internal_token)])
async def rag_ingest_qa(req: QAIngestReq):
    await app.state.rag.ingest_qa(
        qa_id=req.qa_id, question=req.question,
        answer=req.answer, tenant_id=req.tenant_id,
    )
    return {"status": "indexed", "qa_id": req.qa_id}


@app.delete("/internal/rag/qa/{qa_id}", dependencies=[Depends(require_internal_token)])
async def rag_delete_qa(qa_id: str, tenant_id: str):
    await app.state.rag.delete_qa_chunk(qa_id=qa_id, tenant_id=tenant_id)
    return {"status": "deleted", "qa_id": qa_id}

@app.delete("/internal/rag/doc/{doc_id}", dependencies=[Depends(require_internal_token)])
async def rag_delete_doc(doc_id: str, tenant_id: str):
    await app.state.rag.delete_doc_chunks(doc_id=doc_id, tenant_id=tenant_id)
    return {"status": "deleted", "doc_id": doc_id}

@app.post("/internal/gemini/chat", dependencies=[Depends(require_internal_token)])
async def gemini_chat(req: GeminiChatReq):
    result = await app.state.gemini.chat(
        message=req.message, history=req.history,
        context_chunks=req.context_chunks, tenant_config=req.tenant_config,
        language=req.language
    )
    return result

@app.post("/internal/gemini/analyze-intent", dependencies=[Depends(require_internal_token)])
async def gemini_analyze_intent(req: IntentAnalyzeReq):
    result = await app.state.gemini.analyze_intent(
        history=req.history, tenant_config=req.tenant_config
    )
    return result

@app.post("/internal/behavior/process", dependencies=[Depends(require_internal_token)])
async def behavior_process(req: BehaviorProcessReq):
    result = await app.state.behavior.process_event(
        event=req.event, tenant_id=req.tenant_id, gemini=app.state.gemini
    )
    return result

@app.post("/internal/cv/search", dependencies=[Depends(require_internal_token)])
async def cv_search(tenant_id: str = Form(...), top_k: int = Form(3), file: UploadFile = File(...)):
    img_bytes = await file.read()
    matches = await app.state.cv.find_products(
        image_bytes=img_bytes, tenant_id=tenant_id, top_k=top_k
    )
    return {"matches": matches}

@app.post("/internal/stt/process", dependencies=[Depends(require_internal_token)])
async def stt_process(req: STTProcessReq):
    audio_bytes = base64.b64decode(req.audio_b64)
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail=f"Audio too large (max {MAX_AUDIO_BYTES // (1024*1024)} MB)")
    result = await app.state.stt.process_chunk(
        audio_chunk=audio_bytes, session_id=req.session_id, language=req.language
    )
    return result

@app.post("/internal/tts/synthesize", dependencies=[Depends(require_internal_token)])
async def tts_synthesize(req: TTSSynthesizeReq):
    audio_url = await app.state.tts.synthesize(
        text=req.text, language=req.language, session_id=req.session_id
    )
    return {"audio_url": audio_url}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
