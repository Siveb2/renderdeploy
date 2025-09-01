# main.py
import os
import logging
import time
import uuid
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from app.llm_core import LLMEngine, LLMConfig

# Load environment variables from .env file for local development
load_dotenv()

# --- Logging Setup ---
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Global Variables ---
llm_engine: Optional[LLMEngine] = None
app_start_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    global llm_engine
    environment = os.getenv("ENVIRONMENT", "development")
    logger.info(f"🚀 Application startup in {environment} mode...")
    try:
        config = LLMConfig.from_env()
        persona_file = os.getenv("PERSONA_FILE_PATH", "persona.txt")
        llm_engine = LLMEngine(config=config, persona_file_path=persona_file)
        
        health = await llm_engine.health_check()
        if not health.get("healthy"):
             logger.warning(f"Initial health check failed: {health.get('details')}")
        else:
             logger.info("✅ LLM Engine initialized and health check passed.")

    except Exception as e:
        logger.error(f"❌ Critical Error: Failed to initialize LLM Engine: {e}", exc_info=True)
        llm_engine = None
    yield
    logger.info("👋 Application shutdown.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="LLM Logic Microservice",
    version="2.0.0",
    lifespan=lifespan,
    # Docs are disabled in production by default for security
    docs_url="/docs" if os.getenv("ENVIRONMENT", "development") == "development" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT", "development") == "development" else None,
)

# --- Middleware ---
# In a Render environment, Render's proxy handles trusted hosts.
# For simplicity and security, we rely on standard CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins, can be locked down via Render dashboard if needed
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log HTTP requests and add a unique ID."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(f"rid={request_id} method={request.method} path={request.url.path} status={response.status_code} duration={process_time:.2f}ms")
    return response

# --- Pydantic Models ---
class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)

class ChatRequest(BaseModel):
    user_input: str
    history: List[Message] = []
    latest_summary: Optional[str] = None

class SummarizeRequest(BaseModel):
    history_chunk: List[Message] = Field(..., min_items=1)

# --- API Endpoints ---
async def get_llm_engine() -> LLMEngine:
    if not llm_engine:
        raise HTTPException(status_code=503, detail="LLM Engine is not available.")
    return llm_engine

@app.get("/health", tags=["Health Check"])
@app.get("/", tags=["Health Check"])
async def health_check():
    uptime = time.time() - app_start_time
    if not llm_engine:
        return {"status": "unhealthy", "engine_initialized": False, "uptime": uptime}
    
    llm_health = await llm_engine.health_check()
    return {
        "status": "healthy" if llm_health.get("healthy") else "degraded",
        "engine_initialized": True,
        "uptime": uptime,
        "llm_status": llm_health
    }

@app.post("/chat", tags=["LLM Operations"])
async def generate_chat_response(request: ChatRequest, engine: LLMEngine = Depends(get_llm_engine)):
    try:
        history_dicts = [msg.model_dump() for msg in request.history]
        response = await engine.get_response(request.user_input, history_dicts, request.latest_summary)
        return {"response": response}
    except Exception as e:
        logger.error("Chat generation failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize", tags=["LLM Operations"])
async def generate_summary(request: SummarizeRequest, engine: LLMEngine = Depends(get_llm_engine)):
    try:
        history_dicts = [msg.model_dump() for msg in request.history_chunk]
        summary = await engine.get_summary(history_dicts)
        return {"summary": summary}
    except Exception as e:
        logger.error("Summarization failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# This part is for local execution, Gunicorn will run the app in production
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8008))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
