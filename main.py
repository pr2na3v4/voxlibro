import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

from voice import (
    text_to_wav,
    extract_text_from_pdf,
    list_voices,
    run_garbage_collection,
    scheduled_gc,
    DEFAULT_VOICE,
    OUTPUT_DIR,
)

# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background garbage-collection task
    gc_task = asyncio.create_task(scheduled_gc(interval=300))
    yield
    gc_task.cancel()
    try:
        await gc_task
    except asyncio.CancelledError:
        pass


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Edge TTS — Text & PDF to WAV API",
    description=(
        "Convert plain text or PDF documents to natural-sounding WAV audio "
        "using Microsoft Edge TTS neural voices."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response models ─────────────────────────────────────────────────
class TextToSpeechRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000, description="Text to convert")
    voice: str = Field(DEFAULT_VOICE, description="Voice key (see /voices)")
    rate: str = Field("+0%", description="Speed adjustment e.g. +10%, -20%")
    volume: str = Field("+0%", description="Volume adjustment e.g. +5%, -10%")


class HealthResponse(BaseModel):
    status: str
    message: str


# ── Helper: delete file after response is sent ────────────────────────────────
def _cleanup(path: Path):
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════════════════════════════════════

# ── 1. Health check ───────────────────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Utility"],
    summary="API health check",
)
async def health_check():
    """Returns 200 OK if the API is running."""
    return {"status": "ok", "message": "Edge TTS API is running."}


# ── 2. List available voices ──────────────────────────────────────────────────
@app.get(
    "/voices",
    tags=["Utility"],
    summary="List all available voices",
)
async def get_voices():
    """
    Returns all supported voice keys and their details.
    Pass the `voice` key from this list to the TTS endpoints.
    """
    return {"voices": list_voices()}


# ── 3. Text → WAV (JSON body) ─────────────────────────────────────────────────
@app.post(
    "/tts/text",
    tags=["TTS"],
    summary="Convert plain text to WAV",
    response_class=FileResponse,
)
async def tts_from_text(
    request: TextToSpeechRequest,
    background_tasks: BackgroundTasks,
):
    """
    Send a JSON body with `text` and optional `voice`, `rate`, `volume`.
    Returns a downloadable `.wav` audio file.
    """
    try:
        wav_path = await text_to_wav(
            text=request.text,
            voice_key=request.voice,
            rate=request.rate,
            volume=request.volume,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    background_tasks.add_task(_cleanup, wav_path)
    return FileResponse(
        path=str(wav_path),
        media_type="audio/wav",
        filename="output.wav",
    )


# ── 4. Text → WAV (form fields — easy for mobile apps) ───────────────────────
@app.post(
    "/tts/text/form",
    tags=["TTS"],
    summary="Convert plain text to WAV (form-data)",
    response_class=FileResponse,
)
async def tts_from_text_form(
    background_tasks: BackgroundTasks,
    text: str = Form(..., description="Text to convert"),
    voice: str = Form(DEFAULT_VOICE, description="Voice key"),
    rate: str = Form("+0%", description="Speed e.g. +10%"),
    volume: str = Form("+0%", description="Volume e.g. +5%"),
):
    """
    Same as `/tts/text` but accepts `multipart/form-data` fields.
    Useful for mobile apps that send form submissions.
    """
    try:
        wav_path = await text_to_wav(
            text=text,
            voice_key=voice,
            rate=rate,
            volume=volume,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    background_tasks.add_task(_cleanup, wav_path)
    return FileResponse(
        path=str(wav_path),
        media_type="audio/wav",
        filename="output.wav",
    )


# ── 5. PDF → WAV ──────────────────────────────────────────────────────────────
@app.post(
    "/tts/pdf",
    tags=["TTS"],
    summary="Convert PDF document to WAV",
    response_class=FileResponse,
)
async def tts_from_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to convert"),
    voice: str = Form(DEFAULT_VOICE, description="Voice key"),
    rate: str = Form("+0%", description="Speed e.g. +10%"),
    volume: str = Form("+0%", description="Volume e.g. +5%"),
):
    """
    Upload a PDF file. The API extracts the text and returns a WAV audio file.
    - Max recommended PDF size: ~5 MB / ~50,000 characters of text.
    - Scanned/image-only PDFs are not supported (no OCR).
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    # Extract text
    try:
        text = extract_text_from_pdf(pdf_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF parsing error: {e}")

    # Convert to WAV
    try:
        wav_path = await text_to_wav(
            text=text,
            voice_key=voice,
            rate=rate,
            volume=volume,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    background_tasks.add_task(_cleanup, wav_path)
    return FileResponse(
        path=str(wav_path),
        media_type="audio/wav",
        filename="output.wav",
    )


# ── 6. Manual GC trigger (admin / debug) ─────────────────────────────────────
@app.post(
    "/admin/gc",
    tags=["Utility"],
    summary="Manually trigger garbage collection",
)
async def trigger_gc():
    """
    Immediately deletes expired WAV files from the output directory.
    Normally this runs automatically every 5 minutes.
    """
    deleted = run_garbage_collection()
    return {"deleted_files": deleted, "message": f"Removed {deleted} expired file(s)."}


# ── 7. Root ───────────────────────────────────────────────────────────────────
@app.get("/", tags=["Utility"], summary="API info")
async def root():
    return {
        "name": "Edge TTS API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "voices": "/voices",
    }