import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from voice import (
    text_to_wav,
    extract_text_from_pdf,
    list_voices,
    run_garbage_collection,
    scheduled_gc,
    detect_language,
)

# ══════════════════════════════════════════════════════════════════
#  LIFESPAN
# ══════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    gc_task = asyncio.create_task(scheduled_gc(interval=300))
    yield
    gc_task.cancel()
    try:
        await gc_task
    except asyncio.CancelledError:
        pass


# ══════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Edge TTS — Text & PDF to WAV API",
    description=(
        "Convert plain text or PDF to natural-sounding WAV audio. "
        "Supports English, Hindi, and Marathi with automatic language detection."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://voxlibro.onrender.com",
        "https://voxlibro.netlify.app",
        "http://localhost",
        "http://127.0.0.1",
    ],
    allow_origin_regex=r"http://localhost(:[0-9]+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════
#  REQUEST MODELS
# ══════════════════════════════════════════════════════════════════
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    voice: str = Field(
        "auto",
        description=(
            "Voice key from /voices, or 'auto' to detect language automatically. "
            "Hindi text → hi-IN voice, Marathi → mr-IN voice, English → en-IN voice."
        ),
    )
    gender: str = Field("female", description="'female' or 'male' — used when voice='auto'")
    rate: str   = Field("+0%",    description="Speed:  +10%, -20%, etc.")
    volume: str = Field("+0%",    description="Volume: +5%,  -10%, etc.")


# ══════════════════════════════════════════════════════════════════
#  HELPER
# ══════════════════════════════════════════════════════════════════
def _cleanup(path: Path):
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass

def _wav_response(wav_path: Path, used_voice: str, bg: BackgroundTasks) -> FileResponse:
    """Return a FileResponse and schedule file deletion after send."""
    bg.add_task(_cleanup, wav_path)
    return FileResponse(
        path=str(wav_path),
        media_type="audio/wav",
        filename="output.wav",
        headers={"X-Voice-Used": used_voice},   # app can read which voice was picked
    )


# ══════════════════════════════════════════════════════════════════
#  UTILITY ROUTES
# ══════════════════════════════════════════════════════════════════

@app.get("/", tags=["Utility"], summary="API info")
async def root():
    return {
        "name":    "Edge TTS API",
        "version": "2.0.0",
        "docs":    "/docs",
        "health":  "/health",
        "voices":  "/voices",
    }


@app.get("/health", tags=["Utility"], summary="Health check")
async def health_check():
    """Returns 200 OK if the API is running. Use this as Render's health-check URL."""
    return {"status": "ok", "message": "Edge TTS API is running."}


@app.get("/voices", tags=["Utility"], summary="List all available voices")
async def get_voices():
    """
    Returns all supported voice keys.
    Use 'auto' on TTS endpoints to skip this and let the API pick automatically.
    """
    voices = list_voices()
    # Group by language for easier reading
    grouped: dict = {}
    for key, info in voices.items():
        grouped.setdefault(info["language"], {})[key] = info
    return {"voices": voices, "grouped": grouped}


@app.post("/detect-language", tags=["Utility"], summary="Detect language of text")
async def detect_lang(body: dict):
    """
    Pass {"text": "..."} — returns detected language and recommended voice keys.
    Useful for previewing which voice will be used before generating audio.
    """
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="'text' field is required.")
    lang = detect_language(text)
    return {
        "detected_language": lang,
        "recommended_voices": {
            "female": {"hindi": "hi-female", "marathi": "mr-female", "english": "en-IN-female"}[lang],
            "male":   {"hindi": "hi-male",   "marathi": "mr-male",   "english": "en-IN-male"}[lang],
        },
    }


@app.post("/admin/gc", tags=["Utility"], summary="Trigger garbage collection")
async def trigger_gc():
    """Manually delete expired WAV files. Normally runs automatically every 5 min."""
    deleted = run_garbage_collection()
    return {"deleted_files": deleted, "message": f"Removed {deleted} expired file(s)."}


# ══════════════════════════════════════════════════════════════════
#  TTS ROUTES
# ══════════════════════════════════════════════════════════════════

@app.post("/tts/text", tags=["TTS"], summary="Text → WAV (JSON body)")
async def tts_from_text(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    Send JSON with `text`. Use voice='auto' (default) for automatic language detection.
    Hindi text → Hindi voice. Marathi → Marathi. English → English.
    Returns a downloadable .wav file.
    """
    try:
        wav_path, used_voice = await text_to_wav(
            text=request.text,
            voice_key=request.voice,
            rate=request.rate,
            volume=request.volume,
            gender=request.gender,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return _wav_response(wav_path, used_voice, background_tasks)


@app.post("/tts/text/form", tags=["TTS"], summary="Text → WAV (form-data)")
async def tts_from_text_form(
    background_tasks: BackgroundTasks,
    text:   str = Form(...),
    voice:  str = Form("auto"),
    gender: str = Form("female"),
    rate:   str = Form("+0%"),
    volume: str = Form("+0%"),
):
    """
    Same as /tts/text but accepts multipart/form-data.
    Easier to call from mobile apps. voice='auto' enables language detection.
    """
    try:
        wav_path, used_voice = await text_to_wav(
            text=text, voice_key=voice, rate=rate, volume=volume, gender=gender,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return _wav_response(wav_path, used_voice, background_tasks)


@app.post("/tts/pdf", tags=["TTS"], summary="PDF → WAV")
async def tts_from_pdf(
    background_tasks: BackgroundTasks,
    file:   UploadFile = File(...),
    voice:  str = Form("auto"),
    gender: str = Form("female"),
    rate:   str = Form("+0%"),
    volume: str = Form("+0%"),
):
    """
    Upload a PDF file → API extracts text → returns WAV audio.
    voice='auto' detects language from the extracted text automatically.
    Scanned / image-only PDFs are not supported (no OCR).
    Max recommended size: ~5 MB / 50,000 characters.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    try:
        text = extract_text_from_pdf(pdf_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF parsing error: {e}")

    try:
        wav_path, used_voice = await text_to_wav(
            text=text, voice_key=voice, rate=rate, volume=volume, gender=gender,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return _wav_response(wav_path, used_voice, background_tasks)