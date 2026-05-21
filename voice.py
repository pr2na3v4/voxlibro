import os
import uuid
import asyncio
import hashlib
import time
import logging
from pathlib import Path
from typing import Optional
import edge_tts
import PyPDF2
import io

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("generated_audio")
OUTPUT_DIR.mkdir(exist_ok=True)

CACHE_DIR = Path("audio_cache")
CACHE_DIR.mkdir(exist_ok=True)

# How long (seconds) a file lives before garbage collection removes it
FILE_TTL_SECONDS = 600          # 10 minutes for generated files
CACHE_TTL_SECONDS = 3600        # 1 hour for cached files

# ── Available voices (edge-tts) ───────────────────────────────────────────────
# Format: { "label": "edge-tts voice name" }
VOICE_OPTIONS = {
    "en-US-male":    "en-US-GuyNeural",
    "en-US-female":  "en-US-JennyNeural",
    "en-GB-male":    "en-GB-RyanNeural",
    "en-GB-female":  "en-GB-SoniaNeural",
    "en-IN-female":  "en-IN-NeerjaNeural",
    "en-IN-male":    "en-IN-PrabhatNeural",
    "en-AU-female":  "en-AU-NatashaNeural",
    "en-AU-male":    "en-AU-WilliamNeural",
}

DEFAULT_VOICE = "en-US-female"


# ── PDF text extraction ───────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract plain text from a PDF byte stream."""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text.strip())
        full_text = "\n".join(pages_text)
        if not full_text.strip():
            raise ValueError("PDF appears to be empty or contains only images.")
        return full_text
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        raise


# ── Cache helpers ─────────────────────────────────────────────────────────────
def _cache_key(text: str, voice_key: str) -> str:
    """Generate a deterministic cache key from text + voice."""
    raw = f"{voice_key}::{text}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.wav"


def get_cached_file(text: str, voice_key: str) -> Optional[Path]:
    """Return cached WAV path if it exists and is still fresh."""
    key = _cache_key(text, voice_key)
    path = _cache_path(key)
    if path.exists():
        age = time.time() - path.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            logger.info(f"Cache HIT: {key[:12]}...")
            return path
        else:
            path.unlink(missing_ok=True)  # stale – remove
    return None


def save_to_cache(text: str, voice_key: str, source_path: Path) -> None:
    """Copy a generated file into the cache."""
    key = _cache_key(text, voice_key)
    dest = _cache_path(key)
    import shutil
    shutil.copy2(source_path, dest)
    logger.info(f"Cached: {key[:12]}...")


# ── Core TTS ─────────────────────────────────────────────────────────────────
async def text_to_wav(
    text: str,
    voice_key: str = DEFAULT_VOICE,
    rate: str = "+0%",
    volume: str = "+0%",
) -> Path:
    """
    Convert text → WAV using edge-tts.
    Returns the Path to the generated .wav file.
    Raises ValueError for bad inputs, RuntimeError for TTS failures.
    """
    # Validate text
    text = text.strip()
    if not text:
        raise ValueError("Text cannot be empty.")
    if len(text) > 50_000:
        raise ValueError("Text exceeds 50,000 character limit.")

    # Resolve voice
    voice_name = VOICE_OPTIONS.get(voice_key)
    if not voice_name:
        raise ValueError(
            f"Unknown voice '{voice_key}'. "
            f"Available: {list(VOICE_OPTIONS.keys())}"
        )

    # Check cache first
    cached = get_cached_file(text, voice_key)
    if cached:
        return cached

    # Generate unique output file
    filename = OUTPUT_DIR / f"{uuid.uuid4().hex}.wav"

    try:
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice_name,
            rate=rate,
            volume=volume,
        )
        await communicate.save(str(filename))
    except Exception as e:
        logger.error(f"edge-tts error: {e}")
        filename.unlink(missing_ok=True)
        raise RuntimeError(f"TTS generation failed: {e}")

    if not filename.exists() or filename.stat().st_size == 0:
        raise RuntimeError("TTS produced an empty file.")

    # Store in cache (only for shorter texts to keep cache size reasonable)
    if len(text) <= 5_000:
        save_to_cache(text, voice_key, filename)

    logger.info(f"Generated: {filename.name} | voice={voice_key} | chars={len(text)}")
    return filename


# ── Garbage collection ────────────────────────────────────────────────────────
def run_garbage_collection() -> int:
    """
    Delete expired files from OUTPUT_DIR.
    Returns number of files deleted.
    """
    now = time.time()
    deleted = 0
    for f in OUTPUT_DIR.glob("*.wav"):
        try:
            age = now - f.stat().st_mtime
            if age > FILE_TTL_SECONDS:
                f.unlink()
                deleted += 1
        except Exception as e:
            logger.warning(f"GC error on {f.name}: {e}")
    if deleted:
        logger.info(f"GC: removed {deleted} expired file(s)")
    return deleted


async def scheduled_gc(interval: int = 300):
    """Background task: run GC every `interval` seconds."""
    while True:
        await asyncio.sleep(interval)
        run_garbage_collection()


# ── Helpers exposed to routes ─────────────────────────────────────────────────
def list_voices() -> dict:
    """Return available voice options."""
    return {
        key: {
            "voice_id": val,
            "language": key.rsplit("-", 1)[0],
            "gender": key.rsplit("-", 1)[1],
        }
        for key, val in VOICE_OPTIONS.items()
    }