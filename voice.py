import io
import asyncio
import hashlib
import shutil
import time
import uuid
import logging
from pathlib import Path
from typing import Optional

import edge_tts
import PyPDF2

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("generated_audio")
OUTPUT_DIR.mkdir(exist_ok=True)

CACHE_DIR = Path("audio_cache")
CACHE_DIR.mkdir(exist_ok=True)

FILE_TTL_SECONDS  = 600    # 10 min — generated files
CACHE_TTL_SECONDS = 3600   # 1 hour — cached files

# ── Voice table ───────────────────────────────────────────────────────────────
# key → edge-tts voice name
VOICE_OPTIONS: dict[str, str] = {
    # English
    "en-US-female": "en-US-JennyNeural",
    "en-US-male":   "en-US-GuyNeural",
    "en-GB-female": "en-GB-SoniaNeural",
    "en-GB-male":   "en-GB-RyanNeural",
    "en-IN-female": "en-IN-NeerjaNeural",
    "en-IN-male":   "en-IN-PrabhatNeural",
    "en-AU-female": "en-AU-NatashaNeural",
    "en-AU-male":   "en-AU-WilliamNeural",
    # Hindi
    "hi-female":    "hi-IN-SwaraNeural",
    "hi-male":      "hi-IN-MadhurNeural",
    # Marathi
    "mr-female":    "mr-IN-AarohiNeural",
    "mr-male":      "mr-IN-ManoharNeural",
}

DEFAULT_VOICE = "en-US-female"

# ── Auto-detect default voice per language ────────────────────────────────────
# When auto-detect kicks in, we pick female voice as default
LANG_DEFAULT_VOICE: dict[str, str] = {
    "hindi":   "hi-female",
    "marathi": "mr-female",
    "english": "en-IN-female",
}


# ══════════════════════════════════════════════════════════════════════════════
#  LANGUAGE DETECTION  (pure Python, no ML, no extra libraries)
# ══════════════════════════════════════════════════════════════════════════════

def _count_script_chars(text: str) -> dict[str, int]:
    """Count characters belonging to each Unicode script block."""
    counts = {"devanagari": 0, "latin": 0, "other": 0}
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:   # Devanagari block (Hindi + Marathi share this)
            counts["devanagari"] += 1
        elif 0x0041 <= cp <= 0x007A or 0x00C0 <= cp <= 0x024F:  # Latin
            counts["latin"] += 1
        elif ch.isalpha():
            counts["other"] += 1
    return counts


# Marathi-specific words/characters that don't appear in standard Hindi
# ळ (0x0933) is heavily used in Marathi; also common Marathi words
_MARATHI_MARKERS = {
    "\u0933",  # ळ — retroflex lateral, very common in Marathi
    "\u0965",  # । double danda
}
_MARATHI_WORDS = {
    "आहे", "नाही", "आणि", "हे", "ते", "मला", "तुम्ही", "आम्ही",
    "काय", "कसे", "होते", "असे", "येथे", "त्यांनी", "केले", "झाले",
    "मराठी", "महाराष्ट्र",
}
_HINDI_WORDS = {
    "है", "नहीं", "और", "यह", "वह", "मुझे", "आप", "हम",
    "क्या", "कैसे", "था", "ऐसे", "यहाँ", "उन्होंने", "किया", "हुआ",
    "हिंदी", "भारत",
}


def detect_language(text: str) -> str:
    """
    Detect whether the text is 'hindi', 'marathi', or 'english'.
    Uses Unicode script counting + vocabulary heuristics.
    Returns one of: 'hindi' | 'marathi' | 'english'
    """
    sample = text[:2000]  # only check first 2000 chars for speed
    counts = _count_script_chars(sample)
    total_alpha = counts["devanagari"] + counts["latin"] + counts["other"]

    if total_alpha == 0:
        return "english"

    devanagari_ratio = counts["devanagari"] / total_alpha

    # If less than 20% Devanagari → treat as English
    if devanagari_ratio < 0.20:
        return "english"

    # Devanagari detected — distinguish Hindi vs Marathi via vocabulary
    words_in_text = set(sample.split())
    chars_in_text = set(sample)

    marathi_score = (
        len(words_in_text & _MARATHI_WORDS) * 2 +
        len(chars_in_text & _MARATHI_MARKERS) * 3
    )
    hindi_score = len(words_in_text & _HINDI_WORDS) * 2

    if marathi_score > hindi_score:
        return "marathi"
    elif hindi_score > marathi_score:
        return "hindi"
    else:
        # Tie — fall back to ळ presence (strong Marathi signal)
        if "\u0933" in sample:
            return "marathi"
        return "hindi"


def auto_select_voice(text: str, preferred_gender: str = "female") -> str:
    """
    Detect text language and return the best matching voice key.
    preferred_gender: 'female' or 'male'
    """
    lang = detect_language(text)
    gender_suffix = "female" if preferred_gender == "female" else "male"

    voice_map = {
        "hindi":   f"hi-{gender_suffix}",
        "marathi": f"mr-{gender_suffix}",
        "english": f"en-IN-{gender_suffix}",
    }
    voice_key = voice_map.get(lang, DEFAULT_VOICE)
    logger.info(f"Auto-detected language: '{lang}' → voice: '{voice_key}'")
    return voice_key


# ══════════════════════════════════════════════════════════════════════════════
#  PDF EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract plain text from a PDF byte stream."""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages_text.append(t.strip())
        full_text = "\n".join(pages_text)
        if not full_text.strip():
            raise ValueError("PDF appears to be empty or contains only images.")
        return full_text
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        raise


# ══════════════════════════════════════════════════════════════════════════════
#  CACHE
# ══════════════════════════════════════════════════════════════════════════════

def _cache_key(text: str, voice_key: str) -> str:
    raw = f"{voice_key}::{text}"
    return hashlib.sha256(raw.encode()).hexdigest()

def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.wav"

def get_cached_file(text: str, voice_key: str) -> Optional[Path]:
    key = _cache_key(text, voice_key)
    path = _cache_path(key)
    if path.exists():
        age = time.time() - path.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            logger.info(f"Cache HIT: {key[:12]}...")
            return path
        path.unlink(missing_ok=True)
    return None

def save_to_cache(text: str, voice_key: str, source_path: Path) -> None:
    key = _cache_key(text, voice_key)
    shutil.copy2(source_path, _cache_path(key))
    logger.info(f"Cached: {key[:12]}...")


# ══════════════════════════════════════════════════════════════════════════════
#  CORE TTS
# ══════════════════════════════════════════════════════════════════════════════

async def text_to_wav(
    text: str,
    voice_key: str = "auto",        # pass "auto" to enable language detection
    rate: str = "+0%",
    volume: str = "+0%",
    gender: str = "female",         # used only when voice_key == "auto"
) -> tuple[Path, str]:
    """
    Convert text → WAV using edge-tts.
    Returns (Path to WAV file, resolved voice_key that was used).

    voice_key="auto"  → auto-detect language, pick matching voice
    voice_key="hi-female" etc. → use that voice directly
    """
    text = text.strip()
    if not text:
        raise ValueError("Text cannot be empty.")
    if len(text) > 50_000:
        raise ValueError("Text exceeds 50,000 character limit.")

    # Resolve voice
    if voice_key == "auto":
        voice_key = auto_select_voice(text, preferred_gender=gender)

    voice_name = VOICE_OPTIONS.get(voice_key)
    if not voice_name:
        raise ValueError(
            f"Unknown voice '{voice_key}'. "
            f"Available: {list(VOICE_OPTIONS.keys())} or 'auto'"
        )

    # Cache check
    cached = get_cached_file(text, voice_key)
    if cached:
        return cached, voice_key

    # Generate
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

    if len(text) <= 5_000:
        save_to_cache(text, voice_key, filename)

    logger.info(f"Generated: {filename.name} | voice={voice_key} | chars={len(text)}")
    return filename, voice_key


# ══════════════════════════════════════════════════════════════════════════════
#  GARBAGE COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

def run_garbage_collection() -> int:
    now = time.time()
    deleted = 0
    for f in OUTPUT_DIR.glob("*.wav"):
        try:
            if now - f.stat().st_mtime > FILE_TTL_SECONDS:
                f.unlink()
                deleted += 1
        except Exception as e:
            logger.warning(f"GC error on {f.name}: {e}")
    if deleted:
        logger.info(f"GC: removed {deleted} expired file(s)")
    return deleted

async def scheduled_gc(interval: int = 300):
    while True:
        await asyncio.sleep(interval)
        run_garbage_collection()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS FOR ROUTES
# ══════════════════════════════════════════════════════════════════════════════

def list_voices() -> dict:
    result = {}
    for key, val in VOICE_OPTIONS.items():
        parts = key.split("-")
        lang_code = parts[0]
        gender = parts[-1]
        lang_label = {"en": "English", "hi": "Hindi", "mr": "Marathi"}.get(lang_code, lang_code)
        result[key] = {
            "voice_id": val,
            "language": lang_label,
            "language_code": lang_code,
            "gender": gender,
        }
    return result