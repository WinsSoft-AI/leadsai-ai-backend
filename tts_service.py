"""
Winssoft BMA — TTS Service (Local)
Primary:  Coqui TTS  — neural voices, multilingual, 100% offline
          pip install TTS
Fallback: pyttsx3    — system voices, zero dependencies, works everywhere
          pip install pyttsx3
Fallback2: espeak    — ultra-lightweight, available on all Linux/Windows

Models auto-download on first use (~100-400MB, cached in ~/.local/share/tts)

.env options:
    TTS_ENGINE=coqui                     # coqui | pyttsx3 | espeak | auto (default)
    TTS_MODEL=tts_models/en/ljspeech/tacotron2-DDC   # any Coqui model
    TTS_SPEAKER=                         # for multi-speaker models
    TTS_DEVICE=cpu                       # cpu | cuda
"""
import os
import io
import base64
import logging
import hashlib
import tempfile
import asyncio
import collections
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# ── Engine availability ────────────────────────────────────────────────────────
try:
    from TTS.api import TTS as CoquiTTS
    COQUI_AVAILABLE = True
    logger.info("✅ Coqui TTS available")
except ImportError:
    CoquiTTS = None
    COQUI_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
    logger.info("✅ pyttsx3 available")
except ImportError:
    pyttsx3 = None
    PYTTSX3_AVAILABLE = False

# ── Language → Coqui model mapping ───────────────────────────────────────────
COQUI_MODELS = {
    "en": "tts_models/en/ljspeech/tacotron2-DDC",
    "es": "tts_models/es/css10/vits",
    "fr": "tts_models/fr/css10/vits",
    "de": "tts_models/de/thorsten/tacotron2-DDC",
    "zh": "tts_models/zh-CN/baker/tacotron2-DDC-GST",
    "ja": "tts_models/ja/kokoro/tacotron2-DDC",
    "pt": "tts_models/pt/cv/vits",
    "pl": "tts_models/pl/mai_female/vits",
    # For unsupported languages, fall back to multilingual model
    "default": "tts_models/multilingual/multi-dataset/your_tts",
}

# ── pyttsx3 voice language hints ─────────────────────────────────────────────
PYTTSX3_LANG = {
    "en": "en", "es": "es", "fr": "fr", "de": "de",
    "zh": "zh", "ar": "ar", "hi": "hi", "ja": "ja",
    "pt": "pt",
}


class TTSService:
    """
    Local text-to-speech. Priority: Coqui TTS → pyttsx3 → silent (no audio).
    Models are lazy-loaded on first use.
    """

    def __init__(self):
        self._coqui_models: Dict[str, any] = {}   # lang → loaded model
        self._pyttsx3_engine = None
        self._cache: collections.OrderedDict[str, str] = collections.OrderedDict()
        self._engine = os.getenv("TTS_ENGINE", "auto").lower()
        self._tts_model_name = os.getenv("TTS_MODEL", "")
        self._device = os.getenv("TTS_DEVICE", "cpu")
        self._cache_size = 200                     # max cached responses

    # ── Public interface ─────────────────────────────────────────────────────
    async def synthesize(
        self,
        text: str,
        language: str = "en",
        session_id: Optional[str] = None,
        voice_name: Optional[str] = None,
    ) -> Optional[str]:
        """Convert text to speech. Returns base64 data URL or None."""
        text = text.strip()
        if not text:
            return None

        # Truncate long text
        text = text[:600]

        # Cache lookup
        key = hashlib.md5(f"{text}{language}".encode()).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)  # LRU: mark as recently used
            return self._cache[key]

        # Run sync TTS in thread pool to avoid blocking FastAPI event loop
        loop = asyncio.get_event_loop()
        try:
            audio_bytes = await loop.run_in_executor(
                None, self._synthesize_sync, text, language
            )
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return None

        if not audio_bytes:
            return None

        # Encode to data URL
        b64 = base64.b64encode(audio_bytes).decode()
        data_url = f"data:audio/wav;base64,{b64}"

        # Cache with LRU eviction
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)  # evict least-recently-used
        self._cache[key] = data_url

        return data_url

    # ── Sync synthesis (runs in thread executor) ─────────────────────────────
    def _synthesize_sync(self, text: str, language: str) -> Optional[bytes]:
        engine = self._engine

        if engine == "auto":
            if COQUI_AVAILABLE:
                engine = "coqui"
            elif PYTTSX3_AVAILABLE:
                engine = "pyttsx3"
            else:
                engine = "none"

        if engine == "coqui" and COQUI_AVAILABLE:
            result = self._coqui_synthesize(text, language)
            if result:
                return result
            # fallthrough to pyttsx3

        if engine in ("pyttsx3", "coqui") and PYTTSX3_AVAILABLE:
            return self._pyttsx3_synthesize(text, language)

        return None

    # ── Coqui TTS ────────────────────────────────────────────────────────────
    def _coqui_synthesize(self, text: str, language: str) -> Optional[bytes]:
        try:
            model_name = self._tts_model_name or COQUI_MODELS.get(language, COQUI_MODELS["default"])

            if model_name not in self._coqui_models:
                logger.info(f"Loading Coqui model: {model_name}")
                self._coqui_models[model_name] = CoquiTTS(model_name, gpu=(self._device == "cuda"))

            tts = self._coqui_models[model_name]

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                path = tmp.name

            # Multi-speaker models need a speaker
            if hasattr(tts, "speakers") and tts.speakers:
                speaker = tts.speakers[0]
                tts.tts_to_file(text=text, speaker=speaker, language=language[:2], file_path=path)
            else:
                tts.tts_to_file(text=text, file_path=path)

            with open(path, "rb") as f:
                audio = f.read()
            os.unlink(path)
            return audio

        except Exception as e:
            logger.warning(f"Coqui synthesis failed ({e}), trying fallback")
            return None

    # ── pyttsx3 (system voices — works everywhere) ───────────────────────────
    def _pyttsx3_synthesize(self, text: str, language: str) -> Optional[bytes]:
        try:
            if self._pyttsx3_engine is None:
                self._pyttsx3_engine = pyttsx3.init()
                self._pyttsx3_engine.setProperty("rate", 175)
                self._pyttsx3_engine.setProperty("volume", 0.9)

            engine = self._pyttsx3_engine

            # Try to select a voice matching the language
            voices = engine.getProperty("voices")
            lang_hint = PYTTSX3_LANG.get(language, "en")
            for v in voices:
                if lang_hint in (v.id or "").lower() or lang_hint in (v.name or "").lower():
                    engine.setProperty("voice", v.id)
                    break

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                path = tmp.name

            engine.save_to_file(text, path)
            engine.runAndWait()

            with open(path, "rb") as f:
                audio = f.read()
            os.unlink(path)
            return audio if len(audio) > 100 else None

        except Exception as e:
            logger.error(f"pyttsx3 synthesis failed: {e}")
            return None

    @property
    def engine_info(self) -> Dict:
        return {
            "configured_engine": self._engine,
            "coqui_available":   COQUI_AVAILABLE,
            "pyttsx3_available": PYTTSX3_AVAILABLE,
            "cached_items":      len(self._cache),
        }
