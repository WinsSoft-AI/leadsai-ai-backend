"""
Winssoft BMA — STT Service (Local)
Primary:  OpenAI Whisper (runs 100% locally, no API key needed)
Fallback: faster-whisper (CTranslate2 backend — 4× faster, lower RAM)
Model sizes: tiny (75MB) | base (150MB) | small (500MB) | medium (1.5GB) | large (3GB)

Install:
    pip install openai-whisper          # standard whisper
    pip install faster-whisper          # recommended: faster & lighter

Usage in .env:
    WHISPER_MODEL=base                  # default — good speed/quality balance
    WHISPER_DEVICE=cpu                  # or 'cuda' if GPU available
"""
import os
import io
import logging
import tempfile
import hashlib
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Try faster-whisper first (recommended), fall back to openai-whisper ──────
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    WHISPER_BACKEND = "faster"
    logger.info("✅ Using faster-whisper backend")
except ImportError:
    FasterWhisperModel = None
    try:
        import whisper as OpenAIWhisper
        WHISPER_BACKEND = "openai"
        logger.info("✅ Using openai-whisper backend")
    except ImportError:
        OpenAIWhisper = None
        WHISPER_BACKEND = "none"
        logger.warning("⚠️  No Whisper backend found. Run: pip install faster-whisper")

# ── Attempt numpy import for audio processing ─────────────────────────────────
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ── Language code mapping ─────────────────────────────────────────────────────
LANG_MAP = {
    "en": "en", "es": "es", "fr": "fr", "de": "de",
    "zh": "zh", "ar": "ar", "hi": "hi", "ja": "ja",
    "pt": "pt", "ta": "ta", "auto": None,
}


class STTService:
    """
    Local speech-to-text using Whisper.
    Model is loaded once on first use (lazy init) to avoid startup delay.
    """

    def __init__(self):
        self._model = None
        self._model_name    = os.getenv("WHISPER_MODEL",  "base")
        self._device        = os.getenv("WHISPER_DEVICE", "cpu")
        self._compute_type  = os.getenv("WHISPER_COMPUTE", "int8")  # int8 | float16 | float32
        self._sessions: Dict[str, list] = {}   # session_id → buffered audio chunks

    # ── Lazy model loader ────────────────────────────────────────────────────
    def _load_model(self):
        if self._model is not None:
            return True
        if WHISPER_BACKEND == "none":
            return False
        try:
            if WHISPER_BACKEND == "faster":
                self._model = FasterWhisperModel(
                    self._model_name,
                    device=self._device,
                    compute_type=self._compute_type,
                )
                logger.info(f"✅ faster-whisper model '{self._model_name}' loaded on {self._device}")
            else:  # openai
                self._model = OpenAIWhisper.load_model(self._model_name, device=self._device)
                logger.info(f"✅ openai-whisper model '{self._model_name}' loaded")
            return True
        except Exception as e:
            logger.error(f"Whisper model load failed: {e}")
            return False

    # ── Main transcription endpoint ──────────────────────────────────────────
    async def process_chunk(
        self,
        audio_chunk: bytes,
        session_id: str,
        language: str = "auto",
    ) -> Dict:
        """
        Buffer audio chunk. When buffer reaches ~2s of audio, transcribe.
        Returns {text, is_final, language_detected}
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(audio_chunk)

        # ~8KB ≈ 0.5s of 16kHz mono audio — buffer 4 chunks (~2s) before transcribing
        total = sum(len(c) for c in self._sessions[session_id])
        if total < 32_000:
            return {"text": "", "is_final": False, "language_detected": language}

        return await self.transcribe_session(session_id, language)

    async def transcribe_session(self, session_id: str, language: str = "auto") -> Dict:
        """Transcribe all buffered audio for a session."""
        chunks = self._sessions.get(session_id, [])
        if not chunks:
            return {"text": "", "is_final": True, "language_detected": language}

        audio_bytes = b"".join(chunks)
        self._sessions[session_id] = []   # reset buffer

        return await self._transcribe(audio_bytes, language)

    async def _transcribe(self, audio_bytes: bytes, language: str = "auto") -> Dict:
        """Core transcription logic."""
        if not self._load_model():
            # Graceful degradation — return empty so chat still works via text
            return {"text": "", "is_final": True, "language_detected": "en", "error": "Whisper not available"}

        lang_code = LANG_MAP.get(language)  # None = auto-detect

        try:
            # Write to temp file (Whisper works with file paths)
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            if WHISPER_BACKEND == "faster":
                segments, info = self._model.transcribe(
                    tmp_path,
                    language=lang_code,
                    beam_size=5,
                    vad_filter=True,             # Remove silence automatically
                    vad_parameters={"min_silence_duration_ms": 500},
                )
                text = " ".join(s.text.strip() for s in segments)
                detected = info.language or language

            else:  # openai-whisper
                kwargs = {"fp16": False}
                if lang_code:
                    kwargs["language"] = lang_code
                result   = self._model.transcribe(tmp_path, **kwargs)
                text     = result.get("text", "").strip()
                detected = result.get("language", language)

            os.unlink(tmp_path)

            return {
                "text":               text,
                "is_final":           True,
                "language_detected":  detected,
            }

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            return {"text": "", "is_final": True, "language_detected": language, "error": "Transcription failed"}

    def clear_session(self, session_id: str):
        self._sessions.pop(session_id, None)

    @property
    def model_info(self) -> Dict:
        return {
            "backend":    WHISPER_BACKEND,
            "model":      self._model_name,
            "device":     self._device,
            "loaded":     self._model is not None,
        }
