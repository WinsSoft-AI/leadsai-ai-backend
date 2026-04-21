"""
Winssoft BMA — CV Service (Local)
Primary:   YOLOv8  — object detection (ultralytics)   pip install ultralytics
Secondary: CLIP    — zero-shot image understanding     pip install open-clip-torch
Fallback:  Pillow  — basic image metadata extraction   pip install Pillow

Workflow:
  1. YOLO detects objects → class labels + bounding boxes
  2. CLIP encodes image → dense embedding → matched against text descriptions
  3. Results merged → product search query → RAG lookup
  4. If neither available → Pillow extracts color/size hints

Install:
    pip install ultralytics open-clip-torch Pillow
"""
import os
import io
import time
import logging
import hashlib
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ── YOLO ──────────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("✅ YOLOv8 available")
except ImportError:
    YOLO = None
    YOLO_AVAILABLE = False
    logger.warning("⚠️  ultralytics not found. Run: pip install ultralytics")

# ── CLIP ──────────────────────────────────────────────────────────────────────
try:
    import open_clip
    import torch
    CLIP_AVAILABLE = True
    logger.info("✅ OpenCLIP available")
except ImportError:
    open_clip = None
    torch = None
    CLIP_AVAILABLE = False

# ── Pillow fallback ───────────────────────────────────────────────────────────
try:
    from PIL import Image as PILImage
    PILLOW_AVAILABLE = True
except ImportError:
    PILImage = None
    PILLOW_AVAILABLE = False

# ── YOLO model sizes ──────────────────────────────────────────────────────────
# yolov8n (nano, 6MB) | yolov8s (small, 22MB) | yolov8m (medium, 50MB)
# yolov8l (large, 83MB) | yolov8x (xlarge, 130MB)
# Models auto-download on first use to ~/.cache/
YOLO_MODEL_SIZE = os.getenv("YOLO_MODEL", "yolov8n.pt")   # nano for speed
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL", "ViT-B-32")
CLIP_PRETRAINED = os.getenv("CLIP_PRETRAINED", "openai")

MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB


class CVService:
    """
    Local computer vision: detect product types + generate search queries.
    No cloud API, no API key required.
    """

    def __init__(self):
        self._yolo: Optional[any] = None
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._device = "cuda" if (CLIP_AVAILABLE and torch and torch.cuda.is_available()) else "cpu"
        self._cache: Dict[str, List[Dict]] = {}
        self._cache_max = 500  # max cached results

    # ── Lazy loaders ─────────────────────────────────────────────────────────
    def _load_yolo(self) -> bool:
        if self._yolo is not None:
            return True
        if not YOLO_AVAILABLE:
            return False
        try:
            self._yolo = YOLO(YOLO_MODEL_SIZE)
            logger.info(f"✅ YOLO model loaded: {YOLO_MODEL_SIZE}")
            return True
        except Exception as e:
            logger.error(f"YOLO load failed: {e}")
            return False

    def _load_clip(self) -> bool:
        if self._clip_model is not None:
            return True
        if not CLIP_AVAILABLE:
            return False
        try:
            self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
                CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=self._device
            )
            self._clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
            self._clip_model.eval()
            logger.info(f"✅ CLIP model loaded: {CLIP_MODEL_NAME}")
            return True
        except Exception as e:
            logger.error(f"CLIP load failed: {e}")
            return False

    # ── Main interface ────────────────────────────────────────────────────────
    async def find_products(
        self,
        image_bytes: bytes,
        tenant_id: str,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Analyze image → detect objects → generate product search query.
        Returns list of match dicts compatible with existing API response format.
        """
        # Reject oversized images
        if len(image_bytes) > MAX_IMAGE_BYTES:
            return [{"match_score": 0, "error": "Image too large (max 5MB)"}]

        # Cache by full image hash (SHA-256 for collision resistance)
        img_hash = hashlib.sha256(image_bytes).hexdigest()
        cache_key = f"{tenant_id}:{img_hash}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._analyze_sync, image_bytes, top_k)

        # Cache with size limit
        if len(self._cache) >= self._cache_max:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[cache_key] = result
        return result

    def _analyze_sync(self, image_bytes: bytes, top_k: int) -> List[Dict]:
        """Full synchronous analysis pipeline."""
        detections = self._yolo_detect(image_bytes)
        clip_desc  = self._clip_classify(image_bytes)
        color_hint = self._color_hint(image_bytes)

        # Build product type from detections + CLIP
        if detections:
            primary_object = detections[0]["label"]
        elif clip_desc:
            primary_object = clip_desc
        else:
            primary_object = "product"

        # Build rich search query
        parts = [primary_object]
        if color_hint:
            parts.append(color_hint)
        if len(detections) > 1:
            parts.extend(d["label"] for d in detections[1:3])

        search_query = " ".join(dict.fromkeys(parts))  # deduplicated

        # Format for confidence scoring
        confidence = 0.85 if detections else (0.65 if clip_desc else 0.45)

        result = [{
            "match_score":      confidence,
            "product_type":     primary_object,
            "description":      self._build_description(detections, clip_desc, color_hint),
            "search_query":     search_query,
            "visual_features":  [d["label"] for d in detections] + ([clip_desc] if clip_desc else []),
            "detections":       detections[:5],
            "color":            color_hint,
            "message": (
                f"I can see this looks like a **{primary_object}**"
                + (f" in {color_hint}" if color_hint else "")
                + f". Let me search our catalog for matching products!"
            ),
            "detection_engine": (
                "YOLOv8 + CLIP" if (detections and clip_desc) else
                "YOLOv8" if detections else
                "CLIP" if clip_desc else
                "basic"
            ),
        }]

        return result

    # ── YOLO detection ────────────────────────────────────────────────────────
    def _yolo_detect(self, image_bytes: bytes) -> List[Dict]:
        if not self._load_yolo():
            return []
        try:
            if not PILLOW_AVAILABLE:
                return []
            img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
            results = self._yolo(img, verbose=False)[0]
            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label  = results.names[cls_id]
                conf   = float(box.conf[0])
                if conf > 0.3:
                    detections.append({
                        "label":      label,
                        "confidence": round(conf, 3),
                        "bbox":       [round(x, 1) for x in box.xyxy[0].tolist()],
                    })
            # Sort by confidence
            detections.sort(key=lambda x: x["confidence"], reverse=True)
            return detections[:8]
        except Exception as e:
            logger.warning(f"YOLO detection error: {e}")
            return []

    # ── CLIP zero-shot classification ─────────────────────────────────────────
    def _clip_classify(self, image_bytes: bytes) -> Optional[str]:
        if not self._load_clip():
            return None
        try:
            img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
            img_tensor = self._clip_preprocess(img).unsqueeze(0).to(self._device)

            # Product category candidates
            candidates = [
                "clothing apparel shirt dress pants jacket",
                "shoe sneaker boot sandal footwear",
                "bag handbag backpack purse wallet",
                "electronics phone laptop tablet computer",
                "furniture chair sofa table desk bed",
                "food drink beverage meal snack",
                "beauty cosmetics skincare makeup perfume",
                "watch jewelry accessory necklace ring",
                "sports equipment gym fitness outdoor",
                "toy game book stationery office",
                "home decor art plant kitchen appliance",
                "car vehicle automobile part accessory",
            ]

            text_tokens = self._clip_tokenizer(candidates).to(self._device)

            with torch.no_grad():
                img_features  = self._clip_model.encode_image(img_tensor)
                text_features = self._clip_model.encode_text(text_tokens)

                img_features  = img_features  / img_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                probs = (100.0 * img_features @ text_features.T).softmax(dim=-1)

            best_idx  = probs[0].argmax().item()
            best_prob = probs[0][best_idx].item()

            if best_prob > 0.1:
                # Return first word of category as clean label
                return candidates[best_idx].split()[0]
            return None

        except Exception as e:
            logger.warning(f"CLIP classify error: {e}")
            return None

    # ── Color hint from Pillow ────────────────────────────────────────────────
    def _color_hint(self, image_bytes: bytes) -> Optional[str]:
        if not PILLOW_AVAILABLE:
            return None
        try:
            img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
            img = img.resize((50, 50))
            pixels = list(img.getdata())
            avg_r = sum(p[0] for p in pixels) // len(pixels)
            avg_g = sum(p[1] for p in pixels) // len(pixels)
            avg_b = sum(p[2] for p in pixels) // len(pixels)

            # Map to basic color name
            max_c = max(avg_r, avg_g, avg_b)
            min_c = min(avg_r, avg_g, avg_b)

            if max_c - min_c < 30:
                if max_c < 80:
                    return "black"
                elif max_c > 200:
                    return "white"
                else:
                    return "grey"

            if avg_r > avg_g and avg_r > avg_b:
                return "red" if avg_r > 180 else "dark red"
            elif avg_g > avg_r and avg_g > avg_b:
                return "green" if avg_g > 150 else "dark green"
            elif avg_b > avg_r and avg_b > avg_g:
                return "blue" if avg_b > 150 else "dark blue"
            elif avg_r > 150 and avg_g > 150 and avg_b < 100:
                return "yellow"
            elif avg_r > 180 and avg_g > 100 and avg_b < 80:
                return "orange"
            elif avg_r > 150 and avg_b > 150 and avg_g < 100:
                return "purple"

            return None
        except Exception:
            return None

    def _build_description(self, detections, clip_desc, color):
        parts = []
        if detections:
            labels = list(dict.fromkeys(d["label"] for d in detections[:3]))
            parts.append(f"Detected: {', '.join(labels)}")
        if clip_desc:
            parts.append(f"Category: {clip_desc}")
        if color:
            parts.append(f"Color: {color}")
        return ". ".join(parts) + "." if parts else "Product image analyzed."

    @property
    def engine_info(self) -> Dict:
        return {
            "yolo_available":      YOLO_AVAILABLE,
            "clip_available":      CLIP_AVAILABLE,
            "pillow_available":    PILLOW_AVAILABLE,
            "yolo_model":          YOLO_MODEL_SIZE,
            "clip_model":          CLIP_MODEL_NAME,
            "device":              self._device,
            "yolo_loaded":         self._yolo is not None,
            "clip_loaded":         self._clip_model is not None,
        }


# ═══════════════════════════════════════════════════════════════
# BEHAVIOR ANALYZER (unchanged from previous version)
# ═══════════════════════════════════════════════════════════════
class BehaviorAnalyzer:
    def __init__(self):
        self._visitor_data: Dict[str, List] = {}

    async def process_event(self, event, tenant_id: str, gemini) -> Dict:
        vid = event.visitor_id
        if vid not in self._visitor_data:
            self._visitor_data[vid] = []

        self._visitor_data[vid].append({
            "type":  event.event_type,
            "page":  event.page_path,
            "ts":    time.time(),
            "dwell": event.dwell_seconds,
            "meta":  event.metadata,
        })

        events = self._visitor_data[vid]
        if len(events) < 3:
            return {"trigger": False, "message": None}

        if not self._heuristic_check(events):
            return {"trigger": False, "message": None}

        result = await gemini.check_proactive_trigger(
            behavior_events=events,
            tenant_config={"name": tenant_id, "product_summary": "various products"},
        )

        self._visitor_data[vid] = []

        if result.get("should_trigger") and result.get("confidence", 0) > 0.6:
            return {
                "trigger":    True,
                "message":    result.get("message", "Hi! Can I help you find what you're looking for?"),
                "confidence": result.get("confidence", 0.7),
            }
        return {"trigger": False, "message": None}

    def _heuristic_check(self, events: List[Dict]) -> bool:
        pages = set(e["page"] for e in events)
        if len(pages) >= 3:
            return True
        if any(e.get("dwell", 0) and e["dwell"] > 60 for e in events):
            return True
        if any(e["type"] == "return_visit" for e in events):
            return True
        product_views = sum(1 for e in events if "/product" in e.get("page","") or "/pricing" in e.get("page",""))
        return product_views >= 2
