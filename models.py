"""
Pydantic models for Winssoft BMA API
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class LanguageCode(str, Enum):
    EN = "en"
    ES = "es"
    FR = "fr"
    DE = "de"
    ZH = "zh"
    AR = "ar"
    HI = "hi"
    JA = "ja"
    PT = "pt"
    TA = "ta"
    AUTO = "auto"


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    ts: float


class ChatRequest(BaseModel):
    session_id: str
    message: str
    language: LanguageCode = LanguageCode.AUTO
    tts_enabled: bool = True
    image_base64: Optional[str] = None  # For CV queries sent via text mode


class ChatResponse(BaseModel):
    session_id: str
    message: str
    audio_url: Optional[str] = None
    sources: List[Dict[str, Any]] = []
    needs_pii_prompt: bool = False
    language: str = "en"
    suggested_products: List[Dict[str, Any]] = []


class LeadCapture(BaseModel):
    session_id: str
    name: str
    email: str
    phone: Optional[str] = None
    consent: bool = True  # Must be True


class BehaviorEvent(BaseModel):
    session_id: str
    visitor_id: str  # Anonymous fingerprint
    event_type: str  # "page_view" | "scroll" | "click" | "dwell" | "return_visit"
    page_path: str
    metadata: Dict[str, Any] = {}
    dwell_seconds: Optional[float] = None


class TenantConfig(BaseModel):
    name: str
    domain: str
    primary_color: str = "#2E3A8C"
    accent_color: str = "#00B4D8"
    logo_url: Optional[str] = None
    greeting: str = "Hi! I'm your AI assistant. How can I help you today?"
    languages: List[str] = ["en"]
    proactive_enabled: bool = True
    proactive_delay_seconds: int = 30
    pii_prompt_after_messages: int = 3
    notification_email: str
    business_type: str = "general"  # ecommerce | saas | services | general
    tts_voice: str = "en-US-Neural2-F"
    widget_position: str = "bottom-right"  # bottom-right | bottom-left
    auto_close_minutes: int = 30


class IngestJob(BaseModel):
    job_id: str
    status: str  # pending | processing | done | error
    filename: str
    chunks_indexed: int = 0
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


class SessionSummary(BaseModel):
    session_id: str
    tenant_id: str
    message_count: int
    pii_collected: bool
    pii: Optional[Dict[str, str]] = None
    history: List[Dict] = []
    behavior_events: List[Dict] = []
    language: str = "en"
    started_at: float
    last_active: float


class AnalyticsData(BaseModel):
    total_sessions: int
    total_leads: int
    conversion_rate: float
    avg_messages_per_session: float
    top_intents: List[Dict[str, Any]]
    sessions_by_day: List[Dict[str, Any]]
    language_breakdown: Dict[str, int]


class ProactiveCheck(BaseModel):
    trigger: bool
    message: Optional[str] = None
    confidence: float = 0.0
    reason: Optional[str] = None


class PlanCreate(BaseModel):
    id: str
    name: str
    amount_rupee: int
    currency: str = "INR"
    sessions: int
    leads: int
    ticket_limit: int
    api_limit: int
    rate_limit_rpm: int = 60
    rate_limit_rpd: int = 5000
    description: Optional[str] = None
    razorpay_plan_id: Optional[str] = None


class PlanUpdate(BaseModel):
    name: Optional[str] = None
    amount_rupee: Optional[int] = None
    currency: Optional[str] = None
    sessions: Optional[int] = None
    leads: Optional[int] = None
    ticket_limit: Optional[int] = None
    api_limit: Optional[int] = None
    rate_limit_rpm: Optional[int] = None
    rate_limit_rpd: Optional[int] = None
    description: Optional[str] = None
    razorpay_plan_id: Optional[str] = None
