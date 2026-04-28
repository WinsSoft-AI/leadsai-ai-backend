"""
Winssoft BMA — Gemini API Client
Handles: chat generation, intent analysis, proactive message generation
"""
import httpx
import json
import logging
import os
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")


SYSTEM_PROMPT = """You are a friendly, knowledgeable customer support assistant who works for {business_name}.
You know {business_name}'s products, services, and policies inside-out because you are part of the team.

HOW TO RESPOND:
1. Speak naturally and confidently as if you personally know about {business_name}. NEVER say "based on the information provided", "from the given context", "according to my data", or any phrasing that reveals you are reading from documents.
2. If you genuinely cannot find the answer in the knowledge below, say something natural like: "I'm not sure about that specific detail — let me connect you with someone who can help!" Do NOT say "I don't have that information in my context."
3. Answer the user's question directly and completely. Do not be overly brief (one-liners) or overly verbose (paragraphs). Aim for 2-5 sentences for simple questions, more for detailed/technical ones.
4. Always respond in the SAME language the user writes in.
5. If the user shows purchase intent, naturally guide them toward next steps (contacting sales, placing an order, etc.)
6. Never invent prices, features, specifications, or policies that are not in the knowledge below. If a specific detail isn't available, acknowledge it gracefully and offer to help otherwise.

YOUR KNOWLEDGE ABOUT {business_name}:
{context}

CONVERSATION SO FAR:
{history}
"""

INTENT_SYSTEM_PROMPT = """Analyze this customer chat session and return a JSON object with:
{
  "intent": "purchase | enquiry | support | comparison | pricing | general",
  "products_interested": ["array of exact product names mentioned"],
  "product_quantities": {"exact product name": "integer or string quantity (e.g. 20) if mentioned, else omit"},
  "sentiment": "positive | neutral | negative",
  "urgency": "high | medium | low",
  "summary": "2-3 sentence summary of what the visitor wants",
  "recommended_followup": "specific action the sales team should take",
  "lead_quality": "hot | warm | cold"
}

IMPORTANT: "product_quantities" must ALWAYS be a JSON Object (e.g., {}), NEVER a raw string. Ensure you carefully read the user's messages to accurately detect ANY mentioned products and their quantities.
Return ONLY valid JSON, no other text.
"""

PROACTIVE_SYSTEM_PROMPT = """A website visitor has been browsing. Analyze their behavior and determine if we should proactively engage them.

Visitor behavior:
{behavior_data}

Business: {business_name}
Products/Services: {product_summary}

Return JSON:
{
  "should_trigger": true/false,
  "confidence": 0.0-1.0,
  "message": "personalized opening message to send" or null,
  "reason": "why we should/shouldn't engage"
}

Only trigger if confidence > 0.65. Message must be natural, non-pushy, max 20 words.
Return ONLY valid JSON.
"""


class GeminiClient:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        if os.getenv("ENV", "dev") == "prod" and not self.api_key:
            raise RuntimeError("GEMINI_API_KEY must be set in production")
        self._headers = {"x-goog-api-key": self.api_key} if self.api_key else {}
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def chat(
        self,
        message: str,
        history: List[Dict],
        context_chunks: List[Dict],
        tenant_config: Dict,
        language: str = "en"
    ) -> Dict[str, Any]:
        """Generate a chat response using RAG context."""
        
        # Build context string from retrieved chunks
        context_text = "\n\n".join([
            f"[Source: {c.get('source', 'Product Data')}]\n{c['text']}"
            for c in context_chunks
        ]) if context_chunks else "No specific product data available."
        
        # Build history string
        history_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in history[-10:]
        ]) if history else ""
        
        system = SYSTEM_PROMPT.format(
            business_name=tenant_config.get("name", "our company"),
            context=context_text,
            history=history_text
        )
        
        # Add language instruction if not auto
        if language and language != "auto":
            _lang_names = {
                'en':'English','hi':'Hindi','ta':'Tamil','te':'Telugu','bn':'Bengali',
                'mr':'Marathi','gu':'Gujarati','kn':'Kannada','ml':'Malayalam','pa':'Punjabi',
                'es':'Spanish','fr':'French','de':'German','pt':'Portuguese',
                'ar':'Arabic','zh':'Chinese','ja':'Japanese','ko':'Korean','ru':'Russian','it':'Italian'
            }
            lang_name = _lang_names.get(language, language)
            system += f"\n\nIMPORTANT: The user has selected '{lang_name}' as their preferred language. You MUST respond entirely in {lang_name}."
        
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": message}]}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 1024,
                "topP": 0.85
            }
        }
        
        response = await self._call_gemini(GEMINI_MODEL, payload)
        
        if "error" in response:
            return {
                "text": f"⚠️ AI Provider Error: {response['error'].get('message', str(response['error']))}",
                "sources": [],
                "detected_language": language
            }
            
        candidates = response.get("candidates", [])
        if not candidates:
            return {
                "text": "⚠️ The AI provider returned an empty response. This might be due to content policy restrictions.",
                "sources": [],
                "detected_language": language
            }
            
        candidate = candidates[0]
        if candidate.get("finishReason") == "SAFETY":
            return {
                "text": "⚠️ I'm sorry, I cannot answer that query as it triggered our safety systems.",
                "sources": [],
                "detected_language": language
            }
            
        text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "I'm sorry, I couldn't generate a response.")
        
        return {
            "text": text,
            "sources": [{"source": c.get("source"), "score": c.get("score")} for c in context_chunks[:3]],
            "detected_language": language,
            "usage": response.get("usageMetadata", {})
        }
    
    async def analyze_intent(self, history: List[Dict], tenant_config: Dict) -> Dict[str, Any]:
        """Post-session intent analysis for lead brief."""
        
        conversation = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in history
        ])
        
        payload = {
            "systemInstruction": {"parts": [{"text": INTENT_SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": f"SESSION TRANSCRIPT:\n{conversation}"}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 512}
        }
        
        response = await self._call_gemini(GEMINI_MODEL, payload)
        text = response.get("candidates", [{}])[0].get(
            "content", {}).get("parts", [{}])[0].get("text", "{}")
        
        usage = response.get("usageMetadata", {})
        try:
            # Strip markdown if present
            clean = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            res = json.loads(clean)
            res["usage"] = usage
            return res
        except Exception:
            return {
                "intent": "enquiry",
                "sentiment": "neutral",
                "summary": "Customer had a conversation with the AI assistant.",
                "lead_quality": "warm",
                "recommended_followup": "Follow up within 24 hours"
            }
    
    async def check_proactive_trigger(
        self,
        behavior_events: List[Dict],
        tenant_config: Dict
    ) -> Dict[str, Any]:
        """Decide if/when to proactively engage a visitor."""
        
        behavior_summary = json.dumps(behavior_events[-20:], indent=2)
        product_summary = tenant_config.get("product_summary", "various products and services")
        
        system = PROACTIVE_SYSTEM_PROMPT.format(
            behavior_data=behavior_summary,
            business_name=tenant_config.get("name", "the company"),
            product_summary=product_summary
        )
        
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": "Analyze this visitor and decide."}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 256}
        }
        
        response = await self._call_gemini(GEMINI_MODEL, payload)
        text = response.get("candidates", [{}])[0].get(
            "content", {}).get("parts", [{}])[0].get("text", "{}")
        
        try:
            clean = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(clean)
        except Exception:
            return {"should_trigger": False, "confidence": 0.0, "message": None}
    
    async def generate_text(self, prompt: str, temperature: float = 0.3, max_tokens: int = 1024) -> str:
        """General-purpose text generation (e.g. product enrichment)."""
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        response = await self._call_gemini(GEMINI_MODEL, payload)
        candidates = response.get("candidates", [])
        if not candidates:
            return ""
        return candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings for RAG indexing."""
        embeddings = []
        for text in texts:
            payload = {
                "model": f"models/{GEMINI_EMBED_MODEL}",
                "content": {"parts": [{"text": text[:2048]}]}
            }
            url = f"{GEMINI_API_URL}/{GEMINI_EMBED_MODEL}:embedContent"
            
            try:
                resp = await self.client.post(url, json=payload, headers=self._headers)
                data = resp.json()
                embeddings.append(data.get("embedding", {}).get("values", [0.0] * 768))
            except Exception:
                embeddings.append([0.0] * 768)
        
        return embeddings
    
    async def _call_gemini(self, model: str, payload: Dict) -> Dict:
        """Make a call to the Gemini API."""
        url = f"{GEMINI_API_URL}/{model}:generateContent"
        
        try:
            resp = await self.client.post(url, json=payload, headers=self._headers)
            return resp.json()
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return {"error": {"message": "AI service temporarily unavailable"}, "candidates": []}
