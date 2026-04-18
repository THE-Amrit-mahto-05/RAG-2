import os
import json
import re
import time
from typing import List, Dict, Optional, Any
from groq import Groq

# Optional imports for other providers
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

import requests # For Ollama

class LLMService:
    def __init__(self, provider: str = "groq", api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initializes the LLM service.
        Providers: 'groq', 'openai', 'ollama', 'google'
        """
        self.provider = provider
        self.model = model
        
        # --- GROQ Multi-Key Support ---
        self.groq_keys = []
        if provider == "groq":
            primary_key = api_key or os.getenv("GROQ_API_KEY")
            if primary_key: self.groq_keys.append(primary_key)
            
            idx = 2
            while True:
                extra_key = os.getenv(f"GROQ_API_KEY_{idx}")
                if extra_key:
                    self.groq_keys.append(extra_key)
                    idx += 1
                else: break
            
            self.current_groq_idx = 0
            if not self.groq_keys:
                print("WARNING: No Groq API Keys found.")
                self.client = None
            else:
                self.client = Groq(api_key=self.groq_keys[0])
                self.model = model or "llama-3.1-8b-instant"
                print(f"Initialized Groq with {len(self.groq_keys)} available keys. Model: {self.model}")
        
        elif provider == "google":
            if not genai:
                raise ImportError("google-generativeai package not installed.")
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self.model = model or "gemini-1.5-flash"
                print(f"Initialized Google Gemini: {self.model}")
        
        elif provider == "openai":
            if not OpenAI:
                raise ImportError("OpenAI package not installed.")
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if self.api_key:
                self.client = OpenAI(api_key=self.api_key)
                self.model = model or "gpt-4o-mini"
                print(f"Initialized OpenAI: {self.model}")

    def _rotate_groq_key(self):
        """Switches to the next available Groq API key. Returns True if a full cycle is completed."""
        if self.provider == "groq" and len(self.groq_keys) > 1:
            self.current_groq_idx = (self.current_groq_idx + 1) % len(self.groq_keys)
            new_key = self.groq_keys[self.current_groq_idx]
            self.client = Groq(api_key=new_key)
            print(f"🔄 Swapped to Groq Backup Key (Key {self.current_groq_idx + 1})")
            return self.current_groq_idx == 0 # True if we wrapped around to the start
        return False
        
    def generate_answer(self, question: str, context: str, history: List[Dict] = [], has_context: bool = True) -> Dict[str, Any]:
        """Generates a contextualized response with a groundedness check."""
        
        if not has_context:
            system_prompt = """You are a friendly AI tutor for an educational textbook. 
            The student is chatting with you but no specific chapter context was retrieved. 
            Respond briefly, stating that you can only answer questions related to the loaded textbook chapter. Do not provide outside knowledge.
            CRITICAL: Do not generate any URLs, markdown links [text](url), or QR codes.
            """.strip()
        else:
            system_prompt = """
            You are an expert, friendly AI tutor. Your goal is to explain concepts clearly from the provided textbook context.
            
            STRICT RULES:
            1. GROUNDING: You MUST ONLY use the provided context. Do not use outside knowledge. If the context is missing info, say: "I cannot find this in the textbook."
            2. CITATIONS: You MUST include inline citations using exactly this format: [Source: Page X].
            3. NO LINKS: Do not generate ANY URLs, markdown links [text](url), placeholder images, or QR codes. Use only plain text explanations.
            4. PEDAGOGY: Use helpful analogies from the text. For example, explain Inertia using the "bus starting or stopping" analogy if the context relates to motion.
            
            OUTPUT FORMAT:
            - Provide the explanation first.
            - MUST end your response exactly with this tag:
            IMAGE_KEYWORDS: [keyword1, keyword2]
            (Pick 2-3 specific nouns that represent the best diagram for this answer, e.g., [bus inertia], [friction surface], [longitudinal wave])
            """.strip()

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        user_prompt = f"Textbook Context:\n{context}\n\nStudent Question: {question}"
        messages.append({"role": "user", "content": user_prompt})

        max_retries = max(3, len(self.groq_keys) * 3 if self.provider == "groq" else 3)
        retry_delay = 2 # seconds

        for attempt in range(max_retries):
            try:
                full_response = ""
                if self.provider == "groq" or self.provider == "openai":
                    if not self.client:
                        return {"answer": "LLM provider is not configured.", "keywords": []}
                    
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.2,
                        max_tokens=2048
                    )
                    full_response = completion.choices[0].message.content
                
                elif self.provider == "google":
                    if not genai:
                        return {"answer": "Google Generative AI package not installed.", "keywords": []}
                    
                    model = genai.GenerativeModel(self.model)
                    google_history = []
                    for m in messages[:-1]:
                        role = "user" if m["role"] == "user" else "model"
                        google_history.append({"role": role, "parts": [m["content"]]})
                    
                    response = model.generate_content(
                        messages[-1]["content"],
                        generation_config=genai.types.GenerationConfig(temperature=0.2),
                        history=google_history
                    )
                    full_response = response.text

                elif self.provider == "ollama":
                    self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api/chat")
                    resp = requests.post(self.base_url, json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": 0.2}
                    })
                    full_response = resp.json()["message"]["content"]

                break

            except Exception as e:
                error_msg = str(e).lower()
                print(f"LLM Error (Attempt {attempt+1}/{max_retries}): {error_msg}")
                
                if ("429" in error_msg or "rate limit" in error_msg) and attempt < max_retries - 1:
                    wrapped_around = self._rotate_groq_key()
                    
                    if wrapped_around or len(self.groq_keys) <= 1:
                        print(f"Rate limit hit globally. Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    continue
                
                friendly_message = "I encountered an error while thinking. Please try again."
                if "429" in error_msg:
                    friendly_message = "I'm currently receiving too many questions. Please wait a moment."
                
                return {"answer": friendly_message, "keywords": []}

        # Post-Process: Extract Keywords for Image Matching
        keywords = []
        # Support both IMAGE_KEYWORDS: [a, b] and legacy KEYWORDS: a, b
        kw_match = re.search(r'IMAGE_KEYWORDS:\s*\[(.*?)\]', full_response, re.IGNORECASE)
        if not kw_match:
            kw_match = re.search(r'KEYWORDS:\s*(.*)', full_response, re.IGNORECASE)

        if kw_match:
            raw_kws = kw_match.group(1)
            keywords = [k.strip() for k in raw_kws.split(',')]
            # Remove the keywords/metadata block from the display answer
            display_answer = full_response[:kw_match.start()].strip()
        else:
            display_answer = full_response

        # Simple Groundedness Check (Noun/Verb overlap)
        is_grounded = self._check_groundedness(display_answer, context)

        return {
            "answer": display_answer,
            "keywords": keywords,
            "is_grounded": is_grounded
        }

    def extract_toc(self, text: str) -> List[Dict[str, Any]]:
        """Extracts a structured Table of Contents from raw text using the LLM."""
        system_prompt = """
        You are a Document Structure Expert. Your task is to extract a logical Table of Contents (TOC) from the provided textbook text.
        
        RULES:
        1. Identify the 5-8 most important chapter headings or topic titles.
        2. Assign a logical section number if none exists (e.g., 1, 2, 3 or A, B, C).
        3. Return the result STRICTLY as a JSON list of objects with "section", "title", and "page" fields.
        4. If you cannot find a page number, estimate it based on the text flow or use 1.
        
        Example Output:
        [
            {"section": "1", "title": "Introduction to Biology", "page": 1},
            {"section": "2", "title": "Cell Structure", "page": 4}
        ]
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract TOC from this text:\n\n{text[:8000]}"} # Send first 8k chars
        ]

        max_retries = max(3, len(self.groq_keys) * 3 if self.provider == "groq" else 3)
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                raw_content = ""
                if self.provider == "groq" or self.provider == "openai":
                    if not self.client: return []
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.1,
                        response_format={ "type": "json_object" } if self.provider == "openai" else None
                    )
                    raw_content = completion.choices[0].message.content
                
                elif self.provider == "google":
                    if not genai: return []
                    model = genai.GenerativeModel(self.model)
                    response = model.generate_content(
                        f"{system_prompt}\n\nExtract TOC from this text:\n\n{text[:8000]}",
                        generation_config=genai.types.GenerationConfig(temperature=0.1)
                    )
                    raw_content = response.text

                elif self.provider == "ollama":
                    resp = requests.post(self.base_url, json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "format": "json"
                    })
                    raw_content = resp.json()["message"]["content"]

                # Parse JSON from response
                json_match = re.search(r'\[.*\]', raw_content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
                
                return json.loads(raw_content)

            except Exception as e:
                error_msg = str(e).lower()
                if ("429" in error_msg or "rate limit" in error_msg) and attempt < max_retries - 1:
                    wrapped_around = self._rotate_groq_key()
                    
                    if wrapped_around or len(self.groq_keys) <= 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    continue
                print(f"Error extracting TOC: {e}")
                return []
        
        return []

    def _check_groundedness(self, answer: str, context: str) -> bool:
        """Simple check to see if key factual terms in answer exist in context."""
        # This is a basic version; for production, one might use an LLM-based grader.
        # We look for uncommon words (len > 5) in the answer.
        answer_words = set(re.findall(r'\b\w{6,}\b', answer.lower()))
        context_words = set(re.findall(r'\b\w+\b', context.lower()))
        
        if not answer_words: return True
        
        overlap = answer_words.intersection(context_words)
        coverage = len(overlap) / len(answer_words)
        
        return coverage > 0.4 # Minimum 40% keyword overlap for safety
