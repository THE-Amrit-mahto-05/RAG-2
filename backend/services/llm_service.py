import os
import json
import re
from typing import List, Dict, Optional, Any
from groq import Groq

# Optional imports for other providers
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

import requests # For Ollama

class LLMService:
    def __init__(self, provider: str = "groq", api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initializes the LLM service.
        Providers: 'groq', 'openai', 'ollama'
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")

        if provider == "groq":
            if not self.api_key:
                print("WARNING: Groq API Key not found.")
                self.client = None
            else:
                self.client = Groq(api_key=self.api_key)
                self.model = model or "llama-3.1-70b-versatile"
                print(f"Initialized Groq: {self.model}")
        
        elif provider == "openai":
            if not OpenAI:
                raise ImportError("OpenAI package not installed.")
            if not self.api_key:
                print("WARNING: OpenAI API Key not found.")
                self.client = None
            else:
                self.client = OpenAI(api_key=self.api_key)
                self.model = model or "gpt-4o-mini"
                print(f"Initialized OpenAI: {self.model}")
        
        elif provider == "ollama":
            self.model = model or "llama3.1"
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api/chat")
            print(f"Initialized Ollama: {self.model}")

    def generate_answer(self, question: str, context: str, history: List[Dict] = [], has_context: bool = True) -> Dict[str, Any]:
        """Generates a contextualized response with a groundedness check."""
        
        if not has_context:
            system_prompt = """You are a friendly AI tutor for a textbook chapter on Sound. 
            The student is chatting with you but no specific chapter context was retrieved. 
            Respond briefly, stating that you can only answer questions related to the loaded textbook chapter. Do not provide outside knowledge.
            IMPORTANT: Do not generate links, markdown images, or QR codes.
            """.strip()
        else:
            system_prompt = """
            You are an expert, friendly AI tutor. Your goal is to explain concepts clearly from the provided textbook context.
            
            RULES:
            1. STRICT GROUNDING: You MUST ONLY use the provided textbook context. Do not use any outside or general knowledge. If the context does not contain the answer, simply say: "I cannot find this information in the textbook."
            2. You MUST include inline citations using exactly this format: [Source: Page X].
            3. Format answers clearly with line breaks between paragraphs.
            4. Do not generate links, markdown images, or QR codes. Provide a clear text explanation.
            5. MUST end your response exactly with this format:
            KEYWORDS: word1, word2, word3
            (Pick 3 main topics from the answer)
            """.strip()

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        user_prompt = f"Textbook Context:\n{context}\n\nStudent Question: {question}"
        messages.append({"role": "user", "content": user_prompt})

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
            
            elif self.provider == "ollama":
                resp = requests.post(self.base_url, json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.2}
                })
                full_response = resp.json()["message"]["content"]

            # Post-Process: Extract Keywords
            keywords = []
            kw_match = re.search(r'KEYWORDS:\s*(.*)', full_response, re.IGNORECASE)
            if kw_match:
                keywords = [k.strip() for k in kw_match.group(1).split(',')]
                # Remove keywords block from display answer
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

        except Exception as e:
            error_msg = str(e).lower()
            print(f"LLM Error: {error_msg}")
            
            # Map ugly provider errors to friendly student-facing messages
            if "429" in error_msg or "rate limit" in error_msg:
                friendly_message = "I'm currently receiving too many questions at once! Please wait a moment and try asking again."
            elif "401" in error_msg or "authentication" in error_msg:
                friendly_message = "There seems to be an issue with my API credentials. Please contact the administrator."
            else:
                friendly_message = "I encountered an unexpected error while thinking. Please try rephrasing your question."
                
            return {"answer": friendly_message, "keywords": []}

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
