import os
import json
from typing import List, Dict, Optional
from groq import Groq

class LLMService:
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.1-70b-versatile"):
        """Initializes the Groq LLM client."""
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        
        if not self.api_key:
            print("WARNING: Groq API Key not found. LLMService will run in mock mode.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
            print(f"Initialized Groq LLM with model: {self.model}")

    def generate_answer(self, question: str, context: str, history: List[Dict] = []) -> str:
        """Generates a contextualized response using Groq."""
        if not self.client:
            return "I'm currently in offline mode. Please configure the GROQ_API_KEY to enable live tutoring."

        # Define the system personality and rules
        system_prompt = """
        You are an expert, friendly AI tutor. Your goal is to explain concepts clearly from the provided textbook context.
        
        RULES:
        1. Use ONLY the provided context to answer.
        2. If the information isn't in the context, say: "Based on this chapter, I don't have enough information to answer that accurately."
        3. Cite specific page numbers when mentioned in the context (e.g., "As mentioned on page 5...").
        4. Keep explanations concise but thorough.
        5. At the end of your response, list 3-5 keywords representing the main concepts as 'KEYWORDS: word1, word2, word3'.
        """.strip()

        # Construct the conversation messages
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Add conversation history (limited to last 5 exchanges)
        for msg in history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
        # Add current context and question
        user_prompt = f"Context from Textbook:\n---\n{context}\n---\n\nStudent Question: {question}"
        messages.append({"role": "user", "content": user_prompt})

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3, # Lower temperature for factual accuracy
                max_tokens=1024,
                top_p=1,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            return f"I encountered an error while trying to generate an answer. Error: {str(e)}"
