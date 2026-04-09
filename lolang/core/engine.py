import json
import os
from typing import Optional, List, Dict
from nexttoken import NextToken
from lolang.core.logger import logger

class LolangCore:
    def __init__(self, model: str = "gemini-3-flash-preview", seed: int = 279):
        self.client = NextToken()
        self.model = model
        self.seed = seed
        self._system_rules = f"""
You are an AI agent using "LOLANG", a semantic language designed for efficient AI-to-AI communication.
Rules:
1. Names: Do not encrypt names.
2. Identifiers: Do not encrypt identifiers.
3. Method: Suitable for thinking models.
4. Seed: {self.seed}
5. Numbers: Do not encrypt numbers.
6. Context: Rely on long context for full meaning.
7. Target: AI agents only.
8. Human Unreadable: Intentional.
"""

    def _load_prompt(self, template: str) -> str:
        return self._system_rules + template

class Encoder(LolangCore):
    def encode(self, text: str) -> str:
        prompt = self._load_prompt("\nTask: Encrypt the following message into LOLANG to reduce token consumption. Return ONLY the LOLANG encrypted string.")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            max_tokens=4000
        )
        lolang_text = response.choices[0].message.content.strip()
        logger.log_event("ENCODE", {
            "original": text,
            "encoded": lolang_text,
            "seed": self.seed,
            "model": self.model
        })
        return lolang_text

class Decoder(LolangCore):
    def decode(self, lolang_text: str) -> str:
        prompt = self._load_prompt("\nTask: Decrypt the following LOLANG message into natural English. Return ONLY the English translation.")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": lolang_text}
            ],
            max_tokens=4000
        )
        decoded_text = response.choices[0].message.content.strip()
        logger.log_event("DECODE", {
            "lolang_text": lolang_text,
            "decoded": decoded_text,
            "seed": self.seed,
            "model": self.model
        })
        return decoded_text
