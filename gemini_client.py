"""Simple Gemini client wrapper.

Usage:
 - Set environment variables: GEMINI_API_KEY and optionally GEMINI_ENDPOINT.
 - GEMINI_ENDPOINT should be a full URL to the REST endpoint that accepts a POST
   with JSON { "text": "..." } and returns JSON { "score": float, "label": str, ... }.

This module intentionally keeps the HTTP contract generic so you can point it
at any compatible hosted Gemini / LLM endpoint or a small proxy you run.
"""
from typing import Optional, Dict, List
import os
import requests
import logging

logger = logging.getLogger(__name__)


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None, timeout: int = 12):
        # Read from environment if not provided
        self.api_key = api_key or os.environ.get('AIzaSyAp-U4dsThAr-K48hSH-1-ZMYU2vxVoI8A')
        self.endpoint = endpoint or os.environ.get('https://your-gemini-endpoint.example/api/classify')
        self.timeout = timeout

    def is_configured(self) -> bool:
        return bool(self.api_key and self.endpoint)

    def classify(self, text: str) -> Dict:
        """Classify a single text snippet. Returns a dict with at least 'score' and 'label' when available.

        NOTE: This function performs a real HTTP request to the configured endpoint when configured.
        If you point GEMINI_ENDPOINT at a proxy or staging endpoint, make sure it expects JSON {text: str}.
        """
        if not self.is_configured():
            raise RuntimeError('Gemini client not configured (set GEMINI_API_KEY and GEMINI_ENDPOINT)')

        payload = {'text': text}
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

        try:
            resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            # Normalize expected fields
            result = {}
            # Try common field names
            if isinstance(data, dict):
                if 'score' in data:
                    result['score'] = float(data.get('score') or 0.0)
                if 'label' in data:
                    result['label'] = str(data.get('label'))
                # If the API returns a structured classification, include it
                result.update({k: v for k, v in data.items() if k not in ['score', 'label']})
            else:
                # Unexpected shape
                result['raw'] = data

            return result

        except Exception as e:
            logger.warning(f"Gemini classify request failed: {e}")
            raise

    def batch_classify(self, texts: List[str]) -> List[Dict]:
        """Classify multiple texts in one call where supported. Default implementation calls classify per item."""
        if not self.is_configured():
            raise RuntimeError('Gemini client not configured')

        results = []
        for t in texts:
            try:
                results.append(self.classify(t))
            except Exception:
                results.append({'error': 'failed'})
        return results
