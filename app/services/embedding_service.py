from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings
from pydantic import SecretStr
import numpy as np
from typing import List
import os
from collections import OrderedDict
from threading import Lock


class EmbeddingService:
    """Service để tạo embeddings với Matryoshka truncation và normalization"""
    
    def __init__(self):
        # Sử dụng gemini-embedding-001 với output_dimensionality=768
        # Model này tạo 3072 dimensions nhưng truncate về 768 dimensions
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            api_key=SecretStr(settings.GEMINI_API_KEY)
        )
        # Set output dimensionality for Matryoshka truncation
        self.output_dimensionality = settings.DIMENSION_OF_MODEL

        # Reuse genai client across calls to avoid re-creating per-request
        from google import genai
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

        # Simple in-memory LRU cache for embeddings to reduce API calls
        self._embed_cache: OrderedDict[str, List[float]] = OrderedDict()
        self._cache_lock = Lock()
        self._cache_max_size = int(os.environ.get("EMBED_CACHE_SIZE", 1024))
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """
        Normalize embedding vector để đảm bảo chất lượng với smaller dimensions.
        Cần normalize khi dùng dimensions < 3072 (như 768, 1536).
        """
        embedding_np = np.array(embedding)
        norm = np.linalg.norm(embedding_np)
        if norm > 0:
            normalized = embedding_np / norm
            return normalized.tolist()
        return embedding
    
    def embed_text(self, text: str) -> List[float]:
        """
        Tạo embedding cho một đoạn text với truncation và normalization
        """
        # Check LRU cache first
        key = text
        with self._cache_lock:
            if key in self._embed_cache:
                # move to end (most-recent)
                val = self._embed_cache.pop(key)
                self._embed_cache[key] = val
                return list(val)

        # Tạo embedding với output_dimensionality using persistent client
        from google.genai import types
        result = self.client.models.embed_content(
            model=settings.EMBEDDING_MODEL_NAME,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=self.output_dimensionality)
        )
        
        if result.embeddings and len(result.embeddings) > 0:
            values = result.embeddings[0].values
            if values is not None:
                embedding = list(values)
                # Normalize để đảm bảo chất lượng
                normalized = self._normalize_embedding(embedding)

                # store in LRU cache
                with self._cache_lock:
                    if key in self._embed_cache:
                        self._embed_cache.pop(key)
                    self._embed_cache[key] = list(normalized)
                    # evict oldest
                    if len(self._embed_cache) > self._cache_max_size:
                        self._embed_cache.popitem(last=False)

                return normalized

        raise ValueError("Failed to generate embedding")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Tạo embeddings cho nhiều documents với truncation và normalization
        """
        from google.genai import types

        embeddings: List[List[float]] = []
        for text in texts:
            # try cache first
            key = text
            with self._cache_lock:
                if key in self._embed_cache:
                    val = self._embed_cache.pop(key)
                    self._embed_cache[key] = val
                    embeddings.append(list(val))
                    continue

            result = self.client.models.embed_content(
                model=settings.EMBEDDING_MODEL_NAME,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=self.output_dimensionality)
            )

            if result.embeddings and len(result.embeddings) > 0:
                values = result.embeddings[0].values
                if values is not None:
                    embedding = list(values)
                    normalized = self._normalize_embedding(embedding)
                    embeddings.append(normalized)
                    with self._cache_lock:
                        if key in self._embed_cache:
                            self._embed_cache.pop(key)
                        self._embed_cache[key] = list(normalized)
                        if len(self._embed_cache) > self._cache_max_size:
                            self._embed_cache.popitem(last=False)
                else:
                    raise ValueError(f"Failed to generate embedding for text: {text[:50]}...")
            else:
                raise ValueError(f"Failed to generate embedding for text: {text[:50]}...")

        return embeddings


# Singleton instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get singleton instance của EmbeddingService"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
