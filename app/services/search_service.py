"""
Hybrid Search Service using PostgreSQL + pgvector
Implements RRF (Reciprocal Rank Fusion) for combining BM25 and semantic search
"""
import re
import math
import json
import hashlib
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text, func
from app.database.models import Chunk, Document
from app.services.embedding_service import get_embedding_service
from collections import OrderedDict
from threading import Lock
import copy


class SearchService:
    """Service for hybrid search using BM25 + Semantic (pgvector)"""
    
    def __init__(self, db: Session):
        self.db = db
        self.embedding_service = get_embedding_service()
        self._auto_stopwords: Optional[set] = None
        # In-memory caches to reduce repeated DB / embedding work
        self._semantic_cache: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
        self._hybrid_cache: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
        self._cache_lock = Lock()
        self._cache_max_size = 1024
    
    @staticmethod
    def _tokenize_vi(text: str) -> List[str]:
        """Vietnamese-friendly tokenizer"""
        word_pattern = re.compile(r"[0-9A-Za-zÀ-ỹ]+", re.UNICODE)
        if not text:
            return []
        return [t.lower() for t in word_pattern.findall(text)]
    
    def _build_auto_stopwords(
        self, 
        df_threshold: float = 0.35, 
        max_size: int = 250
    ) -> set:
        """
        Build stopwords from corpus based on document frequency
        """
        # Get all chunks
        chunks = self.db.query(Chunk).all()
        n_docs = max(1, len(chunks))
        
        df: Dict[str, int] = {}
        
        for chunk in chunks:
            content = str(chunk.content) if chunk.content else ""  # type: ignore[arg-type]
            tokens = set(self._tokenize_vi(content))
            for t in tokens:
                df[t] = df.get(t, 0) + 1
        
        threshold = int(math.ceil(df_threshold * n_docs))
        high_df = [(t, c) for t, c in df.items() if c >= threshold and len(t) >= 2]
        high_df.sort(key=lambda x: x[1], reverse=True)
        
        stop = set([t for t, _ in high_df[:max_size]])
        print(f"🧹 Auto-stopwords: {len(stop)} tokens (DF≥{threshold}/{n_docs})")
        return stop
    
    def get_stopwords(self) -> set:
        """Get or build stopwords"""
        if self._auto_stopwords is None:
            self._auto_stopwords = self._build_auto_stopwords()
        return self._auto_stopwords
    
    # def bm25_search(
    #     self, 
    #     query: str, 
    #     k: int = 10,
    #     document_ids: Optional[List[str]] = None
    # ) -> List[Dict[str, Any]]:
    #     """
    #     BM25 search using ParadeDB pg_search extension
    #     Native BM25 implementation with better multi-language support
    #     """
    #     print(f"🔍 BM25 query: {query}")
        
    #     # Build ParadeDB search query
    #     # Use the BM25 index created on chunks table
    #     search_query = text("""
    #         SELECT 
    #             c.id,
    #             c.content,
    #             c.document_id,
    #             c.h1,
    #             c.h2,
    #             c.h3,
    #             c.chunk_index,
    #             c.section_id,
    #             c.sub_chunk_id,
    #             c.metadata as meta_data,
    #             paradedb.score(c.id) as rank
    #         FROM chunks c
    #         WHERE c.id @@@ paradedb.parse(:query_text)
    #         ORDER BY rank DESC
    #         LIMIT :limit_k
    #     """)
        
    #     # Execute query
    #     try:
    #         results = self.db.execute(
    #             search_query, 
    #             {"query_text": query, "limit_k": k}
    #         ).fetchall()

    #         print(f"📊 BM25 raw results: {len(results)} chunks")

    #         return [
    #             {
    #                 'id': r.id,
    #                 'content': r.content,
    #                 'document_id': str(r.document_id),
    #                 'h1': r.h1,
    #                 'h2': r.h2,
    #                 'h3': r.h3,
    #                 'chunk_index': r.chunk_index,
    #                 'section_id': r.section_id,
    #                 'sub_chunk_id': r.sub_chunk_id,
    #                 'metadata': r.meta_data,
    #                 'score': float(r.rank) if r.rank else 0.0
    #             }
    #             for r in results
    #         ]
    #     except Exception as e:
    #         try:
    #             # clear any aborted transaction so subsequent queries can run
    #             self.db.rollback()
    #         except Exception:
    #             pass
    #         print(f"❌ BM25 search error: {e}")
    #         return []
        
    #     # Filter by document_ids if provided
    #     if document_ids:
    #         base_query = base_query.filter(Chunk.document_id.in_(document_ids))
        
    #     # Order by rank and limit
    #     results = base_query.order_by(text('rank DESC')).limit(k).all()
        
    #     print(f"📊 BM25 raw results: {len(results)} chunks")
        
    #     return [
    #         {
    #             'id': r.id,
    #             'content': r.content,
    #             'document_id': str(r.document_id),
    #             'h1': r.h1,
    #             'h2': r.h2,
    #             'h3': r.h3,
    #             'chunk_index': r.chunk_index,
    #             'section_id': r.section_id,
    #             'sub_chunk_id': r.sub_chunk_id,
    #             'metadata': r.meta_data,
    #             'score': float(r.rank) if r.rank else 0.0
    #         }
    #         for r in results
    #     ]

    def bm25_search(self, query: str, k: int = 10, document_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        print(f"🔍 Trình tìm kiếm dự phòng (Postgres FTS): {query}")
        
        # Chuẩn hóa query cho Postgres tsquery
        clean_query = " & ".join(re.findall(r"\w+", query))
        
        # Query SQL sử dụng ts_rank mặc định của Postgres
        search_query = text("""
            SELECT 
                c.id, c.content, c.document_id, c.h1, c.h2, c.h3,
                ts_rank_cd(to_tsvector('simple', c.content), to_tsquery('simple', :query_text)) as rank
            FROM chunks c
            WHERE to_tsvector('simple', c.content) @@ to_tsquery('simple', :query_text)
            ORDER BY rank DESC
            LIMIT :limit_k
        """)
        
        try:
            results = self.db.execute(search_query, {"query_text": clean_query, "limit_k": k}).fetchall()
            return [
                {
                    'id': r.id, 'content': r.content, 'document_id': str(r.document_id),
                    'h1': r.h1, 'h2': r.h2, 'h3': r.h3, 'score': float(r.rank)
                } for r in results
            ]
        except Exception as e:
            self.db.rollback()
            print(f"❌ FTS Fallback error: {e}")
            return []
    
    def semantic_search(
        self, 
        query: str, 
        k: int = 10,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search using pgvector cosine similarity
        """
        # Build cache key
        key_obj = {"q": query, "k": k, "docs": document_ids}
        cache_key = hashlib.sha256(json.dumps(key_obj, sort_keys=True, default=str).encode()).hexdigest()
        with self._cache_lock:
            if cache_key in self._semantic_cache:
                val = self._semantic_cache.pop(cache_key)
                # refresh LRU
                self._semantic_cache[cache_key] = val
                return copy.deepcopy(val)

        # Generate query embedding (embedding_service has its own cache)
        query_embedding = self.embedding_service.embed_text(query)
        
        # Base query with cosine distance (pgvector accepts list directly)
        base_query = self.db.query(
            Chunk.id,
            Chunk.content,
            Chunk.document_id,
            Chunk.h1,
            Chunk.h2,
            Chunk.h3,
            Chunk.chunk_index,
            Chunk.section_id,
            Chunk.sub_chunk_id,
            Chunk.meta_data,
            (1 - Chunk.embedding.cosine_distance(query_embedding)).label('similarity')
        )
        
        # Filter by document_ids if provided
        if document_ids:
            base_query = base_query.filter(Chunk.document_id.in_(document_ids))
        
        # Order by similarity and limit
        try:
            results = base_query.order_by(text('similarity DESC')).limit(k).all()
        except Exception as e:
            try:
                self.db.rollback()
            except Exception:
                pass
            print(f"❌ Semantic search error: {e}")
            return []

        out = [
            {
                'id': r.id,
                'content': r.content,
                'document_id': str(r.document_id),
                'h1': r.h1,
                'h2': r.h2,
                'h3': r.h3,
                'chunk_index': r.chunk_index,
                'section_id': r.section_id,
                'sub_chunk_id': r.sub_chunk_id,
                'metadata': r.meta_data,
                'score': float(r.similarity) if r.similarity else 0.0
            }
            for r in results
        ]

        # store in LRU cache
        with self._cache_lock:
            if cache_key in self._semantic_cache:
                self._semantic_cache.pop(cache_key)
            self._semantic_cache[cache_key] = copy.deepcopy(out)
            if len(self._semantic_cache) > self._cache_max_size:
                self._semantic_cache.popitem(last=False)

        return out
    
    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        bm25_k: Optional[int] = None,
        semantic_k: Optional[int] = None,
        rrf_k: int = 60,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search using RRF (Reciprocal Rank Fusion)
        Combines BM25 and semantic search results
        """
        # Cache key for hybrid search
        key_obj = {"q": query, "k": k, "bm25_w": bm25_weight, "sem_w": semantic_weight, "docs": document_ids}
        cache_key = hashlib.sha256(json.dumps(key_obj, sort_keys=True, default=str).encode()).hexdigest()
        with self._cache_lock:
            if cache_key in self._hybrid_cache:
                val = self._hybrid_cache.pop(cache_key)
                self._hybrid_cache[cache_key] = val
                return copy.deepcopy(val)

        # Determine per-stage k values and cap them to avoid heavy queries
        bm25_k = bm25_k or min(max(k * 2, 20), 200)
        semantic_k = semantic_k or min(max(k * 2, 20), 200)

        bm25_res = self.bm25_search(query, k=bm25_k, document_ids=document_ids)
        semantic_res = self.semantic_search(query, k=semantic_k, document_ids=document_ids)
        scores: Dict[str, Dict[str, Any]] = {}
        # semantic_k = semantic_k or max(k, 20)
        # print(f"\n🔍 SEMANTIC-ONLY SEARCH (hybrid disabled)")
        # print(f"Query: {query}")
        # print(f"Semantic_k={semantic_k}")
        # return self.semantic_search(query=query, k=semantic_k, document_ids=document_ids)
        # Trộn danh sách BM25
        for rank, doc in enumerate(bm25_res, start=1):
            doc_id = str(doc['id'])
            if doc_id not in scores:
                scores[doc_id] = {'doc': doc, 'score': 0.0}
            scores[doc_id]['score'] += bm25_weight * (1.0 / (rrf_k + rank))
            
        # Trộn danh sách Semantic
        for rank, doc in enumerate(semantic_res, start=1):
            doc_id = str(doc['id'])
            if doc_id not in scores:
                scores[doc_id] = {'doc': doc, 'score': 0.0}
            scores[doc_id]['score'] += semantic_weight * (1.0 / (rrf_k + rank))
            
        # Sắp xếp lại theo điểm số mới
        fused_results = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
        final_docs = [item['doc'] for item in fused_results[:k]]
        out = final_docs

        # cache hybrid results
        with self._cache_lock:
            if cache_key in self._hybrid_cache:
                self._hybrid_cache.pop(cache_key)
            self._hybrid_cache[cache_key] = copy.deepcopy(out)
            if len(self._hybrid_cache) > self._cache_max_size:
                self._hybrid_cache.popitem(last=False)

        print(f"✅ Hybrid Fusion completed: {len(out)} chunks selected")
        return out


def get_search_service(db: Session) -> SearchService:
    """Factory function for SearchService"""
    return SearchService(db)
