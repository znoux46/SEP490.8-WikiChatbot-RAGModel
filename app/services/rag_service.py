"""
RAG (Retrieval-Augmented Generation) Service
Implements two-pass retrieval with query expansion and RRF fusion
"""
import json
import re
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


from app.config import settings
from app.services.search_service import SearchService
from collections import OrderedDict
from threading import Lock
import copy
import json
import hashlib


class RAGService:
    """Service for RAG chat with advanced retrieval strategies"""
    
    def __init__(
        self,
        db: Session,
        model_name: str | None = None,
        temperature: float = 0.1,
        top_k: int = 20,
        bm25_weight: float = 0.6,
        semantic_weight: float = 0.4,
        first_pass_k: int = 40,
        variant_count: int = 5,
        rrf_k: int = 60
    ):
        self.db = db
        self.search_service = SearchService(db)

        self.top_k = top_k
        self.first_pass_k = first_pass_k
        self.variant_count = variant_count
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight

        # Determine model to use: constructor arg -> env var -> fallback
        effective_model = model_name or settings.GROQ_MODEL_NAME or "openai/gpt-oss-120b"
        if settings.GROQ_MODEL_NAME:
            print(f"🤖 Using GROQ model from settings: {settings.GROQ_MODEL_NAME}")
        print(f"🤖 Initializing {effective_model}...")
        # Support Groq or Google Gemini (if model name suggests gemini-*)
        if effective_model.lower().startswith("gemini"):
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                print(f"🤖 Using Google Gemini model: {effective_model}")
                self.llm = ChatGoogleGenerativeAI(
                    model=effective_model, 
                    google_api_key=settings.GEMINI_API_KEY, 
                    temperature=temperature
                )
            except ImportError:
                print("⚠️ Please install langchain-google-genai")
        else:
            self.llm = ChatGroq(
                model=effective_model,
                api_key=settings.GROQ_API_KEY,
                temperature=temperature
            )
        
        # Main RAG prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("human", """Bạn là trợ lý AI. Trả lời câu hỏi dựa trên CONTEXT được cung cấp.

NGUYÊN TẮC:
1) CHỈ trả lời dựa trên thông tin trong CONTEXT
2) Nếu CONTEXT không có thông tin → trả lời đúng câu: "Tôi không tìm thấy thông tin này trong tài liệu."
3) Trả lời NGẮN GỌN, CHÍNH XÁC, bằng tiếng Việt
4) Nếu trong CONTEXT có nhiều tên gọi (bí danh / tên khai sinh / tên khác) của cùng một người, hãy coi chúng là 1 thực thể khi suy luận.
5) Không bịa đặt.

CONTEXT:
{context}

CÂU HỎI: {question}

TRẢ LỜI:"""),
        ])
        
        # Alias extraction prompt
        self.alias_prompt = ChatPromptTemplate.from_messages([
            ("human", """Bạn sẽ trích xuất thông tin thực thể từ CONTEXT để hỗ trợ truy hồi.

Hãy trả về JSON hợp lệ theo schema:
{{
  "entity": "tên thực thể chính nếu xác định được, nếu không thì rỗng",
  "aliases": ["các tên gọi khác của cùng thực thể, nếu có"],
  "keywords": ["một vài từ khóa quan trọng liên quan đến câu hỏi (không quá chung chung)"]
}}

Ràng buộc:
- Chỉ dùng thông tin xuất hiện trong CONTEXT.
- Nếu không chắc entity là gì → entity = "".
- aliases: chỉ đưa alias thật sự cùng một người/tổ chức với entity trong CONTEXT.
- keywords: tối đa 8 từ/nhóm từ.
- Trả về JSON và CHỈ JSON, không thêm chữ nào khác.

CÂU HỎI: {question}

CONTEXT:
{context}
""")
        ])
        
        # Build chains
        self.rag_chain = (
            {
                "context": lambda x: self._format_docs(x["docs"]),
                "question": lambda x: x["question"]
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        self.alias_chain = self.alias_prompt | self.llm | StrOutputParser()
        
        print("✅ RAG Service ready!")

        # Simple in-memory cache for retrieve results (question -> fused docs)
        self._retrieve_cache: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
        self._retrieve_lock = Lock()
        self._retrieve_cache_max = 256
    
    @staticmethod
    def _tokenize_vi(text: str) -> List[str]:
        """Vietnamese-friendly tokenizer"""
        word_pattern = re.compile(r"[0-9A-Za-zÀ-ỹ]+", re.UNICODE)
        if not text:
            return []
        return [t.lower() for t in word_pattern.findall(text)]
    
    def _format_docs(self, docs: List[Dict[str, Any]]) -> str:
        """Format documents for context"""
        if not docs:
            return ""

        formatted = []
        for i, doc in enumerate(docs, 1):
            header = doc.get('h2') or doc.get('h1') or ''
            content = doc.get('content', '')

            # Lightweight meta extraction to highlight aliases / era names / chữ Hán
            meta_parts = []
            # tên thật
            m_name = re.search(r"tên\s+(?:thật|thực)\s*(?:là|:)\s*([^,\.\n]+)", content, flags=re.IGNORECASE)
            if m_name:
                meta_parts.append(f"tên thật: {m_name.group(1).strip()}")
            # niên hiệu
            m_era = re.search(r"niên\s*hiệu\s*(?:là|:)\s*([^,\.\n]+)", content, flags=re.IGNORECASE)
            if m_era:
                meta_parts.append(f"niên hiệu: {m_era.group(1).strip()}")
            # chữ Hán
            m_ch = re.search(r"chữ\s*Hán\s*[:：]?\s*([^\)\n]+)", content, flags=re.IGNORECASE)
            if m_ch:
                meta_parts.append(f"chữ Hán: {m_ch.group(1).strip()}")

            meta_line = ""
            if meta_parts:
                meta_line = "[META: " + "; ".join(meta_parts) + "]\n"

            formatted.append(f"--- Đoạn {i} | {header} ---\n" + meta_line + f"{content}")

        return "\n\n".join(formatted)
    
    def _extract_entity_info(
        self, 
        question: str, 
        docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract entity, aliases, and keywords using LLM"""
        context = self._format_docs(docs[:min(len(docs), 8)])
        
        try:
            raw = self.alias_chain.invoke({
                "question": question, 
                "context": context
            })
            
            # Try to parse JSON
            data = json.loads(raw)
            entity = (data.get("entity") or "").strip()
            aliases = data.get("aliases") or []
            keywords = data.get("keywords") or []
            
            # Normalize
            aliases = [a.strip() for a in aliases if isinstance(a, str) and a.strip()]
            keywords = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
            
            # Remove duplicates
            aliases = list(dict.fromkeys(aliases))
            keywords = list(dict.fromkeys(keywords))
            
            return {
                "entity": entity,
                "aliases": aliases,
                "keywords": keywords
            }
        except Exception as e:
            print(f"⚠️ Entity extraction failed: {e}")
            return {"entity": "", "aliases": [], "keywords": []}
    
    def _make_variants(
        self, 
        question: str, 
        info: Dict[str, Any]
    ) -> List[str]:
        """Generate query variants from entity/aliases/keywords"""
        entity = info.get("entity", "").strip()
        aliases = info.get("aliases") or []
        keywords = info.get("keywords") or []
        
        # Filter stopwords
        stop = self.search_service.get_stopwords()
        q_tokens = self._tokenize_vi(question)
        q_core = [t for t in q_tokens if t not in stop]
        core_text = " ".join(q_core).strip()
        
        variants = [question]
        
        # Entity-based variants
        if entity:
            if core_text:
                variants.append(f"{entity} {core_text}")
            variants.append(f"{entity} {question}")
        
        # Alias-based variants
        for alias in aliases[:self.variant_count]:
            if core_text:
                variants.append(f"{alias} {core_text}")
            else:
                variants.append(f"{alias} {question}")
        
        # Keyword-based variants
        if entity and keywords:
            variants.append(f"{entity} " + " ".join(keywords[:8]))
        elif keywords:
            variants.append(" ".join(keywords[:8]))
        
        # Dedup
        deduped = []
        seen = set()
        for v in variants:
            v2 = v.strip()
            if not v2 or v2.lower() in seen:
                continue
            seen.add(v2.lower())
            deduped.append(v2)
        
        return deduped[:max(3, self.variant_count)]
    
    def _rrf_fuse(
        self, 
        list_of_results: List[List[Dict[str, Any]]], 
        rrf_k: int = 60, 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion for multiple result lists"""
        scores: Dict[int, Dict[str, Any]] = {}
        
        for results in list_of_results:
            for rank, doc in enumerate(results, start=1):
                doc_id = doc['id']
                scores.setdefault(doc_id, {'doc': doc, 'score': 0.0})
                scores[doc_id]['score'] += 1.0 / (rrf_k + rank)
        
        ranked = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
        fused = [x['doc'] for x in ranked[:top_k]]
        
        # Add fused scores
        for i, doc in enumerate(fused):
            doc['fused_score'] = ranked[i]['score']
        
        return fused
    
    def retrieve(
        self, 
        question: str,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Two-pass retrieval with query expansion:
        1. First pass: retrieve with original question
        2. Extract entity/aliases/keywords from context
        3. Generate query variants
        4. Retrieve for each variant
        5. RRF fusion of all results
        """
        print(f"\n🔍 Retrieve: '{question}'")
        # small retrieve cache check
        cache_key = hashlib.sha256(json.dumps({"q": question, "docs": document_ids}, sort_keys=True, default=str).encode()).hexdigest()
        with self._retrieve_lock:
            if cache_key in self._retrieve_cache:
                val = self._retrieve_cache.pop(cache_key)
                self._retrieve_cache[cache_key] = val
                print("♻️ Returning cached retrieve results")
                return copy.deepcopy(val)
        
        # Pass 1: Initial retrieval
        first_pass = self.search_service.hybrid_search(
            query=question,
            k=self.first_pass_k,
            bm25_weight=self.bm25_weight,
            semantic_weight=self.semantic_weight,
            rrf_k=self.rrf_k,
            document_ids=document_ids
        )
        print(f"📦 Pass 1: {len(first_pass)} chunks")
        
        # Extract entity info from first pass
        info = self._extract_entity_info(question, first_pass)
        entity = info.get("entity", "")
        aliases = info.get("aliases") or []
        keywords = info.get("keywords") or []
        
        print(f"🧠 Entity: {entity or '(none)'} | Aliases: {len(aliases)} | Keywords: {len(keywords)}")
        
        # Generate variants
        variants = self._make_variants(question, info)
        # dedupe variants and keep order
        variants = list(dict.fromkeys(variants))
        print(f"🧩 Variants: {len(variants)}")
        for i, v in enumerate(variants, 1):
            print(f"   Q{i}: {v}")
        
        # Pass 2: Retrieve for each variant
        all_results = [first_pass]
        
        # Increase per-variant k to reduce missing important chunks
        per_variant_k = max(self.top_k, 60)
        for variant in variants:
            results = self.search_service.hybrid_search(
                query=variant,
                k=per_variant_k,
                bm25_weight=self.bm25_weight,
                semantic_weight=self.semantic_weight,
                rrf_k=self.rrf_k,
                document_ids=document_ids
            )
            all_results.append(results)
        
        # Fuse all results
        fused = self._rrf_fuse(all_results, rrf_k=self.rrf_k, top_k=self.top_k)
        print(f"✅ Fused: {len(fused)} chunks")
        # store in retrieve cache
        with self._retrieve_lock:
            if cache_key in self._retrieve_cache:
                self._retrieve_cache.pop(cache_key)
            self._retrieve_cache[cache_key] = copy.deepcopy(fused)
            if len(self._retrieve_cache) > self._retrieve_cache_max:
                self._retrieve_cache.popitem(last=False)

        return fused
    
    def chat(
        self, 
        question: str, 
        document_ids: Optional[List[str]] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        RAG chat: retrieve + generate answer
        
        Returns:
            {
                'answer': str,
                'chunks': List[Dict],
                'metadata': Dict
            }
        """
        # Retrieve relevant chunks
        docs = self.retrieve(question, document_ids=document_ids)

        print("\n💬 Generating answer...")
        # Try deterministic rule-based answers for short factual queries first
        rule_ans = self._rule_based_answer(question, docs)
        if rule_ans:
            answer = rule_ans
        else:
            # If we have no docs, skip calling the LLM and return final fallback later
            if not docs:
                answer = ""
            else:
                answer = self.rag_chain.invoke({"docs": docs, "question": question})
                answer = (answer or "").strip()
            
        if verbose:
            print(f"\n{'='*70}\nCONTEXT:\n{'='*70}")
            for i, doc in enumerate(docs[:5], 1):
                print(f"\n📄 Chunk {i}:")
                print(f"   Headers: {doc.get('h1', '')} / {doc.get('h2', '')}")
                print(f"   Preview: {doc.get('content', '')[:200]}...")
        
        # # Normalize fallback
        # if not answer or ("không tìm thấy" in answer.lower() and "tài liệu" in answer.lower()):
        #     answer = "Tôi không tìm thấy thông tin này trong tài liệu."
        
        # Final fallback: if still no answer or LLM said it couldn't find, normalize message
        if not answer or (isinstance(answer, str) and "không tìm thấy" in answer.lower()):
            answer = "Tôi không tìm thấy thông tin này trong tài liệu."

        return {
            'answer': answer,
            'chunks': docs[:10],  # Return top 10 for reference
            'metadata': {
                'chunks_used': len(docs),
                'model': getattr(self.llm, 'model', 'unknown')
            }
        }


def get_rag_service(db: Session) -> RAGService:
    """Factory function for RAGService"""
    return RAGService(db)
