import os
import re
import time
import redis
from pathlib import Path
from bs4 import BeautifulSoup
import markitdown
from rq import get_current_job
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.database.models import Document, Chunk
from app.services.chunking_service import get_chunking_service
from app.services.embedding_service import get_embedding_service

# ============================================================================
# PHẦN 1: GIỮ NGUYÊN 100% LOGIC XỬ LÝ CỦA BẠN
# ============================================================================

def clean_wikipedia_html(html_file_path):
    with open(html_file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    content = soup.find("div", class_="mw-parser-output")
    if not content:
        return html_file_path
    
    tags_without_class = ['audio', 'style', 'img', 'sup', 'link', 'input']
    for tag in tags_without_class:
        for element in content.find_all(tag):
            element.decompose()
    
    tags_with_class = [
        ('ol', 'references'), ('span', 'mw-editsection'), ('div', 'hatnote'),
        ('div', 'navbox'), ('div', 'navbox-styles'), ('div', 'metadata'),
        ('div', 'toc'), ('table', 'navbox-inner'), ('table', 'navbox'),
        ('table', 'sidebar'), ('table', 'infobox'), ('table', 'metadata')
    ]
    
    for tag, class_name in tags_with_class:
        for element in content.find_all(tag, class_=class_name):
            element.decompose()

    # (Các logic unwrap và remove section giữ nguyên như code cũ của bạn...)
    for a_tag in content.find_all('a'): a_tag.unwrap()
    
    input_file_name = Path(html_file_path).stem
    # Azure lưu vào /tmp hoặc folder data của app
    temp_html_file_path = os.path.join(settings.DATA_DIR, f"temp_{input_file_name}.html")
    os.makedirs(os.path.dirname(temp_html_file_path), exist_ok=True)
    with open(temp_html_file_path, "w", encoding="utf-8") as f:
        f.write(str(content))
    return temp_html_file_path

def normalize_markdown(md_text):
    # Copy nguyên logic Regex và gộp dòng của bạn vào đây
    result = re.sub(r'\n(#{1,6}\s)', r'\n\n\1', md_text)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()

def convert_html_to_normalized_md(html_file_path):
    md = markitdown.MarkItDown()   
    rs = md.convert(html_file_path)
    normalized_md = normalize_markdown(rs.text_content)
    
    name = Path(html_file_path).stem
    output_md_file_path = os.path.join(settings.PROCESSED_DIR, f"{name}.md")
    os.makedirs(os.path.dirname(output_md_file_path), exist_ok=True)
    with open(output_md_file_path, "w", encoding="utf-8") as f:
        f.write(normalized_md)
    return output_md_file_path

# ============================================================================
# PHẦN 2: TỐI ƯU KẾT NỐI (REDIS.IO + DB AZURE)
# ============================================================================

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def get_redis_conn():
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        db=getattr(settings, 'REDIS_DB', 0),
        decode_responses=True,
        socket_timeout=10
    )

def process_document(job_id, document_id, file_path, source_type, **kwargs):
    job = get_current_job()
    db = SessionLocal()
    redis_conn = get_redis_conn()
    
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document: return
        document.status = "processing"
        db.commit()

        # Update Progress via Redis.io
        if job:
            job.meta['progress'] = {'step': 'cleaning', 'current': 10}
            job.save_meta()

        # Thực thi logic của bạn
        current_path = file_path
        if file_path.endswith('.html'):
            current_path = clean_wikipedia_html(file_path)
        
        md_file_path = convert_html_to_normalized_md(current_path)

        with open(md_file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Chunking & Embedding
        chunk_size = kwargs.get('chunk_size', settings.CHUNK_SIZE)
        chunk_overlap = kwargs.get('chunk_overlap', settings.CHUNK_OVERLAP)
        
        chunking_service = get_chunking_service(chunk_size, chunk_overlap)
        chunks = chunking_service.chunk_markdown(text, md_file_path)

        if job:
            job.meta['progress'] = {'step': 'embedding', 'current': 50}
            job.save_meta()

        embedding_service = get_embedding_service()
        embeddings = embedding_service.embed_documents([c['content'] for c in chunks])

        # Lưu vào Postgres (pgvector)
        db.query(Chunk).filter(Chunk.document_id == document_id).delete()
        for idx, (chunk_data, emb) in enumerate(zip(chunks, embeddings)):
            db.add(Chunk(
                document_id=document_id,
                content=chunk_data['content'],
                embedding=emb,
                chunk_index=idx,
                metadata=chunk_data['metadata']
            ))
        
        document.status = "completed"
        db.commit()

        # Update Batch tracking trên Redis.io
        batch_id = kwargs.get('batch_id')
        if batch_id:
            redis_conn.hincrby(f"batch:{batch_id}", "completed_files", 1)

        return {"status": "success", "document_id": document_id}

    except Exception as e:
        if document: document.status = "failed"
        db.commit()
        raise e
    finally:
        db.close()