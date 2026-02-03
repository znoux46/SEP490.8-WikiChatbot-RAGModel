import redis
from rq import Queue
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Khởi tạo Redis connection với cấu hình từ Redis.io
try:
    redis_conn = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD, # Bắt buộc cho Redis.io
        db=getattr(settings, 'REDIS_DB', 0),
        decode_responses=True,
        socket_timeout=5,
        retry_on_timeout=True
    )
    # Kiểm tra kết nối thực tế
    redis_conn.ping()
    logger.info("✅ QueueService: Connected to Redis.io successfully")
except Exception as e:
    logger.error(f"❌ QueueService: Could not connect to Redis: {e}")
    # Dự phòng để app không sập ngay lập tức
    redis_conn = None 

# Tạo queue (chỉ tạo nếu có kết nối)
process_queue = Queue('process', connection=redis_conn) if redis_conn else None

def queue_process_job(job_id: str, document_id: str, file_path: str, source_type: str, **kwargs):
    if not redis_conn or not process_queue:
        return "error: redis_not_connected"

    from app.workers.process_worker import process_document
    
    batch_id = kwargs.get('batch_id')
    if batch_id:
        batch_key = f"batch:{batch_id}"
        # Khởi tạo tracking trong Redis.io
        redis_conn.hset(batch_key, mapping={
            "total_files": str(kwargs.get('total_files', 1)),
            "completed_files": "0",
            "processing_total": "0.0"
        })
        redis_conn.expire(batch_key, 86400) # Tự xóa sau 24h

    # Đẩy job vào hàng đợi Azure worker sẽ lấy ra làm
    job = process_queue.enqueue(
        process_document,
        kwargs={
            "job_id": job_id,
            "document_id": document_id,
            "file_path": file_path,
            "source_type": source_type,
            "chunk_size": kwargs.get('chunk_size', 800),
            "chunk_overlap": kwargs.get('chunk_overlap', 150),
            "batch_id": batch_id
        },
        job_id=job_id,
        job_timeout='2h'
    )
    
    redis_conn.set(f"job:{job_id}", job.id, ex=86400)
    return job.id

def get_job_status(job_id: str):
    from rq.job import Job
    rq_job_id = redis_conn.get(f"job:{job_id}") if redis_conn else None
    if not rq_job_id: return None
    
    try:
        job = Job.fetch(str(rq_job_id), connection=redis_conn)
        status_map = {'queued': 'pending', 'started': 'processing', 'finished': 'completed', 'failed': 'failed'}
        status = status_map.get(job.get_status(), 'pending')
        
        return {
            'job_id': job_id,
            'status': status,
            'progress': job.meta.get('progress'),
            'document_id': job.meta.get('document_id')
        }
    except Exception as e:
        return {'job_id': job_id, 'status': 'failed', 'message': str(e)}