from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "RAG Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str = "postgresql://rag_user:rag_password@127.0.0.1:5433/rag_db"
    
    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))

    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"    

    # Google AI
    GEMINI_API_KEY: str = ""
    # Sử dụng text-embedding-004 với Matryoshka truncation về 768 dimensions
    # Model tạo 3072d nhưng truncate về 768d để tiết kiệm storage
    EMBEDDING_MODEL_NAME: str = "models/text-embedding-004"
    DIMENSION_OF_MODEL: int = 768

    # Groq AI
    GROQ_API_KEY: str = ""
    # Groq model name (set to a supported model via .env if needed)
    GROQ_MODEL_NAME: str = ""
    
    # Pinecone (Legacy - optional, for backward compatibility)
    PINECONE_API_KEY: Optional[str] = None
    
    # Chunking
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    
    # File Storage
    DATA_DIR: str = "./data"
    UPLOAD_DIR: str = "./data/uploads"
    PROCESSED_DIR: str = "./data/processed_data"
    RAW_DIR: str = "./data/raw_data"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Auto-detect Docker environment and adjust paths
        if os.path.exists('/app'):
            self.DATA_DIR = "/app/data"
            self.UPLOAD_DIR = "/app/data/uploads"
            self.PROCESSED_DIR = "/app/data/processed_data"
            self.RAW_DIR = "/app/data/raw_data"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from .env


settings = Settings()
