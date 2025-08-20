import os
from pydantic import BaseModel

class Settings(BaseModel):
    # 데이터베이스 설정
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://admin:password123@localhost:5432/ai_tlsdlp"
    )
    
    # ChromaDB 설정
    CHROMADB_HOST: str = os.getenv("CHROMADB_HOST", "localhost")
    CHROMADB_PORT: int = int(os.getenv("CHROMADB_PORT", "8001"))
    CHROMADB_COLLECTION_NAME: str = os.getenv("CHROMADB_COLLECTION_NAME", "sensitive_documents")
    
    # Redis 설정
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # AI 모델 설정
    PII_MODEL_NAME: str = os.getenv("PII_MODEL_NAME", "psh3333/roberta-large-korean-pii5")
    SIMILARITY_MODEL_NAME: str = os.getenv("SIMILARITY_MODEL_NAME", "BM-K/KoSimCSE-roberta-multitask")
    
    # 파일 업로드 설정
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "storage/uploads")
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    ALLOWED_EXTENSIONS: list[str] = ["txt", "pdf", "docx", "md"]
    
    # 분석 설정
    DEFAULT_SIMILARITY_THRESHOLD: float = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.85"))
    DEFAULT_PII_THRESHOLD: float = float(os.getenv("DEFAULT_PII_THRESHOLD", "0.59"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # 캐싱 설정
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))  # 1시간
    
    # 실제 모델을 붙일 경우 필요한 엔드포인트/토큰 등도 여기에
    MODEL_MODE: str = os.getenv("MODEL_MODE", "LOCAL")  # LOCAL | REMOTE

settings = Settings()