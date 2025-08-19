import os
from pydantic import BaseModel

class Settings(BaseModel):
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://user:password@localhost:5432/dlpdb"
    )
    # 실제 모델을 붙일 경우 필요한 엔드포인트/토큰 등도 여기에
    MODEL_MODE: str = os.getenv("MODEL_MODE", "ECHO_INPUT")  # ECHO_INPUT | REMOTE

settings = Settings()