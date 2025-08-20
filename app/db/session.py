from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from typing import AsyncGenerator
from app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL, 
    future=True, 
    echo=False,
    pool_pre_ping=True,  # 연결 상태 확인
    pool_recycle=300     # 5분마다 연결 재활용
)
AsyncSessionLocal = sessionmaker(
    bind=engine, 
    class_=AsyncSession, 
    expire_on_commit=False,
    autoflush=False,     # 자동 플러시 비활성화
    autocommit=False     # 자동 커밋 비활성화
)

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    안전한 비동기 세션 생성기
    async with가 자동으로 close()를 호출하므로 중복 호출 방지
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        # finally 블록에서 session.close() 제거
        # async with AsyncSessionLocal()이 __aexit__에서 자동으로 close() 호출

# 기존 함수명 호환성을 위해 별칭 유지
get_session = get_async_session