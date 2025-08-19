# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routers.pii import router as pii_router
from app.ai.model_manager import preload_models, cleanup_models
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI-TLS-DLP Backend", 
    description="한국어 개인정보 탐지 API",
    version="1.0.0"
)

# CORS 미들웨어 추가 (개발환경용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PII 탐지 라우터 추가
app.include_router(pii_router, prefix="/api/v1/pii", tags=["PII Detection"])

# 애플리케이션 시작/종료 이벤트 핸들러
@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행"""
    logger.info("Starting AI-TLS-DLP Backend...")
    preload_models()  # 모델 사전 로딩
    logger.info("AI-TLS-DLP Backend startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 실행"""
    logger.info("Shutting down AI-TLS-DLP Backend...")
    cleanup_models()  # 모델 메모리 정리
    logger.info("AI-TLS-DLP Backend shutdown completed")

@app.get("/", summary="API 상태 확인")
async def root():
    """루트 엔드포인트 - API 상태 확인"""
    return {
        "message": "AI-TLS-DLP Backend API is running",
        "status": "ok",
        "version": "1.0.0"
    }