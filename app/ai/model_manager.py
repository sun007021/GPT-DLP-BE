from functools import lru_cache
from typing import Optional
import logging
from .pii_detector import RobertaKoreanPIIDetector

logger = logging.getLogger(__name__)

# 전역 모델 인스턴스 저장소
_detector_instance: Optional[RobertaKoreanPIIDetector] = None

@lru_cache(maxsize=1)
def get_pii_detector() -> RobertaKoreanPIIDetector:
    """
    PII 탐지 모델을 싱글톤으로 관리
    
    장점:
    - 앱 시작 시 한번만 모델 로딩 (2-5초 절약)
    - 메모리 효율성 (중복 로딩 방지)
    - 멀티프로세스 환경에서도 안전
    
    Returns:
        RobertaKoreanPIIDetector: PII 탐지 모델 인스턴스
    """
    global _detector_instance
    
    if _detector_instance is None:
        logger.info("Loading PII detection model (singleton initialization)...")
        try:
            _detector_instance = RobertaKoreanPIIDetector()
            logger.info("PII detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load PII detection model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")
    
    return _detector_instance

def preload_models():
    """
    앱 시작 시 모델을 미리 로딩
    FastAPI startup event에서 호출
    """
    logger.info("Preloading AI models...")
    get_pii_detector()
    logger.info("AI models preloaded successfully")

def cleanup_models():
    """
    앱 종료 시 모델 메모리 정리
    FastAPI shutdown event에서 호출
    """
    global _detector_instance
    if _detector_instance is not None:
        logger.info("Cleaning up AI models...")
        # GPU 메모리 해제 (필요시)
        if hasattr(_detector_instance, 'model') and hasattr(_detector_instance.model, 'cpu'):
            _detector_instance.model.cpu()
        _detector_instance = None
        get_pii_detector.cache_clear()
        logger.info("AI models cleaned up")