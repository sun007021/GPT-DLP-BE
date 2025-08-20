from functools import lru_cache
from typing import Optional
import logging
from .pii_detector import RobertaKoreanPIIDetector
from .similarity_detector import KoSimCSESimilarityDetector

logger = logging.getLogger(__name__)

# 전역 모델 인스턴스 저장소
_pii_detector_instance: Optional[RobertaKoreanPIIDetector] = None
_similarity_detector_instance: Optional[KoSimCSESimilarityDetector] = None

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
    global _pii_detector_instance
    
    if _pii_detector_instance is None:
        logger.info("Loading PII detection model (singleton initialization)...")
        try:
            _pii_detector_instance = RobertaKoreanPIIDetector()
            logger.info("PII detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load PII detection model: {str(e)}")
            raise RuntimeError(f"PII model initialization failed: {str(e)}")
    
    return _pii_detector_instance

@lru_cache(maxsize=1)
def get_similarity_detector() -> KoSimCSESimilarityDetector:
    """
    유사도 탐지 모델을 싱글톤으로 관리
    
    Returns:
        KoSimCSeSimilarityDetector: 유사도 탐지 모델 인스턴스
    """
    global _similarity_detector_instance
    
    if _similarity_detector_instance is None:
        logger.info("Loading similarity detection model (singleton initialization)...")
        try:
            _similarity_detector_instance = KoSimCSESimilarityDetector()
            logger.info("Similarity detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load similarity detection model: {str(e)}")
            raise RuntimeError(f"Similarity model initialization failed: {str(e)}")
    
    return _similarity_detector_instance

def preload_models():
    """
    앱 시작 시 모델을 미리 로딩
    FastAPI startup event에서 호출
    """
    logger.info("Preloading AI models...")
    
    try:
        # PII 탐지 모델 로딩
        get_pii_detector()
        logger.info("✓ PII detection model loaded")
        
        # 유사도 탐지 모델 로딩  
        get_similarity_detector()
        logger.info("✓ Similarity detection model loaded")
        
        logger.info("All AI models preloaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to preload models: {str(e)}")
        # 모델 로딩 실패 시에도 서버는 시작하되, 런타임에 에러 발생하도록 함
        raise

def cleanup_models():
    """
    앱 종료 시 모델 메모리 정리
    FastAPI shutdown event에서 호출
    """
    global _pii_detector_instance, _similarity_detector_instance
    
    logger.info("Cleaning up AI models...")
    
    # PII 모델 정리
    if _pii_detector_instance is not None:
        if hasattr(_pii_detector_instance, 'model') and hasattr(_pii_detector_instance.model, 'cpu'):
            _pii_detector_instance.model.cpu()
        _pii_detector_instance = None
        get_pii_detector.cache_clear()
        logger.info("✓ PII detection model cleaned up")
    
    # 유사도 모델 정리
    if _similarity_detector_instance is not None:
        if hasattr(_similarity_detector_instance, 'model') and hasattr(_similarity_detector_instance.model, 'cpu'):
            _similarity_detector_instance.model.cpu()
        _similarity_detector_instance = None
        get_similarity_detector.cache_clear()
        logger.info("✓ Similarity detection model cleaned up")
    
    logger.info("All AI models cleaned up")