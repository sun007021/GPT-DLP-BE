from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from app.schemas.pii import PIIDetectionRequest, PIIDetectionResponse
from app.services.pii_service import PIIDetectionService
from app.ai.model_manager import get_pii_detector
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# 서비스 인스턴스 생성 (이제 모델은 싱글톤으로 관리됨)
pii_service = PIIDetectionService()

@router.post("/detect", 
             response_model=PIIDetectionResponse,
             summary="PII 탐지",
             description="입력된 텍스트에서 개인정보를 탐지하고 결과를 반환합니다.",
             status_code=status.HTTP_200_OK)
async def detect_pii(request: PIIDetectionRequest) -> PIIDetectionResponse:
    """
    텍스트에서 개인정보 탐지 API
    
    - **text**: 분석할 텍스트 (1-10,000자)
    
    반환값:
    - **has_pii**: 개인정보 탐지 여부 (boolean)
    - **reason**: 탐지 결과 이유
    - **details**: 구체적인 탐지 내용
    - **entities**: 탐지된 개인정보 엔티티 목록
    """
    try:
        # 입력 검증
        if not request.text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="입력 텍스트가 비어있습니다."
            )
        
        # PII 탐지 수행
        logger.info(f"PII detection started for text length: {len(request.text)}")
        result = await pii_service.analyze_text(request.text.strip())
        
        logger.info(f"PII detection completed. Has PII: {result.has_pii}, Entities: {len(result.entities)}")
        
        # 디버깅을 위해 원시 예측 결과 로그 출력
        from app.ai.model_manager import get_pii_detector
        detector = get_pii_detector()
        raw_predictions = await detector._predict_tokens(request.text.strip())
        logger.info(f"Raw predictions sample: {raw_predictions[:20]}")
        
        # 임시: 디버깅을 위해 raw_predictions를 응답에 포함
        response_dict = result.model_dump()
        response_dict["debug_raw_predictions"] = raw_predictions[:20]  # 처음 20개만
        
        return response_dict
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PII detection failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PII 탐지 중 오류가 발생했습니다."
        )

@router.get("/health", 
            summary="PII 탐지 서비스 상태 확인",
            description="PII 탐지 모델이 정상적으로 로드되었는지 확인합니다.")
async def health_check():
    """PII 탐지 서비스 헬스체크"""
    try:
        # 모델 인스턴스 상태 확인 (실제 추론 없이 빠른 체크)
        detector = get_pii_detector()
        model_loaded = detector.model is not None and detector.tokenizer is not None
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "healthy",
                "message": "PII detection service is running",
                "model_loaded": model_loaded,
                "model_name": detector.model_name
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy", 
                "message": "PII detection service is not available",
                "model_loaded": False,
                "error": str(e)
            }
        )