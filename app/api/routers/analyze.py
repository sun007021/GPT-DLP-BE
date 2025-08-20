"""
통합 분석 API 라우터 (PII + 유사도)
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse

from app.schemas.analyze import (
    AnalysisRequest, AnalysisResponse, 
    AnalysisHistoryResponse, AnalysisStatsResponse,
    SimilarityOnlyRequest, SimilarityOnlyResponse
)
from app.services.analyze_service import IntegratedAnalysisService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analyze", tags=["Analysis"])

# 서비스 인스턴스 (싱글톤)
_analysis_service: Optional[IntegratedAnalysisService] = None


def get_analysis_service() -> IntegratedAnalysisService:
    """분석 서비스 인스턴스를 반환합니다."""
    global _analysis_service
    if _analysis_service is None:
        _analysis_service = IntegratedAnalysisService()
    return _analysis_service


def get_client_ip(request: Request) -> str:
    """클라이언트 IP 주소를 추출합니다."""
    # X-Forwarded-For 헤더 확인 (프록시 환경)
    x_forwarded_for = request.headers.get("x-forwarded-for")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    
    # X-Real-IP 헤더 확인
    x_real_ip = request.headers.get("x-real-ip")
    if x_real_ip:
        return x_real_ip
    
    # 기본 클라이언트 IP
    return request.client.host if request.client else "unknown"


@router.post(
    "/comprehensive",
    response_model=AnalysisResponse,
    summary="종합 분석",
    description="텍스트에 대한 PII 탐지와 유사도 분석을 동시에 수행합니다."
)
async def analyze_comprehensive(
    request_data: AnalysisRequest,
    request: Request,
    service: IntegratedAnalysisService = Depends(get_analysis_service)
) -> AnalysisResponse:
    """
    텍스트에 대한 종합 분석을 수행합니다.
    
    - **PII 탐지**: 개인정보(이름, 전화번호 등) 탐지
    - **유사도 분석**: 기등록된 민감 문서와의 유사도 분석
    - **차단 결정**: 설정된 임계값에 따른 자동 차단
    """
    try:
        logger.info(f"Comprehensive analysis request: text_length={len(request_data.text)}")
        
        # 클라이언트 IP 추출
        client_ip = get_client_ip(request)
        
        # 종합 분석 수행 (단순화된 API - text만 필요)
        result = await service.analyze_comprehensive(
            text=request_data.text,
            user_ip=client_ip
        )
        
        logger.info(f"Analysis completed: blocked={result['blocked']}, time={result['analysis_time_ms']}ms")
        return AnalysisResponse(**result)
        
    except ValueError as e:
        logger.warning(f"Invalid request data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Analysis failed: {str(e)}"
        )


@router.get(
    "/history",
    response_model=AnalysisHistoryResponse,
    summary="분석 이력 조회",
    description="최근 분석 이력을 조회합니다."
)
async def get_analysis_history(
    limit: int = 50,
    blocked_only: bool = False,
    service: IntegratedAnalysisService = Depends(get_analysis_service)
) -> AnalysisHistoryResponse:
    """
    분석 이력을 조회합니다.
    
    - **limit**: 조회할 최대 결과 수 (기본값: 50)
    - **blocked_only**: 차단된 결과만 조회할지 여부 (기본값: False)
    """
    try:
        if limit <= 0 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000")
        
        logger.info(f"Getting analysis history: limit={limit}, blocked_only={blocked_only}")
        
        history = await service.get_analysis_history(
            limit=limit,
            blocked_only=blocked_only
        )
        
        return AnalysisHistoryResponse(
            total=len(history),
            items=history
        )
        
    except ValueError as e:
        logger.warning(f"Invalid request parameters: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(f"Failed to get analysis history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get analysis history: {str(e)}"
        )


@router.get(
    "/stats",
    response_model=AnalysisStatsResponse,
    summary="분석 통계",
    description="전체 분석 통계를 조회합니다."
)
async def get_analysis_stats(
    service: IntegratedAnalysisService = Depends(get_analysis_service)
) -> AnalysisStatsResponse:
    """
    분석 통계를 조회합니다.
    
    - 총 분석 수
    - 차단된 분석 수  
    - PII/유사도 탐지 수
    - 차단율 및 평균 분석 시간
    """
    try:
        logger.info("Getting analysis statistics")
        
        stats = await service.get_analysis_stats()
        
        # 차단율 계산
        total_analyses = stats.get("total_analyses", 0)
        blocked_analyses = stats.get("blocked_analyses", 0)
        block_rate = (blocked_analyses / total_analyses * 100) if total_analyses > 0 else 0.0
        
        return AnalysisStatsResponse(
            total_analyses=total_analyses,
            blocked_analyses=blocked_analyses,
            pii_detections=stats.get("pii_detections", 0),
            similarity_detections=stats.get("similarity_detections", 0),
            block_rate=round(block_rate, 2),
            avg_analysis_time_ms=stats.get("avg_analysis_time_ms")
        )
        
    except Exception as e:
        logger.error(f"Failed to get analysis stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get analysis stats: {str(e)}"
        )



@router.post(
    "/similarity-only", 
    response_model=SimilarityOnlyResponse,
    summary="유사도 전용 분석 (단순화)",
    description="텍스트만 입력받아 KoSimCSE 유사도 분석을 수행합니다."
)
async def analyze_similarity_only(
    request_data: SimilarityOnlyRequest,
    service: IntegratedAnalysisService = Depends(get_analysis_service)
) -> SimilarityOnlyResponse:
    """
    유사도 분석만 수행하는 단순화된 API
    
    - **입력**: 텍스트만 필요 (다른 옵션들 비활성화)
    - **출력**: KoSimCSE 유사도 수치 포함한 상세 결과
    - **자동**: 기본 임계값 0.85 사용, 자동 차단 여부 결정
    """
    try:
        logger.info(f"Similarity-only analysis request: text_length={len(request_data.text)}")
        
        # 통합 분석 수행 (두 모델 모두 실행하지만 유사도 결과만 반환)
        result = await service.analyze_comprehensive(
            text=request_data.text,
            user_ip=None  # IP 추적하지 않음 (단순화)
        )
        
        # 유사도 분석 결과 추출
        similarity_result = result.get("similarity_analysis", {})
        
        # 응답 데이터 구성
        response_data = SimilarityOnlyResponse(
            input_text=request_data.text,
            is_similar=similarity_result.get("is_similar", False),
            max_similarity=similarity_result.get("max_similarity", 0.0),
            threshold=similarity_result.get("threshold", 0.85),
            blocked=result.get("blocked", False),
            matched_count=similarity_result.get("matched_count", 0),
            total_found_count=similarity_result.get("total_found_count", 0),
            matched_documents=similarity_result.get("matched_documents", []),
            analysis_time_ms=result.get("analysis_time_ms", 0)
        )
        
        logger.info(f"Similarity analysis completed: is_similar={response_data.is_similar}, "
                   f"max_similarity={response_data.max_similarity:.4f}, "
                   f"time={response_data.analysis_time_ms}ms")
        
        return response_data
        
    except ValueError as e:
        logger.warning(f"Invalid request data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(f"Similarity-only analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Similarity analysis failed: {str(e)}"
        )


@router.get(
    "/health",
    summary="분석 서비스 상태 확인",
    description="AI 모델 로딩 상태 및 서비스 상태를 확인합니다."
)
async def health_check(
    service: IntegratedAnalysisService = Depends(get_analysis_service)
):
    """분석 서비스의 상태를 확인합니다."""
    try:
        # 모델 로딩 상태 확인
        service._ensure_models_loaded()
        
        # 간단한 테스트 분석 수행
        test_result = await service.analyze_comprehensive(
            text="테스트 텍스트입니다."
        )
        
        return {
            "status": "healthy",
            "models_loaded": True,
            "pii_detector": "loaded",
            "similarity_detector": "loaded",
            "test_analysis_time_ms": test_result.get("analysis_time_ms", 0),
            "message": "All systems operational"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "models_loaded": False,
                "error": str(e),
                "message": "Service unavailable"
            }
        )