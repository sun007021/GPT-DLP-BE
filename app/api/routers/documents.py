"""
문서 관리 API 라우터
"""

import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse

from app.schemas.documents import (
    DocumentUploadResponse, DocumentListResponse, DocumentDetailResponse,
    DocumentDeleteResponse, ReindexResponse
)
from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])

# 서비스 인스턴스 (싱글톤)
_document_service: Optional[DocumentService] = None


def get_document_service() -> DocumentService:
    """문서 서비스 인스턴스를 반환합니다."""
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    summary="민감 문서 업로드",
    description="민감 문서를 업로드하고 유사도 분석을 위한 임베딩을 생성합니다."
)
async def upload_document(
    file: UploadFile = File(..., description="업로드할 문서 파일"),
    title: Optional[str] = Form(None, description="문서 제목 (선택적)"),
    service: DocumentService = Depends(get_document_service)
) -> DocumentUploadResponse:
    """
    민감 문서를 업로드합니다.
    
    - **지원 파일 형식**: .txt, .md, .pdf, .docx
    - **최대 파일 크기**: 설정에 따라 결정 (기본 10MB)
    - **처리 과정**: 파일 저장 → 텍스트 추출 → 청킹 → 임베딩 생성
    """
    try:
        # 파일 정보 로깅
        logger.info(f"Document upload request: filename={file.filename}, size={getattr(file, 'size', 'unknown')}")
        
        # 문서 업로드 처리
        result = await service.upload_document(
            file=file,
            title=title
        )
        
        logger.info(f"Document upload completed: {result['document_id']}")
        return DocumentUploadResponse(**result)
        
    except ValueError as e:
        logger.warning(f"Invalid upload request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Document upload failed: {str(e)}"
        )


@router.get(
    "/list",
    response_model=DocumentListResponse,
    summary="문서 목록 조회",
    description="등록된 민감 문서 목록을 조회합니다."
)
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    service: DocumentService = Depends(get_document_service)
) -> DocumentListResponse:
    """
    문서 목록을 조회합니다.
    
    - **skip**: 건너뛸 항목 수 (페이징)
    - **limit**: 반환할 최대 항목 수 (최대 1000)
    - **status**: 임베딩 상태로 필터링 (pending, processing, completed, failed)
    """
    try:
        # 파라미터 검증
        if skip < 0:
            raise ValueError("Skip must be non-negative")
        if limit <= 0 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000")
        if status and status not in ["pending", "processing", "completed", "failed"]:
            raise ValueError("Status must be one of: pending, processing, completed, failed")
        
        logger.info(f"Getting documents list: skip={skip}, limit={limit}, status={status}")
        
        # 문서 목록 조회
        documents = await service.get_documents_list(
            skip=skip,
            limit=limit,
            status_filter=status
        )
        
        return DocumentListResponse(
            total=len(documents),
            items=documents
        )
        
    except ValueError as e:
        logger.warning(f"Invalid list request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {str(e)}"
        )


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="문서 상세 조회",
    description="특정 문서의 상세 정보를 조회합니다."
)
async def get_document_detail(
    document_id: str,
    service: DocumentService = Depends(get_document_service)
) -> DocumentDetailResponse:
    """
    문서 상세 정보를 조회합니다.
    
    - 문서 메타데이터
    - 청크 정보
    - 임베딩 상태
    """
    try:
        logger.info(f"Getting document detail: {document_id}")
        
        # 문서 상세 정보 조회
        document = await service.get_document_detail(document_id)
        
        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )
        
        return DocumentDetailResponse(**document)
        
    except HTTPException:
        raise  # FastAPI 예외는 그대로 전달
        
    except Exception as e:
        logger.error(f"Failed to get document detail: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get document detail: {str(e)}"
        )


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="문서 삭제",
    description="문서와 관련된 모든 데이터를 삭제합니다."
)
async def delete_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service)
) -> DocumentDeleteResponse:
    """
    문서를 삭제합니다.
    
    - 파일 시스템에서 파일 삭제
    - 데이터베이스에서 메타데이터 삭제
    - 벡터 저장소에서 임베딩 삭제
    """
    try:
        logger.info(f"Deleting document: {document_id}")
        
        # 문서 삭제
        success = await service.delete_document(document_id)
        
        if success:
            return DocumentDeleteResponse(
                success=True,
                message="Document deleted successfully",
                document_id=document_id
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )
        
    except HTTPException:
        raise  # FastAPI 예외는 그대로 전달
        
    except Exception as e:
        logger.error(f"Failed to delete document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )


@router.post(
    "/reindex",
    response_model=ReindexResponse,
    summary="전체 문서 재인덱싱",
    description="모든 문서의 임베딩을 다시 생성합니다."
)
async def reindex_all_documents(
    service: DocumentService = Depends(get_document_service)
) -> ReindexResponse:
    """
    모든 문서를 재인덱싱합니다.
    
    - 기존 벡터 데이터 삭제
    - 문서 다시 읽기
    - 새로운 임베딩 생성
    - 벡터 저장소에 저장
    
    **주의**: 시간이 오래 걸릴 수 있습니다.
    """
    try:
        logger.info("Starting reindexing of all documents")
        
        # 재인덱싱 수행
        result = await service.reindex_all_documents()
        
        logger.info(f"Reindexing completed: {result}")
        return ReindexResponse(**result)
        
    except Exception as e:
        logger.error(f"Reindexing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Reindexing failed: {str(e)}"
        )


@router.get(
    "/stats/overview",
    summary="문서 통계",
    description="문서 관련 전체 통계를 조회합니다."
)
async def get_documents_stats(
    service: DocumentService = Depends(get_document_service)
):
    """
    문서 관리 통계를 조회합니다.
    
    - 총 문서 수
    - 임베딩 상태별 문서 수  
    - 총 청크 수
    - 평균 청크 수
    """
    try:
        logger.info("Getting document statistics")
        
        # 문서 목록으로부터 통계 계산
        all_documents = await service.get_documents_list(limit=1000)
        
        # 상태별 집계
        status_counts = {}
        total_chunks = 0
        total_size = 0
        
        for doc in all_documents:
            status = doc["embedding_status"]
            status_counts[status] = status_counts.get(status, 0) + 1
            total_chunks += doc["chunk_count"]
            if doc["file_size"]:
                total_size += doc["file_size"]
        
        avg_chunks = total_chunks / len(all_documents) if all_documents else 0
        avg_size = total_size / len(all_documents) if all_documents else 0
        
        stats = {
            "total_documents": len(all_documents),
            "status_breakdown": status_counts,
            "total_chunks": total_chunks,
            "average_chunks_per_document": round(avg_chunks, 2),
            "total_file_size_bytes": total_size,
            "average_file_size_bytes": round(avg_size, 2)
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get document stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get document stats: {str(e)}"
        )


@router.get(
    "/health",
    summary="문서 서비스 상태 확인",
    description="문서 관리 서비스의 상태를 확인합니다."
)
async def health_check(
    service: DocumentService = Depends(get_document_service)
):
    """문서 서비스의 상태를 확인합니다."""
    try:
        # 업로드 디렉토리 존재 확인
        upload_dir_exists = service.upload_dir.exists()
        
        # 벡터 저장소 연결 확인
        try:
            service.vector_repo.connect()
            vector_store_connected = True
        except Exception:
            vector_store_connected = False
        
        # 문서 수 확인
        documents = await service.get_documents_list(limit=1)
        
        return {
            "status": "healthy" if upload_dir_exists and vector_store_connected else "degraded",
            "upload_directory_exists": upload_dir_exists,
            "upload_directory_path": str(service.upload_dir),
            "vector_store_connected": vector_store_connected,
            "sample_documents_accessible": len(documents) >= 0,
            "message": "Document service operational"
        }
        
    except Exception as e:
        logger.error(f"Document service health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "message": "Document service unavailable"
            }
        )