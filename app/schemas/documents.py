"""
문서 관리 API용 Pydantic 스키마
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    """문서 업로드 응답 스키마"""
    document_id: str = Field(..., description="생성된 문서 ID")
    title: str = Field(..., description="문서 제목")
    file_size: int = Field(..., description="파일 크기 (bytes)")
    content_length: int = Field(..., description="텍스트 내용 길이")
    status: str = Field(..., description="업로드 상태")
    embedding_status: str = Field(..., description="임베딩 처리 상태")
    
    class Config:
        schema_extra = {
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "title": "기밀_문서_샘플.txt",
                "file_size": 2048,
                "content_length": 1856,
                "status": "uploaded",
                "embedding_status": "processing"
            }
        }


class DocumentChunkInfo(BaseModel):
    """문서 청크 정보"""
    chunk_index: int = Field(..., description="청크 인덱스")
    content_preview: str = Field(..., description="내용 미리보기")
    content_length: int = Field(..., description="청크 내용 길이")
    embedding_id: Optional[str] = Field(None, description="임베딩 ID")


class DocumentListItem(BaseModel):
    """문서 목록 항목"""
    id: str = Field(..., description="문서 ID")
    title: str = Field(..., description="문서 제목")
    file_size: Optional[int] = Field(None, description="파일 크기")
    chunk_count: int = Field(..., description="청크 개수")
    embedding_status: str = Field(..., description="임베딩 상태")
    created_at: str = Field(..., description="생성 시간")
    updated_at: str = Field(..., description="업데이트 시간")


class DocumentListResponse(BaseModel):
    """문서 목록 응답"""
    total: int = Field(..., description="총 문서 수")
    items: List[DocumentListItem] = Field(..., description="문서 목록")
    
    class Config:
        schema_extra = {
            "example": {
                "total": 2,
                "items": [
                    {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "title": "기밀_문서_샘플.txt",
                        "file_size": 2048,
                        "chunk_count": 3,
                        "embedding_status": "completed",
                        "created_at": "2024-03-15T10:30:00",
                        "updated_at": "2024-03-15T10:35:00"
                    }
                ]
            }
        }


class DocumentDetailResponse(BaseModel):
    """문서 상세 정보 응답"""
    id: str = Field(..., description="문서 ID")
    title: str = Field(..., description="문서 제목")
    file_path: str = Field(..., description="파일 경로")
    file_size: Optional[int] = Field(None, description="파일 크기")
    content_hash: Optional[str] = Field(None, description="내용 해시")
    chunk_count: int = Field(..., description="청크 개수")
    embedding_status: str = Field(..., description="임베딩 상태")
    chunks: List[DocumentChunkInfo] = Field(default_factory=list, description="청크 정보들")
    created_at: str = Field(..., description="생성 시간")
    updated_at: str = Field(..., description="업데이트 시간")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "title": "기밀_문서_샘플.txt",
                "file_path": "/storage/uploads/550e8400-e29b-41d4-a716-446655440000.txt",
                "file_size": 2048,
                "content_hash": "a1b2c3d4...",
                "chunk_count": 3,
                "embedding_status": "completed",
                "chunks": [
                    {
                        "chunk_index": 0,
                        "content_preview": "이 문서는 우리 회사의 기밀 정보를 포함하고 있습니다...",
                        "content_length": 512,
                        "embedding_id": "doc_chunk_0"
                    }
                ],
                "created_at": "2024-03-15T10:30:00",
                "updated_at": "2024-03-15T10:35:00"
            }
        }


class DocumentDeleteResponse(BaseModel):
    """문서 삭제 응답"""
    success: bool = Field(..., description="삭제 성공 여부")
    message: str = Field(..., description="응답 메시지")
    document_id: str = Field(..., description="삭제된 문서 ID")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Document deleted successfully",
                "document_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }


class ReindexResponse(BaseModel):
    """재인덱싱 응답"""
    total_documents: int = Field(..., description="총 문서 수")
    successful: int = Field(..., description="성공한 문서 수")
    failed: int = Field(..., description="실패한 문서 수")
    errors: List[str] = Field(default_factory=list, description="오류 메시지들")
    
    class Config:
        schema_extra = {
            "example": {
                "total_documents": 5,
                "successful": 4,
                "failed": 1,
                "errors": [
                    "Failed to reindex document doc-123: File not found"
                ]
            }
        }


class DocumentUploadRequest(BaseModel):
    """문서 업로드 요청 (멀티파트 폼이 아닌 경우)"""
    title: Optional[str] = Field(None, description="문서 제목")
    
    @validator('title')
    def validate_title(cls, v):
        if v is not None and len(v.strip()) == 0:
            raise ValueError("Title cannot be empty")
        return v.strip() if v else None