"""
통합 분석 API용 Pydantic 스키마
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime


class AnalysisRequest(BaseModel):
    """통합 검사 요청 스키마 (단순화)"""
    text: str = Field(..., min_length=1, max_length=10000, description="분석할 텍스트")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "text": "제 번호는 010-1234-5678이고, 우리 회사의 기밀 프로젝트 X 정보입니다."
            }
        }


class PIIEntity(BaseModel):
    """PII 엔티티 정보"""
    type: str = Field(..., description="PII 엔티티 타입")
    value: str = Field(..., description="탐지된 값")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도")
    token_count: int = Field(..., ge=1, description="토큰 개수")


class PIIAnalysisResult(BaseModel):
    """PII 분석 결과"""
    has_pii: bool = Field(..., description="PII 탐지 여부")
    threshold: float = Field(..., description="사용된 임계값")
    entities: List[PIIEntity] = Field(default_factory=list, description="탐지된 PII 엔티티들")
    total_entities: int = Field(..., description="총 탐지된 엔티티 수")
    reason: str = Field(default="", description="분석 결과 요약")
    details: str = Field(default="", description="상세 분석 내용")


class MatchedText(BaseModel):
    """매칭된 텍스트 정보"""
    content: str = Field(..., description="매칭된 텍스트 내용")
    similarity: float = Field(..., ge=0.0, le=1.0, description="KoSimCSE 유사도 점수")
    similarity_percentage: str = Field(..., description="유사도 백분율 표시")
    chunk_index: int = Field(..., description="청크 인덱스")


class MatchedDocument(BaseModel):
    """매칭된 문서 정보"""
    document_id: str = Field(..., description="문서 ID")
    document_title: str = Field(..., description="문서 제목")
    max_similarity: float = Field(..., ge=0.0, le=1.0, description="최대 유사도")
    matched_chunks_count: int = Field(..., description="매칭된 청크 수")
    matched_texts: List[MatchedText] = Field(default_factory=list, description="매칭된 텍스트들")


class SimilarityAnalysisResult(BaseModel):
    """유사도 분석 결과"""
    is_similar: bool = Field(..., description="유사도 탐지 여부 (임계값 기준)")
    max_similarity: float = Field(..., ge=0.0, le=1.0, description="최대 유사도 (전체 검색 결과 중)")
    threshold: float = Field(..., description="사용된 임계값")
    matched_count: int = Field(..., description="임계값 이상 청크 수 (차단 결정 기준)")
    total_found_count: Optional[int] = Field(None, description="전체 검색된 청크 수")
    matched_documents: List[MatchedDocument] = Field(default_factory=list, description="매칭된 문서들 (임계값 이하 포함)")


class AnalysisResponse(BaseModel):
    """통합 분석 응답 스키마"""
    input_text: str = Field(..., description="분석 대상 텍스트")
    input_hash: str = Field(..., description="입력 텍스트 해시")
    blocked: bool = Field(..., description="차단 여부")
    block_reasons: List[str] = Field(default_factory=list, description="차단 사유들")
    analysis_time_ms: int = Field(..., description="분석 소요 시간 (ms)")
    pii_analysis: Optional[PIIAnalysisResult] = Field(None, description="PII 분석 결과")
    similarity_analysis: Optional[SimilarityAnalysisResult] = Field(None, description="유사도 분석 결과")
    
    class Config:
        schema_extra = {
            "example": {
                "input_text": "우리 회사의 기밀 프로젝트 X에 대한 상세 기술 사양서입니다.",
                "input_hash": "a1b2c3d4...",
                "blocked": True,
                "block_reasons": ["similarity_detected"],
                "analysis_time_ms": 450,
                "pii_analysis": {
                    "has_pii": False,
                    "threshold": 0.90,
                    "entities": [],
                    "total_entities": 0,
                    "reason": "개인정보가 탐지되지 않았습니다.",
                    "details": ""
                },
                "similarity_analysis": {
                    "is_similar": True,
                    "max_similarity": 0.92,
                    "threshold": 0.85,
                    "matched_count": 2,
                    "matched_documents": [
                        {
                            "document_id": "doc-123",
                            "document_title": "프로젝트X_기술사양서.txt",
                            "max_similarity": 0.92,
                            "matched_chunks_count": 2,
                            "matched_texts": [
                                {
                                    "content": "기밀 프로젝트 X 기술 사양서...",
                                    "similarity": 0.92,
                                    "chunk_index": 0
                                }
                            ]
                        }
                    ]
                }
            }
        }


class AnalysisHistoryItem(BaseModel):
    """분석 이력 항목"""
    id: str = Field(..., description="분석 ID")
    input_text: str = Field(..., description="분석 텍스트 (축약)")
    blocked: bool = Field(..., description="차단 여부")
    block_reasons: List[str] = Field(default_factory=list, description="차단 사유들")
    pii_detected: bool = Field(..., description="PII 탐지 여부")
    similarity_detected: bool = Field(..., description="유사도 탐지 여부")
    max_similarity: Optional[float] = Field(None, description="최대 유사도")
    analysis_time_ms: Optional[int] = Field(None, description="분석 시간")
    created_at: str = Field(..., description="분석 시간")


class AnalysisHistoryResponse(BaseModel):
    """분석 이력 응답"""
    total: int = Field(..., description="총 이력 수")
    items: List[AnalysisHistoryItem] = Field(..., description="분석 이력 목록")


class AnalysisStatsResponse(BaseModel):
    """분석 통계 응답"""
    total_analyses: int = Field(..., description="총 분석 수")
    blocked_analyses: int = Field(..., description="차단된 분석 수")
    pii_detections: int = Field(..., description="PII 탐지 수")
    similarity_detections: int = Field(..., description="유사도 탐지 수")
    block_rate: float = Field(..., description="차단율")
    avg_analysis_time_ms: Optional[float] = Field(None, description="평균 분석 시간")


class SimilarityOnlyRequest(BaseModel):
    """유사도 전용 분석 요청 스키마"""
    text: str = Field(..., min_length=1, max_length=10000, description="분석할 텍스트")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "text": "우리 회사의 기밀 프로젝트 X에 대한 상세 기술 사양서입니다."
            }
        }


class SimilarityOnlyResponse(BaseModel):
    """유사도 전용 분석 응답 스키마"""
    input_text: str = Field(..., description="분석 대상 텍스트")
    is_similar: bool = Field(..., description="유사도 탐지 여부 (임계값 기준)")
    max_similarity: float = Field(..., ge=0.0, le=1.0, description="최대 유사도 (KoSimCSE, 전체 결과 중)")
    threshold: float = Field(..., description="사용된 임계값")
    blocked: bool = Field(..., description="차단 여부 (임계값 기준)")
    matched_count: int = Field(..., description="임계값 이상 청크 수")
    total_found_count: Optional[int] = Field(None, description="전체 검색된 청크 수")
    matched_documents: List[MatchedDocument] = Field(default_factory=list, description="매칭된 문서들 (임계값 이하 포함)")
    analysis_time_ms: int = Field(..., description="분석 소요 시간 (ms)")
    
    class Config:
        schema_extra = {
            "example": {
                "input_text": "우리 회사의 기밀 프로젝트 X에 대한 상세 기술 사양서입니다.",
                "is_similar": True,
                "max_similarity": 0.92,
                "threshold": 0.85,
                "blocked": True,
                "matched_count": 2,
                "matched_documents": [
                    {
                        "document_id": "doc-123",
                        "document_title": "프로젝트X_기술사양서.txt",
                        "max_similarity": 0.92,
                        "matched_chunks_count": 2,
                        "matched_texts": [
                            {
                                "content": "기밀 프로젝트 X 기술 사양서...",
                                "similarity": 0.92,
                                "similarity_percentage": "92.0%",
                                "chunk_index": 0
                            }
                        ]
                    }
                ],
                "analysis_time_ms": 250
            }
        }

