"""
민감 문서 관련 SQLAlchemy 모델
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID
import uuid

from sqlalchemy import String, Integer, DateTime, Text, Boolean, DECIMAL, func, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import expression

from app.db.base import Base


class SensitiveDocument(Base):
    """민감 문서 정보"""
    __tablename__ = "sensitive_documents"
    
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(500), nullable=False, comment="문서 제목")
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False, comment="파일 저장 경로")
    file_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment="파일 크기 (bytes)")
    content_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, comment="파일 내용 해시")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, comment="청크 개수")
    embedding_status: Mapped[str] = mapped_column(
        String(20), 
        default="pending",
        comment="임베딩 처리 상태: pending, processing, completed, failed"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    
    # 관계: 문서 청크들
    chunks: Mapped[List["DocumentChunk"]] = relationship(
        "DocumentChunk", 
        back_populates="document",
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<SensitiveDocument(id={self.id}, title='{self.title}', status='{self.embedding_status}')>"


class DocumentChunk(Base):
    """문서 청크 (문서를 작은 단위로 나눈 것)"""
    __tablename__ = "document_chunks"
    
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("sensitive_documents.id"), nullable=False, comment="소속 문서 ID")
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, comment="청크 순서")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="청크 텍스트 내용")
    embedding_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, comment="ChromaDB 벡터 ID")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    # 관계: 소속 문서
    document: Mapped["SensitiveDocument"] = relationship("SensitiveDocument", back_populates="chunks")
    
    def __repr__(self) -> str:
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, index={self.chunk_index})>"


class AnalysisLog(Base):
    """통합 분석 결과 로그"""
    __tablename__ = "analysis_logs"
    
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    input_text: Mapped[str] = mapped_column(Text, nullable=False, comment="분석 대상 텍스트")
    input_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, comment="중복 분석 방지용 해시")
    
    # 차단 관련
    blocked: Mapped[bool] = mapped_column(Boolean, default=False, comment="차단 여부")
    block_reasons: Mapped[List[str]] = mapped_column(ARRAY(String), default=list, comment="차단 사유들")
    
    # PII 분석 결과
    pii_detected: Mapped[bool] = mapped_column(Boolean, default=False, comment="PII 탐지 여부")
    pii_entities: Mapped[dict] = mapped_column(JSONB, default=dict, comment="탐지된 PII 엔티티들")
    
    # 유사도 분석 결과
    similarity_detected: Mapped[bool] = mapped_column(Boolean, default=False, comment="유사도 탐지 여부")
    max_similarity: Mapped[Optional[float]] = mapped_column(DECIMAL(5, 4), nullable=True, comment="최대 유사도")
    matched_documents: Mapped[dict] = mapped_column(JSONB, default=dict, comment="매칭된 문서들")
    
    # 메타데이터
    user_ip: Mapped[Optional[str]] = mapped_column(String(45), nullable=True, comment="사용자 IP")
    analysis_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment="분석 소요 시간 (ms)")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    def __repr__(self) -> str:
        return f"<AnalysisLog(id={self.id}, blocked={self.blocked}, pii={self.pii_detected}, similarity={self.similarity_detected})>"


class SystemConfig(Base):
    """시스템 전역 설정 (단일 레코드)"""
    __tablename__ = "system_config"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1, comment="항상 1 (단일 레코드)")
    similarity_threshold: Mapped[float] = mapped_column(DECIMAL(3, 2), default=0.85, comment="유사도 임계값")
    pii_threshold: Mapped[float] = mapped_column(DECIMAL(3, 2), default=0.90, comment="PII 탐지 임계값")
    auto_block_pii: Mapped[bool] = mapped_column(Boolean, default=True, comment="PII 자동 차단 활성화")
    auto_block_similarity: Mapped[bool] = mapped_column(Boolean, default=True, comment="유사도 자동 차단 활성화")
    max_file_size_mb: Mapped[int] = mapped_column(Integer, default=10, comment="최대 업로드 파일 크기 (MB)")
    settings: Mapped[dict] = mapped_column(JSONB, default=dict, comment="기타 설정들")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    
    def __repr__(self) -> str:
        return f"<SystemConfig(similarity_threshold={self.similarity_threshold}, pii_threshold={self.pii_threshold})>"