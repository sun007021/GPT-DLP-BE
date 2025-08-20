"""
민감 문서 관리 서비스
"""

import logging
import hashlib
import os
import aiofiles
from typing import List, Dict, Any, Optional, BinaryIO
from pathlib import Path
import uuid
from datetime import datetime

from fastapi import UploadFile

from app.ai.similarity_detector import KoSimCSESimilarityDetector
from app.repository.vector_repo import get_vector_repository
from app.repository.detection_repo import DetectionRepository
from app.models.documents import SensitiveDocument, DocumentChunk
from app.utils.text_chunker import TextChunker
from app.core.config import settings

logger = logging.getLogger(__name__)


class DocumentService:
    """민감 문서 관리 서비스"""
    
    def __init__(self):
        """서비스 초기화"""
        self.similarity_detector: Optional[KoSimCSESimilarityDetector] = None
        self.vector_repo = get_vector_repository()
        self.detection_repo = DetectionRepository()
        self.text_chunker = TextChunker()
        
        # 업로드 디렉토리 생성
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("DocumentService initialized")
    
    def _ensure_similarity_detector_loaded(self) -> None:
        """유사도 모델이 로드되어 있는지 확인하고, 없으면 로드"""
        if self.similarity_detector is None:
            logger.info("Loading similarity detector for document processing...")
            self.similarity_detector = KoSimCSESimilarityDetector()
    
    async def upload_document(
        self,
        file: UploadFile,
        title: str = None
    ) -> Dict[str, Any]:
        """
        민감 문서를 업로드하고 처리합니다.
        
        Args:
            file: 업로드된 파일
            title: 문서 제목 (선택적)
            
        Returns:
            업로드 결과
        """
        try:
            # 파일 검증
            self._validate_uploaded_file(file)
            
            # 파일 제목 설정
            if not title:
                title = file.filename or f"uploaded_document_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Starting document upload: {title}")
            
            # 고유 문서 ID 생성
            document_id = str(uuid.uuid4())
            
            # 파일 저장
            file_path = await self._save_uploaded_file(file, document_id)
            
            # 파일 내용 읽기
            text_content = await self._extract_text_content(file_path)
            content_hash = hashlib.sha256(text_content.encode()).hexdigest()
            
            # 데이터베이스에 문서 정보 저장
            document = SensitiveDocument(
                id=document_id,
                title=title,
                file_path=str(file_path),
                file_size=len(text_content.encode()),
                content_hash=content_hash,
                embedding_status="pending"
            )
            
            saved_document = await self.detection_repo.create_sensitive_document(document)
            
            # 백그라운드에서 임베딩 생성 (비동기 처리)
            await self._process_document_embeddings(saved_document, text_content)
            
            result = {
                "document_id": document_id,
                "title": title,
                "file_size": len(text_content.encode()),
                "content_length": len(text_content),
                "status": "uploaded",
                "embedding_status": "processing"
            }
            
            logger.info(f"Document upload completed: {document_id}")
            return result
            
        except Exception as e:
            logger.error(f"Document upload failed: {str(e)}")
            raise RuntimeError(f"Document upload failed: {str(e)}")
    
    def _validate_uploaded_file(self, file: UploadFile) -> None:
        """업로드된 파일을 검증합니다."""
        # 파일 크기 검증
        if hasattr(file, 'size') and file.size:
            max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024  # MB to bytes
            if file.size > max_size:
                raise ValueError(f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB}MB")
        
        # 파일 확장자 검증
        if file.filename:
            file_extension = Path(file.filename).suffix.lower().lstrip('.')
            if file_extension not in settings.ALLOWED_EXTENSIONS:
                raise ValueError(f"File type '{file_extension}' not allowed. Allowed types: {settings.ALLOWED_EXTENSIONS}")
    
    async def _save_uploaded_file(self, file: UploadFile, document_id: str) -> Path:
        """업로드된 파일을 저장합니다."""
        # 파일명 생성 (중복 방지)
        file_extension = Path(file.filename).suffix if file.filename else '.txt'
        filename = f"{document_id}{file_extension}"
        file_path = self.upload_dir / filename
        
        # 파일 저장
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        logger.debug(f"File saved: {file_path}")
        return file_path
    
    async def _extract_text_content(self, file_path: Path) -> str:
        """파일에서 텍스트 내용을 추출합니다."""
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.txt':
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    return await f.read()
            
            elif file_extension == '.md':
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    return await f.read()
            
            # TODO: PDF, DOCX 파일 처리 추가
            elif file_extension == '.pdf':
                # pypdf 사용 예정
                raise NotImplementedError("PDF processing not yet implemented")
            
            elif file_extension == '.docx':
                # python-docx 사용 예정
                raise NotImplementedError("DOCX processing not yet implemented")
            
            else:
                # 기본적으로 텍스트 파일로 처리 시도
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    return await f.read()
                    
        except UnicodeDecodeError:
            logger.error(f"Failed to decode file as UTF-8: {file_path}")
            raise ValueError("File encoding not supported. Please upload UTF-8 encoded text files.")
        
        except Exception as e:
            logger.error(f"Text extraction failed for {file_path}: {str(e)}")
            raise RuntimeError(f"Failed to extract text from file: {str(e)}")
    
    async def _process_document_embeddings(self, document: SensitiveDocument, text_content: str) -> None:
        """문서 임베딩을 생성하고 저장합니다."""
        try:
            logger.info(f"Starting embedding processing for document: {document.id}")
            
            # 문서 상태를 처리 중으로 업데이트
            await self.detection_repo.update_document_embedding_status(
                document.id, "processing"
            )
            
            # 유사도 모델 로드
            self._ensure_similarity_detector_loaded()
            
            # 텍스트 청킹
            chunks_with_positions = self.text_chunker.chunk_text(text_content)
            
            if not chunks_with_positions:
                logger.warning(f"No chunks created for document: {document.id}")
                await self.detection_repo.update_document_embedding_status(
                    document.id, "failed"
                )
                return
            
            # 청크들과 임베딩 생성
            chunk_texts = [chunk[0] for chunk in chunks_with_positions]
            chunk_indices = list(range(len(chunk_texts)))
            
            # 배치로 임베딩 생성
            embeddings = self.similarity_detector.encode_texts(chunk_texts)
            
            # 벡터 저장소에 저장
            embedding_ids = await self.vector_repo.store_document_chunks(
                document_id=str(document.id),
                chunks=chunk_texts,
                embeddings=embeddings.tolist(),
                chunk_indices=chunk_indices,
                document_title=document.title
            )
            
            # 데이터베이스에 청크 정보 저장
            for i, (chunk_text, position) in enumerate(chunks_with_positions):
                chunk = DocumentChunk(
                    document_id=document.id,
                    chunk_index=i,
                    content=chunk_text,
                    embedding_id=embedding_ids[i]
                )
                await self.detection_repo.create_document_chunk(chunk)
            
            # 문서 정보 업데이트
            await self.detection_repo.update_document_embedding_status(
                document.id, "completed", chunk_count=len(chunks_with_positions)
            )
            
            logger.info(f"Embedding processing completed for document: {document.id}, {len(chunks_with_positions)} chunks")
            
        except Exception as e:
            logger.error(f"Embedding processing failed for document {document.id}: {str(e)}")
            await self.detection_repo.update_document_embedding_status(
                document.id, "failed"
            )
            raise
    
    async def get_documents_list(
        self,
        skip: int = 0,
        limit: int = 100,
        status_filter: str = None
    ) -> List[Dict[str, Any]]:
        """문서 목록을 조회합니다."""
        try:
            documents = await self.detection_repo.get_sensitive_documents(
                skip=skip,
                limit=limit,
                embedding_status=status_filter
            )
            
            result = []
            for doc in documents:
                result.append({
                    "id": str(doc.id),
                    "title": doc.title,
                    "file_size": doc.file_size,
                    "chunk_count": doc.chunk_count,
                    "embedding_status": doc.embedding_status,
                    "created_at": doc.created_at.isoformat(),
                    "updated_at": doc.updated_at.isoformat()
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get documents list: {str(e)}")
            return []
    
    async def get_document_detail(self, document_id: str) -> Optional[Dict[str, Any]]:
        """문서 상세 정보를 조회합니다."""
        try:
            document = await self.detection_repo.get_sensitive_document_by_id(document_id)
            
            if not document:
                return None
            
            # 청크 정보 조회
            chunks = await self.detection_repo.get_document_chunks(document_id)
            
            chunk_info = []
            for chunk in chunks:
                chunk_info.append({
                    "chunk_index": chunk.chunk_index,
                    "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    "content_length": len(chunk.content),
                    "embedding_id": chunk.embedding_id
                })
            
            return {
                "id": str(document.id),
                "title": document.title,
                "file_path": document.file_path,
                "file_size": document.file_size,
                "content_hash": document.content_hash,
                "chunk_count": document.chunk_count,
                "embedding_status": document.embedding_status,
                "chunks": chunk_info,
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get document detail: {str(e)}")
            return None
    
    async def delete_document(self, document_id: str) -> bool:
        """문서를 삭제합니다."""
        try:
            logger.info(f"Deleting document: {document_id}")
            
            # 데이터베이스에서 문서 조회
            document = await self.detection_repo.get_sensitive_document_by_id(document_id)
            
            if not document:
                logger.warning(f"Document not found: {document_id}")
                return False
            
            # 벡터 저장소에서 삭제
            await self.vector_repo.delete_document_chunks(document_id)
            
            # 파일 시스템에서 삭제
            if os.path.exists(document.file_path):
                os.remove(document.file_path)
                logger.debug(f"File deleted: {document.file_path}")
            
            # 데이터베이스에서 삭제 (CASCADE로 청크도 함께 삭제됨)
            await self.detection_repo.delete_sensitive_document(document_id)
            
            logger.info(f"Document deleted successfully: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {str(e)}")
            return False
    
    async def reindex_all_documents(self) -> Dict[str, Any]:
        """모든 문서를 재인덱싱합니다."""
        try:
            logger.info("Starting reindexing of all documents...")
            
            # 모든 문서 조회
            documents = await self.detection_repo.get_sensitive_documents(limit=1000)
            
            reindex_results = {
                "total_documents": len(documents),
                "successful": 0,
                "failed": 0,
                "errors": []
            }
            
            for document in documents:
                try:
                    # 파일에서 텍스트 다시 추출
                    text_content = await self._extract_text_content(Path(document.file_path))
                    
                    # 기존 벡터 데이터 삭제
                    await self.vector_repo.delete_document_chunks(str(document.id))
                    
                    # 새로운 임베딩 생성
                    await self._process_document_embeddings(document, text_content)
                    
                    reindex_results["successful"] += 1
                    logger.info(f"Reindexed document: {document.id}")
                    
                except Exception as e:
                    reindex_results["failed"] += 1
                    error_msg = f"Failed to reindex document {document.id}: {str(e)}"
                    reindex_results["errors"].append(error_msg)
                    logger.error(error_msg)
            
            logger.info(f"Reindexing completed: {reindex_results}")
            return reindex_results
            
        except Exception as e:
            logger.error(f"Reindexing failed: {str(e)}")
            return {
                "total_documents": 0,
                "successful": 0,
                "failed": 0,
                "error": str(e)
            }