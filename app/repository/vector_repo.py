"""
ChromaDB 벡터 저장소 Repository
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import uuid
from datetime import datetime

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.api.models.Collection import Collection

from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorRepository:
    """ChromaDB를 이용한 벡터 저장소 Repository"""
    
    def __init__(self):
        """ChromaDB 클라이언트 초기화"""
        self.client: Optional[chromadb.HttpClient] = None
        self.collection: Optional[Collection] = None
        self.collection_name = settings.CHROMADB_COLLECTION_NAME
        
    def connect(self) -> None:
        """ChromaDB 서버에 연결합니다."""
        try:
            logger.info(f"Connecting to ChromaDB at {settings.CHROMADB_HOST}:{settings.CHROMADB_PORT}")
            
            self.client = chromadb.HttpClient(
                host=settings.CHROMADB_HOST,
                port=settings.CHROMADB_PORT,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 헬스체크
            self.client.heartbeat()
            logger.info("Successfully connected to ChromaDB")
            
            # 컬렉션 생성 또는 가져오기
            self._get_or_create_collection()
            
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {str(e)}")
            raise RuntimeError(f"ChromaDB connection failed: {str(e)}")
    
    def _ensure_connected(self) -> None:
        """연결이 되어 있는지 확인하고, 없으면 연결합니다."""
        if self.client is None:
            self.connect()
    
    def _get_or_create_collection(self) -> None:
        """컬렉션을 가져오거나 생성합니다."""
        try:
            # 기존 컬렉션 가져오기 시도
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Retrieved existing collection: {self.collection_name}")
            
        except Exception:
            # 컬렉션이 없으면 새로 생성
            try:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "description": "Sensitive document embeddings for similarity detection",
                        "created_at": datetime.now().isoformat()
                    }
                )
                logger.info(f"Created new collection: {self.collection_name}")
                
            except Exception as e:
                logger.error(f"Failed to create collection: {str(e)}")
                raise RuntimeError(f"Collection creation failed: {str(e)}")
    
    async def store_document_chunks(
        self,
        document_id: str,
        chunks: List[str],
        embeddings: List[List[float]],
        chunk_indices: List[int],
        document_title: str = ""
    ) -> List[str]:
        """
        문서 청크들과 임베딩을 저장합니다.
        
        Args:
            document_id: 문서 ID
            chunks: 텍스트 청크들
            embeddings: 임베딩 벡터들
            chunk_indices: 청크 인덱스들
            document_title: 문서 제목
            
        Returns:
            생성된 임베딩 ID들
        """
        self._ensure_connected()
        
        if not chunks or not embeddings or len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length")
        
        try:
            # 임베딩 ID 생성
            embedding_ids = [f"{document_id}_chunk_{idx}" for idx in chunk_indices]
            
            # 메타데이터 생성
            metadatas = [
                {
                    "document_id": document_id,
                    "document_title": document_title,
                    "chunk_index": chunk_idx,
                    "chunk_text_length": len(chunk),
                    "created_at": datetime.now().isoformat()
                }
                for chunk, chunk_idx in zip(chunks, chunk_indices)
            ]
            
            # ChromaDB에 추가
            self.collection.add(
                ids=embedding_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas
            )
            
            logger.info(f"Stored {len(chunks)} chunks for document {document_id}")
            return embedding_ids
            
        except Exception as e:
            logger.error(f"Failed to store document chunks: {str(e)}")
            raise RuntimeError(f"Document chunks storage failed: {str(e)}")
    
    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        similarity_threshold: float = 0.8
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        쿼리 임베딩과 유사한 청크들을 검색합니다.
        
        Args:
            query_embedding: 쿼리 임베딩 벡터
            top_k: 반환할 최대 결과 수
            similarity_threshold: 유사도 임계값 (차단 결정용)
            
        Returns:
            Tuple[임계값 이상 청크들, 전체 상위 청크들]
        """
        self._ensure_connected()
        
        try:
            # ChromaDB에서 유사도 검색 (임계값 상관없이 top_k 가져오기)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # 결과 포맷팅
            all_chunks = []
            threshold_chunks = []
            
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    # ChromaDB는 거리(distance)를 반환하므로, 유사도로 변환
                    # 코사인 거리에서 유사도로 변환: similarity = 1 - distance
                    distance = results["distances"][0][i]
                    similarity = 1.0 - distance
                    
                    chunk_info = {
                        "chunk_id": results["ids"][0][i],
                        "document_id": results["metadatas"][0][i]["document_id"],
                        "document_title": results["metadatas"][0][i].get("document_title", ""),
                        "chunk_index": results["metadatas"][0][i]["chunk_index"],
                        "content": results["documents"][0][i],
                        "similarity": float(similarity),
                        "distance": float(distance)
                    }
                    
                    # 전체 결과에 추가
                    all_chunks.append(chunk_info)
                    
                    # 임계값 이상인 것만 따로 저장
                    if similarity >= similarity_threshold:
                        threshold_chunks.append(chunk_info)
            
            logger.debug(f"Found {len(all_chunks)} total chunks, {len(threshold_chunks)} above threshold {similarity_threshold}")
            return threshold_chunks, all_chunks
            
        except Exception as e:
            logger.error(f"Similar chunks search failed: {str(e)}")
            raise RuntimeError(f"Similar chunks search failed: {str(e)}")
    
    async def delete_document_chunks(self, document_id: str) -> int:
        """
        특정 문서의 모든 청크를 삭제합니다.
        
        Args:
            document_id: 삭제할 문서 ID
            
        Returns:
            삭제된 청크 수
        """
        self._ensure_connected()
        
        try:
            # 해당 문서의 모든 청크 찾기
            results = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )
            
            if results["ids"]:
                # 청크들 삭제
                self.collection.delete(ids=results["ids"])
                deleted_count = len(results["ids"])
                logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
                return deleted_count
            else:
                logger.info(f"No chunks found for document {document_id}")
                return 0
                
        except Exception as e:
            logger.error(f"Failed to delete document chunks: {str(e)}")
            raise RuntimeError(f"Document chunks deletion failed: {str(e)}")
    
    async def get_all_document_texts(self) -> List[str]:
        """
        저장된 모든 문서 텍스트를 가져옵니다.
        
        Returns:
            모든 청크 텍스트들의 리스트
        """
        self._ensure_connected()
        
        try:
            # 모든 문서 가져오기
            results = self.collection.get(
                include=["documents"]
            )
            
            texts = results["documents"] if results["documents"] else []
            logger.debug(f"Retrieved {len(texts)} document texts")
            return texts
            
        except Exception as e:
            logger.error(f"Failed to get all document texts: {str(e)}")
            return []


# 전역 벡터 repository 인스턴스
_vector_repo_instance: Optional[VectorRepository] = None


def get_vector_repository() -> VectorRepository:
    """벡터 repository 인스턴스를 반환합니다. (싱글톤 패턴)"""
    global _vector_repo_instance
    
    if _vector_repo_instance is None:
        _vector_repo_instance = VectorRepository()
    
    return _vector_repo_instance