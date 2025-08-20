from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func, and_
from app.models.detection import Detection
from app.models.documents import SensitiveDocument, DocumentChunk, AnalysisLog
from app.db.session import get_async_session
from typing import List, Dict, Any, Optional
from uuid import UUID
import logging

logger = logging.getLogger(__name__)

class DetectionRepository:
    def __init__(self):
        """Repository 초기화"""
        pass
        
    async def _get_session(self) -> AsyncSession:
        """데이터베이스 세션 가져오기"""
        # 이 부분은 dependency injection으로 개선 가능
        async for session in get_async_session():
            return session
    
    # === 기존 Detection 관련 메서드들 ===
    async def create(self, db: AsyncSession, det: Detection) -> Detection:
        db.add(det)
        await db.commit()
        await db.refresh(det)
        return det

    async def get(self, db: AsyncSession, detection_id: int) -> Detection | None:
        res = await db.execute(select(Detection).where(Detection.id == detection_id))
        return res.scalar_one_or_none()

    async def update(self, db: AsyncSession, det: Detection) -> Detection:
        await db.commit()
        await db.refresh(det)
        return det
    
    async def get_by_user_ip(self, db: AsyncSession, user_ip: str, limit: int = 100) -> List[Detection]:
        """사용자 IP 기준으로 탐지 기록 조회 (최신순)"""
        res = await db.execute(
            select(Detection)
            .where(Detection.user_ip == user_ip)
            .order_by(desc(Detection.created_at))
            .limit(limit)
        )
        return list(res.scalars().all())
    
    async def get_pii_detections_by_user_ip(self, db: AsyncSession, user_ip: str, limit: int = 100) -> List[Detection]:
        """사용자 IP 기준으로 개인정보가 탐지된 기록만 조회 (최신순)"""
        res = await db.execute(
            select(Detection)
            .where(Detection.user_ip == user_ip, Detection.has_pii == True)
            .order_by(desc(Detection.created_at))
            .limit(limit)
        )
        return list(res.scalars().all())
    
    async def count_by_user_ip(self, db: AsyncSession, user_ip: str) -> int:
        """사용자 IP 기준 총 탐지 횟수"""
        res = await db.execute(
            select(func.count(Detection.id))
            .where(Detection.user_ip == user_ip)
        )
        return res.scalar() or 0
    
    async def count_pii_detections_by_user_ip(self, db: AsyncSession, user_ip: str) -> int:
        """사용자 IP 기준 개인정보 탐지 횟수"""
        res = await db.execute(
            select(func.count(Detection.id))
            .where(and_(Detection.user_ip == user_ip, Detection.has_pii == True))
        )
        return res.scalar() or 0
    
    # === 새로운 문서 관리 메서드들 ===
    async def create_sensitive_document(self, document: SensitiveDocument) -> SensitiveDocument:
        """민감 문서 생성"""
        try:
            session = await self._get_session()
            session.add(document)
            await session.commit()
            await session.refresh(document)
            return document
        except Exception as e:
            logger.error(f"Failed to create sensitive document: {str(e)}")
            await session.rollback()
            raise
        finally:
            await session.close()
    
    async def get_sensitive_document_by_id(self, document_id: str) -> Optional[SensitiveDocument]:
        """문서 ID로 민감 문서 조회"""
        try:
            session = await self._get_session()
            res = await session.execute(
                select(SensitiveDocument).where(SensitiveDocument.id == UUID(document_id))
            )
            return res.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get sensitive document: {str(e)}")
            return None
        finally:
            await session.close()
    
    async def get_sensitive_documents(
        self, 
        skip: int = 0, 
        limit: int = 100, 
        embedding_status: str = None
    ) -> List[SensitiveDocument]:
        """민감 문서 목록 조회"""
        try:
            session = await self._get_session()
            query = select(SensitiveDocument)
            
            if embedding_status:
                query = query.where(SensitiveDocument.embedding_status == embedding_status)
            
            query = query.order_by(desc(SensitiveDocument.created_at)).offset(skip).limit(limit)
            
            res = await session.execute(query)
            return list(res.scalars().all())
        except Exception as e:
            logger.error(f"Failed to get sensitive documents: {str(e)}")
            return []
        finally:
            await session.close()
    
    async def update_document_embedding_status(
        self, 
        document_id: UUID, 
        status: str, 
        chunk_count: int = None
    ) -> bool:
        """문서 임베딩 상태 업데이트"""
        try:
            session = await self._get_session()
            res = await session.execute(
                select(SensitiveDocument).where(SensitiveDocument.id == document_id)
            )
            document = res.scalar_one_or_none()
            
            if document:
                document.embedding_status = status
                if chunk_count is not None:
                    document.chunk_count = chunk_count
                await session.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to update document embedding status: {str(e)}")
            await session.rollback()
            return False
        finally:
            await session.close()
    
    async def delete_sensitive_document(self, document_id: str) -> bool:
        """민감 문서 삭제"""
        try:
            session = await self._get_session()
            res = await session.execute(
                select(SensitiveDocument).where(SensitiveDocument.id == UUID(document_id))
            )
            document = res.scalar_one_or_none()
            
            if document:
                await session.delete(document)
                await session.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete sensitive document: {str(e)}")
            await session.rollback()
            return False
        finally:
            await session.close()
    
    # === 문서 청크 관련 메서드들 ===
    async def create_document_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """문서 청크 생성"""
        try:
            session = await self._get_session()
            session.add(chunk)
            await session.commit()
            await session.refresh(chunk)
            return chunk
        except Exception as e:
            logger.error(f"Failed to create document chunk: {str(e)}")
            await session.rollback()
            raise
        finally:
            await session.close()
    
    async def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        """문서의 모든 청크 조회"""
        try:
            session = await self._get_session()
            res = await session.execute(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == UUID(document_id))
                .order_by(DocumentChunk.chunk_index)
            )
            return list(res.scalars().all())
        except Exception as e:
            logger.error(f"Failed to get document chunks: {str(e)}")
            return []
        finally:
            await session.close()
    
    # === 분석 로그 관련 메서드들 ===
    async def create_analysis_log(self, log: AnalysisLog) -> AnalysisLog:
        """분석 로그 생성"""
        try:
            session = await self._get_session()
            session.add(log)
            await session.commit()
            await session.refresh(log)
            return log
        except Exception as e:
            logger.error(f"Failed to create analysis log: {str(e)}")
            await session.rollback()
            raise
        finally:
            await session.close()
    
    async def get_recent_analysis_logs(
        self, 
        limit: int = 50, 
        blocked_only: bool = False
    ) -> List[AnalysisLog]:
        """최근 분석 로그 조회"""
        try:
            session = await self._get_session()
            query = select(AnalysisLog)
            
            if blocked_only:
                query = query.where(AnalysisLog.blocked == True)
            
            query = query.order_by(desc(AnalysisLog.created_at)).limit(limit)
            
            res = await session.execute(query)
            return list(res.scalars().all())
        except Exception as e:
            logger.error(f"Failed to get recent analysis logs: {str(e)}")
            return []
        finally:
            await session.close()
    
    async def get_analysis_statistics(self) -> Dict[str, Any]:
        """분석 통계 조회"""
        try:
            session = await self._get_session()
            
            # 총 분석 수
            total_res = await session.execute(select(func.count(AnalysisLog.id)))
            total_analyses = total_res.scalar() or 0
            
            # 차단된 분석 수
            blocked_res = await session.execute(
                select(func.count(AnalysisLog.id)).where(AnalysisLog.blocked == True)
            )
            blocked_analyses = blocked_res.scalar() or 0
            
            # PII 탐지 수
            pii_res = await session.execute(
                select(func.count(AnalysisLog.id)).where(AnalysisLog.pii_detected == True)
            )
            pii_detections = pii_res.scalar() or 0
            
            # 유사도 탐지 수
            similarity_res = await session.execute(
                select(func.count(AnalysisLog.id)).where(AnalysisLog.similarity_detected == True)
            )
            similarity_detections = similarity_res.scalar() or 0
            
            # 평균 분석 시간
            avg_time_res = await session.execute(
                select(func.avg(AnalysisLog.analysis_time_ms))
                .where(AnalysisLog.analysis_time_ms.isnot(None))
            )
            avg_analysis_time = avg_time_res.scalar()
            
            return {
                "total_analyses": total_analyses,
                "blocked_analyses": blocked_analyses,
                "pii_detections": pii_detections,
                "similarity_detections": similarity_detections,
                "avg_analysis_time_ms": float(avg_analysis_time) if avg_analysis_time else None
            }
        except Exception as e:
            logger.error(f"Failed to get analysis statistics: {str(e)}")
            return {
                "total_analyses": 0,
                "blocked_analyses": 0,
                "pii_detections": 0,
                "similarity_detections": 0,
                "avg_analysis_time_ms": None
            }
        finally:
            await session.close()