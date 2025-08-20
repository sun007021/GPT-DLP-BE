"""
PII 탐지와 유사도 분석을 통합한 분석 서비스
"""

import logging
import hashlib
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.ai.pii_detector import RobertaKoreanPIIDetector
from app.ai.similarity_detector import KoSimCSESimilarityDetector
from app.repository.vector_repo import get_vector_repository
from app.repository.detection_repo import DetectionRepository
from app.models.documents import AnalysisLog
from app.core.config import settings

logger = logging.getLogger(__name__)


class IntegratedAnalysisService:
    """PII 탐지 + 유사도 분석 통합 서비스"""
    
    def __init__(self):
        """서비스 초기화"""
        self.pii_detector: Optional[RobertaKoreanPIIDetector] = None
        self.similarity_detector: Optional[KoSimCSESimilarityDetector] = None
        self.vector_repo = get_vector_repository()
        self.detection_repo = DetectionRepository()
        
        logger.info("IntegratedAnalysisService initialized")
    
    def _ensure_models_loaded(self) -> None:
        """AI 모델들이 로드되어 있는지 확인하고, 없으면 로드"""
        if self.pii_detector is None:
            logger.info("Loading PII detector...")
            self.pii_detector = RobertaKoreanPIIDetector()
            
        if self.similarity_detector is None:
            logger.info("Loading similarity detector...")
            self.similarity_detector = KoSimCSESimilarityDetector()
    
    async def analyze_comprehensive(
        self,
        text: str,
        user_ip: str = None
    ) -> Dict[str, Any]:
        """
        통합 검사: 텍스트만 입력받아 PII와 유사도를 무조건 모두 분석
        
        Args:
            text: 분석할 텍스트
            user_ip: 사용자 IP 주소
            
        Returns:
            통합 분석 결과 (reason과 details 포함)
        """
        start_time = time.time()
        
        # 기본 임계값 사용
        pii_threshold = settings.DEFAULT_PII_THRESHOLD
        similarity_threshold = settings.DEFAULT_SIMILARITY_THRESHOLD
        
        # 입력 텍스트 해시 생성 (중복 분석 방지용)
        input_hash = hashlib.sha256(text.encode()).hexdigest()
        
        logger.info(f"Starting comprehensive analysis: text_length={len(text)}")
        
        try:
            # 모델 로드 확인
            self._ensure_models_loaded()
            
            # 무조건 두 모델 모두 실행
            pii_result = await self._analyze_pii(text, pii_threshold)
            similarity_result = await self._analyze_similarity(text, similarity_threshold)
            
            # 차단 여부 결정
            blocked = pii_result["has_pii"] or similarity_result["is_similar"]
            block_reasons = []
            if pii_result["has_pii"]:
                block_reasons.append("pii_detected")
            if similarity_result["is_similar"]:
                block_reasons.append("similarity_detected")
            
            # reason과 details 통합 생성
            pii_reason, pii_details = self._generate_pii_text(pii_result)
            sim_reason, sim_details = self._generate_similarity_text(similarity_result)
            
            # 최종 reason 결합
            if blocked:
                if pii_result["has_pii"] and similarity_result["is_similar"]:
                    reason = "개인정보와 유사 문서가 동시에 탐지되어 차단되었습니다."
                elif pii_result["has_pii"]:
                    reason = "개인정보가 탐지되어 차단되었습니다."
                else:
                    reason = "유사 문서가 탐지되어 차단되었습니다."
            else:
                parts = []
                if pii_reason:
                    parts.append(pii_reason)
                if sim_reason:
                    parts.append(sim_reason)
                reason = "검사를 통과했습니다. " + ", ".join(parts) + "."
            
            # 최종 details 결합
            detail_parts = []
            if pii_details:
                detail_parts.append(pii_details)
            if sim_details:
                detail_parts.append(sim_details)
            details = ". ".join(detail_parts) + "." if detail_parts else "추가 상세 정보 없음."
            
            # PII analysis에 통합된 reason/details 추가
            pii_result["reason"] = reason
            pii_result["details"] = details
            
            # 분석 시간 계산
            analysis_time_ms = int((time.time() - start_time) * 1000)
            
            # 분석 결과 구성
            analysis_result = {
                "input_text": text,
                "input_hash": input_hash,
                "blocked": blocked,
                "block_reasons": block_reasons,
                "analysis_time_ms": analysis_time_ms,
                "pii_analysis": pii_result,
                "similarity_analysis": similarity_result
            }
            
            # 분석 로그 저장
            await self._save_analysis_log(analysis_result, user_ip)
            
            logger.info(f"Analysis completed: blocked={blocked}, time={analysis_time_ms}ms")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {str(e)}")
            raise RuntimeError(f"Analysis failed: {str(e)}")
    
    async def _analyze_pii(self, text: str, threshold: float) -> Dict[str, Any]:
        """PII 분석 수행"""
        try:
            logger.debug("Performing PII analysis...")
            
            # PII 탐지 실행
            pii_result = await self.pii_detector.detect_pii(text)
            
            # 임계값 필터링
            filtered_entities = []
            for entity in pii_result.get("entities", []):
                if entity.get("confidence", 0.0) >= threshold:
                    filtered_entities.append(entity)
            
            # 결과 포맷팅
            result = {
                "has_pii": len(filtered_entities) > 0,
                "threshold": threshold,
                "entities": filtered_entities,
                "total_entities": len(filtered_entities),
                "reason": pii_result.get("reason", ""),
                "details": pii_result.get("details", "")
            }
            
            logger.debug(f"PII analysis completed: {len(filtered_entities)} entities found")
            return result
            
        except Exception as e:
            logger.error(f"PII analysis failed: {str(e)}")
            return {
                "has_pii": False,
                "threshold": threshold,
                "entities": [],
                "total_entities": 0,
                "error": str(e)
            }
    
    async def _analyze_similarity(self, text: str, threshold: float) -> Dict[str, Any]:
        """유사도 분석 수행"""
        try:
            logger.debug("Performing similarity analysis...")
            
            # 쿼리 텍스트 임베딩 생성
            query_embedding = self.similarity_detector.encode_text(text)
            
            # 벡터 저장소에서 유사한 청크 검색 (임계값 이상과 전체 결과 모두 받기)
            threshold_chunks, all_chunks = await self.vector_repo.search_similar_chunks(
                query_embedding=query_embedding.tolist(),
                top_k=5,
                similarity_threshold=threshold
            )
            
            # 결과 포맷팅 - 차단 여부는 임계값 이상 청크 기준
            is_similar = len(threshold_chunks) > 0
            
            # 최대 유사도는 전체 결과에서 가져오기 (임계값과 상관없이)
            max_similarity = all_chunks[0]["similarity"] if all_chunks else 0.0
            
            # 문서별로 매칭 결과 그룹화 - 전체 결과 사용 (임계값 이하도 포함)
            matched_documents = self._group_chunks_by_document(all_chunks)
            
            result = {
                "is_similar": is_similar,
                "max_similarity": float(max_similarity),
                "threshold": float(threshold),
                "matched_count": len(threshold_chunks),  # 임계값 이상 청크 수
                "total_found_count": len(all_chunks),    # 전체 검색된 청크 수
                "matched_documents": matched_documents,
                "matched_chunks": threshold_chunks       # 차단 여부 결정에 사용된 청크들
            }
            
            logger.debug(f"Similarity analysis completed: {len(all_chunks)} total chunks found, {len(threshold_chunks)} above threshold")
            return result
            
        except Exception as e:
            logger.error(f"Similarity analysis failed: {str(e)}")
            return {
                "is_similar": False,
                "max_similarity": 0.0,
                "threshold": float(threshold),
                "matched_count": 0,
                "matched_documents": [],
                "error": str(e)
            }
    
    def _group_chunks_by_document(self, similar_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """유사한 청크들을 문서별로 그룹화"""
        document_groups = {}
        
        for chunk in similar_chunks:
            doc_id = chunk["document_id"]
            doc_title = chunk["document_title"]
            
            if doc_id not in document_groups:
                document_groups[doc_id] = {
                    "document_id": doc_id,
                    "document_title": doc_title,
                    "max_similarity": chunk["similarity"],
                    "matched_chunks_count": 0,
                    "matched_texts": []
                }
            
            # 문서의 최대 유사도 업데이트
            if chunk["similarity"] > document_groups[doc_id]["max_similarity"]:
                document_groups[doc_id]["max_similarity"] = chunk["similarity"]
            
            # 매칭된 텍스트 추가
            document_groups[doc_id]["matched_chunks_count"] += 1
            document_groups[doc_id]["matched_texts"].append({
                "content": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                "similarity": chunk["similarity"],
                "similarity_percentage": f"{chunk['similarity'] * 100:.1f}%",
                "chunk_index": chunk["chunk_index"]
            })
        
        # 유사도 순으로 정렬
        return sorted(
            document_groups.values(),
            key=lambda x: x["max_similarity"],
            reverse=True
        )
    
    async def _save_analysis_log(self, analysis_result: Dict[str, Any], user_ip: str = None) -> None:
        """분석 결과를 데이터베이스에 저장"""
        try:
            pii_analysis = analysis_result.get("pii_analysis", {})
            similarity_analysis = analysis_result.get("similarity_analysis", {})
            
            # AnalysisLog 객체 생성
            log_entry = AnalysisLog(
                input_text=analysis_result["input_text"],
                input_hash=analysis_result["input_hash"],
                blocked=analysis_result["blocked"],
                block_reasons=analysis_result["block_reasons"],
                pii_detected=pii_analysis.get("has_pii", False),
                pii_entities=pii_analysis.get("entities", []),
                similarity_detected=similarity_analysis.get("is_similar", False),
                max_similarity=similarity_analysis.get("max_similarity"),
                matched_documents=similarity_analysis.get("matched_documents", []),
                user_ip=user_ip,
                analysis_time_ms=analysis_result["analysis_time_ms"]
            )
            
            # 데이터베이스에 저장
            await self.detection_repo.create_analysis_log(log_entry)
            logger.debug(f"Analysis log saved: {analysis_result['input_hash'][:8]}")
            
        except Exception as e:
            logger.error(f"Failed to save analysis log: {str(e)}")
            # 로그 저장 실패는 치명적이지 않으므로 예외를 다시 발생시키지 않음
    
    async def get_analysis_history(
        self,
        limit: int = 50,
        blocked_only: bool = False
    ) -> List[Dict[str, Any]]:
        """분석 이력 조회"""
        try:
            logs = await self.detection_repo.get_recent_analysis_logs(
                limit=limit,
                blocked_only=blocked_only
            )
            
            # 결과 포맷팅
            history = []
            for log in logs:
                history.append({
                    "id": str(log.id),
                    "input_text": log.input_text[:100] + "..." if len(log.input_text) > 100 else log.input_text,
                    "blocked": log.blocked,
                    "block_reasons": log.block_reasons,
                    "pii_detected": log.pii_detected,
                    "similarity_detected": log.similarity_detected,
                    "max_similarity": float(log.max_similarity) if log.max_similarity else None,
                    "analysis_time_ms": log.analysis_time_ms,
                    "created_at": log.created_at.isoformat()
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get analysis history: {str(e)}")
            return []
    
    async def get_analysis_stats(self) -> Dict[str, Any]:
        """분석 통계 조회"""
        try:
            stats = await self.detection_repo.get_analysis_statistics()
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get analysis stats: {str(e)}")
            return {
                "total_analyses": 0,
                "blocked_analyses": 0,
                "pii_detections": 0,
                "similarity_detections": 0,
                "error": str(e)
            }
    
    def _generate_pii_text(self, pii_result: Dict[str, Any]) -> tuple[str, str]:
        """PII 분석 결과에서 reason과 details 텍스트 생성"""
        if pii_result["has_pii"]:
            pii_count = len(pii_result["entities"])
            reason = f"개인정보 {pii_count}개 탐지됨"
            
            # PII 상세 정보
            pii_details = []
            for entity in pii_result["entities"]:
                pii_details.append(f"{entity['type']} '{entity['value']}' (신뢰도: {entity['confidence']*100:.1f}%)")
            details = f"개인정보 탐지: {', '.join(pii_details)}"
            return reason, details
        else:
            return "개인정보 없음", ""
    
    def _generate_similarity_text(self, similarity_result: Dict[str, Any]) -> tuple[str, str]:
        """유사도 분석 결과에서 reason과 details 텍스트 생성"""
        if similarity_result["is_similar"]:
            reason = f"유사 문서 탐지됨 (최대 {similarity_result['max_similarity']*100:.1f}%)"
            
            # 매칭된 문서 정보
            if similarity_result["matched_documents"]:
                matched_docs = []
                for doc in similarity_result["matched_documents"][:3]:  # 상위 3개만
                    matched_docs.append(f"'{doc['document_title']}' ({doc['max_similarity']*100:.1f}%)")
                details = f"유사 문서: {', '.join(matched_docs)}"
            else:
                details = ""
            return reason, details
        else:
            if similarity_result["max_similarity"] > 0:
                reason = f"유사도 {similarity_result['max_similarity']*100:.1f}% (임계값 미만)"
                
                # 임계값 이하지만 최고 유사도 문서 정보 제공
                if similarity_result["matched_documents"]:
                    top_doc = similarity_result["matched_documents"][0]
                    details = f"최고 유사 문서: '{top_doc['document_title']}' ({top_doc['max_similarity']*100:.1f}%)"
                else:
                    details = ""
                return reason, details
            else:
                return "유사 문서 없음", ""