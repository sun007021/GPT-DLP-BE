"""
KoSimCSE 기반 한국어 문장 유사도 탐지 모델
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import settings

logger = logging.getLogger(__name__)


class KoSimCSESimilarityDetector:
    """한국어 문장 유사도 탐지를 위한 KoSimCSE 모델 래퍼"""
    
    def __init__(self, model_name: str = None):
        """
        Args:
            model_name: 사용할 KoSimCSE 모델 이름
        """
        self.model_name = model_name or settings.SIMILARITY_MODEL_NAME
        self.model: Optional[SentenceTransformer] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"KoSimCSE detector initialized with device: {self.device}")
    
    def load_model(self) -> None:
        """KoSimCSE 모델을 로드합니다."""
        try:
            logger.info(f"Loading KoSimCSE model: {self.model_name}")
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            logger.info("KoSimCSE model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load KoSimCSE model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def _ensure_model_loaded(self) -> None:
        """모델이 로드되어 있는지 확인하고, 없으면 로드합니다."""
        if self.model is None:
            self.load_model()
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        단일 텍스트를 임베딩 벡터로 변환합니다.
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            임베딩 벡터 (numpy array)
        """
        self._ensure_model_loaded()
        
        try:
            # 텍스트 전처리
            text = self._preprocess_text(text)
            
            # 임베딩 생성
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True  # 코사인 유사도 계산을 위한 정규화
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Text encoding failed for text: {text[:50]}... Error: {str(e)}")
            raise RuntimeError(f"Text encoding failed: {str(e)}")
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        여러 텍스트를 배치로 임베딩 벡터로 변환합니다.
        
        Args:
            texts: 임베딩할 텍스트들
            
        Returns:
            임베딩 벡터들 (numpy array, shape: [len(texts), embedding_dim])
        """
        self._ensure_model_loaded()
        
        if not texts:
            return np.array([])
        
        try:
            # 텍스트들 전처리
            preprocessed_texts = [self._preprocess_text(text) for text in texts]
            
            # 배치 임베딩 생성
            embeddings = self.model.encode(
                preprocessed_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=32,  # 배치 크기 설정
                show_progress_bar=len(texts) > 10  # 텍스트가 많을 때만 진행바 표시
            )
            
            logger.debug(f"Successfully encoded {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Batch text encoding failed. Error: {str(e)}")
            raise RuntimeError(f"Batch text encoding failed: {str(e)}")
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        두 텍스트 간의 코사인 유사도를 계산합니다.
        
        Args:
            text1: 첫 번째 텍스트
            text2: 두 번째 텍스트
            
        Returns:
            코사인 유사도 (0.0 ~ 1.0)
        """
        try:
            # 두 텍스트를 임베딩으로 변환
            embedding1 = self.encode_text(text1)
            embedding2 = self.encode_text(text2)
            
            # 코사인 유사도 계산
            similarity = cosine_similarity(
                embedding1.reshape(1, -1),
                embedding2.reshape(1, -1)
            )[0][0]
            
            # 음수 유사도를 0으로 클리핑 (코사인 유사도는 -1~1이지만 우리는 0~1 범위 사용)
            similarity = max(0.0, float(similarity))
            
            logger.debug(f"Calculated similarity: {similarity:.4f}")
            return similarity
            
        except Exception as e:
            logger.error(f"Similarity calculation failed. Error: {str(e)}")
            raise RuntimeError(f"Similarity calculation failed: {str(e)}")
    
    def find_similar_texts(
        self, 
        query_text: str, 
        candidate_texts: List[str],
        threshold: float = 0.85,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        주어진 텍스트와 유사한 텍스트들을 찾습니다.
        
        Args:
            query_text: 검색할 기준 텍스트
            candidate_texts: 후보 텍스트들
            threshold: 유사도 임계값
            top_k: 반환할 최대 결과 수
            
        Returns:
            (텍스트, 유사도) 튜플들의 리스트, 유사도 내림차순 정렬
        """
        if not candidate_texts:
            return []
        
        try:
            # 쿼리 텍스트 임베딩
            query_embedding = self.encode_text(query_text)
            
            # 후보 텍스트들 배치 임베딩
            candidate_embeddings = self.encode_texts(candidate_texts)
            
            # 유사도 계산
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                candidate_embeddings
            )[0]
            
            # 임계값 이상인 결과들 필터링
            results = []
            for i, similarity in enumerate(similarities):
                similarity = max(0.0, float(similarity))  # 음수 클리핑
                if similarity >= threshold:
                    results.append((candidate_texts[i], similarity))
            
            # 유사도 내림차순 정렬하여 top_k 반환
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:top_k]
            
            logger.debug(f"Found {len(results)} similar texts above threshold {threshold}")
            return results
            
        except Exception as e:
            logger.error(f"Similar text search failed. Error: {str(e)}")
            raise RuntimeError(f"Similar text search failed: {str(e)}")
    
    def _preprocess_text(self, text: str) -> str:
        """
        텍스트 전처리를 수행합니다.
        
        Args:
            text: 원본 텍스트
            
        Returns:
            전처리된 텍스트
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 기본적인 정규화
        text = text.strip()
        text = " ".join(text.split())  # 연속된 공백 제거
        
        # 너무 긴 텍스트는 자르기 (모델의 최대 길이 고려)
        max_length = 512  # KoSimCSE 모델의 일반적인 최대 길이
        if len(text) > max_length:
            text = text[:max_length]
            logger.debug(f"Text truncated to {max_length} characters")
        
        return text
    
    async def detect_similarity(
        self,
        input_text: str,
        reference_texts: List[str],
        threshold: float = None
    ) -> Dict[str, Any]:
        """
        비동기 유사도 탐지 메서드 (API용)
        
        Args:
            input_text: 분석할 입력 텍스트
            reference_texts: 비교할 참조 텍스트들
            threshold: 유사도 임계값 (None이면 기본값 사용)
            
        Returns:
            유사도 분석 결과 딕셔너리
        """
        if threshold is None:
            threshold = settings.DEFAULT_SIMILARITY_THRESHOLD
        
        try:
            # 유사한 텍스트들 찾기
            similar_texts = self.find_similar_texts(
                query_text=input_text,
                candidate_texts=reference_texts,
                threshold=threshold,
                top_k=5  # 상위 5개만 반환
            )
            
            # 결과 포맷팅
            is_similar = len(similar_texts) > 0
            max_similarity = similar_texts[0][1] if similar_texts else 0.0
            
            matched_texts = [
                {
                    "text": text,
                    "similarity": float(similarity),
                    "similarity_percentage": f"{similarity * 100:.1f}%"
                }
                for text, similarity in similar_texts
            ]
            
            result = {
                "is_similar": is_similar,
                "max_similarity": float(max_similarity),
                "threshold": float(threshold),
                "matched_count": len(similar_texts),
                "matched_texts": matched_texts
            }
            
            logger.info(f"Similarity detection completed: similar={is_similar}, max_sim={max_similarity:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Async similarity detection failed: {str(e)}")
            raise RuntimeError(f"Similarity detection failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보를 반환합니다."""
        self._ensure_model_loaded()
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_seq_length": getattr(self.model, 'max_seq_length', 512),
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "is_loaded": self.model is not None
        }