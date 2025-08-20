"""
문서 텍스트 청킹 유틸리티
"""

import re
import logging
from typing import List, Tuple, Optional
import kss  # Korean Sentence Splitter

from app.core.config import settings

logger = logging.getLogger(__name__)


class TextChunker:
    """문서 텍스트를 청크로 분할하는 클래스"""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        min_chunk_size: int = 50
    ):
        """
        Args:
            chunk_size: 청크 최대 크기 (토큰 수)
            chunk_overlap: 청크 간 겹치는 부분 크기
            min_chunk_size: 최소 청크 크기
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.min_chunk_size = min_chunk_size
        
        logger.info(f"TextChunker initialized: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def chunk_text(self, text: str) -> List[Tuple[str, int]]:
        """
        텍스트를 청크로 분할합니다.
        
        Args:
            text: 분할할 텍스트
            
        Returns:
            (청크 텍스트, 시작 위치) 튜플들의 리스트
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            logger.warning("Text too short for chunking")
            return []
        
        try:
            # 텍스트 전처리
            text = self._preprocess_text(text)
            
            # 문장 단위로 분할
            sentences = self._split_into_sentences(text)
            
            if not sentences:
                return []
            
            # 문장들을 청크로 그룹화
            chunks = self._group_sentences_into_chunks(sentences)
            
            logger.info(f"Created {len(chunks)} chunks from {len(sentences)} sentences")
            return chunks
            
        except Exception as e:
            logger.error(f"Text chunking failed: {str(e)}")
            raise RuntimeError(f"Text chunking failed: {str(e)}")
    
    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        # 연속된 공백 정리
        text = re.sub(r'\s+', ' ', text)
        
        # 연속된 줄바꿈 정리
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """텍스트를 문장 단위로 분할"""
        try:
            # KSS를 이용한 한국어 문장 분할
            sentences = kss.split_sentences(text)
            
            # 빈 문장이나 너무 짧은 문장 필터링
            sentences = [
                sent.strip() 
                for sent in sentences 
                if sent.strip() and len(sent.strip()) >= 10
            ]
            
            logger.debug(f"Split text into {len(sentences)} sentences")
            return sentences
            
        except Exception as e:
            logger.warning(f"KSS sentence splitting failed, using simple split: {str(e)}")
            # KSS 실패 시 간단한 문장 분할
            return self._simple_sentence_split(text)
    
    def _simple_sentence_split(self, text: str) -> List[str]:
        """간단한 문장 분할 (KSS 실패 시 백업)"""
        # 기본적인 문장 끝 문자로 분할
        sentences = re.split(r'[.!?]\s+', text)
        
        # 빈 문장 필터링
        sentences = [sent.strip() for sent in sentences if sent.strip()]
        
        return sentences
    
    def _group_sentences_into_chunks(self, sentences: List[str]) -> List[Tuple[str, int]]:
        """문장들을 청크로 그룹화"""
        chunks = []
        current_chunk = []
        current_length = 0
        start_position = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # 청크 크기 초과 시 새 청크 시작
            if current_chunk and (current_length + sentence_length) > self.chunk_size:
                # 현재 청크 완성
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append((chunk_text, start_position))
                
                # 겹치는 부분 계산하여 다음 청크 시작
                overlap_sentences = self._calculate_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
                
                # 새 청크의 시작 위치 계산
                start_position = i - len(overlap_sentences)
                
            else:
                # 현재 청크에 문장 추가
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # 마지막 청크 처리
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append((chunk_text, start_position))
        
        return chunks
    
    def _calculate_overlap_sentences(self, current_chunk: List[str]) -> List[str]:
        """청크 간 겹치는 문장들 계산"""
        if not current_chunk or self.chunk_overlap <= 0:
            return []
        
        # 겹치는 부분의 문자 수만큼 문장들 선택
        overlap_sentences = []
        overlap_length = 0
        
        # 뒤에서부터 문장들을 선택
        for sentence in reversed(current_chunk):
            if overlap_length + len(sentence) <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_length += len(sentence)
            else:
                break
        
        return overlap_sentences
    
    def chunk_by_paragraphs(self, text: str) -> List[Tuple[str, int]]:
        """
        단락 단위로 청킹 (간단한 방식)
        
        Args:
            text: 분할할 텍스트
            
        Returns:
            (청크 텍스트, 시작 위치) 튜플들의 리스트
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            return []
        
        try:
            # 단락으로 분할 (연속된 줄바꿈 기준)
            paragraphs = re.split(r'\n\s*\n', text.strip())
            
            chunks = []
            for i, paragraph in enumerate(paragraphs):
                paragraph = paragraph.strip()
                
                if len(paragraph) >= self.min_chunk_size:
                    # 단락이 너무 크면 문장 단위로 재분할
                    if len(paragraph) > self.chunk_size:
                        sub_chunks = self.chunk_text(paragraph)
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append((paragraph, i))
            
            logger.info(f"Created {len(chunks)} chunks from {len(paragraphs)} paragraphs")
            return chunks
            
        except Exception as e:
            logger.error(f"Paragraph chunking failed: {str(e)}")
            raise RuntimeError(f"Paragraph chunking failed: {str(e)}")
    
    def get_chunk_stats(self, chunks: List[Tuple[str, int]]) -> dict:
        """청킹 통계 정보 반환"""
        if not chunks:
            return {"total_chunks": 0, "total_length": 0, "avg_chunk_size": 0}
        
        total_length = sum(len(chunk[0]) for chunk in chunks)
        avg_chunk_size = total_length / len(chunks)
        
        chunk_sizes = [len(chunk[0]) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_length": total_length,
            "avg_chunk_size": avg_chunk_size,
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "chunk_size_limit": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }