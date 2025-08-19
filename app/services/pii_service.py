from app.ai.model_manager import get_pii_detector
from app.schemas.pii import PIIDetectionResponse, DetectedEntity

class PIIDetectionService:
    """PII 탐지 비즈니스 로직을 처리하는 서비스"""
    
    def __init__(self):
        # 싱글톤 패턴으로 모델 인스턴스 재사용
        self.detector = None
    
    async def analyze_text(self, text: str) -> PIIDetectionResponse:
        """텍스트를 분석하여 PII 탐지 결과 반환"""
        
        # 싱글톤 모델 인스턴스 가져오기
        detector = get_pii_detector()
        
        # AI 모델로 PII 탐지
        detection_result = await detector.detect_pii(text)
        
        # 비즈니스 로직에 맞게 응답 구성
        has_pii = detection_result["has_pii"]
        entities = [
            DetectedEntity(
                type=entity["type"],
                value=entity["value"], 
                confidence=entity["confidence"],
                token_count=entity["token_count"]
            )
            for entity in detection_result["entities"]
        ]
        
        # reason과 details 생성
        reason = self._generate_reason(has_pii, entities)
        details = self._generate_details(has_pii, entities)
        
        return PIIDetectionResponse(
            has_pii=has_pii,
            reason=reason,
            details=details,
            entities=entities
        )
    
    def _generate_reason(self, has_pii: bool, entities: list[DetectedEntity]) -> str:
        """탐지 결과에 대한 이유 생성"""
        if not has_pii:
            return "개인정보가 탐지되지 않았습니다"
        
        if len(entities) == 1:
            entity_type = entities[0].type
            return f"개인정보 1개 탐지됨 ({entity_type})"
        
        entity_types = list(set(entity.type for entity in entities))
        type_str = ", ".join(entity_types)
        return f"개인정보 {len(entities)}개 탐지됨 ({type_str})"
    
    def _generate_details(self, has_pii: bool, entities: list[DetectedEntity]) -> str:
        """탐지된 개인정보에 대한 상세 설명 생성"""
        if not has_pii:
            return "입력된 텍스트에서 개인정보가 발견되지 않았습니다."
        
        details_parts = []
        for entity in entities:
            confidence_pct = f"{entity.confidence:.1%}"
            details_parts.append(f"{entity.type} '{entity.value}' (신뢰도: {confidence_pct})")
        
        details_str = ", ".join(details_parts)
        return f"다음 개인정보가 탐지되었습니다: {details_str}"