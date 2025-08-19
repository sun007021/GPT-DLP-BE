from pydantic import BaseModel, Field

class PIIDetectionRequest(BaseModel):
    text: str = Field(..., description="분석할 텍스트", min_length=1, max_length=10000)

class DetectedEntity(BaseModel):
    type: str = Field(..., description="PII 타입 (예: PERSON, PHONE, EMAIL 등)")
    value: str = Field(..., description="탐지된 개인정보 값")
    confidence: float = Field(..., description="탐지 신뢰도 (0.0 ~ 1.0)", ge=0.0, le=1.0)
    token_count: int = Field(..., description="해당 엔티티의 토큰 개수", gt=0)

class PIIDetectionResponse(BaseModel):
    has_pii: bool = Field(..., description="개인정보 탐지 여부")
    reason: str = Field(..., description="탐지 결과에 대한 이유")
    details: str = Field(..., description="구체적인 설명 및 탐지된 데이터")
    entities: list[DetectedEntity] = Field(default_factory=list, description="탐지된 개인정보 엔티티 목록")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "has_pii": True,
                    "reason": "개인정보 2개 탐지됨 (PERSON, PHONE)",
                    "details": "이름 '홍길동'과 전화번호 '010-1234-5678'이 탐지되었습니다.",
                    "entities": [
                        {
                            "type": "PERSON",
                            "value": "홍길동",
                            "confidence": 0.95,
                            "token_count": 2
                        },
                        {
                            "type": "PHONE", 
                            "value": "010-1234-5678",
                            "confidence": 0.89,
                            "token_count": 3
                        }
                    ]
                }
            ]
        }
    }