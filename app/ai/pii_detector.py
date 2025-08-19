# app/ai/pii_detector.py
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

class RobertaKoreanPIIDetector:
    """
    한국어 PII 탐지를 위한 RoBERTa 모델
    psh3333/roberta-large-korean-pii3 모델 사용
    """
    
    def __init__(self):
        self.model_name = "psh3333/roberta-large-korean-pii3"
        self.tokenizer: AutoTokenizer | None = None
        self.model: AutoModelForTokenClassification | None = None
        self._load_model()
    
    def _load_model(self):
        """모델과 토크나이저 로드"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load PII detection model: {str(e)}")
    
    async def detect_pii(self, text: str) -> dict[str, any]:
        """
        텍스트에서 PII 탐지
        
        Returns:
            Dict containing:
            - has_pii: bool
            - entities: List of detected PII entities
            - raw_predictions: Raw model predictions
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("PII detection model not loaded")
        
        # 토큰화 및 예측
        predictions = await self._predict_tokens(text)
        
        # PII 존재 여부 확인
        has_pii = self._has_pii_tokens(predictions)
        
        # 엔티티 추출
        entities = self._extract_entities(text, predictions) if has_pii else []
        
        return {
            "has_pii": has_pii,
            "entities": entities,
            "raw_predictions": predictions
        }
    
    async def _predict_tokens(self, text: str) -> list[dict[str, any]]:
        """토큰별 PII 라벨 예측"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_classes = predictions.argmax(-1)
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predictions_list = predicted_classes[0].tolist()
        
        results = []
        for i, (token, prediction) in enumerate(zip(tokens, predictions_list)):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            
            label = self.model.config.id2label[prediction]
            confidence = float(predictions[0][i][prediction])
            
            results.append({
                "token": token,
                "label": label,
                "confidence": confidence,
                "position": i
            })
        
        return results
    
    def _has_pii_tokens(self, predictions: list[dict[str, any]]) -> bool:
        """O 태그를 제외한 PII 태그 존재 여부 확인"""
        return any(pred["label"] != "O" for pred in predictions)
    
    def _extract_entities(self, text: str, predictions: list[dict[str, any]]) -> list[dict[str, any]]:
        """BIO 태그를 기반으로 PII 엔티티 추출"""
        entities = []
        current_entity = None
        
        for pred in predictions:
            label = pred["label"]
            token = pred["token"]
            
            if label.startswith("B-"):
                # 새로운 엔티티 시작
                if current_entity:
                    entities.append(self._finalize_entity(current_entity))
                
                current_entity = {
                    "type": label[2:],
                    "tokens": [token],
                    "confidences": [pred["confidence"]],
                    "start_pos": pred["position"]
                }
                
            elif label.startswith("I-") and current_entity:
                # 기존 엔티티 확장
                entity_type = label[2:]
                if current_entity["type"] == entity_type:
                    current_entity["tokens"].append(token)
                    current_entity["confidences"].append(pred["confidence"])
                else:
                    # 타입이 다르면 기존 엔티티 종료하고 새로 시작
                    entities.append(self._finalize_entity(current_entity))
                    current_entity = None
            else:
                # O 태그 또는 연속되지 않는 경우
                if current_entity:
                    entities.append(self._finalize_entity(current_entity))
                    current_entity = None
        
        # 마지막 엔티티 처리
        if current_entity:
            entities.append(self._finalize_entity(current_entity))
        
        return entities
    
    def _finalize_entity(self, entity: dict[str, any]) -> dict[str, any]:
        """엔티티 정보 완성"""
        # 토큰들을 문자열로 변환
        value = self.tokenizer.convert_tokens_to_string(entity["tokens"])
        
        # 평균 신뢰도 계산
        avg_confidence = sum(entity["confidences"]) / len(entity["confidences"])
        
        return {
            "type": entity["type"],
            "value": value.strip(),
            "confidence": avg_confidence,
            "token_count": len(entity["tokens"])
        }