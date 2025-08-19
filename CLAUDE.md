# AI-TLS-DLP Backend - 상세 코드 검토 및 개발 가이드

## 프로젝트 개요
한국어 개인정보(PII) 탐지를 위한 FastAPI 기반 백엔드 서비스입니다.
허깅페이스의 `psh3333/roberta-large-korean-pii3` 모델을 사용하여 토큰 기반 PII 탐지를 수행합니다.

## 📁 프로젝트 구조 분석

### 전체 아키텍처
클린 아키텍처를 적용하여 계층별로 구성되었습니다:

```
AI-TLS-DLP-BE/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI 애플리케이션 진입점
│   ├── api/                       # 🌐 API 레이어 (Presentation)
│   │   ├── __init__.py
│   │   └── routers/
│   │       ├── __init__.py
│   │       └── pii.py            # PII 탐지 REST API 엔드포인트
│   ├── services/                  # 💼 비즈니스 로직 레이어 (Use Cases)
│   │   ├── __init__.py
│   │   └── pii_service.py        # PII 탐지 비즈니스 로직
│   ├── ai/                        # 🤖 AI 모델 레이어 (Infrastructure)
│   │   ├── __init__.py
│   │   └── pii_detector.py       # RoBERTa 한국어 PII 탐지 모델
│   ├── schemas/                   # 📋 데이터 스키마 (Interface Adapters)
│   │   ├── __init__.py
│   │   └── pii.py                # Request/Response 모델
│   ├── core/                      # ⚙️ 핵심 설정 (Infrastructure)
│   │   ├── __init__.py
│   │   └── config.py             # 애플리케이션 설정
│   ├── db/                        # 🗄️ 데이터베이스 레이어 (Infrastructure)
│   │   ├── __init__.py
│   │   ├── base.py               # SQLAlchemy 기본 모델
│   │   └── session.py            # 데이터베이스 세션 관리
│   ├── models/                    # 📊 데이터 모델 (Entities)
│   │   ├── __init__.py
│   │   └── detection.py         # PII 탐지 결과 엔티티
│   ├── repository/                # 📦 데이터 접근 레이어 (Interface Adapters)
│   │   ├── __init__.py
│   │   └── detection_repo.py    # PII 탐지 데이터 저장소
│   └── utils/                     # 🔧 유틸리티 (Infrastructure)
│       ├── __init__.py
│       └── tag_parser.py         # BIO 태그 파싱 유틸리티
├── pyproject.toml                 # Python 패키지 설정
├── uv.lock                        # 의존성 잠금 파일
├── test_main.http                 # API 테스트 파일
└── CLAUDE.md                      # 프로젝트 문서
```

### 📊 코드 품질 메트릭
- **총 Python 파일**: 21개 (앱 코드만)
- **코드 라인**: 약 800줄 (주요 로직)
- **의존성**: 7개 (핵심 라이브러리)
- **Python 버전**: 3.13 (최신 문법 적용)
- **아키텍처 패턴**: Clean Architecture + Layered Architecture

## 🔍 상세 코드 검토 결과

### ✅ 잘 구현된 부분

#### 1. **아키텍처 설계** (⭐⭐⭐⭐⭐)
- **클린 아키텍처 적용**: 계층별 책임이 명확히 분리됨
- **의존성 방향**: 외부에서 내부로 의존성이 올바르게 흐름
- **단일 책임 원칙**: 각 모듈이 하나의 책임만 가짐
- **확장성**: 새로운 AI 모델 추가 시 쉽게 확장 가능한 구조

#### 2. **Python 3.13 최신 문법 활용** (⭐⭐⭐⭐⭐)
- **타입 힌트**: `list[T]`, `dict[K,V]`, `str | None` 등 최신 문법 사용
- **Pydantic V2**: `model_config`, `json_schema_extra` 등 최신 API 활용
- **자동화된 의존성 관리**: uv를 사용한 현대적 패키지 관리

#### 3. **REST API 설계** (⭐⭐⭐⭐)
- **표준 HTTP 메서드**: POST for detection, GET for health check
- **명확한 URL 구조**: `/api/v1/pii/detect` (버전 관리 포함)
- **일관된 응답 형식**: `has_pii`, `reason`, `details`, `entities` 구조
- **적절한 상태 코드**: 200, 400, 500 등 HTTP 표준 준수
- **자동 문서화**: FastAPI의 OpenAPI 스키마 활용

#### 4. **데이터 모델 설계** (⭐⭐⭐⭐)
- **SQLAlchemy 2.0**: 최신 ORM 기능 활용
- **타입 안전성**: `Mapped[T]` 타입 힌트로 컴파일 타임 검증
- **PostgreSQL 특화**: JSONB, INET 등 고급 데이터 타입 활용
- **비동기 지원**: AsyncSession으로 성능 최적화

#### 5. **AI 모델 통합** (⭐⭐⭐⭐)
- **허깅페이스 모델 활용**: 검증된 한국어 PII 탐지 모델 사용
- **BIO 태깅 처리**: Named Entity Recognition 표준 방식 구현
- **신뢰도 처리**: 토큰별 confidence score 적절히 집계
- **에러 처리**: 모델 로딩 실패 시 명확한 예외 처리

### ⚠️ 개선이 필요한 부분

#### 1. **성능 이슈** (🔴 중요도: 높음)
```python
# ❌ 문제: 매 요청마다 모델 초기화
class PIIDetectionService:
    def __init__(self):
        self.detector = RobertaKoreanPIIDetector()  # 매번 모델 로딩

# ✅ 개선안: 싱글톤 패턴으로 모델 공유
_detector_instance = None
def get_detector():
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = RobertaKoreanPIIDetector()
    return _detector_instance
```

#### 2. **데이터베이스 연동 미완성** (🟡 중요도: 중간)
```python
# ❌ 문제: 서비스 레이어에서 DB 사용하지 않음
class PIIDetectionService:
    async def analyze_text(self, text: str):
        # DB 저장 로직 없음
        return result

# ✅ 개선안: Repository 패턴 활용
class PIIDetectionService:
    def __init__(self, repo: DetectionRepository):
        self.repo = repo
    
    async def analyze_text(self, text: str, user_ip: str):
        result = await self.detector.detect_pii(text)
        # DB에 결과 저장
        await self.repo.create(...)
        return result
```

#### 3. **설정 관리 개선 필요** (🟡 중요도: 중간)
```python
# ❌ 현재: 하드코딩된 설정
class RobertaKoreanPIIDetector:
    def __init__(self):
        self.model_name = "psh3333/roberta-large-korean-pii3"  # 하드코딩

# ✅ 개선안: 환경변수 기반 설정
class RobertaKoreanPIIDetector:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.PII_MODEL_NAME
```

#### 4. **예외 처리 강화 필요** (🟡 중요도: 중간)
- 모델 로딩 실패 시 대체 방안 필요
- 네트워크 오류 시 재시도 로직 부족
- 입력 검증 강화 (XSS, 악성 스크립트 방지)

### 🔧 코드베이스별 상세 분석

#### `app/ai/pii_detector.py` (⭐⭐⭐⭐)
**장점:**
- BIO 태깅 로직이 정확히 구현됨
- 토큰 후처리가 적절히 처리됨
- 비동기 메서드로 성능 고려

**개선점:**
- 모델 로딩이 매번 발생 (싱글톤 패턴 적용 필요)
- 하드코딩된 모델명 (설정으로 분리 필요)

#### `app/services/pii_service.py` (⭐⭐⭐)
**장점:**
- 비즈니스 로직이 명확히 분리됨
- reason/details 생성 로직이 체계적

**개선점:**
- 데이터베이스 연동 미구현
- 의존성 주입 패턴 미적용

#### `app/api/routers/pii.py` (⭐⭐⭐⭐)
**장점:**
- RESTful API 설계 원칙 준수
- 적절한 에러 처리 및 로깅
- FastAPI 기능을 잘 활용

**개선점:**
- IP 주소 추출 로직 부족
- Rate limiting 미구현

#### `app/models/detection.py` (⭐⭐⭐⭐⭐)
**장점:**
- SQLAlchemy 2.0 최신 기능 활용
- 적절한 인덱싱 고려
- PostgreSQL 고급 타입 활용

#### `app/repository/detection_repo.py` (⭐⭐⭐)
**장점:**
- Repository 패턴 올바른 구현
- 다양한 쿼리 메서드 제공

**개선점:**
- Count 쿼리 비효율 (`len(list(...))` 대신 `func.count()` 사용)
- 페이징 처리 미구현

#### `app/utils/tag_parser.py` (⭐⭐⭐)
**장점:**
- 복잡한 BIO 태그 파싱 로직 구현
- 다양한 태그 형식 지원

**개선점:**
- 정규식 성능 최적화 필요
- 테스트 케이스 부족으로 검증 어려움

### 🏗️ 아키텍처 평가

#### 클린 아키텍처 준수도: 85/100
- ✅ **계층 분리**: 각 레이어가 명확히 구분됨
- ✅ **의존성 방향**: 외부→내부 의존성 올바름
- ✅ **단일 책임**: 각 클래스가 하나의 책임만 가짐
- ⚠️ **의존성 주입**: 일부 하드 의존성 존재
- ⚠️ **인터페이스**: 추상화 레이어 일부 부족

#### 확장성 평가: 90/100
- ✅ **모듈성**: 새로운 기능 추가가 쉬운 구조
- ✅ **플러그인 아키텍처**: AI 모델 교체 용이
- ✅ **API 버저닝**: URL에 버전 정보 포함
- ✅ **데이터베이스 스키마**: 확장 가능한 JSONB 활용

#### 유지보수성 평가: 80/100
- ✅ **코드 가독성**: 명확한 변수명과 주석
- ✅ **타입 안전성**: 타입 힌트로 버그 방지
- ⚠️ **테스트 코드**: 단위 테스트 부족
- ⚠️ **문서화**: API 문서는 있으나 코드 문서 부족

## API 엔드포인트

### PII 탐지
- **POST** `/api/v1/pii/detect`
- **Request**: `{"text": "분석할 텍스트"}`
- **Response**:
  ```json
  {
    "has_pii": true,
    "reason": "개인정보 2개 탐지됨 (PERSON, PHONE)",
    "details": "다음 개인정보가 탐지되었습니다: PERSON '홍길동' (신뢰도: 95.0%), PHONE '010-1234-5678' (신뢰도: 89.0%)",
    "entities": [
      {
        "type": "PERSON",
        "value": "홍길동",
        "confidence": 0.95,
        "token_count": 2
      }
    ]
  }
  ```

### 헬스체크
- **GET** `/api/v1/pii/health` - 모델 로딩 상태 확인

## 후처리 작업 TODO

### 1. 성능 최적화
- [ ] **모델 로딩 최적화**: 앱 시작시 한번만 로딩하도록 싱글톤 패턴 적용
- [ ] **배치 처리**: 여러 텍스트를 한번에 처리할 수 있는 배치 API 추가
- [ ] **비동기 처리**: CPU 집약적인 모델 추론을 별도 스레드풀에서 처리
- [ ] **캐싱**: 동일한 텍스트에 대한 결과 캐싱 (Redis 활용)

### 2. 후처리 개선
- [ ] **토큰 재조합**: 서브워드 토큰을 원문 단어로 정확히 매핑
- [ ] **위치 정보**: 탐지된 개인정보의 원문 내 정확한 시작/끝 위치 제공
- [ ] **신뢰도 조정**: 토큰별 신뢰도를 엔티티별로 적절히 집계
- [ ] **중복 제거**: 겹치거나 중복된 엔티티 탐지 결과 정리

### 3. 확장성 준비
- [ ] **멀티 모델 지원**: OCR, 맥락 기반 탐지 모델 추가를 위한 인터페이스 설계
- [ ] **모델 선택 API**: 요청시 사용할 모델을 선택할 수 있는 기능
- [ ] **설정 관리**: 환경별 모델 설정 및 임계값 조정 기능

### 4. 품질 개선
- [ ] **단위 테스트**: 각 레이어별 테스트 코드 작성
- [ ] **통합 테스트**: API 엔드투엔드 테스트 
- [ ] **에러 처리**: 더 세밀한 예외 처리 및 사용자 친화적 에러 메시지
- [ ] **검증 로직**: 입력값 검증 강화 (HTML 태그 제거, 길이 제한 등)

### 5. 운영 지원
- [ ] **로깅**: 구조화된 로깅 (JSON 형태)
- [ ] **모니터링**: 메트릭 수집 (처리 시간, 성공/실패율 등)
- [ ] **설정 파일**: 환경변수 기반 설정 관리
- [ ] **도커화**: 컨테이너 배포를 위한 Dockerfile 작성

## 🚨 발견된 잠재적 이슈

### 1. **보안 취약점**
- **입력 검증 부족**: 악성 스크립트나 SQL 인젝션 방어 미흡
- **Rate Limiting 부재**: API 남용 방지 메커니즘 없음
- **CORS 설정**: 모든 도메인 허용 (프로덕션에서 위험)

### 2. **데이터 일관성 문제**
- **트랜잭션 미처리**: 데이터베이스 작업 시 원자성 보장 안됨
- **동시성 제어**: 여러 요청 간 데이터 충돌 가능성

### 3. **메모리 누수 위험**
- **모델 메모리**: GPU/CPU 메모리 해제 로직 부족
- **대용량 텍스트**: 메모리 제한 없이 처리 시 OOM 가능성

## 📈 성능 벤치마크 예상

### 현재 구조 기준 예상 성능
- **요청 처리 시간**: 2-5초 (모델 로딩 포함)
- **동시 처리**: 1-2 요청 (메모리 제약)
- **처리 가능 텍스트**: 최대 512 토큰 (약 1000자)

### 최적화 후 목표 성능
- **요청 처리 시간**: 100-300ms (모델 사전 로딩)
- **동시 처리**: 10-20 요청 (비동기 처리)
- **처리 가능 텍스트**: 최대 2048 토큰 (약 4000자)

## 📋 우선순위별 개발 로드맵

### Phase 1: 기본 기능 완성 (1-2주)
1. **모델 싱글톤 패턴 적용** - 성능 개선
2. **데이터베이스 연동 완성** - 기본 기능 구현
3. **입력 검증 강화** - 보안 강화
4. **기본 테스트 작성** - 품질 보증

### Phase 2: 성능 최적화 (2-3주)
1. **비동기 처리 개선** - 동시 요청 처리
2. **캐싱 시스템 구축** - 응답 속도 향상
3. **배치 처리 API** - 대량 처리 지원
4. **모니터링 시스템** - 운영 지원

### Phase 3: 확장 기능 (3-4주)
1. **OCR 모델 통합** - 이미지 PII 탐지
2. **맥락 기반 탐지** - 고도화된 PII 탐지
3. **다국어 지원** - 영어/중국어 등 확장
4. **관리자 대시보드** - 사용 통계 및 관리

## 🛠️ 추천 기술 스택 확장

### 현재 스택
- **백엔드**: FastAPI + Python 3.13
- **AI**: Transformers + PyTorch
- **데이터베이스**: PostgreSQL + SQLAlchemy 2.0
- **패키지 관리**: uv

### 추천 추가 스택
- **캐싱**: Redis (결과 캐싱용)
- **큐**: Celery + Redis (비동기 작업용)
- **모니터링**: Prometheus + Grafana
- **로깅**: Elasticsearch + Logstash + Kibana (ELK)
- **컨테이너**: Docker + Docker Compose
- **CI/CD**: GitHub Actions
- **테스팅**: pytest + pytest-asyncio

## 📊 최종 평가

### 종합 점수: 82/100

#### 강점 (85점)
- ✅ **현대적 기술 스택**: Python 3.13, FastAPI, SQLAlchemy 2.0
- ✅ **클린 아키텍처**: 확장 가능한 구조 설계
- ✅ **AI 모델 통합**: 실용적인 PII 탐지 기능
- ✅ **타입 안전성**: 컴파일 타임 오류 방지

#### 개선점 (78점)
- ⚠️ **성능 최적화**: 모델 로딩 및 동시 처리 개선 필요
- ⚠️ **테스트 커버리지**: 단위/통합 테스트 부족
- ⚠️ **운영 준비도**: 모니터링, 로깅 시스템 미흡
- ⚠️ **보안 강화**: 입력 검증, Rate limiting 필요

### 결론
**매우 잘 설계된 프로젝트**입니다. 클린 아키텍처를 올바르게 적용했고, 최신 Python 기술을 효과적으로 활용했습니다. 

몇 가지 성능과 보안 이슈를 해결하면 **프로덕션 환경에서도 안정적으로 동작할 수 있는 수준**으로 판단됩니다.

특히 OCR과 맥락 기반 탐지 모델 추가를 위한 **확장성이 뛰어난 구조**로 설계되어 있어, 향후 기능 확장이 용이할 것으로 예상됩니다.

## 개발 명령어

### 서버 실행
```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 의존성 설치
```bash
uv sync
```

### API 문서 확인
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc