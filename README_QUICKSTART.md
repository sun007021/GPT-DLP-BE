# AI-TLS-DLP Backend 빠른 시작 가이드

## 🚀 단일 기업 배포용 유사도 기반 DLP 시스템

PII 탐지 + 유사도 분석을 통합한 AI 기반 민감정보 유출 차단 시스템입니다.

## 📋 주요 기능

- ✅ **한국어 PII 탐지**: RoBERTa 기반 개인정보 탐지
- ✅ **유사도 분석**: KoSimCSE 기반 문서 유사도 검증
- ✅ **문서 관리**: 기업 민감문서 업로드 및 임베딩 생성
- ✅ **통합 분석**: PII + 유사도 동시 분석 및 차단
- ✅ **실시간 처리**: 벡터 검색 기반 고속 처리

## 🏗️ 시스템 구조

```
AI-TLS-DLP-BE/
├── app/
│   ├── ai/                    # AI 모델 (PII + 유사도)
│   ├── api/routers/          # API 엔드포인트
│   ├── services/             # 비즈니스 로직
│   ├── repository/           # 데이터 접근 계층
│   └── models/              # SQLAlchemy 모델
├── storage/uploads/         # 업로드 파일 저장소
├── docker-compose.yml       # 인프라 설정
└── pyproject.toml          # 의존성 설정
```

## 🚀 빠른 시작

### 1. 인프라 시작

```bash
# PostgreSQL, ChromaDB, Redis 시작
docker-compose up -d

# 서비스 상태 확인
docker-compose ps
```

### 2. 의존성 설치

```bash
# uv를 사용한 의존성 설치
uv sync

# 또는 pip 사용
pip install -e .
```

### 3. 서버 실행

```bash
# 개발 서버 시작
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 서버 준비 확인
curl http://localhost:8000/
```

### 4. API 문서 확인

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🔍 주요 API 엔드포인트

### 통합 분석 API

#### 종합 분석 (메인 기능)
```bash
curl -X POST "http://localhost:8000/api/v1/analyze/comprehensive" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "우리 회사 기밀 프로젝트 X의 상세 사양서입니다. 담당자: 김철수 (010-1234-5678)",
    "similarity_threshold": 0.85,
    "pii_threshold": 0.90,
    "enable_pii": true,
    "enable_similarity": true
  }'
```

#### PII 전용 분석
```bash
curl -X POST "http://localhost:8000/api/v1/analyze/pii-only" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "담당자: 김철수 (010-1234-5678)",
    "pii_threshold": 0.90
  }'
```

#### 유사도 전용 분석
```bash
curl -X POST "http://localhost:8000/api/v1/analyze/similarity-only" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "기밀 프로젝트 상세 기술 사양서",
    "similarity_threshold": 0.85
  }'
```

### 문서 관리 API

#### 민감 문서 업로드
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@기밀문서.txt" \
  -F "title=기밀 프로젝트 사양서"
```

#### 문서 목록 조회
```bash
curl "http://localhost:8000/api/v1/documents/list?limit=10"
```

#### 문서 상세 조회
```bash
curl "http://localhost:8000/api/v1/documents/{document_id}"
```

### 상태 확인 API

```bash
# 전체 시스템 상태
curl "http://localhost:8000/"

# 분석 서비스 상태
curl "http://localhost:8000/api/v1/analyze/health"

# 문서 서비스 상태  
curl "http://localhost:8000/api/v1/documents/health"
```

## 💡 사용 시나리오

### 1. 초기 설정

1. **민감 문서 등록**
   ```bash
   # 기업의 민감 문서들을 업로드
   curl -X POST "http://localhost:8000/api/v1/documents/upload" \
     -F "file=@기밀프로젝트_사양서.txt"
   
   curl -X POST "http://localhost:8000/api/v1/documents/upload" \
     -F "file=@내부_기술문서.txt"
   ```

2. **임베딩 처리 확인**
   ```bash
   # 문서 처리 상태 확인
   curl "http://localhost:8000/api/v1/documents/list"
   ```

### 2. 실시간 분석

```bash
# 사용자 입력 텍스트 종합 분석
curl -X POST "http://localhost:8000/api/v1/analyze/comprehensive" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "분석할 사용자 입력 텍스트",
    "enable_pii": true,
    "enable_similarity": true
  }'
```

**예상 응답:**
```json
{
  "blocked": true,
  "block_reasons": ["similarity_detected"],
  "pii_analysis": {
    "has_pii": false,
    "entities": []
  },
  "similarity_analysis": {
    "is_similar": true,
    "max_similarity": 0.92,
    "matched_documents": [
      {
        "document_title": "기밀프로젝트_사양서.txt",
        "max_similarity": 0.92
      }
    ]
  },
  "analysis_time_ms": 250
}
```

### 3. 모니터링

```bash
# 분석 이력 조회
curl "http://localhost:8000/api/v1/analyze/history?limit=50&blocked_only=true"

# 분석 통계
curl "http://localhost:8000/api/v1/analyze/stats"

# 문서 통계
curl "http://localhost:8000/api/v1/documents/stats/overview"
```

## ⚙️ 설정

### 환경변수

`.env` 파일 또는 환경변수로 설정 가능:

```bash
# 데이터베이스
DATABASE_URL=postgresql+asyncpg://admin:password123@localhost:5432/ai_tlsdlp

# ChromaDB
CHROMADB_HOST=localhost
CHROMADB_PORT=8001

# AI 모델
PII_MODEL_NAME=psh3333/roberta-large-korean-pii5
SIMILARITY_MODEL_NAME=BM-K/KoSimCSE-roberta-multitask

# 분석 임계값
DEFAULT_SIMILARITY_THRESHOLD=0.85
DEFAULT_PII_THRESHOLD=0.90

# 파일 업로드
MAX_FILE_SIZE_MB=10
UPLOAD_DIR=storage/uploads
```

### 성능 튜닝

- **청킹 설정**: `CHUNK_SIZE=512`, `CHUNK_OVERLAP=50`
- **캐싱**: `CACHE_TTL_SECONDS=3600`
- **배치 크기**: 모델별 최적화된 배치 크기 사용

## 🔧 개발 및 확장

### 새로운 PII 엔티티 추가

`app/ai/pii_detector.py`에서 엔티티 타입 확장 가능

### 새로운 임베딩 모델 사용

`app/core/config.py`에서 `SIMILARITY_MODEL_NAME` 변경

### API 확장

`app/api/routers/`에 새로운 라우터 추가

## 📊 예상 성능

| 기능 | 응답 시간 | 동시 처리 |
|------|-----------|-----------|
| 종합 분석 | 200-400ms | 15-25 req/sec |
| PII 분석 | 100-200ms | 25-40 req/sec |
| 유사도 분석 | 150-300ms | 20-30 req/sec |
| 문서 업로드 | 1MB/3-5초 | 5-10 files/min |

## 🎯 프로덕션 배포 시 고려사항

1. **보안**: CORS 설정, API 키 인증 추가
2. **성능**: GPU 사용, 모델 양자화, 로드 밸런싱
3. **모니터링**: Prometheus/Grafana 연동
4. **백업**: PostgreSQL, ChromaDB 데이터 백업
5. **스케일링**: Docker Swarm 또는 Kubernetes 배포

---

## 🆘 문제 해결

### 모델 로딩 실패
- 네트워크 연결 확인
- 허깅페이스 모델 접근 권한 확인
- 메모리 부족 시 CUDA 메모리 확인

### ChromaDB 연결 실패
- `docker-compose ps`로 서비스 상태 확인
- 포트 8001 사용 가능 여부 확인

### 의존성 설치 실패
- Python 3.13 설치 확인
- `uv` 또는 `pip` 최신 버전 사용

이제 완전한 유사도 기반 DLP 시스템이 준비되었습니다! 🎉