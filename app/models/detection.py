# app/models/detection.py
from sqlalchemy import Boolean, DateTime, func, Text
from sqlalchemy.dialects.postgresql import INET, JSONB
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base

class Detection(Base):
    __tablename__ = "pii_detections"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_ip: Mapped[str | None] = mapped_column(INET, nullable=True)
    has_pii: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # 입력/가공 결과
    original_text: Mapped[str] = mapped_column(Text, nullable=False)
    tagged_text: Mapped[str | None] = mapped_column(Text, nullable=True)  # 모델 반환 원문(태그 포함)

    # 엔티티 [{label, value, start, end}] (cleaned_text 기준 오프셋)
    entities: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())