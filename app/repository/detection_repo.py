from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.models.detection import Detection
from typing import List

class DetectionRepository:
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
            select(Detection)
            .where(Detection.user_ip == user_ip)
        )
        return len(list(res.scalars().all()))
    
    async def count_pii_detections_by_user_ip(self, db: AsyncSession, user_ip: str) -> int:
        """사용자 IP 기준 개인정보 탐지 횟수"""
        res = await db.execute(
            select(Detection)
            .where(Detection.user_ip == user_ip, Detection.has_pii == True)
        )
        return len(list(res.scalars().all()))