from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Text
from datetime import datetime

from app.database import Base


class Farmer(Base):
    __tablename__ = "farmers"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    name = Column(
        String(100),
        nullable=False
    )

    phone = Column(
        String(15),
        unique=True,
        nullable=False
    )

    village = Column(
        String(100),
        nullable=True
    )

    district = Column(
        String(100),
        nullable=True
    )

    state = Column(
        String(100),
        nullable=True
    )

    preferred_language = Column(
        String(30),
        default="Hindi"
    )


class Field(Base):
    __tablename__ = "fields"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    farmer_id = Column(
        Integer,
        ForeignKey("farmers.id"),
        nullable=False
    )

    field_name = Column(
        String(100),
        nullable=False
    )

    area_acres = Column(
        Float,
        nullable=True
    )

    latitude = Column(
        Float,
        nullable=True
    )

    longitude = Column(
        Float,
        nullable=True
    )

    soil_type = Column(
        String(50),
        nullable=True
    )


class Crop(Base):
    __tablename__ = "crops"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    field_id = Column(
        Integer,
        ForeignKey("fields.id"),
        nullable=False
    )

    crop_name = Column(
        String(100),
        nullable=False
    )

    variety = Column(
        String(100),
        nullable=True
    )

    sowing_date = Column(
        String(20),
        nullable=True
    )

    expected_harvest_date = Column(
        String(20),
        nullable=True
    )

    growth_stage = Column(
        String(50),
        nullable=True
    )

    status = Column(
        String(30),
        default="Active"
    )
class DiseaseDetection(Base):
    __tablename__ = "disease_detections"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    crop_id = Column(
        Integer,
        ForeignKey("crops.id"),
        nullable=False
    )

    disease_name = Column(
        String(100),
        nullable=False
    )

    confidence = Column(
        Float,
        nullable=False
    )

    risk_level = Column(
        String(30),
        nullable=False
    )

    image_path = Column(
        String(255),
        nullable=True
    )

    recommendation = Column(
        Text,
        nullable=True
    )

    detected_at = Column(
        DateTime,
        default=datetime.utcnow
    )