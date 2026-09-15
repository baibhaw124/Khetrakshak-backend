from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import Base, engine
from app import models
# from fastapi import FastAPI
from app.routers import disease

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Khetrakshak AI API",
    description="AI-powered crop health, disease detection and risk assessment system",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(disease.router)

@app.get("/")
def root():
    return {
        "project": "Khetrakshak AI",
        "status": "online",
        "message": "Khetrakshak AI backend is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "khetrakshak-backend"
    }