from fastapi import FastAPI
from app.core.lifespan import lifespan
from app.api.routes import router as api_router
from app.core.config import settings

# Use the config settings here
app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION, lifespan=lifespan)

app.include_router(api_router, prefix="/api")