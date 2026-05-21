"""Точка входа: python -m src.service."""

import os
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.service.app:app",
        host=os.getenv("SERVICE_HOST", "0.0.0.0"),
        port=int(os.getenv("SERVICE_PORT", "8000")),
        reload=False,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
