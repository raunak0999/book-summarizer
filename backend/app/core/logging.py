"""
Structured JSON logging + a request-id middleware so every log line and
every error response can be correlated to one HTTP request (observability
requirement #7).
"""
import logging
import sys
import time
import uuid
import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


def configure_logging():
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


log = structlog.get_logger()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attaches a request_id to every log line and logs latency + status."""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
        start = time.time()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled_exception")
            raise
        duration_ms = round((time.time() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        log.info("request_completed", status=response.status_code, duration_ms=duration_ms)
        structlog.contextvars.clear_contextvars()
        return response
