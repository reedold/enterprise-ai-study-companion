"""Domain exceptions and their FastAPI handlers."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    status_code = 500
    error_code = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidInputError(AppError):
    status_code = 400
    error_code = "invalid_input"


class PayloadTooLargeError(AppError):
    status_code = 413
    error_code = "payload_too_large"


class NoContentError(AppError):
    status_code = 422
    error_code = "no_extractable_content"


class UpstreamServiceError(AppError):
    status_code = 502
    error_code = "upstream_error"


def _error(status: int, code: str, message: str, details=None) -> JSONResponse:
    body = {"error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError):
        if exc.status_code >= 500:
            logger.error("%s: %s", exc.error_code, exc.message)
        return _error(exc.status_code, exc.error_code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError):
        return _error(422, "validation_error", "Request validation failed", exc.errors())

    @app.exception_handler(Exception)
    async def unhandled_handler(_: Request, exc: Exception):
        logger.exception("Unhandled error: %s", exc)
        return _error(500, "internal_error", "An unexpected error occurred")
