"""Standardized error responses for the dubbing service API.

所有 API 错误统一使用 ``{"detail": {"code": "...", "message": "..."}}`` 格式。
前端 fetchJson 已兼容 detail 为 string 或 object（含 .message）的情况。
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


def error_response(
    status_code: int,
    code: str,
    message: str,
    extra: dict[str, Any] | None = None,
) -> JSONResponse:
    """构建标准化错误 JSON 响应。"""
    body: dict[str, Any] = {"code": code, "message": message}
    if extra:
        body.update(extra)
    return JSONResponse(status_code=status_code, content={"detail": body})


def install_exception_handlers(app: FastAPI) -> None:
    """为 FastAPI 应用安装全局异常处理器。"""

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """统一 HTTPException 响应格式。"""
        detail = exc.detail
        # 如果 detail 已经是结构化的 dict（由路由手动构造），保持原样
        if isinstance(detail, dict):
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": detail},
            )
        # 否则包装为标准格式
        code = _status_to_code(exc.status_code)
        return error_response(exc.status_code, code, str(detail))

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """统一请求参数校验错误响应格式。"""
        # Pydantic includes raw exception instances under ``ctx.error`` for
        # value validators. Those cannot be encoded by JSONResponse and used
        # to turn ordinary 422 validation failures into unhandled 500s.
        errors = [
            {key: value for key, value in item.items() if key != "ctx"}
            for item in exc.errors()
        ]
        # 提取第一个错误作为主消息
        first = errors[0] if errors else {}
        loc = " → ".join(str(x) for x in first.get("loc", []))
        msg = first.get("msg", "参数校验失败")
        message = f"{loc}: {msg}" if loc else msg

        return error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="VALIDATION_ERROR",
            message=message,
            extra={"errors": errors},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """捕获所有未处理异常，避免泄漏堆栈信息。"""
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="INTERNAL_ERROR",
            message="服务器内部错误，请稍后重试。",
        )


def _status_to_code(status_code: int) -> str:
    """将 HTTP 状态码映射为简短的错误码字符串。"""
    mapping = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        409: "CONFLICT",
        413: "PAYLOAD_TOO_LARGE",
        422: "VALIDATION_ERROR",
        429: "TOO_MANY_REQUESTS",
        500: "INTERNAL_ERROR",
        502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE",
    }
    return mapping.get(status_code, f"HTTP_{status_code}")
