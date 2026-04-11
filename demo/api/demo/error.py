from fastapi import Request
from fastapi.responses import JSONResponse


async def handle(request: Request, exc: Exception) -> JSONResponse:
    code = getattr(exc, "status_code", 500)
    detail = getattr(exc, "detail", str(exc))
    return JSONResponse(
        status_code=code,
        content={"error": {"code": code, "message": detail}},
    )
