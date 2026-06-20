"""FastAPI 应用入口，提供与赛题一致的 `POST /chat` 接口。"""

from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException, status

from app.core.config import settings
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.pipeline import ChatPipeline
from app.services.session_store import ensure_session_id

app = FastAPI(
    title="多模态客服 Agent · V1 骨架",
    version="0.1.0",
)

pipeline = ChatPipeline()


def verify_bearer_token(authorization: str = Header(default="")) -> None:
    """校验 Bearer Token。

    请求头格式：Authorization: Bearer <token>
    """

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少或格式错误的 Authorization 请求头。",
        )

    token = authorization.removeprefix("Bearer ").strip()
    if token != settings.api_bearer_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer Token 无效。",
        )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """轻量健康检查，用于部署探活。"""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    _: None = Depends(verify_bearer_token),
) -> ChatResponse:
    """赛题兼容的对话接口。

    自行扩展时可考虑：
    - `stream=true` 时改为流式响应或明确拒绝；
    - `images` 在此已接收，后续可接多模态理解与检索。
    """
    if req.stream:
        # 后续可改为流式返回；V1 骨架先明确拒绝，避免语义不清。
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="V1 骨架暂不支持 stream=true。",
        )
    # 确保 session_id 存在（无则生成）
    sid = ensure_session_id(req.session_id)
    # 执行主流程
    result = pipeline.run(
        question=req.question,
        images=req.images,
        session_id=sid,
        user_id=req.user_id,
    )
    return ChatResponse.success(answer=result.answer, session_id=sid, images=result.images)
