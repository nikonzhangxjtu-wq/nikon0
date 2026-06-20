"""兼容旧 import 路径的上下文压缩入口。

新实现位于 ``app.services.context`` 包中；保留这个文件，避免 pipeline 和测试
需要同时迁移 import 路径。
"""

from app.services.context import (
    AssembledPromptContext,
    CompressionTrace,
    ContextAssembler,
    ContextAssemblyTrace,
    estimate_tokens,
)
from app.core.config import settings

__all__ = [
    "AssembledPromptContext",
    "CompressionTrace",
    "ContextAssembler",
    "ContextAssemblyTrace",
    "estimate_tokens",
    "settings",
]
