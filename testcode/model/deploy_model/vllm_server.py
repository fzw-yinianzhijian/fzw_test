import os
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PublicAPI")

app = FastAPI(title="Public LLM Service")

# CORS 配置，允许 OPTIONS 预检请求
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # 如仅本地调试，也可以改为 ["*"]，生产环境不建议
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # 或 ["*"]
    allow_credentials=True,
    allow_methods=["*"],        # 包含 OPTIONS
    allow_headers=["*"],        # 包含 Authorization 等
)

# 配置
VLLM_API_URL = "http://localhost:8000/v1/completions"
API_KEY_SECRET = "test_fzw_key"
MAX_CONCURRENT_REQUESTS = 100 # 应用层队列限制

# instruction one term

# 信号量，用于应用层排队调度
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

class ChatRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    # 预检请求直接放行，由 CORS 中间件处理
    if request.method == "OPTIONS":
        return await call_next(request)

    # 简单的鉴权中间件
    # 实际生产中建议在 Nginx 层或使用专门的 Auth 服务
    token = request.headers.get("Authorization")
    if request.url.path != "/docs" and request.url.path != "/openapi.json":
        if token != f"Bearer {API_KEY_SECRET}":
            print(f" !!!!!SSSS Invalid or missing API key: {token}")
            # raise HTTPException(status_code=401, detail="Invalid or missing API key")
            # 这里为了演示简单，实际可根据需求开启或关闭鉴权
            pass 
    response = await call_next(request)
    return response

@app.post("/api/v1/generate")
async def generate_response(request: ChatRequest):
    """
    Public API 入口
    """
    # 1. 队列调度逻辑：获取信号量
    # 如果当前并发超过 MAX_CONCURRENT_REQUESTS，后续请求会在这里 "等待" (排队)
    async with request_semaphore:
        logger.info("Processing request...")
        
        payload = {
            "model": "gemma_test", # 需要与 vLLM 启动的模型名一致
            "prompt": request.prompt,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        # 2. 转发给 vLLM 后端
        timeout_config = httpx.Timeout(60.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            try:
                # 能够处理流式 (Streaming) 或 非流式
                if request.stream:
                    assert False 
                    # 针对流式输出的特殊处理（需配合 StreamingResponse）
                    # 这里为了代码简洁，演示非流式
                    pass 
                
                response = await client.post(VLLM_API_URL, json=payload)
                response.raise_for_status()
                
                vllm_data = response.json()
                
                # 3. 提取结果并返回标准格式
                return {
                    "status": "success",
                    "data": vllm_data['choices'][0]['text'],
                    "usage": vllm_data['usage']
                }

            except httpx.RequestError as exc:
                logger.error(f"Backend connection error: {exc}")
                raise HTTPException(status_code=503, detail="Inference engine unavailable")
            except Exception as e:
                logger.error(f"Error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 启动 Public API 服务，端口 5000
    uvicorn.run(app, host="0.0.0.0", port=5000)