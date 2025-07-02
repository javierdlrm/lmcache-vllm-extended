import vllm.entrypoints.openai.api_server as base_api
from vllm.entrypoints.openai.protocol import *
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import List
import time

# You should use the following file to implement all APIs you may require in the project.
# Note that the two important ones are already implemented here simply by calling the default v1 implementation in VLLM.
# You may need to modify these functions to enable pre-processing of requests, before running the inference.

extended_router = APIRouter()


@extended_router.get("/models")
async def show_available_models(request: Request):
    print("v2 models is called!")
    return await base_api.show_available_models(request)


@extended_router.post("/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    print("v2 completion is called")
    return await base_api.create_chat_completion(request, raw_request)


class ExtendedChatCompletionRequest(BaseModel):
    request: ChatCompletionRequest
    seq_length: int
    context_key: str


class BatchExtendedChatCompletionRequest(BaseModel):
    requests: List[ExtendedChatCompletionRequest]
    sort_before_forwarding: bool


@extended_router.post("/batch/chat/completions")
async def create_batch_chat_completion(
    batch_request: BatchExtendedChatCompletionRequest, raw_request: Request
):
    print("v2 batch completion is called")
    responses = []

    if batch_request.sort_before_forwarding:
        batch_request.requests.sort(key=lambda x: x.context_key)

    for request in batch_request.requests:
        start = time.perf_counter()
        end = None

        streaming_response = await base_api.create_chat_completion(
            request.request, raw_request
        )

        async for _ in streaming_response.body_iterator:
            pass

        end = time.perf_counter()
        latency = end - start

        responses.append({"seq_length": request.seq_length, "latency": latency})
        # responses.append(response)

    return responses
