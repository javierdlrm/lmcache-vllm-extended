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


#######################################################################################
# Batch endpoints
#######################################################################################


class ExtendedChatCompletionRequest(BaseModel):
    request_idx: int
    request: ChatCompletionRequest
    context_key: str
    rag_latency: float = None
    rag_encode_latency: float = None
    rag_search_latency: float = None


class BatchExtendedChatCompletionRequest(BaseModel):
    requests: List[ExtendedChatCompletionRequest]
    sort_before_forwarding: bool
    use_rag: bool = False  # Whether to use RAG for context retrieval


@extended_router.post("/batch/chat/completions")
async def create_batch_chat_completion(
    batch_request: BatchExtendedChatCompletionRequest, raw_request: Request
):
    print("v2 batch completion is called")
    responses = []
    rag_accuracy = None

    if batch_request.sort_before_forwarding:
        # Sort batched requests by context_key if specified
        if batch_request.use_rag:
            rag_accuracy = 0.0

            # If using RAG, retrieve context for each request before sorting
            for request in batch_request.requests:
                question = request.request.messages[-1]["content"]

                start = time.perf_counter()
                question_np = extended_router.rag_instance.encode(question)
                encode_latency = time.perf_counter() - start

                start = time.perf_counter()
                context_key, context = extended_router.rag_instance.search(
                    question, question_np, top_k=1
                )
                search_latency = time.perf_counter() - start

                latency = encode_latency + search_latency

                # Add the context to the last user message
                request.request.messages[-1] = {
                    "role": "user",
                    "content": f"User prompt: {question}. "
                    + "\nPlease, answer given the following context: "
                    + context,
                }

                rag_accuracy += 1 if context_key == request.context_key else 0

                request.context_key = context_key
                request.rag_latency = latency
                request.rag_encode_latency = encode_latency
                request.rag_search_latency = search_latency

            rag_accuracy /= len(batch_request.requests)

        # Sort requests by context_key
        batch_request.requests.sort(key=lambda x: x.context_key)

    # Process each request in the batch
    for request in batch_request.requests:
        start = time.perf_counter()
        end = None

        streaming_response = await base_api.create_chat_completion(
            request.request, raw_request
        )

        # Consume the streaming response to ensure the whole request is processed
        async for _ in streaming_response.body_iterator:
            pass

        end = time.perf_counter()
        latency = end - start

        # Build response with metrics
        metrics = {
            "request_idx": request.request_idx,
            "latency": latency,
        }
        if batch_request.use_rag:
            metrics["rag_accuracy"] = rag_accuracy
            metrics["rag_latency"] = request.rag_latency
            metrics["rag_encode_latency"] = request.rag_encode_latency
            metrics["rag_search_latency"] = request.rag_search_latency
            metrics["rag_context_key"] = request.context_key

        responses.append(metrics)

    return responses


#######################################################################################
# RAG endpoints
#######################################################################################


class RAGIndexRequest(BaseModel):
    context_key: str
    context: str


class RAGSearchRequest(BaseModel):
    prompt: str


@extended_router.post("/rag/index")
async def rag_index(request: RAGIndexRequest):
    try:
        context_np = extended_router.rag_instance.encode(request.context)
        extended_router.rag_instance.index(
            request.context_key, request.context, context_np
        )
        return {
            "status": "success",
            "message": f"Context '{request.context_key}' indexed.",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@extended_router.post("/rag/search")
async def rag_search(request: RAGSearchRequest):
    try:
        start = time.perf_counter()
        prompt_np = extended_router.rag_instance.encode(request.prompt)
        encode_latency = time.perf_counter() - start

        start = time.perf_counter()
        context_key, _ = extended_router.rag_instance.search(request.prompt, prompt_np)
        search_latency = time.perf_counter() - start

        latency = encode_latency + search_latency

        return {
            "status": "success",
            "rag_context_key": context_key,
            "rag_latency": latency,
            "rag_encode_latency": encode_latency,
            "rag_search_latency": search_latency,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
