from typing import Literal, Optional

from fastapi import APIRouter, Depends
from openai import AsyncOpenAI
from openai.types.beta import Thread

app = APIRouter()
ai = AsyncOpenAI()


@app.get("/api/thread")
async def create_thread():
    """
    Create a new thread.
    """
    threads = ai.beta.threads
    response = await threads.create()
    return response


@app.delete("/api/thread/{thread_id}")
async def delete_thread(*, thread_id: str):
    """
    Delete a thread.
    """
    threads = ai.beta.threads
    response = await threads.delete(thread_id=thread_id)
    return response


@app.post("/api/messages/{thread_id}")
async def create_message(
    *,
    content: str,
    thread_id: str,
    role: Literal["user"] = "user",
    file_ids: list[str] = [],
    metadata: Optional[dict[str, str]] = {}
):
    """
    Create a message.
    """
    messages = ai.beta.threads.messages
    response = await messages.create(
        thread_id=thread_id,
        content=content,
        role=role,
        file_ids=file_ids,
        metadata=metadata,
    )
    return response


@app.get("/api/messages/{thread_id}")
async def retrieve_messages(*, thread_id: str):
    """
    Retrieve messages.
    """
    messages = ai.beta.threads.messages
    response = await messages.list(thread_id=thread_id)
    return response
