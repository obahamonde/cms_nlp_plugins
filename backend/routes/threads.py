from fastapi import APIRouter
from openai import AsyncOpenAI

app = APIRouter()
ai = AsyncOpenAI()


async def create_thread(*, ai: AsyncOpenAI):
    """
    Create a new thread.
    """
    threads = ai.beta.threads
    response = await threads.create()
    return response


async def delete_thread(*, ai: AsyncOpenAI, thread_id: str):
    """
    Delete a thread.
    """
    threads = ai.beta.threads
    response = await threads.delete(thread_id=thread_id)
    return response


@app.get("/api/thread")
async def thread_endpoint():
    """
    Returns a new thread.
    """
    response = await create_thread(ai=ai)
    return response


@app.delete("/api/thread/{thread_id}")
async def delete_thread_endpoint(thread_id: str):
    """
    Deletes a thread.
    """
    return await ai.beta.threads.delete(thread_id=thread_id)
