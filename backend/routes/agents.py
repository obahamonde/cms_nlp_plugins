from typing import Literal

from fastapi import APIRouter, Depends
from openai import AsyncOpenAI
from openai.types.beta import Assistant, Thread

from .files import delete_file, get_files, retrieve_files, upload_file
from .threads import create_thread, delete_thread

app = APIRouter()
ai = AsyncOpenAI()


@app.post("/api/assistant")
async def create_assistant(
    name: str,
    instructions: str,
    model: Literal["gpt-3.5-turbo-1106", "gpt-4-1106-preview"] = "gpt-3.5-turbo-1106",
):
    """
    Create a new assistant.
    """
    assistants = ai.beta.assistants
    response = await assistants.create(
        name=name,
        instructions=instructions,
        model=model,
    )
    return response


@app.delete("/api/assistant/{assistant_id}")
async def delete_assistant(assistant_id: str):
    """
    Delete an assistant.
    """
    assistants = ai.beta.assistants
    response = await assistants.delete(assistant_id=assistant_id)
    return response


@app.get("/api/assistant")
async def retrieve_assistant(assistant_id: str):
    """
    Retrieve an assistant.
    """
    assistants = ai.beta.assistants
    response = await assistants.retrieve(assistant_id=assistant_id)
    return response
