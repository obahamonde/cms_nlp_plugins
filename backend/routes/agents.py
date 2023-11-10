from typing import Literal

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
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


@app.get("/api/assistant/{assistant_id}")
async def retrieve_assistant(assistant_id: str):
    """
    Retrieve an assistant.
    """
    assistants = ai.beta.assistants
    response = await assistants.retrieve(assistant_id=assistant_id)
    return response


@app.get("/api/assistant")
async def retrieve_all_assistants():
    """
    Retrieve all assistants.
    """
    assistants = ai.beta.assistants
    response = await assistants.list()

    async def generator():
        async for assistant in response:
            yield f"data: {assistant.json()}"

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.put("/api/assistant/files/{assistant_id}")
async def attach_file(assistant_id: str, file_id: str):
    """
    Attach a file to an assistant.
    """
    assistants = ai.beta.assistants
    response = await assistants.files.create(
        assistant_id=assistant_id,
        file_id=file_id,
    )
    return response


@app.delete("/api/assistant/files/{assistant_id}")
async def detach_file(assistant_id: str, file_id: str):
    """
    Detach a file from an assistant.
    """
    assistants = ai.beta.assistants
    response = await assistants.files.delete(
        assistant_id=assistant_id,
        file_id=file_id,
    )
    return response


@app.get("/api/assistant/files/{assistant_id}")
async def retrieve_attached_files(assistant_id: str):
    """
    Retrieve all files attached to an assistant.
    """
    assistants = ai.beta.assistants
    response = await assistants.files.list(assistant_id=assistant_id)

    async def generator():
        async for file in response:
            yield f"data: {file.json()}"

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.post("/api/run/{thread_id}")
async def run_thread(thread_id: str, assistant_id: str):
    """
    Run a thread.
    """
    threads = ai.beta.threads
    response = await threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )
    return response


@app.get("/api/run")
async def retrieve_run(thread_id: str, run_id: str):
    """
    Retrieve a run.
    """
    threads = ai.beta.threads
    response = await threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run_id,
    )
    return response


@app.get("/api/run/{thread_id}")
async def retrieve_all_runs(thread_id: str):
    response = await ai.beta.threads.runs.list(thread_id=thread_id)

    async def generator():
        async for run in response:
            yield f"data: {run.json()}"

    return StreamingResponse(generator(), media_type="text/event-stream")
