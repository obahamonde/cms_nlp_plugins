from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

from backend import (
    use_chat,
    use_chat_stream,
    use_function,
    use_image,
    use_instruction,
    use_instruction_stream,
    use_tts,
    use_vision,
)
from backend.routes import agents_app, files_app, threads_app

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
ai = AsyncOpenAI()

##### TTS #############################################


@app.get("/api/audio/{text}")
async def tts_endpoint(
    text: str,
    context: Optional[str] = None,
    model: Literal["gpt-4-1106-preview", "gpt-3.5-turbo-1106"] = "gpt-4-1106-preview",
    voice_model: Literal["tts-1", "tts-1-hd"] = "tts-1",
    max_tokens: int = 1024,
    temperature: float = 0.9,
    voice: Literal["nova", "alloy", "echo", "fable", "onyx", "shimmer"] = "shimmer",
    response_format: Literal["mp3", "opus", "aac", "flac"] = "opus",
):
    """
    Returns a synthesized audio file of the input text.
    """
    result = await use_chat(
        ai=ai,
        text=text,
        context=context,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
    )
    audio = use_tts(
        ai=ai,
        text=result,
        voice=voice,
        model=voice_model,
        response_format=response_format,
    )

    return StreamingResponse(audio, media_type=f"audio/{response_format}")


####### AUTOCOMPLETE ENDPOINTS #######


@app.get("/api/autocomplete/blog")
async def autocomplete_blog_endpoint(
    text: str,
    max_tokens: int = 128,
    temperature: float = 0.9,
    model: Literal["gpt-3.5-turbo-instruct"] = "gpt-3.5-turbo-instruct",
):
    """
    Returns a list of blog post titles based on the input text.
    """
    text = f"You are assisting the user to write a blog, please write the three following paragraphs according to user input, dont repeat any of the text that the user has already mentioned, this is the input of the user: \n\n{text}\n\n"
    response = await use_instruction(
        ai=ai, text=text, max_tokens=max_tokens, temperature=temperature, model=model
    )
    return PlainTextResponse(response)


@app.get("/api/autocomplete/code")
async def autocomplete_code_endpoint(
    text: str,
    max_tokens: int = 128,
    temperature: float = 0.9,
    model: Literal["gpt-3.5-turbo-instruct"] = "gpt-3.5-turbo-instruct",
):
    """
    Returns a list of code snippets based on the input text.
    """
    text = f"You are assisting the user to write a code, please write the three following liunes of codeaccording to user input, dont repeat any of the text that the user has already mentioned, this is the input of the user: \n\n{text}\n\n"
    response = await use_instruction(
        ai=ai, text=text, max_tokens=max_tokens, temperature=temperature, model=model
    )
    return PlainTextResponse(response)


######### CHAT #########


@app.get("/api/chat")
async def chat_endpoint(text: str):
    """
    Returns a response to the input text.
    """
    response = use_chat_stream(
        ai=ai,
        text=text,
        context=None,
        temperature=0.7,
        max_tokens=1024,
        model="gpt-3.5-turbo-1106",
    )

    async def generator():
        async for item in response:
            yield f"data: {item}\n\n"
        yield "event: done\ndata: \n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


app.include_router(threads_app, tags=["threads,messages"])
app.include_router(files_app, tags=["files"])
app.include_router(agents_app, tags=["assistants"])
