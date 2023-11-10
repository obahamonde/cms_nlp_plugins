"""OpenAI Functions: Natural Language triggered schema driven functions"""
from abc import ABC
from typing import Any, AsyncGenerator, List, Literal, Optional, Type, TypeVar, cast

from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, Field  # pylint: disable=no-name-in-module
from typing_extensions import ParamSpec

from .utils import handle, setup_logging  # pylint: disable=E0401

T = TypeVar("T")
P = ParamSpec("P")

logger = setup_logging(__name__)


class FunctionCall(BaseModel):
    """
    Datastructure for rendering function calls
    """

    name: str
    data: Any
    comments: str


class OpenAIFunction(BaseModel, ABC):
    """
    OpenAI Function
    ---------------

    Schema-driven natural language function orchestration engine, powered by OpenAI functions feature and Pydantic.

    It enables the users to easily create and orchestrate functions that can be called from a natural language interface by defining the input schema and implementing the execution logic.

    The schema is defined using Pydantic models, and the execution logic is implemented by overriding the `run` method.

    It's an abstract class, so it can't be instantiated directly, but it can be subclassed to create new functions.

    Example:

    ```python

    import asyncio

    from typing import Any, Optional

    from pydantic import BaseModel

    from liferay_llm import OpenAIFunction

    class AddFunction(OpenAIFunction):
                                                                                                                                                                                                    '''
                                                                                                                                                                                                    Adds two numbers
                                                                                                                                                                                                    '''

                                                                                                                                                                                                    x: int
                                                                                                                                                                                                    y: int

                                                                                                                                                                                                    async def run(self, **kwargs: Any):
                                                                                                                                                                                                                                                                                                                                    return self.x + self.y

    async def main():
                                                                    print(await function_call("add 2 and 3"))

    if __name__ == "__main__":
                                                                    asyncio.run(main())

    >>> 5

    ```
    """

    @classmethod
    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        _schema = cls.schema()
        if cls.__doc__ is None:
            raise ValueError(
                f"OpenAIFunction subclass {cls.__name__} must have a docstring"
            )
        cls.openaischema = {
            "name": cls.__name__,
            "description": cls.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    k: v for k, v in _schema["properties"].items() if k != "self"
                },
                "required": _schema.get("required", []),
            },
        }

    @handle
    async def __call__(self,*,comments:str, **kwargs: Any) -> FunctionCall:
        response = await self.run(**kwargs)
        return FunctionCall(name=self.__class__.__name__, data=response,comments=comments)

    async def run(self, **kwargs: Any) -> Any:
        """Main function to be implemented by subclasses"""
        raise NotImplementedError


@handle
async def use_function(
    *,
    ai: AsyncOpenAI,
    text: str,
    model: Literal["gpt-4-1106-preview", "gpt-3.5-turbo-1106"] = "gpt-3.5-turbo-1106",
    temperature: float = 0.2,
    max_tokens: int = 2048,
    functions: Optional[List[Type[OpenAIFunction]]] = None,
    **kwargs: Any,
) -> FunctionCall:
    if functions is None:
        functions = OpenAIFunction.__subclasses__()
    response = await ai.chat.completions.create(  # type: ignore
        model=model,
        messages=[
            ChatCompletionUserMessageParam(content=text, role="user"),
        ],
        functions=[func.openaischema for func in functions],  # type: ignore
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if response.choices[0].message.function_call is None:
        return FunctionCall(name="chat", data=response.choices[0].message.content,comments="No Function was called, defaulting to chat")
    for func in functions:
        if func.__name__ == response.choices[0].message.function_call.name:
            instance = func.parse_raw(
                response.choices[0].message.function_call.arguments
            )
            data = response.choices[0].message.content
            if not isinstance(data, str):
                comments = "No comments were provided"
            else:
                comments = data
            return await instance(comments=comments,**kwargs)
    raise ValueError("Function not found")


@handle
async def use_chat(
    *,
    ai: AsyncOpenAI,
    text: str,
    context: Optional[str] = None,
    model: Literal["gpt-4-1106-preview", "gpt-3.5-turbo-1106"] = "gpt-3.5-turbo-1106",
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    if context is not None:
        messages = [
            ChatCompletionUserMessageParam(content=text, role="user"),
            ChatCompletionSystemMessageParam(content=context, role="system"),
        ]
    else:
        messages = [ChatCompletionUserMessageParam(content=text, role="user")]
    response = await ai.chat.completions.create(
        model=model,
        messages=cast(List[ChatCompletionMessageParam], messages),
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    model_output = response.choices[0].message.content
    if not isinstance(model_output, str):
        raise ValueError("Expected text output to be a string")
    return model_output


@handle
async def use_instruction(
    *,
    ai: AsyncOpenAI,
    text: str,
    temperature: float = 0.2,
    max_tokens: int = 512,
    model: str = "gpt-3.5-turbo-instruct",
) -> str:
    response = await ai.completions.create(  # type: ignore
        model=model,
        prompt=text,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    return response.choices[0].text  # type: ignore


@handle
async def use_vision(
    *,
    ai: AsyncOpenAI,
    text: str,
    image: str,
    model: Literal["gpt-4-vision-preview"],
    max_tokens: int,
    temperature: float,
) -> str:
    response = await ai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image}},
                    {"type": "text", "text": text},
                ],
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    model_result = response.choices[0].message.content
    if not isinstance(model_result, str):
        raise ValueError("Expected text output to be a string")
    return model_result


@handle
async def use_image(
    *,
    ai: AsyncOpenAI,
    text: str,
    model: Literal["dall-e-3", "dall-e-2"],
    n: int,
    response_format: Literal["b64_json", "url"],
    quality: Literal["hd", "standard"],
) -> List[str]:
    response = await ai.images.generate(
        prompt=text,
        model=model,
        n=n,
        response_format=response_format,
        quality=quality,
    )
    model_result: List[str] = []
    for i in response.data:
        if i.url is not None:
            model_result.append(i.url)
        if i.b64_json is not None:
            model_result.append(i.b64_json)
        else:
            raise ValueError("Expected image output to be a string")
    return model_result


async def use_instruction_stream(
    *,
    ai: AsyncOpenAI,
    text: str,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    model: str = "gpt-3.5-turbo-instruct",
) -> AsyncGenerator[str, None]:
    response = await ai.completions.create(  # type: ignore
        model=model,
        prompt=text,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    async for i in response:  # type: ignore
        yield i.choices[0].text  # type: ignore


async def use_chat_stream(
    *,
    ai: AsyncOpenAI,
    text: str,
    context: Optional[str],
    temperature: float,
    max_tokens: int,
    model: Literal["gpt-4-1106-preview", "gpt-3.5-turbo-1106"],
) -> AsyncGenerator[str, None]:
    if context is not None:
        messages = [
            ChatCompletionUserMessageParam(content=text, role="user"),
            ChatCompletionSystemMessageParam(content=context, role="system"),
        ]
    else:
        messages = [ChatCompletionUserMessageParam(content=text, role="user")]
    response = await ai.chat.completions.create(
        model=model,
        messages=cast(List[ChatCompletionMessageParam], messages),
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    async for chunk in response:
        data = chunk.choices[0].delta.content
        if not isinstance(data, str):
            continue
        yield data


async def use_tts(
    *,
    ai: AsyncOpenAI,
    text: str,
    model: Literal["tts-1", "tts-1-hd"],
    voice: Literal["nova", "alloy", "echo", "fable", "onyx", "shimmer"],
    response_format: Literal["mp3", "opus", "aac", "flac"],
) -> AsyncGenerator[bytes, None]:
    stream = await ai.audio.speech.create(
        input=text, model=model, voice=voice, response_format=response_format
    )
    response = await stream.aiter_bytes()
    async for chunk in response:
        yield chunk

'''
class UseChat(OpenAIFunction):
    """
    Chat with a ChatGPT the best large language model in the industry, you can customize the context of the conversation by passing on the context parameter custom text to tweak the behavior of ChatGPT to solve especific problems. For example you can tell him that he is an specialist in a topic and he will try to answer your questions about that topic as best as he can.
    """

    text: str
    context: str
    model: Literal["gpt-4-1106-preview", "gpt-3.5-turbo-1106"] = Field(
        default="gpt-3.5-turbo-1106"
    )
    temperature: float = Field(default=0.2)
    max_tokens: int = Field(default=1024)

    @property
    def ai(self) -> AsyncOpenAI:
        return AsyncOpenAI()

    async def run(self, **kwargs: Any) -> str:
        return await use_chat(
            ai=self.ai,
            text=self.text,
            context=self.context,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
'''

class UseInstruction(OpenAIFunction):
    """
    Give an Instruction to ChatGPT to do a specific task, for example you can tell him to write a poem about a specific topic.
    """

    text: str
    model: Literal["gpt-3.5-turbo-instruct"] = Field(default="gpt-3.5-turbo-instruct")
    temperature: float = Field(default=0.2)
    max_tokens: int = Field(default=1024)

    @property
    def ai(self) -> AsyncOpenAI:
        return AsyncOpenAI()

    async def run(self, **kwargs: Any) -> str:
        return await use_instruction(
            ai=self.ai,
            text=self.text,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


class UseVision(OpenAIFunction):
    """
    Use the power of ChatGPT Vision to give details or explanation about the image url provided on the image parameter based on the instruction provided on the text parameter.
    """

    text: str
    image: str
    model: Literal["gpt-4-vision-preview"] = Field(default="gpt-4-vision-preview")
    max_tokens: int = Field(default=512)
    temperature: float = Field(default=0.5)

    @property
    def ai(self) -> AsyncOpenAI:
        return AsyncOpenAI()

    async def run(self, **kwargs: Any) -> str:
        return await use_vision(
            ai=self.ai,
            text=self.text,
            image=self.image,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )


class UseImage(OpenAIFunction):
    """
    Use the power of Dall-E to generate images based on the text provided on the text parameter.
    """

    text: str
    model: Literal["dall-e-3", "dall-e-2"] = Field(default="dall-e-3")
    n: int = Field(default=1)
    response_format: Literal["b64_json", "url"] = Field(default="url")
    quality: Literal["hd", "standard"] = Field(default="standard")

    @property
    def ai(self) -> AsyncOpenAI:
        return AsyncOpenAI()

    async def run(self, **kwargs: Any) -> List[str]:
        return await use_image(
            ai=self.ai,
            text=self.text,
            model=self.model,
            n=self.n,
            response_format=self.response_format,
            quality=self.quality,
        )
