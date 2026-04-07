import asyncio
import importlib.metadata
import os

import agent_framework
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

# agent-framework 1.0.0 expects __version__ to exist on the top-level package.
if not getattr(agent_framework, "__version__", None):
    agent_framework.__version__ = importlib.metadata.version("agent-framework")

from agent_framework._agents import Agent
from agent_framework.openai import OpenAIChatClient

load_dotenv()


def first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def normalize_api_version(value: str | None) -> str | None:
    if not value:
        return None
    if value == "preview":
        return value
    if value[:4].isdigit():
        return "preview"
    return value


async def main() -> None:
    endpoint = first_env("AZURE_OPENAI_ENDPOINT")
    model = first_env(
        "AZURE_OPENAI_MODEL",
        "AZURE_OPENAI_CHAT_MODEL",
        "AZURE_OPENAI_MODEL_NAME",
        "AZURE_OPENAI_DEPLOYMENT",
    )
    api_key = first_env("AZURE_OPENAI_API_KEY")
    api_version = normalize_api_version(first_env("AZURE_OPENAI_API_VERSION"))

    if not endpoint:
        raise RuntimeError("Set AZURE_OPENAI_ENDPOINT in .env or your shell.")
    if not model:
        raise RuntimeError(
            "Set AZURE_OPENAI_MODEL, AZURE_OPENAI_MODEL_NAME, or AZURE_OPENAI_DEPLOYMENT in .env or your shell."
        )

    agent = Agent(
        client=OpenAIChatClient(
            model=model,
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            credential=None if api_key else AzureCliCredential(),
        ),
        name="HaikuBot",
        instructions="You are an upbeat assistant that writes beautiful poetry.",
    )

    print(await agent.run("Write a haiku about the Nurse."))


if __name__ == "__main__":
    asyncio.run(main())
