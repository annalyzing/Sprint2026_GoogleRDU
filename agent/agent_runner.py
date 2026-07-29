from dotenv import load_dotenv

load_dotenv()

from agents import process_education_query


def run_agent(message: str, persona: str = "general", history: list = None) -> str:
    """
    Main entry point for chatbot.

    Passes:
    - current user question
    - user persona
    - conversation history
    """

    if history is None:
        history = []

    return process_education_query(
        message=message,
        persona=persona,
        history=history
    )