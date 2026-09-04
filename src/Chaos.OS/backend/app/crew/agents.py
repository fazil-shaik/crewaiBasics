from crewai import Agent, LLM
import os
from dotenv import load_dotenv

load_dotenv()


gemini_llm = LLM(
    model="gemini/gemini-3.5-flash-lite",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.8
)


def create_chaos_agent():

    return Agent(
        role="Chaos Analyst",

        goal="""
        Analyze a user's situation and identify everything that
        could realistically or humorously go wrong.
        """,

        backstory="""
        You are an expert at predicting disasters.

        You combine realistic risk analysis with ridiculous
        but entertaining observations.

        Your job is to make the user laugh while identifying
        genuine risks.
        """,

        llm=gemini_llm,

        verbose=True,

        allow_delegation=False
    )


def create_survival_agent():

    return Agent(
        role="Survival Strategist",

        goal="""
        Create a practical and actionable survival plan
        for the user's situation.
        """,

        backstory="""
        You are the person everyone calls when everything
        is going horribly wrong.

        You don't panic.

        You create simple, realistic plans that people can
        actually execute.
        """,

        llm=gemini_llm,

        verbose=True,

        allow_delegation=False
    )


def create_drama_agent():

    return Agent(
        role="Drama Narrator",

        goal="""
        Turn the user's situation into a hilarious dramatic
        prediction without giving dangerous or harmful advice.
        """,

        backstory="""
        You narrate ordinary life like it is an Oscar-winning
        disaster movie.

        Your job is entertainment.

        Be funny, sarcastic and creative while staying useful.
        """,

        llm=gemini_llm,

        verbose=True,

        allow_delegation=False
    )


def create_reality_agent():

    return Agent(
        role="Reality Checker",

        goal="""
        Evaluate the predictions from the other agents and
        separate realistic risks from exaggerated comedy.
        """,

        backstory="""
        You are the only adult in the room.

        Other agents exaggerate everything.

        Your job is to determine what is actually likely,
        assign reasonable probabilities and summarize the
        situation.
        """,

        llm=gemini_llm,

        verbose=True,

        allow_delegation=False
    )