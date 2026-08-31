from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import os
from crewai import LLM
from crewai.tools import BaseTool



llm =LLM(
    model="gemini/gemini-3.6-flash",
    temperature=0.7,
    api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
)

# res = llm.call("Write a short poem about the beauty of nature.")

# print(res)


class ReplaceJargonsTool(BaseTool):
    name: str = "Jargon replacement tool"
    description : str = "Replaces jargon with more specific terms. "

    def _run(self, email: str) -> str:
        replacements = {
            "PRX": "Project Phoenix (internal AI revamp project)",
            "TAS": "technical architecture stack",
            "DBX": "client database cluster",
            "SDS": "Smart Data Syncer",
            "SYNCBOT": "internal standup assistant bot",
            "WIP": "in progress",
            "POC": "proof of concept",
            "ping": "reach out"
        }
        suggestions = []
        email_lower = email.lower()
        for jargon, replacement in replacements.items():
            if jargon.lower() in email_lower:
                suggestions.append(f"Consider replacing '{jargon}' with '{replacement}'")

        return "\n".join(suggestions) if suggestions else "No jargon or internal abbreviations detected."

jt = ReplaceJargonsTool()

original_email = """
looping in Priya. TAS and PRX updates are in the deck. ETA for SDS integration is Friday.
Let's sync up tomorrow if SYNCBOT allows 😄. ping me if any blockers.
"""

jt.run(original_email)


from crewai import Agent,Task,Crew


email_assistant = Agent(
    role="email_assistant",
    goal="improve emails and make them sound professional",
    backstory="A highly experienced communication expert skilled in professional email writing",
    verbose=True,
    tools=[jt],
    llm=llm 
)

original_email = """
hey team, just wanted to tell u that the demo is kind of ready, but there's still stuff left.
Maybe we can show what we have and say rest is WIP.
Let me know what u think. thanks
"""

email_task = Task(
    description=f"""Take the following rough email and rewrite it into a professional and polished version.
    Expand abbreviations:
    '''{original_email}'''""",
    agent=email_assistant,
    expected_output="A professional written email with proper formatting and content.",
)

crew = Crew(
    agents=[email_assistant],
    tasks=[email_task],
    verbose=True
)

result = crew.kickoff()
print(result)