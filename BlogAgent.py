from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import os
from crewai import LLM
from crewai.tools import BaseTool
from crewai_tools import EXASearchTool




llm =LLM(
    model="gemini/gemini-3.5-flash-lite",
    temperature=0.7,
    api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
)


print(type(os.getenv("SERPER_API_KEY")))

from crewai import Agent,Task,Crew


research_agent = Agent(
    role="Research Specialist",
    goal="Research interesting facts about the topic: {topic}",
    backstory="You are an expert at finding relevant and factual data.",
    tools=[EXASearchTool()],
    verbose=True,
    llm=llm
)



writer_agent = Agent(
    role="Creative Writer",
    goal="Write a short blog summary using the research",
    backstory="You are skilled at writing engaging summaries based on provided content.",
    llm=llm,
    verbose=True,
)


task1 = Task(
    agent=research_agent,
    description="find 3-5 best of the best related to {topic} and best facts as of 2026 and future years also ",
    expected_output="list top 3 bullet points from each facts as of 2026",
)

task2 = Task(
    agent=writer_agent,
    description="Write a nice nice 100 worded blog post of {topic} using the facts from the research we have done and it must very updated result as of 2026",
    expected_output="a blog post summary as of 2026 standards",
    context=[task1]
)


crew = Crew(
        agents=[research_agent,writer_agent],
        tasks=[task1,task2],
        verbose=True
)


crew.kickoff(inputs={"topic":"Future of the EV vehicles"})