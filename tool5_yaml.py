# from crewai import Agent, Crew, Process, Task, LLM
# from crewai.project import CrewBase, agent, crew, task
# import os
# from crewai_tools import EXASearchTool, ScrapeWebsiteTool, DirectoryReadTool, FileWriterTool, FileReadTool

# from dotenv import load_dotenv
# load_dotenv()


# @CrewBase
# class BlogCrew():
#     """Blog writing crew"""

#     agents_config = "config/agents.yaml"
#     tasks_config = "config/tasks.yaml"


#     @agent
#     def researcher(self)->Agent:
#         return Agent(
#             config = self.agents_config['research_agent'],
#             tools=[(EXASearchTool())],
#             reasoning=True,
#             verbose=True,
#             # respect_context_window=True
#         )

#     @agent
#     def writer(self) -> Agent:
#         return Agent(
#             config=self.agents_config['writer_agent'], # type: ignore[index]
#             reasoning=True,
#             verbose=True,
#             tools=[DirectoryReadTool(),FileWriterTool()],
#         )


#     @task
#     def research_task(self) -> Task:
#         return Task(
#             config=self.tasks_config['research_task'], # type: ignore[index]
#             agent = self.researcher()
#         )

#     @task
#     def blog_task(self) -> Task:
#         return Task(
#             config=self.tasks_config['blog_task'], # type: ignore[index]
#             agent = self.writer()
#         )


#     @crew
#     def crew(self)->Crew:
#         return Crew(
#             agents=[self.researcher(),self.writer()],
#             tasks=[self.research_task(),self.blog_task()],
#             memory=True,
#             embedder={
#                 "provider":"google-generativeai",
#                 "config":{
#                     "model":"gemini-embedding-001",
#                     "api_key":os.environ["GOOGLE_API_KEY"]
#                 }
#             }
#         )
# print(os.getenv("EMBEDDINGS_GOOGLE_GENERATIVE_AI_MODEL_NAME"))

# if __name__=="__main__":
#     blog_crew = BlogCrew()
#     blog_crew.crew().kickoff(inputs={"topic":"The future of electrical vehicles"})

from crewai import Agent, Crew, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import EXASearchTool, DirectoryReadTool, FileWriterTool
from dotenv import load_dotenv
import os

load_dotenv()


llm = LLM(
    model="gemini/gemini-3.5-flash-lite",
    api_key=os.environ["GOOGLE_API_KEY"]
)


@CrewBase
class BlogCrew:

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["research_agent"],
            tools=[EXASearchTool()],
            reasoning=True,
            verbose=True,
            llm=llm
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config["writer_agent"],
            tools=[DirectoryReadTool(), FileWriterTool()],
            reasoning=True,
            verbose=True,
            llm=llm
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],
            agent=self.researcher()
        )

    @task
    def blog_task(self) -> Task:
        return Task(
            config=self.tasks_config["blog_task"],
            agent=self.writer()
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.researcher(),
                self.writer()
            ],
            tasks=[
                self.research_task(),
                self.blog_task()
            ],
            memory=True,
            embedder={
                "provider": "google-generativeai",
                "config": {
                    "model": "gemini-embedding-001",
                    "api_key": os.environ["GOOGLE_API_KEY"]
                }
            }
        )


if __name__ == "__main__":
    blog_crew = BlogCrew()

    blog_crew.crew().kickoff(
        inputs={
            "topic": "The future of electrical vehicles"
        }
    )


