from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

from crewai_tools import ExaSearchTool, ScrapeWebsiteTool, DirectoryReadTool, FileWriterTool, FileReadTool

from dotenv import load_dotenv
load_dotenv()


@CrewBase
class BlogCrew():
    """Blog writing crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"


    @agent
    def researcher(self)->Agent:
        return Agent(
            config = self.agents_config['research_agent'],
            tools=[ExaSearchTool()],
            verbose=True
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config['writer_agent'], # type: ignore[index]
            verbose=True,
            tools=[DirectoryReadTool(),FileWriterTool()],
            
        )


    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
            agent = self.researcher()
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
