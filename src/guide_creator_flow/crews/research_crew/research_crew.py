# src/guide_creator_flow/crews/research_crew/research_crew.py

from crewai import Agent, Crew, Task
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

# Define structured output model for research
class ResearchOutput(BaseModel):
    raw_research: str = Field(description="The raw research report")
    synthesized_findings: str = Field(description="Synthesized and categorized findings")
    key_takeaways: list = Field(default_factory=list, description="Most important takeaways")
    beginner_summary: str = Field(description="Simple 2-3 sentence summary for beginners")

@CrewBase
class ResearchCrew:
    """Research Crew for gathering and synthesizing information on any topic"""
    
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    
    def __init__(self):
        self._crew = None
    
    @agent
    def research_analyst(self) -> Agent:
        """Agent responsible for gathering raw research"""
        return Agent(
            config=self.agents_config["research_analyst"],
            verbose=True,
            allow_delegation=False,
        )
    
    @agent
    def research_synthesizer(self) -> Agent:
        """Agent responsible for synthesizing research into structured format"""
        return Agent(
            config=self.agents_config["research_synthesizer"],
            verbose=True,
            allow_delegation=False,
        )
    
    @task
    def gather_research_task(self) -> Task:
        """Task to gather raw research on the topic"""
        return Task(
            config=self.tasks_config["gather_research_task"],
            agent=self.research_analyst(),
        )
    
    @task
    def synthesize_research_task(self) -> Task:
        """Task to synthesize and organize research findings"""
        return Task(
            config=self.tasks_config["synthesize_research_task"],
            agent=self.research_synthesizer(),
        )
    
    @crew
    def crew(self) -> Crew:
        """Create the research crew with sequential task execution"""
        return Crew(
            agents=[self.research_analyst(), self.research_synthesizer()],
            tasks=[self.gather_research_task(), self.synthesize_research_task()],
            verbose=True,
            process="sequential",  # Gather first, then synthesize
        )
    
    def kickoff(self, inputs: Dict[str, Any]) -> Any:
        """Run the research crew with given inputs"""
        crew_instance = self.crew()
        result = crew_instance.kickoff(inputs=inputs)
        return result