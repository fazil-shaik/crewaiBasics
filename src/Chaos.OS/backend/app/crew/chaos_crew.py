from crewai import Crew, Process

from .agents import (
    create_chaos_agent,
    create_survival_agent,
    create_drama_agent,
    create_reality_agent
)

from .tasks import (
    chaos_task,
    survival_task,
    drama_task,
    reality_task
)


def create_chaos_crew(situation):

    chaos_agent = create_chaos_agent()

    survival_agent = create_survival_agent()

    drama_agent = create_drama_agent()

    reality_agent = create_reality_agent()


    task1 = chaos_task(
        chaos_agent,
        situation
    )

    task2 = survival_task(
        survival_agent,
        situation
    )

    task3 = drama_task(
        drama_agent,
        situation
    )

    task4 = reality_task(
        reality_agent,
        situation
    )


    crew = Crew(

        agents=[
            chaos_agent,
            survival_agent,
            drama_agent,
            reality_agent
        ],

        tasks=[
            task1,
            task2,
            task3,
            task4
        ],

        process=Process.sequential,

        verbose=True
    )


    return crew