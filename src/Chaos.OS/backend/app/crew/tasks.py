from crewai import Task


def chaos_task(agent, situation):

    return Task(

        description=f"""
        Analyze this user's situation:

        "{situation}"

        Identify:

        1. The top 5 things that could go wrong.
        2. Realistic risks.
        3. Funny potential disasters.
        4. The most dangerous or urgent issue.
        5. A rough chaos severity from 0 to 100.

        Be entertaining but useful.
        """,

        expected_output="""
        A structured chaos analysis containing:

        - Top risks
        - Funny disasters
        - Most urgent problem
        - Preliminary chaos score
        """,

        agent=agent
    )


def survival_task(agent, situation):

    return Task(

        description=f"""
        The user is experiencing this situation:

        "{situation}"

        Create a practical survival plan.

        Include:

        1. Immediate actions.
        2. Short-term actions.
        3. Things the user should avoid.
        4. Backup plan.
        5. Emergency plan.

        Do not make the advice unnecessarily complicated.
        """,

        expected_output="""
        A practical step-by-step survival plan.
        """,

        agent=agent
    )


def drama_task(agent, situation):

    return Task(

        description=f"""
        Turn this situation into a hilarious disaster movie:

        "{situation}"

        Produce:

        - Dramatic opening
        - 3 ridiculous predictions
        - A dramatic final scene

        Keep it funny and harmless.
        """,

        expected_output="""
        A short comedic disaster narrative.
        """,

        agent=agent
    )


def reality_task(agent, situation):

    return Task(

        description=f"""
        Analyze this situation:

        "{situation}"

        You are the final reality checker.

        Determine:

        - Which risks are realistic.
        - Which risks are exaggerated.
        - Probability of each important risk.
        - Overall chaos severity from 0 to 100.
        - Most important recommendation.
        """,

        expected_output="""
        A reality-checked risk assessment with a chaos score
        between 0 and 100.
        """,

        agent=agent
    )