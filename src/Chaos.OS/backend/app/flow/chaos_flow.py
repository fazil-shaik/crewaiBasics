from crewai.flow.flow import Flow, start, listen, router

from pydantic import BaseModel

from app.crew.chaos_crew import create_chaos_crew

from app.services.scoring import (
    get_severity,
    clamp_score
)


class ChaosState(BaseModel):

    situation: str = ""

    chaos_score: int = 0

    severity: str = ""

    crew_report: str = ""

    final_report: str = ""


class ChaosFlow(Flow[ChaosState]):

    @start()
    def receive_situation(self):

        print("\n🚨 CHAOS.OS STARTED")

        print(
            f"Situation: {self.state.situation}"
        )

        return self.state.situation


    @listen(receive_situation)
    def validate_situation(self, situation):

        print("\n🔍 Validating situation...")

        if not situation.strip():

            raise ValueError(
                "Situation cannot be empty"
            )

        return situation


    @listen(validate_situation)
    def run_chaos_crew(self, situation):

        print("\n🤖 Activating CHAOS CREW...")

        crew = create_chaos_crew(
            situation
        )

        result = crew.kickoff()

        self.state.crew_report = str(result)

        return result


    @listen(run_chaos_crew)
    def calculate_score(self, result):

        print("\n📊 Calculating chaos score...")

        text = str(result)

        score = 50

        # Simple initial scoring.
        # Later we can replace this with
        # structured Gemini output.

        keywords = {
            "urgent": 10,
            "emergency": 10,
            "deadline": 8,
            "tomorrow": 8,
            "failed": 7,
            "crisis": 10,
            "lost": 5,
            "broken": 5,
            "panic": 8
        }

        lower_text = text.lower()

        for keyword, points in keywords.items():

            if keyword in lower_text:

                score += points


        score = clamp_score(score)

        self.state.chaos_score = score

        self.state.severity = get_severity(
            score
        )

        return score


    @router(calculate_score)
    def choose_path(self, score):

        if score >= 80:

            return "emergency"

        return "normal"


    @listen("emergency")
    def emergency_plan(self):

        print(
            "\n🚑 CHAOS LEVEL CRITICAL"
        )

        self.state.final_report = (
            f"""
🚨 CHAOS.OS EMERGENCY MODE

CHAOS SCORE: {self.state.chaos_score}/100

{self.state.severity}

The situation is officially too chaotic
for normal advice.

IMMEDIATE ACTION:

1. Stop making the situation worse.
2. Identify the biggest real-world risk.
3. Handle that problem first.
4. Execute the survival plan from the crew.
5. Keep a backup plan ready.

--------------------------------

AI CREW REPORT

{self.state.crew_report}
"""
        )

        return self.state.final_report


    @listen("normal")
    def normal_plan(self):

        print(
            "\n😌 Chaos is manageable."
        )

        self.state.final_report = (
            f"""
🔥 CHAOS.OS REPORT

CHAOS SCORE: {self.state.chaos_score}/100

{self.state.severity}

You are not completely doomed.

--------------------------------

AI CREW REPORT

{self.state.crew_report}
"""
        )

        return self.state.final_report