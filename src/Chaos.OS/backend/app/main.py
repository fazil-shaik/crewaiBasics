from fastapi import FastAPI

from app.models.schemas import (
    ChaosRequest,
    ChaosResponse
)

from app.flow.chaos_flow import ChaosFlow


app = FastAPI(
    title="CHAOS.OS",
    description="AI-powered disaster prediction system",
    version="1.0.0"
)


@app.get("/")
def home():

    return {
        "message": "🚨 CHAOS.OS is alive",
        "status": "operational"
    }


@app.post(
    "/api/chaos",
    response_model=ChaosResponse
)
def analyze_chaos(
    request: ChaosRequest
):

    flow = ChaosFlow()

    flow.state.situation = (
        request.situation
    )

    result = flow.kickoff()

    return ChaosResponse(

        situation=request.situation,

        chaos_score=flow.state.chaos_score,

        severity=flow.state.severity,

        report=flow.state.final_report
    )