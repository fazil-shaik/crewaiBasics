from pydantic import BaseModel, Field


class ChaosRequest(BaseModel):

    situation: str = Field(
        min_length=5,
        max_length=2000
    )


class ChaosResponse(BaseModel):

    situation: str

    chaos_score: int

    severity: str

    report: str