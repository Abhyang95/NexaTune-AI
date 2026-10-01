from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Customer support message",
    )


class GenerateResponse(BaseModel):
    response: str
    model: str
    guardrail_flagged: bool = False
    guardrail_matches: list[str] = []
    guardrail_categories: list[str] = []
    original_response: str | None = None