"""
Task 3: Pydantic schema for structured LLM output.

This defines the exact shape we require every LLM response to match.
If the LLM's JSON doesn't fit this shape (missing field, wrong type, extra
constraint violated), Pydantic raises a ValidationError -- that's the
"validation" step in LLM -> JSON -> Pydantic validation -> Save output.
"""
from pydantic import BaseModel, Field, field_validator


class PolicyFact(BaseModel):
    """A single fact extracted from a company policy document."""

    policy_name: str = Field(..., description="Which policy document this fact is from")
    category: str = Field(..., description="Short topic label, e.g. 'vacation', 'security'")
    fact: str = Field(..., description="The specific fact/rule, in one sentence")
    key_number: float | None = Field(
        None, description="The single most important number in this fact, if any (e.g. 15 for '15 days')"
    )
    source_grounded: bool = Field(
        ..., description="True only if this fact is directly supported by the retrieved context"
    )

    @field_validator("category")
    @classmethod
    def category_must_be_short(cls, v: str) -> str:
        v = v.strip().lower()

        if len(v.split()) > 4:
            raise ValueError("category must be a short label (<=4 words), not a sentence")

        return v

    @field_validator("fact")
    @classmethod
    def fact_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("fact cannot be empty")
        return v