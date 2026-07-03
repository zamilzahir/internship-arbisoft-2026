"""
Task 4: Validation tests that intentionally fail when LLM output does not
conform to the expected schema.

These tests feed deliberately broken/malformed data into PolicyFact and
assert that Pydantic correctly rejects it (raises ValidationError). This
proves the validation step in the pipeline actually catches bad LLM output
rather than silently accepting it.
"""
import sys
from pathlib import Path
import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from schemas import PolicyFact


class TestValidPolicyFact:
    """Sanity check: well-formed data should NOT raise."""

    def test_valid_fact_is_accepted(self):
        fact = PolicyFact(
            policy_name="leave_policy.pdf",
            category="vacation",
            fact="Employees accrue 15 days of paid vacation per year.",
            key_number=15,
            source_grounded=True,
        )
        assert fact.key_number == 15
        assert fact.source_grounded is True

    def test_valid_fact_with_null_key_number(self):
        """key_number is Optional -- None should be accepted."""
        fact = PolicyFact(
            policy_name="onboarding_policy.pdf",
            category="buddy program",
            fact="New hires are assigned a buddy from a different team.",
            key_number=None,
            source_grounded=True,
        )
        assert fact.key_number is None


class TestMissingFields:
    """Missing required fields should always fail validation."""

    def test_missing_policy_name_fails(self):
        with pytest.raises(ValidationError):
            PolicyFact(
                category="vacation",
                fact="Some fact.",
                key_number=15,
                source_grounded=True,
            )

    def test_missing_fact_fails(self):
        with pytest.raises(ValidationError):
            PolicyFact(
                policy_name="leave_policy.pdf",
                category="vacation",
                key_number=15,
                source_grounded=True,
            )

    def test_missing_source_grounded_fails(self):
        """source_grounded has no default -- must always be explicitly stated."""
        with pytest.raises(ValidationError):
            PolicyFact(
                policy_name="leave_policy.pdf",
                category="vacation",
                fact="Some fact.",
                key_number=15,
            )

    def test_completely_empty_input_fails(self):
        with pytest.raises(ValidationError):
            PolicyFact()


class TestWrongTypes:
    """Fields with the wrong data type should fail validation."""

    def test_key_number_as_string_fails(self):
        """key_number must be a float/int, not a string like 'fifteen'."""
        with pytest.raises(ValidationError):
            PolicyFact(
                policy_name="leave_policy.pdf",
                category="vacation",
                fact="Some fact.",
                key_number="fifteen",
                source_grounded=True,
            )

    def test_source_grounded_as_string_fails(self):
        """source_grounded must be a real boolean, not the string 'true'."""
        with pytest.raises(ValidationError):
            PolicyFact(
                policy_name="leave_policy.pdf",
                category="vacation",
                fact="Some fact.",
                key_number=15,
                source_grounded="yes probably",
            )

    def test_policy_name_as_number_fails(self):
        with pytest.raises(ValidationError):
            PolicyFact(
                policy_name=12345,
                category="vacation",
                fact="Some fact.",
                key_number=15,
                source_grounded=True,
            )


class TestCustomValidators:
    """Custom business-logic validators we defined ourselves in schemas.py."""

    def test_category_too_long_fails(self):
        """Our custom validator rejects categories longer than 4 words."""
        with pytest.raises(ValidationError):
            PolicyFact(
                policy_name="leave_policy.pdf",
                category="this is way too many words for a category",
                fact="Some fact.",
                key_number=15,
                source_grounded=True,
            )

    def test_empty_fact_string_fails(self):
        """Our custom validator rejects a fact that's just whitespace."""
        with pytest.raises(ValidationError):
            PolicyFact(
                policy_name="leave_policy.pdf",
                category="vacation",
                fact="   ",
                key_number=15,
                source_grounded=True,
            )

    def test_short_category_is_accepted(self):
        """Boundary check: exactly 4 words should pass (not fail off-by-one)."""
        fact = PolicyFact(
            policy_name="leave_policy.pdf",
            category="paid vacation time off",
            fact="Some fact.",
            key_number=15,
            source_grounded=True,
        )
        assert fact.category == "paid vacation time off"


class TestMalformedLLMOutputScenarios:
    """
    Simulates realistic ways an LLM might produce malformed JSON, as if
    parsed from a real (bad) response, to prove the pipeline's validation
    step would catch each one.
    """

    def test_llm_returns_extra_hallucinated_field_but_missing_required_one(self):
        """LLM invents a field we never asked for, while omitting a required one."""
        malformed = {
            "policy_name": "leave_policy.pdf",
            "category": "vacation",
            "confidence_score": 0.95,  # not part of our schema -- ignored, not the issue
            "key_number": 15,
            "source_grounded": True,
            # "fact" is missing entirely -- this SHOULD fail
        }
        with pytest.raises(ValidationError):
            PolicyFact(**malformed)

    def test_llm_returns_key_number_with_units_as_string(self):
        """LLM writes '15 days' instead of the plain number 15."""
        malformed = {
            "policy_name": "leave_policy.pdf",
            "category": "vacation",
            "fact": "Employees get 15 days of vacation.",
            "key_number": "15 days",
            "source_grounded": True,
        }
        with pytest.raises(ValidationError):
            PolicyFact(**malformed)
