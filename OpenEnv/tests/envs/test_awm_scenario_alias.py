"""Regression tests for generated AWM response-model aliases."""

from envs.agent_world_model_env.server.scenario_manager import (
    _fix_generated_code_errors,
)


def test_metadata_alias_is_not_duplicated() -> None:
    code = """\
class TicketResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    metadata: Optional[str] = Field(None, validation_alias="metadata_", serialization_alias="metadata")
"""

    fixed = _fix_generated_code_errors(code)

    compile(fixed, "generated.py", "exec")
    assert fixed.count("validation_alias=") == 1
    assert 'validation_alias="metadata_"' in fixed

