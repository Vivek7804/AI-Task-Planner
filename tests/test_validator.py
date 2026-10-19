import pytest
from pydantic import ValidationError

from models import Task, TaskPlan
from validator import PlanValidationError, parse_and_validate_plan, parse_subjects, validate_goal


def valid_payload():
    return {
        "original_goal": "Plan a practical project.",
        "summary": "Complete project in a safe sequence.",
        "assumptions": [],
        "tasks": [
            {
                "task_id": "first",
                "title": "Define the requirements",
                "description": "Write down the user and system requirements.",
                "priority": "high",
                "estimated_duration_hours": 2,
                "dependencies": [],
                "expected_outcome": "A reviewed requirements list.",
                "status": "Pending",
            },
            {
                "task_id": "second",
                "title": "Review the requirements",
                "description": "Check the requirements with the project stakeholders.",
                "priority": "Medium",
                "estimated_duration_hours": 1.5,
                "dependencies": ["first"],
                "expected_outcome": "Approved requirements.",
                "status": "Pending",
            },
        ],
        "milestones": [
            {
                "title": "Requirements approved",
                "description": "Requirements have been reviewed.",
                "task_ids": ["first", "second"],
                "target_date": None,
            }
        ],
        "total_estimated_effort_hours": 3.5,
        "schedule": [],
    }


def test_goal_validation_normalizes_and_rejects_empty():
    assert validate_goal("  Plan my project  ") == "Plan my project"
    with pytest.raises(ValueError, match="more detailed"):
        validate_goal(" ")


def test_subject_parser_splits_deduplicates_and_limits_subject_count():
    assert parse_subjects("Python, DBMS\npython; Data Structures") == [
        "Python",
        "DBMS",
        "Data Structures",
    ]
    with pytest.raises(ValueError, match="no more than 10"):
        parse_subjects(", ".join(f"Subject {index}" for index in range(11)))


def test_task_model_rejects_invalid_priority_and_duration():
    payload = valid_payload()["tasks"][0]
    payload["priority"] = "Urgent"
    with pytest.raises(ValidationError):
        Task.model_validate(payload)
    payload["priority"] = "High"
    payload["estimated_duration_hours"] = 0
    with pytest.raises(ValidationError):
        Task.model_validate(payload)


def test_plan_validation_accepts_case_insensitive_priority_and_calculates_effort():
    plan = parse_and_validate_plan(valid_payload())
    assert plan.tasks[0].priority.value == "High"
    assert plan.total_estimated_effort_hours == 3.5


def test_plan_validation_rejects_unknown_dependency():
    payload = valid_payload()
    payload["tasks"][1]["dependencies"] = ["missing"]
    with pytest.raises(PlanValidationError, match="unknown task"):
        parse_and_validate_plan(payload)


def test_plan_validation_rejects_circular_dependency():
    payload = valid_payload()
    payload["tasks"][0]["dependencies"] = ["second"]
    with pytest.raises(PlanValidationError, match="circular"):
        parse_and_validate_plan(payload)


def test_plan_validation_requires_prerequisites_to_appear_first():
    payload = valid_payload()
    payload["tasks"][0], payload["tasks"][1] = payload["tasks"][1], payload["tasks"][0]
    with pytest.raises(PlanValidationError, match="listed after"):
        parse_and_validate_plan(payload)


def test_plan_validation_rejects_duplicate_ids_and_effort_mismatch():
    payload = valid_payload()
    payload["tasks"][1]["task_id"] = "first"
    with pytest.raises(PlanValidationError, match="unique"):
        parse_and_validate_plan(payload)

    payload = valid_payload()
    payload["total_estimated_effort_hours"] = 99
    with pytest.raises(PlanValidationError, match="sum"):
        parse_and_validate_plan(payload)


def test_task_plan_requires_at_least_one_task():
    payload = valid_payload()
    payload["tasks"] = []
    with pytest.raises(ValidationError):
        TaskPlan.model_validate(payload)
