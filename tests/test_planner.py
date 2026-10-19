import json
from datetime import date
from types import SimpleNamespace

import pytest

from models import PlanningContext
from planner import PlannerConfig, PlannerError, TaskPlanner
from validator import PlanValidationError


def test_demo_generation_creates_distinct_goal_plans():
    planner = TaskPlanner(PlannerConfig(demo_mode=True))
    workshop = planner.generate_plan("Organize a two-day technical workshop for MCA students.")
    exam = planner.generate_plan("Prepare for my Python examination in seven days.")

    assert len(workshop.tasks) >= 8
    assert "Select topics and speakers" in {task.title for task in workshop.tasks}
    assert "Review the syllabus" not in {task.title for task in workshop.tasks}
    assert "Review the syllabus" in {task.title for task in exam.tasks}
    assert workshop.tasks != exam.tasks


def test_subject_input_generates_separate_subject_specific_tracks_without_goal():
    planner = TaskPlanner(PlannerConfig(demo_mode=True))
    plan = planner.generate_plan(
        "",
        PlanningContext(subjects=["Python", "DBMS", "Data Structures"]),
    )
    titles = {task.title for task in plan.tasks}

    assert plan.original_goal == "Prepare for these subjects: Python, DBMS, Data Structures"
    assert "Study core Python topics" in titles
    assert "Practice DBMS" in titles
    assert "Solve past questions in Data Structures" in titles
    assert "Review the Python syllabus" in titles
    assert plan.total_estimated_effort_hours == sum(
        task.estimated_duration_hours for task in plan.tasks
    )
    tasks_by_id = {task.task_id: task for task in plan.tasks}
    assert "syllabus_python" in tasks_by_id["concepts_python"].dependencies
    assert "practice_dbms" in tasks_by_id["questions_dbms"].dependencies


def test_demo_subject_plan_handles_unknown_subjects_without_generic_goal_plan():
    planner = TaskPlanner(PlannerConfig(demo_mode=True))
    plan = planner.generate_plan(
        "Prepare for exams",
        PlanningContext(subjects=["Environmental Studies"]),
    )
    assert any("Environmental Studies" in task.title for task in plan.tasks)
    assert any("Environmental Studies syllabus" in task.description for task in plan.tasks)


def test_demo_plan_has_valid_dependencies_and_effort():
    planner = TaskPlanner(PlannerConfig(demo_mode=True))
    plan = planner.generate_plan("Organize a two-day technical workshop.")
    ids = {task.task_id for task in plan.tasks}
    assert all(set(task.dependencies) <= ids for task in plan.tasks)
    assert plan.total_estimated_effort_hours == sum(
        task.estimated_duration_hours for task in plan.tasks
    )


def test_deadline_schedule_respects_dependency_order_and_parallel_tasks():
    planner = TaskPlanner(PlannerConfig(demo_mode=True))
    plan = planner.generate_plan(
        "Organize a two-day technical workshop.",
        PlanningContext(deadline=date(2030, 1, 1), available_hours_per_day=4),
    )
    schedule = {item.task_id: item for item in plan.schedule}
    assert schedule["objectives"].start_day == 1
    assert schedule["budget"].start_day == 1
    assert schedule["topics_speakers"].start_day == 1
    assert schedule["conduct"].start_day >= schedule["equipment"].end_day


def test_demo_revision_adds_task_and_preserves_existing_work():
    planner = TaskPlanner(PlannerConfig(demo_mode=True))
    original = planner.generate_plan("Organize a technical workshop.")
    revised = planner.revise_plan(original, "Add a hands-on Python session.")
    assert len(revised.tasks) == len(original.tasks) + 1
    assert {task.task_id for task in original.tasks} <= {task.task_id for task in revised.tasks}
    assert revised.total_estimated_effort_hours == original.total_estimated_effort_hours + 2


def test_demo_revision_removes_requested_task_and_updates_dependencies():
    planner = TaskPlanner(PlannerConfig(demo_mode=True))
    original = planner.generate_plan("Organize a technical workshop.")
    revised = planner.revise_plan(original, "Remove the feedback task.")
    assert "feedback" not in {task.task_id for task in revised.tasks}


def test_revision_rejects_ambiguous_removal_without_changing_input_plan():
    planner = TaskPlanner(PlannerConfig(demo_mode=True))
    original = planner.generate_plan("Organize a technical workshop.")
    before = original.model_dump()
    with pytest.raises(PlannerError, match="could not identify"):
        planner.revise_plan(original, "Remove something.")
    assert original.model_dump() == before


def test_api_mode_parses_mock_response_and_recomputes_effort():
    payload = {
        "original_goal": "wrong model goal",
        "summary": "A concise generated plan.",
        "assumptions": [],
        "tasks": [
            {
                "task_id": "plan",
                "title": "Plan the work",
                "description": "Write down the work needed to reach the goal.",
                "priority": "High",
                "estimated_duration_hours": 2,
                "dependencies": [],
                "expected_outcome": "A written plan.",
                "status": "Pending",
            }
        ],
        "milestones": [],
        "total_estimated_effort_hours": 400,
        "schedule": [],
    }
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response)
        )
    )
    planner = TaskPlanner(
        PlannerConfig(demo_mode=False, api_key="test", model="mock-model"),
        client=client,
    )
    plan = planner.generate_plan("Build a small planning tool.")
    assert plan.original_goal == "Build a small planning tool."
    assert plan.total_estimated_effort_hours == 2


def test_api_mode_handles_malformed_json():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response)
        )
    )
    planner = TaskPlanner(
        PlannerConfig(demo_mode=False, api_key="test"),
        client=client,
    )
    with pytest.raises(PlannerError, match="malformed"):
        planner.generate_plan("Build a small planning tool.")


def test_api_mode_rejects_silently_discarded_tasks():
    planner = TaskPlanner(PlannerConfig(demo_mode=True))
    original = planner.generate_plan("Organize a technical workshop.")
    from models import TaskPlan

    dropped = original.model_copy(update={"tasks": original.tasks[:-1]})
    with pytest.raises(PlanValidationError, match="dropped existing tasks"):
        planner._ensure_revision_preserves_tasks(original, dropped, "Add a hands-on demo.")
