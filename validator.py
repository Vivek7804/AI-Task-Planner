"""Input and task-plan validation helpers."""

from collections import defaultdict, deque
from math import isclose
import re

from pydantic import ValidationError

from models import TaskPlan


class PlanValidationError(ValueError):
    """Raised when a plan is structurally invalid or internally inconsistent."""


def validate_goal(goal: str) -> str:
    """Return a normalized goal or raise a useful validation error."""
    if not isinstance(goal, str):
        raise ValueError("Enter your goal as text.")
    normalized = goal.strip()
    if len(normalized) < 5:
        raise ValueError("Enter a more detailed goal (at least 5 characters).")
    if len(normalized) > 2000:
        raise ValueError("The goal must be 2,000 characters or fewer.")
    if not any(character.isalnum() for character in normalized):
        raise ValueError("The goal must include words or numbers.")
    return normalized


def parse_subjects(raw_subjects: str) -> list[str]:
    """Parse a comma-, semicolon-, or newline-separated subject list."""
    if not isinstance(raw_subjects, str):
        raise ValueError("Enter subjects as text.")
    subjects: list[str] = []
    seen: set[str] = set()
    for candidate in re.split(r"[,;\n]+", raw_subjects):
        subject = candidate.strip()
        if subject and subject.casefold() not in seen:
            if len(subject) > 100:
                raise ValueError("Each subject must be 100 characters or fewer.")
            subjects.append(subject)
            seen.add(subject.casefold())
    if len(subjects) > 10:
        raise ValueError("Enter no more than 10 subjects.")
    return subjects


def parse_and_validate_plan(payload: object) -> TaskPlan:
    """Validate model data and enforce cross-task plan invariants."""
    try:
        plan = TaskPlan.model_validate(payload)
    except ValidationError as exc:
        raise PlanValidationError(f"The plan data is invalid: {exc}") from exc
    validate_plan(plan)
    return plan


def validate_plan(plan: TaskPlan) -> None:
    task_ids = [task.task_id for task in plan.tasks]
    if len(task_ids) != len(set(task_ids)):
        raise PlanValidationError("Task IDs must be unique.")

    task_by_id = {task.task_id: task for task in plan.tasks}
    task_positions = {task_id: index for index, task_id in enumerate(task_ids)}
    graph: dict[str, list[str]] = defaultdict(list)
    indegree = {task_id: 0 for task_id in task_ids}

    for task in plan.tasks:
        if len(task.dependencies) != len(set(task.dependencies)):
            raise PlanValidationError(f"Task '{task.task_id}' has duplicate dependencies.")
        for dependency in task.dependencies:
            if dependency not in task_by_id:
                raise PlanValidationError(
                    f"Task '{task.task_id}' depends on unknown task '{dependency}'."
                )
            if dependency == task.task_id:
                raise PlanValidationError(f"Task '{task.task_id}' cannot depend on itself.")
            graph[dependency].append(task.task_id)
            indegree[task.task_id] += 1

    ready = deque(task_id for task_id, count in indegree.items() if count == 0)
    visited = 0
    while ready:
        current = ready.popleft()
        visited += 1
        for dependent in graph[current]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
    if visited != len(task_ids):
        raise PlanValidationError("The plan contains a circular task dependency.")

    for task in plan.tasks:
        for dependency in task.dependencies:
            if task_positions[dependency] >= task_positions[task.task_id]:
                raise PlanValidationError(
                    "Tasks must be listed after their prerequisites. "
                    f"Move '{task.task_id}' after '{dependency}'."
                )

    for milestone in plan.milestones:
        unknown = set(milestone.task_ids) - set(task_ids)
        if unknown:
            raise PlanValidationError(
                f"Milestone '{milestone.title}' refers to unknown tasks: "
                f"{', '.join(sorted(unknown))}."
            )

    calculated_effort = sum(task.estimated_duration_hours for task in plan.tasks)
    if not isclose(plan.total_estimated_effort_hours, calculated_effort, rel_tol=1e-4, abs_tol=0.01):
        raise PlanValidationError(
            "Total estimated effort must equal the sum of all task durations "
            f"({calculated_effort:g} hours)."
        )
