"""Validated data models for task plans."""

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Priority(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class TaskStatus(str, Enum):
    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"


class Task(BaseModel):
    """One actionable item in a plan."""

    model_config = ConfigDict(str_strip_whitespace=True)

    task_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,49}$")
    title: str = Field(min_length=3, max_length=160)
    description: str = Field(min_length=5, max_length=2000)
    priority: Priority
    estimated_duration_hours: float = Field(gt=0, le=1000)
    dependencies: list[str] = Field(default_factory=list)
    expected_outcome: str = Field(min_length=3, max_length=1000)
    status: TaskStatus = TaskStatus.PENDING

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower()
            mapping = {"high": Priority.HIGH, "medium": Priority.MEDIUM, "low": Priority.LOW}
            if normalized in mapping:
                return mapping[normalized]
        return value

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower().replace("_", " ")
            mapping = {
                "pending": TaskStatus.PENDING,
                "in progress": TaskStatus.IN_PROGRESS,
                "completed": TaskStatus.COMPLETED,
            }
            if normalized in mapping:
                return mapping[normalized]
        return value


class Milestone(BaseModel):
    title: str = Field(min_length=3, max_length=160)
    description: str = Field(min_length=3, max_length=1000)
    task_ids: list[str] = Field(default_factory=list)
    target_date: Optional[date] = None


class ScheduleItem(BaseModel):
    task_id: str
    start_day: int = Field(ge=1)
    end_day: int = Field(ge=1)
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class TaskPlan(BaseModel):
    """Complete, user-facing plan returned by the planner."""

    model_config = ConfigDict(str_strip_whitespace=True)

    original_goal: str = Field(min_length=5, max_length=2000)
    summary: str = Field(min_length=5, max_length=2000)
    assumptions: list[str] = Field(default_factory=list)
    tasks: list[Task] = Field(min_length=1)
    milestones: list[Milestone] = Field(default_factory=list)
    total_estimated_effort_hours: float = Field(gt=0)
    schedule: list[ScheduleItem] = Field(default_factory=list)


class PlanningContext(BaseModel):
    """Optional constraints supplied alongside a goal."""

    subjects: list[str] = Field(default_factory=list, max_length=10)
    deadline: Optional[date] = None
    available_hours_per_day: float = Field(default=4.0, gt=0, le=24)
    budget: Optional[str] = Field(default=None, max_length=300)
    constraints: Optional[str] = Field(default=None, max_length=2000)
    planning_style: Optional[str] = Field(default=None, max_length=300)

    @field_validator("subjects")
    @classmethod
    def normalize_subjects(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for subject in value:
            subject = subject.strip()
            if not subject:
                continue
            if len(subject) > 100:
                raise ValueError("Each subject must be 100 characters or fewer.")
            key = subject.casefold()
            if key not in seen:
                normalized.append(subject)
                seen.add(key)
        if len(normalized) > 10:
            raise ValueError("Enter no more than 10 subjects.")
        return normalized

    def as_prompt_data(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude_none=True)
