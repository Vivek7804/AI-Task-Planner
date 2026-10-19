"""AI-backed and deterministic demo planning implementations."""

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Optional

from models import PlanningContext, Priority, ScheduleItem, Task, TaskPlan, TaskStatus
from prompts import SYSTEM_PROMPT, planning_prompt, revision_prompt
from validator import PlanValidationError, parse_and_validate_plan, validate_goal


class PlannerError(RuntimeError):
    """An actionable planning or AI-provider error."""


@dataclass(frozen=True)
class PlannerConfig:
    provider: str = "openai_compatible"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout_seconds: float = 30.0
    demo_mode: bool = True

    @classmethod
    def from_env(cls) -> "PlannerConfig":
        api_key = os.getenv("AI_API_KEY") or os.getenv("OPENAI_API_KEY")
        demo_setting = os.getenv("DEMO_MODE")
        demo_mode = demo_setting.strip().lower() in {"1", "true", "yes", "on"} if demo_setting else not bool(api_key)
        try:
            timeout = float(os.getenv("AI_TIMEOUT_SECONDS", "30"))
        except ValueError as exc:
            raise PlannerError("AI_TIMEOUT_SECONDS must be a positive number.") from exc
        if timeout <= 0:
            raise PlannerError("AI_TIMEOUT_SECONDS must be a positive number.")
        return cls(
            provider=os.getenv("AI_PROVIDER", "openai_compatible").strip().lower(),
            model=os.getenv("AI_MODEL", "gpt-4o-mini").strip(),
            api_key=api_key,
            base_url=os.getenv("AI_BASE_URL") or None,
            timeout_seconds=timeout,
            demo_mode=demo_mode,
        )


class TaskPlanner:
    """Create and revise plans in demo mode or through an OpenAI-compatible API."""

    def __init__(self, config: Optional[PlannerConfig] = None, client: Any = None):
        self.config = config or PlannerConfig.from_env()
        self._client = client
        if not self.config.demo_mode and not self.config.api_key and client is None:
            raise PlannerError(
                "AI_API_KEY is missing. Add a key to .env or enable DEMO_MODE=true."
            )
        if self.config.provider != "openai_compatible" and not self.config.demo_mode:
            raise PlannerError(
                "Unsupported AI_PROVIDER. Use 'openai_compatible' with your provider's base URL."
            )

    @property
    def is_demo_mode(self) -> bool:
        return self.config.demo_mode

    def generate_plan(self, goal: str, context: Optional[PlanningContext] = None) -> TaskPlan:
        planning_context = context or PlanningContext()
        if goal.strip():
            normalized_goal = validate_goal(goal)
        elif planning_context.subjects:
            normalized_goal = validate_goal(
                f"Prepare for these subjects: {', '.join(planning_context.subjects)}"
            )
        else:
            normalized_goal = validate_goal(goal)
        if self.config.demo_mode:
            plan = self._demo_plan(normalized_goal, planning_context)
        else:
            payload = self._request_json(planning_prompt(normalized_goal, planning_context.as_prompt_data()))
            payload["original_goal"] = normalized_goal
            payload["total_estimated_effort_hours"] = self._sum_effort(payload)
            payload["schedule"] = []
            plan = parse_and_validate_plan(payload)
        return self._with_schedule(plan, planning_context)

    def revise_plan(
        self,
        plan: TaskPlan,
        change_request: str,
        context: Optional[PlanningContext] = None,
    ) -> TaskPlan:
        normalized_change = validate_goal(change_request)
        planning_context = context or PlanningContext()
        if self.config.demo_mode:
            payload = self._demo_revision_payload(plan, normalized_change)
        else:
            payload = self._request_json(
                revision_prompt(
                    plan.original_goal,
                    planning_context.as_prompt_data(),
                    plan.model_dump(mode="json"),
                    normalized_change,
                )
            )
            payload["original_goal"] = plan.original_goal
            payload["total_estimated_effort_hours"] = self._sum_effort(payload)
            payload["schedule"] = []
        revised = parse_and_validate_plan(payload)
        self._ensure_revision_preserves_tasks(plan, revised, normalized_change)
        return self._with_schedule(revised, planning_context)

    def _request_json(self, user_message: str) -> dict[str, Any]:
        client = self._get_client()
        try:
            from openai import APIError
        except ImportError as exc:
            raise PlannerError(
                "The OpenAI client package is missing. Install dependencies from requirements.txt."
            ) from exc
        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                timeout=self.config.timeout_seconds,
            )
        except (APIError, TimeoutError, ConnectionError) as exc:
            raise PlannerError(
                "The AI provider request failed. Check the key, model, network, and service status."
            ) from exc
        try:
            content = response.choices[0].message.content
            if not content:
                raise ValueError("The provider returned an empty response.")
            parsed = json.loads(self._strip_json_fence(content))
            if not isinstance(parsed, dict):
                raise ValueError("The response must be a JSON object.")
            return parsed
        except (AttributeError, IndexError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
            raise PlannerError(
                "The AI returned malformed or incomplete JSON. Retry the request or use demo mode."
            ) from exc

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise PlannerError(
                    "The OpenAI client package is missing. Install dependencies from requirements.txt."
                ) from exc
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout_seconds,
            )
        return self._client

    @staticmethod
    def _strip_json_fence(content: str) -> str:
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)
        return stripped

    @staticmethod
    def _sum_effort(payload: dict[str, Any]) -> float:
        tasks = payload.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise PlannerError("The AI response must include at least one task.")
        try:
            return round(sum(float(task["estimated_duration_hours"]) for task in tasks), 2)
        except (KeyError, TypeError, ValueError) as exc:
            raise PlannerError("The AI response contains an invalid task duration.") from exc

    def _with_schedule(self, plan: TaskPlan, context: PlanningContext) -> TaskPlan:
        schedule: list[ScheduleItem] = []
        if context.deadline is not None:
            task_by_id = {task.task_id: task for task in plan.tasks}
            finish_hours: dict[str, float] = {}
            for task in plan.tasks:
                start_hour = max(
                    (finish_hours[dependency] for dependency in task.dependencies),
                    default=0.0,
                )
                finish_hours[task.task_id] = start_hour + task.estimated_duration_hours
                start_day = math.floor(start_hour / context.available_hours_per_day) + 1
                end_day = max(
                    start_day,
                    math.ceil(finish_hours[task.task_id] / context.available_hours_per_day),
                )
                schedule.append(
                    ScheduleItem(
                        task_id=task.task_id,
                        start_day=start_day,
                        end_day=end_day,
                        start_date=date.today() + timedelta(days=start_day - 1),
                        end_date=date.today() + timedelta(days=end_day - 1),
                    )
                )
            if not task_by_id:
                return plan
        return plan.model_copy(update={"schedule": schedule})

    def _demo_plan(self, goal: str, context: PlanningContext) -> TaskPlan:
        lowered = goal.lower()
        if context.subjects:
            tasks = self._subject_study_tasks(context.subjects)
            summary = (
                "A subject-wise study plan with syllabus review, focused learning, "
                "practice, and assessment for each selected subject."
            )
            assumptions = [
                "Study topics should be checked against the learner's actual syllabus.",
                "Subjects can be studied in parallel when time and energy allow.",
            ]
        elif any(word in lowered for word in ("exam", "examination", "study", "python")):
            tasks = self._exam_tasks()
            summary = "A focused seven-day Python revision plan covering concepts, practice, and a mock test."
            assumptions = ["The learner can study about 3 hours per day.", "The syllabus includes core Python topics."]
        elif any(word in lowered for word in ("workshop", "conference", "seminar")):
            tasks = self._workshop_tasks()
            summary = "A practical sequence for planning, preparing, running, and evaluating a two-day technical workshop."
            assumptions = ["The workshop is intended for MCA students.", "A suitable venue and speakers are available to book."]
        elif any(word in lowered for word in ("website", "web site", "application", "software")):
            tasks = self._website_tasks()
            summary = "A staged software project plan from requirements through testing and launch."
            assumptions = ["The project is a small web application.", "A basic deployment environment is available."]
        else:
            tasks = self._generic_tasks()
            summary = "A clear sequence of planning, execution, review, and completion tasks tailored to the stated goal."
            assumptions = ["The user will confirm any missing resources or stakeholder requirements."]

        if context.budget:
            assumptions.append(f"Budget constraint: {context.budget}.")
        if context.constraints:
            assumptions.append(f"Additional constraints: {context.constraints}.")
        if context.planning_style:
            assumptions.append(f"Planning style requested: {context.planning_style}.")
        total = round(sum(task.estimated_duration_hours for task in tasks), 2)
        plan = TaskPlan(
            original_goal=goal,
            summary=summary,
            assumptions=assumptions,
            tasks=tasks,
            milestones=self._make_milestones(tasks),
            total_estimated_effort_hours=total,
        )
        return self._with_schedule(plan, context)

    def _subject_study_tasks(self, subjects: list[str]) -> list[Task]:
        make = self._make_task
        tracks = {
            "python": (
                "variables, control flow, functions, collections, object-oriented programming, "
                "file handling, and exceptions",
                "Write short programs using functions and collections, then solve debugging exercises.",
            ),
            "data structures": (
                "arrays, linked lists, stacks, queues, trees, graphs, hashing, and complexity",
                "Implement representative structures and compare search, insertion, and traversal costs.",
            ),
            "algorithms": (
                "complexity analysis, searching, sorting, recursion, greedy methods, and graph traversal",
                "Solve timed problems and explain the correctness and complexity of each approach.",
            ),
            "dbms": (
                "relational models, keys, normalization, SQL, transactions, and indexing",
                "Write SQL queries and practice schema design, normalization, and transaction scenarios.",
            ),
            "database management systems": (
                "relational models, keys, normalization, SQL, transactions, and indexing",
                "Write SQL queries and practice schema design, normalization, and transaction scenarios.",
            ),
            "operating systems": (
                "processes, CPU scheduling, synchronization, memory management, and file systems",
                "Work through scheduling and memory problems and explain synchronization scenarios.",
            ),
            "computer networks": (
                "OSI and TCP/IP layers, addressing, routing, transport protocols, and network services",
                "Practice subnetting and trace how sample data travels through the network stack.",
            ),
            "mathematics": (
                "definitions, core theorems, worked examples, and the problem types in the syllabus",
                "Solve a graded set of problems and record the reasoning for each solution.",
            ),
            "machine learning": (
                "data preparation, supervised and unsupervised learning, model evaluation, and overfitting",
                "Work through small examples and interpret metrics, model outputs, and common failure cases.",
            ),
            "computer organization": (
                "number systems, instruction execution, CPU components, memory hierarchy, and I/O",
                "Solve representation and performance problems and trace instruction execution.",
            ),
        }

        def slug_for(subject: str, index: int) -> str:
            slug = re.sub(r"[^a-z0-9]+", "_", subject.casefold()).strip("_")[:28]
            return slug or f"subject_{index}"

        subject_slugs: list[tuple[str, str]] = []
        seen_slugs: set[str] = set()
        for index, subject in enumerate(subjects, start=1):
            base_slug = slug_for(subject, index)
            slug = base_slug
            suffix = 2
            while slug in seen_slugs:
                slug = f"{base_slug[:24]}_{suffix}"
                suffix += 1
            seen_slugs.add(slug)
            subject_slugs.append((subject, slug))

        syllabus_ids = [f"syllabus_{slug}" for _, slug in subject_slugs]
        tasks = [
            make(
                f"syllabus_{slug}",
                f"Review the {subject} syllabus",
                f"List the units, learning outcomes, exam weight, and available resources for {subject}.",
                Priority.HIGH,
                1,
                None,
                f"A topic checklist for {subject}.",
            )
            for subject, slug in subject_slugs
        ]
        tasks.append(
            make(
                "study_schedule",
                "Create a study timetable across subjects",
                "Allocate daily sessions based on exam dates, subject difficulty, and the syllabus workload.",
                Priority.HIGH,
                1,
                syllabus_ids,
                "A balanced timetable with time reserved for practice and revision in every subject.",
            )
        )

        for subject, slug in subject_slugs:
            normalized_subject = re.sub(r"\s+", " ", subject.casefold()).strip()
            topics, practice = tracks.get(
                normalized_subject,
                (
                    f"the core concepts, terminology, methods, and units listed in the {subject} syllabus",
                    f"Answer representative {subject} questions, explain each solution, and review mistakes.",
                ),
            )
            syllabus_id = f"syllabus_{slug}"
            concept_id = f"concepts_{slug}"
            practice_id = f"practice_{slug}"
            questions_id = f"questions_{slug}"
            tasks.extend(
                [
                    make(
                        concept_id,
                        f"Study core {subject} topics",
                        f"Learn and summarize {topics}. Use the syllabus to choose the exact depth and order.",
                        Priority.HIGH,
                        4,
                        ["study_schedule", syllabus_id],
                        f"Concise notes and explanations for the main {subject} topics.",
                    ),
                    make(
                        practice_id,
                        f"Practice {subject}",
                        practice,
                        Priority.HIGH,
                        3,
                        [concept_id],
                        f"Completed practice set with mistakes recorded for {subject}.",
                    ),
                    make(
                        questions_id,
                        f"Solve past questions in {subject}",
                        f"Attempt subject-wise past-paper or sample questions under timed conditions; review every error.",
                        Priority.MEDIUM,
                        2,
                        [practice_id],
                        f"A checked question set and weak-topic list for {subject}.",
                    ),
                ]
            )

        question_ids = [f"questions_{slug}" for _, slug in subject_slugs]
        tasks.extend(
            [
                make(
                    "mock_assessment",
                    "Take a combined mock assessment",
                    "Attempt a timed assessment covering all selected subjects and compare results with the syllabus checklist.",
                    Priority.HIGH,
                    2,
                    question_ids,
                    "A score breakdown showing strengths and gaps by subject.",
                ),
                make(
                    "final_revision",
                    "Revise weak topics by subject",
                    "Use the mock results to revisit the weakest topics and redo incorrect questions in each subject.",
                    Priority.HIGH,
                    3,
                    ["mock_assessment"],
                    "A final revision checklist with the highest-priority gaps addressed.",
                ),
            ]
        )
        return tasks

    @staticmethod
    def _make_task(
        task_id: str,
        title: str,
        description: str,
        priority: Priority,
        hours: float,
        dependencies: Optional[list[str]],
        outcome: str,
    ) -> Task:
        return Task(
            task_id=task_id,
            title=title,
            description=description,
            priority=priority,
            estimated_duration_hours=hours,
            dependencies=dependencies or [],
            expected_outcome=outcome,
            status=TaskStatus.PENDING,
        )

    def _workshop_tasks(self) -> list[Task]:
        make = self._make_task
        return [
            make("objectives", "Define workshop objectives", "Specify learning outcomes, target audience, and success measures.", Priority.HIGH, 2, None, "A concise set of measurable workshop objectives."),
            make("budget", "Prepare the budget", "Estimate venue, speaker, equipment, catering, and contingency costs.", Priority.HIGH, 3, ["objectives"], "An approved itemized budget."),
            make("topics_speakers", "Select topics and speakers", "Choose relevant sessions and invite speakers aligned with the objectives.", Priority.HIGH, 5, ["objectives"], "Confirmed sessions and speaker commitments."),
            make("venue", "Book the venue", "Compare accessible venues and reserve one with capacity for expected attendance.", Priority.HIGH, 3, ["budget"], "A confirmed venue booking."),
            make("equipment", "Arrange technical equipment", "Confirm projection, audio, internet, power, and hands-on lab requirements.", Priority.MEDIUM, 3, ["venue", "topics_speakers"], "A tested equipment checklist and support plan."),
            make("registration", "Register participants", "Publish registration information, collect attendee details, and send confirmations.", Priority.MEDIUM, 4, ["objectives", "venue"], "A confirmed participant list."),
            make("schedule", "Prepare the workshop schedule", "Sequence sessions, breaks, setup, and transitions across both days.", Priority.HIGH, 3, ["topics_speakers", "venue"], "A shared two-day agenda."),
            make("conduct", "Conduct the workshop", "Coordinate speakers, participants, facilities, and issue resolution during the event.", Priority.HIGH, 16, ["equipment", "registration", "schedule"], "The two-day workshop is delivered."),
            make("feedback", "Collect feedback", "Gather attendee and speaker feedback and summarize improvements.", Priority.LOW, 2, ["conduct"], "A feedback summary with follow-up actions."),
        ]

    def _exam_tasks(self) -> list[Task]:
        make = self._make_task
        return [
            make("review_syllabus", "Review the syllabus", "List exam topics, marks distribution, and available study resources.", Priority.HIGH, 1, None, "A complete topic checklist."),
            make("identify_weak_topics", "Identify weak topics", "Use a short diagnostic quiz or past questions to rank topics by confidence.", Priority.HIGH, 1, ["review_syllabus"], "A prioritized list of topics to improve."),
            make("study_schedule", "Create a seven-day study schedule", "Allocate daily study blocks for concepts, coding practice, and revision.", Priority.HIGH, 1, ["identify_weak_topics"], "A realistic day-by-day study timetable."),
            make("fundamentals", "Study Python fundamentals", "Review variables, types, operators, control flow, and core syntax with examples.", Priority.HIGH, 4, ["study_schedule"], "Fundamental concepts recalled and practiced."),
            make("functions_data", "Practice functions and data structures", "Solve exercises using functions, lists, tuples, dictionaries, sets, and comprehensions.", Priority.HIGH, 5, ["fundamentals"], "Working solutions for representative practice problems."),
            make("files_exceptions", "Practice files and exception handling", "Implement file read/write examples and handle common runtime errors.", Priority.MEDIUM, 3, ["fundamentals"], "File and exception exercises completed."),
            make("past_questions", "Solve previous examination questions", "Attempt timed questions and review errors against the syllabus checklist.", Priority.HIGH, 4, ["functions_data", "files_exceptions"], "Past questions completed with an error log."),
            make("mock_test", "Take a mock test", "Complete a timed mock paper without notes and score each section.", Priority.HIGH, 2, ["past_questions"], "A mock score and list of remaining gaps."),
            make("revise_topics", "Revise difficult topics", "Revisit weak concepts and redo missed questions before the examination.", Priority.HIGH, 3, ["mock_test"], "Priority gaps addressed with final notes."),
        ]

    def _website_tasks(self) -> list[Task]:
        make = self._make_task
        return [
            make("requirements", "Gather requirements", "Identify users, core workflows, accessibility needs, and acceptance criteria.", Priority.HIGH, 4, None, "A reviewed requirements checklist."),
            make("design", "Design the application", "Create user flows, page sketches, and a simple data model.", Priority.HIGH, 6, ["requirements"], "Approved screen and data designs."),
            make("setup", "Set up the project", "Create the repository, development environment, and initial application structure.", Priority.HIGH, 3, ["design"], "A runnable project baseline."),
            make("implementation", "Implement core features", "Build the primary user flows and connect them to the required data storage.", Priority.HIGH, 20, ["setup"], "Core functionality meets acceptance criteria."),
            make("testing", "Test and fix defects", "Run unit and user-flow tests, address defects, and verify key browsers/devices.", Priority.HIGH, 8, ["implementation"], "A passing test report and resolved critical defects."),
            make("deployment", "Deploy and document", "Deploy the tested application and document setup, usage, and support.", Priority.MEDIUM, 4, ["testing"], "A reachable application with a concise user guide."),
        ]

    def _generic_tasks(self) -> list[Task]:
        make = self._make_task
        return [
            make("clarify_goal", "Clarify the goal and success criteria", "Define the desired outcome, scope, stakeholders, and measurable completion criteria.", Priority.HIGH, 1, None, "A clear definition of success."),
            make("identify_resources", "Identify resources and constraints", "List required time, people, materials, budget, and known limitations.", Priority.HIGH, 1, ["clarify_goal"], "A resource and constraint checklist."),
            make("create_steps", "Break the goal into actionable steps", "Create a practical sequence of deliverables and identify dependencies.", Priority.HIGH, 2, ["identify_resources"], "A sequenced task breakdown."),
            make("execute", "Complete the main work", "Carry out the planned tasks and track progress against their expected outcomes.", Priority.HIGH, 6, ["create_steps"], "The main deliverables are completed."),
            make("review", "Review results and address gaps", "Compare results with success criteria and resolve any remaining issues.", Priority.MEDIUM, 2, ["execute"], "A checked result that meets the agreed criteria."),
        ]

    @staticmethod
    def _make_milestones(tasks: list[Task]) -> list[dict[str, object]]:
        midpoint = max(1, len(tasks) // 2)
        return [
            {
                "title": "Preparation complete",
                "description": "Core planning and prerequisite work is ready for execution.",
                "task_ids": [task.task_id for task in tasks[:midpoint]],
                "target_date": None,
            },
            {
                "title": "Goal completed and reviewed",
                "description": "The main deliverables are complete and checked against the goal.",
                "task_ids": [tasks[-1].task_id],
                "target_date": None,
            },
        ]

    def _demo_revision_payload(self, plan: TaskPlan, change_request: str) -> dict[str, Any]:
        payload = plan.model_dump(mode="json")
        lowered = change_request.lower()
        remove_action = re.search(r"\b(remove|delete|drop|omit|cancel)\b", lowered)
        add_action = re.search(r"\b(add|include|introduce)\b", lowered)
        reduced_count = re.search(r"\b(?:reduce|fewer|less)\b.*?\b(\d+)\b|\b(\d+)\s+(?:tasks|steps)\b", lowered)

        if remove_action:
            terms = self._meaningful_words(lowered)
            candidates = [
                task
                for task in payload["tasks"]
                if terms.intersection(self._meaningful_words(task["title"]))
            ]
            if not candidates:
                raise PlannerError(
                    "I could not identify which task to remove. Mention a task title or its key words."
                )
            removed_ids = {task["task_id"] for task in candidates}
            payload["tasks"] = [
                task for task in payload["tasks"] if task["task_id"] not in removed_ids
            ]
            for task in payload["tasks"]:
                task["dependencies"] = [
                    dependency for dependency in task["dependencies"] if dependency not in removed_ids
                ]
            for milestone in payload["milestones"]:
                milestone["task_ids"] = [
                    task_id for task_id in milestone["task_ids"] if task_id not in removed_ids
                ]
            payload["summary"] += " Explicitly requested tasks were removed."
        elif reduced_count:
            requested_count = int(reduced_count.group(1) or reduced_count.group(2))
            if requested_count < 1:
                raise PlannerError("The revised plan must contain at least one task.")
            if requested_count < len(payload["tasks"]):
                removable = [
                    task
                    for task in reversed(payload["tasks"])
                    if task["priority"] == Priority.LOW.value
                ]
                for task in removable:
                    if len(payload["tasks"]) <= requested_count:
                        break
                    removed_id = task["task_id"]
                    payload["tasks"].remove(task)
                    for remaining in payload["tasks"]:
                        remaining["dependencies"] = [
                            dep for dep in remaining["dependencies"] if dep != removed_id
                        ]
                    for milestone in payload["milestones"]:
                        milestone["task_ids"] = [
                            task_id for task_id in milestone["task_ids"] if task_id != removed_id
                        ]
        if add_action:
            title = change_request.strip()
            task_id = "added_" + re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:35]
            existing_ids = {task["task_id"] for task in payload["tasks"]}
            suffix = 2
            base_id = task_id or "added_task"
            while task_id in existing_ids:
                task_id = f"{base_id}_{suffix}"
                suffix += 1
            payload["tasks"].append(
                self._make_task(
                    task_id,
                    f"Added requirement: {title[:120]}",
                    f"Complete the requested addition: {title}",
                    Priority.MEDIUM,
                    2,
                    [payload["tasks"][-1]["task_id"]],
                    f"The requested addition is completed: {title}",
                ).model_dump(mode="json")
            )
            payload["milestones"][-1]["task_ids"].append(task_id)
        payload["total_estimated_effort_hours"] = round(
            sum(task["estimated_duration_hours"] for task in payload["tasks"]), 2
        )
        payload["schedule"] = []
        return payload

    @staticmethod
    def _meaningful_words(text: str) -> set[str]:
        ignored = {
            "remove", "delete", "drop", "omit", "cancel", "the", "a", "an", "and",
            "from", "please", "task", "program", "session", "item", "this", "that",
        }
        return {
            word for word in re.findall(r"[a-z0-9]+", text.lower())
            if len(word) > 2 and word not in ignored
        }

    def _ensure_revision_preserves_tasks(
        self, original: TaskPlan, revised: TaskPlan, change_request: str
    ) -> None:
        old_ids = {task.task_id for task in original.tasks}
        new_ids = {task.task_id for task in revised.tasks}
        removed_ids = old_ids - new_ids
        if not removed_ids:
            return
        lowered = change_request.lower()
        if not re.search(r"\b(remove|delete|drop|omit|cancel|replace)\b", lowered):
            raise PlanValidationError(
                "The revised plan dropped existing tasks without an explicit removal request. "
                "The previous plan has been preserved."
            )
        terms = self._meaningful_words(lowered)
        permitted = {
            task.task_id
            for task in original.tasks
            if terms.intersection(self._meaningful_words(task.title))
        }
        unexpected = removed_ids - permitted
        if unexpected:
            raise PlanValidationError(
                "The revised plan removed unrelated tasks. The previous plan has been preserved."
            )
