"""Streamlit interface for the AI task-planning agent."""

import csv
import io
import json
from datetime import date

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from models import PlanningContext, TaskPlan
from planner import PlannerConfig, PlannerError, TaskPlanner
from validator import PlanValidationError, parse_subjects


load_dotenv()

st.set_page_config(
    page_title="AI Task Planner",
    page_icon="🗂️",
    layout="wide",
)


def planner_for_request() -> TaskPlanner:
    return TaskPlanner(PlannerConfig.from_env())


def task_table(plan: TaskPlan) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID": task.task_id,
                "Task": task.title,
                "Priority": task.priority.value,
                "Duration (hours)": task.estimated_duration_hours,
                "Depends on": ", ".join(task.dependencies) or "None",
                "Status": task.status.value,
                "Expected outcome": task.expected_outcome,
            }
            for task in plan.tasks
        ]
    )


def plan_csv(plan: TaskPlan) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "task_id",
            "title",
            "description",
            "priority",
            "estimated_duration_hours",
            "dependencies",
            "expected_outcome",
            "status",
        ],
    )
    writer.writeheader()
    for task in plan.tasks:
        row = task.model_dump(mode="json")
        row["priority"] = task.priority.value
        row["status"] = task.status.value
        row["dependencies"] = "; ".join(task.dependencies)
        writer.writerow(row)
    return output.getvalue()


def render_plan(plan: TaskPlan, heading: str, deadline: date | None = None) -> None:
    st.subheader(heading)
    first, second, third = st.columns(3)
    first.metric("Tasks", len(plan.tasks))
    second.metric("Estimated effort", f"{plan.total_estimated_effort_hours:g} hours")
    second.caption("Total effort is not calendar duration; independent tasks can overlap.")
    third.metric("Milestones", len(plan.milestones))
    st.write(plan.summary)

    st.markdown("#### Tasks")
    st.dataframe(task_table(plan), use_container_width=True, hide_index=True)

    with st.expander("Milestones and dependency-aware schedule", expanded=True):
        if plan.milestones:
            for milestone in plan.milestones:
                task_names = [
                    task.title for task in plan.tasks if task.task_id in milestone.task_ids
                ]
                st.markdown(
                    f"**{milestone.title}** — {milestone.description}  \n"
                    f"Tasks: {', '.join(task_names) if task_names else 'No linked tasks'}"
                )
        if plan.schedule:
            schedule_rows = []
            task_titles = {task.task_id: task.title for task in plan.tasks}
            for item in plan.schedule:
                schedule_rows.append(
                    {
                        "Task": task_titles[item.task_id],
                        "Start day": item.start_day,
                        "End day": item.end_day,
                        "Start date": item.start_date.isoformat() if item.start_date else "",
                        "End date": item.end_date.isoformat() if item.end_date else "",
                    }
                )
            st.dataframe(pd.DataFrame(schedule_rows), use_container_width=True, hide_index=True)
            st.caption(
                "Schedule uses earliest dependency completion; tasks with the same start window "
                "may run in parallel. Review resource availability before committing."
            )
            projected_end = max(
                (item.end_date for item in plan.schedule if item.end_date is not None),
                default=None,
            )
            if deadline and projected_end and projected_end > deadline:
                st.warning(
                    f"The estimated schedule ends on {projected_end.isoformat()}, "
                    f"after the requested deadline of {deadline.isoformat()}. "
                    "Consider increasing daily availability, reducing scope, or changing dependencies."
                )
        else:
            st.info("Add a deadline to calculate a dependency-aware suggested schedule.")

    with st.expander("Assumptions"):
        if plan.assumptions:
            for assumption in plan.assumptions:
                st.markdown(f"- {assumption}")
        else:
            st.write("No additional assumptions.")


if "original_plan" not in st.session_state:
    st.session_state.original_plan = None
if "current_plan" not in st.session_state:
    st.session_state.current_plan = None
if "change_summary" not in st.session_state:
    st.session_state.change_summary = ""
if "is_demo" not in st.session_state:
    st.session_state.is_demo = True

st.title("AI-Powered Task Planning Agent")
st.write(
    "Turn a goal into an actionable, prioritized plan with estimates, dependencies, "
    "milestones, and a revision workflow."
)

with st.form("goal_form"):
    goal = st.text_area(
        "What do you want to accomplish?",
        placeholder="Example: Organize a two-day technical workshop for MCA students.",
        height=100,
    )
    subject_text = st.text_area(
        "Subjects to plan for (optional)",
        placeholder="Enter subjects separated by commas or new lines, e.g.:\nPython\nDBMS\nData Structures",
        help="When subjects are entered, the planner creates a separate syllabus, learning, practice, and assessment track for each one. You can leave the goal blank.",
        height=90,
    )
    with st.expander("Optional planning details"):
        deadline = st.date_input("Deadline (optional)", value=None, min_value=date.today())
        available_hours = st.number_input(
            "Available hours per day", min_value=0.5, max_value=24.0, value=4.0, step=0.5
        )
        budget = st.text_input("Budget, if applicable", placeholder="e.g., Rs. 10,000")
        constraints = st.text_area(
            "Additional constraints", placeholder="Resources, people, or requirements to consider"
        )
        planning_style = st.selectbox(
            "Planning style",
            ["Balanced", "Detailed", "Minimal", "Deadline-focused"],
        )
    generate_clicked = st.form_submit_button("Generate plan", type="primary")

if generate_clicked:
    try:
        context = PlanningContext(
            subjects=parse_subjects(subject_text),
            deadline=deadline,
            available_hours_per_day=available_hours,
            budget=budget.strip() or None,
            constraints=constraints.strip() or None,
            planning_style=planning_style,
        )
        planner = planner_for_request()
        generated_plan = planner.generate_plan(goal, context)
        st.session_state.original_plan = generated_plan
        st.session_state.current_plan = generated_plan
        st.session_state.change_summary = ""
        st.session_state.is_demo = planner.is_demo_mode
        st.success("Plan generated successfully.")
    except (ValueError, PlannerError, PlanValidationError) as exc:
        st.error(str(exc))

current_plan = st.session_state.current_plan
if current_plan is not None:
    if st.session_state.is_demo:
        st.warning("Demo mode: this plan was generated by deterministic rules, not an AI model.")
    else:
        st.success("AI mode: plan generated by the configured language model and validated.")
    render_plan(current_plan, "Current plan", deadline)

    st.markdown("### Revise this plan")
    with st.form("revision_form"):
        change_request = st.text_area(
            "What would you like to change?",
            placeholder='Example: "Add a hands-on Python session" or "Remove the feedback task."',
        )
        revise_clicked = st.form_submit_button("Revise plan")
    if revise_clicked:
        try:
            planner = planner_for_request()
            revised_plan = planner.revise_plan(
                st.session_state.current_plan,
                change_request,
                PlanningContext(
                    subjects=parse_subjects(subject_text),
                    deadline=deadline,
                    available_hours_per_day=available_hours,
                    budget=budget.strip() or None,
                    constraints=constraints.strip() or None,
                    planning_style=planning_style,
                ),
            )
            st.session_state.current_plan = revised_plan
            st.session_state.change_summary = change_request.strip()
            st.session_state.is_demo = planner.is_demo_mode
            st.success("Plan revised. The previous plan is preserved below for comparison.")
            st.rerun()
        except (ValueError, PlannerError, PlanValidationError) as exc:
            st.error(f"Revision failed; the previous plan is unchanged. {exc}")

    if st.session_state.original_plan is not None:
        original_plan = st.session_state.original_plan
        if original_plan != current_plan:
            st.markdown("### Original versus revised")
            if st.session_state.change_summary:
                st.caption(f"Latest change request: {st.session_state.change_summary}")
            left, right = st.columns(2)
            with left:
                render_plan(original_plan, "Original plan", deadline)
            with right:
                render_plan(current_plan, "Revised plan", deadline)

    json_data = json.dumps(current_plan.model_dump(mode="json"), indent=2, ensure_ascii=False)
    download_json, download_csv, reset_column = st.columns(3)
    download_json.download_button(
        "Download plan as JSON",
        data=json_data,
        file_name="task_plan.json",
        mime="application/json",
    )
    download_csv.download_button(
        "Download tasks as CSV",
        data=plan_csv(current_plan),
        file_name="task_plan.csv",
        mime="text/csv",
    )
    if reset_column.button("Reset", type="secondary"):
        st.session_state.original_plan = None
        st.session_state.current_plan = None
        st.session_state.change_summary = ""
        st.rerun()

with st.sidebar:
    st.header("Configuration")
    try:
        configured = PlannerConfig.from_env()
        if configured.demo_mode:
            st.info("Demo mode is active. No API key is required.")
        else:
            st.success(f"AI mode: {configured.provider} / {configured.model}")
    except PlannerError as exc:
        st.error(str(exc))
    st.caption("Set provider credentials in a local .env file; never paste API keys into this page.")
