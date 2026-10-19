# AI-Powered Task Planning Agent

A local Streamlit application that turns a natural-language goal into an ordered, prioritized plan. It validates task details and dependencies, estimates effort, calculates a dependency-aware schedule when a deadline is supplied, and supports plan revisions.

## Features

- Goal planning with optional deadline, daily availability, budget, constraints, and planning style.
- Subject-wise exam planning: enter up to 10 subjects (comma- or newline-separated) to generate separate syllabus review, focused study, practice, and assessment tasks for each subject. Subject input can be used without a separate goal.
- OpenAI-compatible LLM provider/model configuration with JSON output validation.
- Deterministic, clearly labeled demo mode when no API key is configured.
- Task priorities, estimates, expected outcomes, dependencies, milestones, and schedule.
- Revision workflow that preserves the previous plan if a revision fails and guards against silently dropped tasks.
- JSON and CSV downloads.
- Pydantic validation, dependency checks, circular-dependency detection, and pytest tests.

## Project structure

```text
app.py              Streamlit interface and exports
planner.py          AI calls, demo plans, revision, and schedule calculation
prompts.py          System prompt and request construction
models.py           Pydantic input and output models
validator.py        Goal, plan, and dependency validation
requirements.txt    Runtime and test dependencies
.env.example        Safe configuration template
tests/              Unit tests with a mock AI response
```

## Installation

Python 3.10 or later is required. In PowerShell, run from the project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Demo mode works without an API key. Keep `DEMO_MODE=true` in `.env` to test with the deterministic rule-based planner.

## Configure an AI provider

The app accepts an OpenAI-compatible chat-completions endpoint. Add a valid key to `.env` and set:

```dotenv
AI_PROVIDER=openai_compatible
AI_API_KEY=your-provider-key
AI_MODEL=gpt-4o-mini
AI_BASE_URL=
AI_TIMEOUT_SECONDS=30
DEMO_MODE=false
```

Set `AI_BASE_URL` when using a compatible provider other than OpenAI. The provider must support JSON-object response mode. The key can also be provided as `OPENAI_API_KEY`. Never commit `.env` or place credentials in the UI/source code. If no key exists and `DEMO_MODE` is not set, demo mode is selected automatically.

## Run the application

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. Enter a goal or list subjects such as `Python`, `DBMS`, and `Data Structures`; subjects can be separated with commas or new lines. The planner creates a separate study track for each selected subject. Optionally set a deadline/constraints and choose **Generate plan**. Use the separate revision form to update it. The original plan remains available for comparison; downloads export the current plan.

## Run tests

```powershell
python -m pytest -q
```

The tests use deterministic demo output and a mock AI response; they do not require credentials or a paid API.

## Demonstration / expected sample output

With demo mode enabled:

**Goal:** Organize a two-day technical workshop for MCA students.

The plan includes defining objectives (High, 2h), preparing a budget (High, 3h), selecting topics and speakers (High, 5h), booking the venue (High, 3h), arranging equipment (Medium, 3h), registering participants (Medium, 4h), preparing the schedule (High, 3h), conducting the workshop (High, 16h), and collecting feedback (Low, 2h). Effort is 41 hours. For example, venue booking depends on the budget; arranging equipment depends on the venue and confirmed speakers; conducting depends on equipment, registration, and the schedule.

**Goal:** Prepare for a Python examination in seven days.

The plan is different: review the syllabus (High, 1h), identify weak topics (High, 1h), create a seven-day study schedule (High, 1h), study fundamentals (High, 4h), practice functions and data structures (High, 5h), practice file/exception handling (Medium, 3h), solve past questions (High, 4h), take a mock test (High, 2h), and revise difficult topics (High, 3h). Effort is 24 hours. The study schedule follows syllabus review and weak-topic identification; the mock follows past-question practice.

**Subject input example:** Enter `Python`, `DBMS`, and `Data Structures` in the subjects box. The plan produces a distinct track for each subject, including Python programming topics and coding exercises, DBMS normalization/SQL and query practice, and data-structure implementations/complexity practice. Each track then leads into subject-wise past questions and a combined mock assessment. For unrecognized subject names, the planner builds a subject-labeled syllabus and practice plan and asks learners to verify exact topics against their syllabus.

**Revision example:** On the workshop plan, submit **“Add a hands-on Python session.”** Demo mode adds a two-hour task linked after the existing plan. Submit **“Remove the feedback task.”** to explicitly remove that task; dependent links and milestones are updated. In AI mode, the model proposes a complete replacement plan and the application validates dependencies and task preservation before replacing the current plan.

## Architecture

`app.py` handles Streamlit UI state and exports. `planner.py` separates provider communication and rule-based planning from the UI. `prompts.py` defines the system instructions and sends user content as JSON data. `models.py` defines validated Pydantic models, while `validator.py` checks goal input and plan-wide invariants. The schedule uses the earliest completion time of task prerequisites; independent tasks can share a schedule window, so effort hours are not presented as calendar time.

## Limitations and future enhancements

- Demo mode uses fixed templates selected by goal keywords; it is intended for demonstration and does not understand arbitrary goals as deeply as an LLM.
- The schedule assumes tasks can run in parallel when dependencies allow. Resource contention, weekends, holidays, and working-hour calendars are not modeled.
- Deadline feasibility is not automatically guaranteed; review the suggested schedule and adjust scope or availability where needed.
- The application currently supports OpenAI-compatible chat-completions APIs. Native provider-specific APIs and other authentication flows are not included.
- Possible extensions include user accounts, persistent plan storage, task completion tracking, calendar export, resource assignment, and richer schedule optimization.
