import json

from fastapi import APIRouter, HTTPException, status
from groq import Groq
from pydantic import ValidationError

from src.app.api.routes.tasks import tasks
from src.app.core.config import get_settings
from src.app.schemas.voice import InstructionPayload, InstructionRequest

router = APIRouter(tags=["instruction"])


SYSTEM_PROMPT = """You route spoken task-list commands. Return only one valid JSON object with exactly
these keys: endpoint, method, params. Do not include markdown or free text.

Supported actions:
- List tasks: endpoint /tasks, method GET, params {}.
- Create a task: endpoint /tasks, method POST, params {"title": string} and optionally {"done": boolean}.
- Replace a task: endpoint /tasks/{task_id}, method PUT, params with both title and done.
- Update a task or mark it done: endpoint /tasks/{task_id}, method PATCH, params with title and/or done.
- Delete a task: endpoint /tasks/{task_id}, method DELETE, params {}.

Use the numeric task ID in the endpoint path for actions on an existing task. The user may
refer to a task by its title, its position, or its ID: resolve it against the current task
list provided below and use that task's ID. Put only request-body fields in params. Never
invent a task ID that is not in the current task list. Keep task titles in the user's
language, without filler words such as "add" or "to my list".

If the request is not a task command, or refers to a task that does not exist, return the
list action: {"endpoint": "/tasks", "method": "GET", "params": {}}.

Examples:
"añade comprar leche a mi lista" -> {"endpoint": "/tasks", "method": "POST", "params": {"title": "Comprar leche"}}
"what's on my list" -> {"endpoint": "/tasks", "method": "GET", "params": {}}
"marca la tarea 2 como hecha" -> {"endpoint": "/tasks/2", "method": "PATCH", "params": {"done": true}}
"delete task 3" -> {"endpoint": "/tasks/3", "method": "DELETE", "params": {}}"""


def _current_tasks_message() -> str:
    task_list = json.dumps(tasks, ensure_ascii=False) if tasks else "[] (empty)"
    return f"Current task list: {task_list}"


def generate_instruction(transcription: str) -> InstructionPayload:
    settings = get_settings()
    client = Groq(
        api_key=settings.groq_api_key,
        timeout=settings.request_timeout_seconds,
    )
    try:
        completion = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": _current_tasks_message()},
                {"role": "user", "content": transcription},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The instruction model could not process the request.",
        ) from exc

    content = completion.choices[0].message.content
    try:
        instruction = InstructionPayload.model_validate(json.loads(content or ""))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The instruction model returned an invalid response.",
        ) from exc

    method = instruction.method.upper()
    is_collection_route = instruction.endpoint == "/tasks" and method in {"GET", "POST"}
    is_task_route = (
        len(instruction.endpoint.split("/")) == 3
        and instruction.endpoint.startswith("/tasks/")
        and instruction.endpoint.removeprefix("/tasks/").isdigit()
        and method in {"PUT", "PATCH", "DELETE"}
    )
    if not (is_collection_route or is_task_route):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The instruction model returned an unsupported task route.",
        )

    instruction.method = method
    return instruction


@router.post("/instruction", response_model=InstructionPayload)
def route_instruction(payload: InstructionRequest) -> InstructionPayload:
    return generate_instruction(payload.transcription)
