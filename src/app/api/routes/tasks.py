from typing import Any

from fastapi import APIRouter, HTTPException, status

from src.app.schemas.voice import Task, TaskCreate, TaskReplace, TaskUpdate

router = APIRouter(prefix="/tasks", tags=["tasks"])
tasks: list[dict[str, Any]] = []
_next_task_id = 1


@router.get("", response_model=list[Task])
def get_tasks() -> list[Task]:
    return [Task.model_validate(task) for task in tasks]


@router.post("", response_model=Task, status_code=status.HTTP_201_CREATED)
def create_task(payload: TaskCreate) -> Task:
    global _next_task_id
    task = {"id": _next_task_id, **payload.model_dump()}
    _next_task_id += 1
    tasks.append(task)
    return Task.model_validate(task)


@router.put("/{task_id}", response_model=Task)
def replace_task(
    task_id: int,
    payload: TaskReplace,
) -> Task:
    task = _find_task(task_id)
    task.update(payload.model_dump())
    return Task.model_validate(task)


@router.patch("/{task_id}", response_model=Task)
def update_task(
    task_id: int,
    payload: TaskUpdate,
) -> Task:
    task = _find_task(task_id)
    task.update(payload.model_dump(exclude_unset=True, exclude_none=True))
    return Task.model_validate(task)


@router.delete("/{task_id}")
def delete_task(task_id: int) -> dict[str, str]:
    task = _find_task(task_id)
    tasks.remove(task)
    return {"message": f"Task {task_id} deleted successfully"}


def execute_task_instruction(endpoint: str, method: str, params: dict[str, Any]) -> Any:
    """Execute a validated task route returned by the instruction model."""
    method = method.upper()
    if endpoint == "/tasks" and method == "GET":
        return get_tasks()
    if endpoint == "/tasks" and method == "POST":
        return create_task(TaskCreate.model_validate(params))

    parts = endpoint.strip("/").split("/")
    if len(parts) != 2 or parts[0] != "tasks" or not parts[1].isdigit():
        raise HTTPException(status_code=422, detail="Unsupported task endpoint")

    task_id = int(parts[1])
    if method == "PUT":
        return replace_task(task_id, TaskReplace.model_validate(params))
    if method == "PATCH":
        return update_task(task_id, TaskUpdate.model_validate(params))
    if method == "DELETE":
        return delete_task(task_id)
    raise HTTPException(status_code=422, detail="Unsupported task method")


def _find_task(task_id: int) -> dict[str, Any]:
    for task in tasks:
        if task["id"] == task_id:
            return task
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
