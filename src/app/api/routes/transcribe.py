import asyncio

from fastapi import APIRouter, HTTPException, Request, status
from groq import Groq
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from src.app.api.routes.instruction import generate_instruction
from src.app.api.routes.tasks import execute_task_instruction
from src.app.core.config import get_settings
from src.app.schemas.voice import InstructionRequest, TranscribeFlowResponse
from src.app.utils.language import normalize_transcription_language

router = APIRouter(tags=["transcribe"])


@router.get("/")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/transcribe", response_model=TranscribeFlowResponse)
async def transcribe_and_run_flow(request: Request) -> TranscribeFlowResponse:
    if request.headers.get("content-type", "").startswith("application/json"):
        try:
            payload = InstructionRequest.model_validate(await request.json())
        except (ValueError, ValidationError) as exc:
            raise HTTPException(status_code=422, detail="Invalid transcription payload") from exc
        transcription = payload.transcription
    elif request.headers.get("content-type", "").startswith("multipart/form-data"):
        form = await request.form()
        audio = form.get("file")
        if not isinstance(audio, UploadFile):
            raise HTTPException(status_code=422, detail="An audio file is required")
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="The uploaded audio file is empty")

        language = normalize_transcription_language(form.get("language"))
        settings = get_settings()
        client = Groq(
            api_key=settings.groq_api_key,
            timeout=settings.request_timeout_seconds,
        )
        transcription_args = {
            "file": (audio.filename or "recording.webm", audio_bytes, audio.content_type),
            "model": settings.groq_transcription_model,
        }
        if language:
            transcription_args["language"] = language
        try:
            result = await asyncio.to_thread(
                client.audio.transcriptions.create,
                **transcription_args,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The transcription model could not process the audio.",
            ) from exc
        transcription = result.text.strip()
        if not transcription:
            raise HTTPException(status_code=422, detail="No speech was detected")
    else:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Send JSON transcription or multipart audio",
        )

    instruction = await asyncio.to_thread(generate_instruction, transcription)
    task_result = await asyncio.to_thread(
        execute_task_instruction,
        instruction.endpoint,
        instruction.method,
        instruction.params,
    )
    return TranscribeFlowResponse(
        transcription=transcription,
        instruction=instruction,
        result=task_result,
    )
