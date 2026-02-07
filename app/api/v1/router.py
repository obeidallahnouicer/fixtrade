from fastapi import APIRouter
from app.schemas.example import Message
from app.services.example_service import get_message

router = APIRouter()


@router.get("/health", response_model=Message)
def health():
    return get_message()
