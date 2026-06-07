"""SQLAlchemy user repository adapter."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.auth.entities import User
from app.domain.auth.ports import UserRepository
from app.infrastructure.auth.models import UserModel


class SQLAlchemyUserRepository(UserRepository):
    """Persist and load users through SQLAlchemy."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_email(self, email: str) -> User | None:
        stmt = select(UserModel).where(UserModel.email == email)
        model = self._session.execute(stmt).scalar_one_or_none()
        return self._to_entity(model) if model is not None else None

    def get_by_id(self, user_id: str) -> User | None:
        model = self._session.get(UserModel, user_id)
        return self._to_entity(model) if model is not None else None

    def create(self, user: User) -> User:
        model = UserModel(
            email=user.email,
            hashed_password=user.hashed_password,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
        )
        self._session.add(model)
        self._session.commit()
        self._session.refresh(model)
        return self._to_entity(model)

    def _to_entity(self, model: UserModel) -> User:
        return User(
            id=model.id,
            email=model.email,
            hashed_password=model.hashed_password,
            full_name=model.full_name,
            role=model.role,
            is_active=model.is_active,
            created_at=model.created_at,
        )