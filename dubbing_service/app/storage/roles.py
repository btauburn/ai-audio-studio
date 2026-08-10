"""File-based helpers for managing roles within a project."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List

from dubbing_service.app.models.enums import RoleType
from dubbing_service.app.models.project import Role
from dubbing_service.app.storage.projects import (
    ProjectStorageError,
    load_project,
    save_project,
)


class RoleStorageError(Exception):
    """与角色存储相关的异常。"""


def create_role(
    project_id: str,
    name: str,
    role_type: RoleType = RoleType.CHARACTER,
    description: str | None = None,
) -> Role:
    """在指定项目下创建一个新角色。"""

    try:
        project = load_project(project_id)
    except ProjectStorageError as exc:
        raise RoleStorageError(f"项目不存在: {project_id}") from exc

    normalized_name = name.strip()
    for existing in project.roles:
        if existing.name == normalized_name:
            raise RoleStorageError(f"项目 {project_id} 中已存在同名角色：{normalized_name}")

    role_id = uuid.uuid4().hex[:8]
    role = Role(
        id=role_id,
        name=normalized_name,
        role_type=role_type,
        description=description,
    )

    project.roles.append(role)
    project.updated_at = datetime.now(timezone.utc)
    save_project(project)

    return role


def list_roles(project_id: str) -> List[Role]:
    """列出项目下的所有角色。"""

    try:
        project = load_project(project_id)
    except ProjectStorageError as exc:
        raise RoleStorageError(f"项目不存在: {project_id}") from exc

    return project.roles


def update_role_speaker_audio(
    project_id: str,
    role_id: str,
    speaker_audio_rel_path: str | None,
    speaker_voice_id: str | None = None,
) -> Role:
    """更新角色的参考音频路径。"""

    try:
        project = load_project(project_id)
    except ProjectStorageError as exc:
        raise RoleStorageError(f"项目不存在: {project_id}") from exc

    for role in project.roles:
        if role.id == role_id:
            role.speaker_audio = speaker_audio_rel_path
            role.speaker_voice_id = speaker_voice_id
            project.updated_at = datetime.now(timezone.utc)
            save_project(project)
            return role

    raise RoleStorageError(f"项目 {project_id} 中未找到角色 {role_id}")


def update_role(
    project_id: str,
    role_id: str,
    name: str,
    role_type: RoleType = RoleType.CHARACTER,
    description: str | None = None,
) -> Role:
    """更新指定项目下某个角色的基础信息（名称 / 类型 / 描述）。"""

    try:
        project = load_project(project_id)
    except ProjectStorageError as exc:
        raise RoleStorageError(f"项目不存在: {project_id}") from exc

    normalized_name = name.strip()

    # 保证同一项目下角色名称唯一（排除自身）
    for existing in project.roles:
        if existing.id != role_id and existing.name == normalized_name:
            raise RoleStorageError(f"项目 {project_id} 中已存在同名角色：{normalized_name}")

    target: Role | None = None
    for role in project.roles:
        if role.id == role_id:
            target = role
            break

    if target is None:
        raise RoleStorageError(f"项目 {project_id} 中未找到角色 {role_id}")

    target.name = normalized_name
    target.role_type = role_type
    target.description = description
    project.updated_at = datetime.now(timezone.utc)
    save_project(project)

    return target


def update_role_voice_profile(
    project_id: str,
    role_id: str,
    updates: dict[str, str | None],
) -> Role:
    """Partially update a role's backend-specific voice profile."""

    try:
        project = load_project(project_id)
    except ProjectStorageError as exc:
        raise RoleStorageError(f"项目不存在: {project_id}") from exc

    target = next((role for role in project.roles if role.id == role_id), None)
    if target is None:
        raise RoleStorageError(f"项目 {project_id} 中未找到角色 {role_id}")

    allowed = {
        "tts_backend",
        "qwen3_speaker",
        "qwen3_language",
        "voice_description",
        "style_instruction",
    }
    for field_name, value in updates.items():
        if field_name not in allowed:
            continue
        normalized = str(value).strip() if value is not None else None
        setattr(target, field_name, normalized or None)

    project.updated_at = datetime.now(timezone.utc)
    save_project(project)
    return target


def delete_role(project_id: str, role_id: str) -> None:
    """删除指定项目下的一个角色。

    注意：与该角色关联的 segments.role_id 清理逻辑放在路由层处理。
    """

    try:
        project = load_project(project_id)
    except ProjectStorageError as exc:
        raise RoleStorageError(f"项目不存在: {project_id}") from exc

    original_count = len(project.roles)
    project.roles = [r for r in project.roles if r.id != role_id]

    if len(project.roles) == original_count:
        raise RoleStorageError(f"项目 {project_id} 中未找到角色 {role_id}")

    project.updated_at = datetime.now(timezone.utc)
    save_project(project)
