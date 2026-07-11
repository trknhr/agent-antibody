from __future__ import annotations

from agent_antibody.targets.base import TargetAdapter
from agent_antibody.targets.opsmate import OpsMateAdapter
from agent_antibody.targets.repomate import RepoMateAdapter
from agent_antibody.targets.supportmate import SUPPORTMATE_ADAPTER

TARGET_ADAPTERS: dict[str, TargetAdapter] = {
    "opsmate": OpsMateAdapter(),
    "repomate": RepoMateAdapter(),
    "supportmate": SUPPORTMATE_ADAPTER,
}

TARGET_IDS: tuple[str, ...] = tuple(TARGET_ADAPTERS)


def get_target_adapter(target_id: str) -> TargetAdapter:
    try:
        return TARGET_ADAPTERS[target_id]
    except KeyError as error:
        raise ValueError(f"unknown target agent: {target_id!r}") from error
