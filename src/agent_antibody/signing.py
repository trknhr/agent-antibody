from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime
from typing import Final
from uuid import uuid4

from pydantic import BaseModel

from agent_antibody.contracts import (
    ApprovalGrant,
    JsonValue,
    SignedApproval,
    SignedTaskContract,
    TaskContract,
)
from agent_antibody.core_types import ToolId

_DIGEST: Final = hashlib.sha256


def canonical_bytes(model: BaseModel) -> bytes:
    payload = model.model_dump(mode="json", exclude_none=True)
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def digest_model(model: BaseModel) -> str:
    return hashlib.sha256(canonical_bytes(model)).hexdigest()


def digest_arguments(arguments: dict[str, JsonValue]) -> str:
    payload = json.dumps(arguments, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


class ContractSigner:
    def __init__(self, secret: bytes) -> None:
        if len(secret) < 32:
            raise ValueError("contract signing secret must be at least 32 bytes")
        self._secret = secret

    def issue(self, contract: TaskContract) -> SignedTaskContract:
        return SignedTaskContract(contract=contract, signature=self._sign(contract))

    def verify(self, envelope: SignedTaskContract) -> bool:
        return hmac.compare_digest(envelope.signature, self._sign(envelope.contract))

    def _sign(self, contract: TaskContract) -> str:
        return hmac.new(self._secret, canonical_bytes(contract), _DIGEST).hexdigest()


class ApprovalSigner:
    def __init__(self, secret: bytes) -> None:
        if len(secret) < 32:
            raise ValueError("approval signing secret must be at least 32 bytes")
        self._secret = secret

    def issue(
        self,
        *,
        contract: TaskContract,
        tool: ToolId,
        arguments: dict[str, JsonValue],
        expires_at: datetime,
    ) -> SignedApproval:
        grant = ApprovalGrant(
            approval_id=str(uuid4()),
            contract_digest=digest_model(contract),
            tool=tool,
            arguments_digest=digest_arguments(arguments),
            expires_at=expires_at,
        )
        return SignedApproval(grant=grant, signature=self._sign(grant))

    def verify(
        self,
        approval: SignedApproval,
        *,
        contract: TaskContract,
        tool: ToolId,
        arguments: dict[str, JsonValue],
        now: datetime | None = None,
    ) -> bool:
        current_time = (now or datetime.now(UTC)).astimezone(UTC)
        grant = approval.grant
        return all(
            (
                hmac.compare_digest(approval.signature, self._sign(grant)),
                hmac.compare_digest(grant.contract_digest, digest_model(contract)),
                grant.tool == tool,
                hmac.compare_digest(grant.arguments_digest, digest_arguments(arguments)),
                grant.expires_at > current_time,
            )
        )

    def _sign(self, grant: ApprovalGrant) -> str:
        return hmac.new(self._secret, canonical_bytes(grant), _DIGEST).hexdigest()
