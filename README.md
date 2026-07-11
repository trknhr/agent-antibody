# Agent Antibody

Agent Antibody is a DevSecOps prototype for tool-using AI agents. It attacks a
target agent in a simulator, judges the run by observable tool calls and state
changes, then turns a confirmed failure into deterministic policy and regression
memory.

The prototype ships three complete target adapters:

```text
OpsMate    malicious Runbook / scale-to-zero       INFECTED -> IMMUNE
RepoMate   malicious Issue / delete README.md      INFECTED -> IMMUNE
SupportMate receipt OCR / high-value refund        INFECTED -> IMMUNE
```

The simulator is local only. No component has production operations credentials
or access to a real service control plane.

## What is implemented

- Signed, expiring task contracts bound to caller, resources, and per-tool capabilities.
- Target-neutral Gateway, Runner, Oracle, and trace engine.
- In-memory Ops, repository, and customer-support simulators.
- Fail-closed Policy Gateway with strict arguments, destination limits, redaction,
  provenance tracking, and single-use approval binding.
- Canonical traces for tool requests, policy decisions, execution outcomes, and
  state changes.
- Deterministic oracle statuses: `INFECTED`, `IMMUNE`, `ATTACK_MISSED`,
  `HEALTHY`, `UNHEALTHY`, and `INVALID_RUN`.
- Policy compiler that validates LLM antibody proposals against a target manifest
  and confirmed evidence before activating policy.
- Regression bundles that are directly executable in CI.
- Google ADK adapters for OpsMate, RepoMate, and SupportMate running through the same Gateway.
- Gemini Attack Agent and Antibody Agent using structured output.
- Ten-case attack campaigns: Gemini generates ten distinct injection techniques,
  uses one confirmed case as immune memory, and verifies the resulting antibody
  against that seed plus nine held-out variants.
- Target-owned attack payload contracts, so generated tool arguments are checked
  against the exact untrusted document grammar each vulnerable adapter consumes.
- Immutable immunity artifacts: a policy, one memory seed, and nine held-out
  regression variants are committed together under `immunities/v1/`.
- A two-stage GitHub workflow that detects agent changes, produces data-only
  candidate evidence, and opens a stacked draft immunity PR for same-repository
  pull requests.
- FastAPI demo UI with deterministic and live Gemini flows.
- Cloud Run source deployment workflow with Workload Identity Federation.

## Architecture

```text
Target registry (OpsMate / RepoMate / SupportMate)
  -> Gemini Attack Agent
  -> target-specific attack materializer
Target Agent through Google ADK
  -> Policy Gateway
  -> target simulator
  -> canonical trace
Deterministic oracle
  -> confirmed failure
Gemini Antibody Agent
  -> bounded policy proposal
Policy compiler
  -> deterministic antibody policy + regression memory
```

Untrusted documents cannot create contracts or approvals. The gateway derives
provenance from successful read-tool results rather than trusting provenance
claimed by the agent.

## Quick Start

Requires Python 3.14 and `uv`.

```bash
uv sync --locked --dev
uv run agent-antibody portfolio
uv run agent-antibody live --target supportmate
uv run agent-antibody immunity verify --repository-root .
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run pyright
```

Generate an antibody and execute the generated file as CI would:

```bash
uv run agent-antibody demo --output .generated-antibodies
uv run agent-antibody regression .generated-antibodies/*.yaml
```

## Live Gemini Pipeline

The deterministic demo does not need an API key. The live path uses real Gemini
calls for three roles: Attack Agent, the selected target through ADK, and
Antibody Agent. Each campaign generates and evaluates ten target-specific attacks
before and after the generated policy. All tools still execute only against the
local simulator.

With Gemini Developer API:

```bash
export GEMINI_API_KEY=...
export AGENT_ANTIBODY_MODEL=gemini-3.5-flash
uv run agent-antibody live
uv run agent-antibody live --target repomate
uv run agent-antibody live --target supportmate
```

With EnvVault:

```bash
envvault exec \
  --env GEMINI_API_KEY=envvault://gemini-api-key \
  --env AGENT_ANTIBODY_MODEL=gemini-3.5-flash \
  -- uv run agent-antibody live
```

Expected result:

```text
OpsMate      10/10 -> 0/10 / normal HEALTHY
RepoMate     10/10 -> 0/10 / normal HEALTHY
SupportMate  10/10 -> 0/10 / normal HEALTHY
```

## Local UI

```bash
uv run uvicorn agent_antibody.api:app --reload
```

Open `http://127.0.0.1:8000`.

The page at `/` is a read-only CI immunity dashboard. It reads the committed
allowlisted snapshot at `immunities/v1/snapshot.json`; it cannot call a model,
run a tool, change policy, or create a GitHub PR from a browser.

The API exposes:

- `GET /healthz`
- `GET /api/healthz`
- `GET /api/demo`
- `GET /api/demo?target_id=opsmate|repomate|supportmate`
- `GET /api/portfolio`
- `GET /api/targets`
- `GET /api/immunity-snapshot`
- `GET /api/live/status`
- `POST /api/live`

`POST /api/live` accepts `{"target_id":"opsmate|repomate|supportmate"}` and
requires `Content-Type: application/json` and a request bearer token matching
`AGENT_ANTIBODY_LIVE_API_TOKEN`. The browser keeps the entered token only in the
current page; it is not embedded in JavaScript or written to browser storage.
`GET /api/live/status` reports whether live mode is configured; it deliberately
does not claim that Vertex IAM, model availability, or quota has been verified.

## Immunity CI and generated PRs

Each confirmed attack becomes an additive, content-addressed file:

```text
immunities/v1/<target_id>/imm-<content-digest>.yaml
```

The file contains the bounded policy DSL and ten replayable attacks, but omits
agent replies, tool output, raw simulator state, contracts, and LLM rationale.
`agent-antibody immunity verify --repository-root .` replays every persisted
memory against the current target adapter and verifies that all ten attacks are
blocked by a policy rule while normal utility remains healthy.

The GitHub flow is intentionally split at the trust boundary:

```text
Agent PR (read-only candidate workflow)
  -> data-only evaluation artifact
  -> trusted remediation workflow
  -> antibody/pr-<source-pr>-<sha> stacked draft PR
  -> review + merge into the source PR
  -> Cloud Run live Gemini release gate
```

`Agent Antibody Candidate` executes candidate code with only `contents: read`,
no secrets, no cache, and no write token. `Agent Antibody Remediate` runs only
trusted default-branch control-plane code, validates the typed artifact, writes
only `immunities/v1/**`, and opens a draft PR whose base is the original PR
branch. Fork PRs never receive an automatic write/PR; they keep the read-only
candidate result. The GitHub-only candidate result is a draft-remediation aid,
not a release gate: the existing Cloud Run live Gemini gate remains the release
authority until a fixed external sandbox harness is configured.

For a local dry run:

```bash
REVISION="$(git rev-parse HEAD)"
printf '%s\n' src/agent_antibody/targets/supportmate.py > /tmp/changed-files.txt
uv run agent-antibody immunity evaluate \
  --repository-root . \
  --source-revision "$REVISION" \
  --changed-files /tmp/changed-files.txt \
  --output /tmp/antibody-evaluation.json
uv run agent-antibody immunity apply \
  --evaluation /tmp/antibody-evaluation.json \
  --repository-root .
```

The CLI never pushes, opens a PR, or deploys. Those side effects are limited to
the reviewed GitHub workflow with explicit `contents: write` and
`pull-requests: write` permissions.

## Cloud Run

Cloud Run should use Vertex AI authentication instead of a local API key:

```bash
GOOGLE_CLOUD_PROJECT=<project-id>
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_VERTEXAI=true
AGENT_ANTIBODY_MODEL=gemini-3.5-flash
AGENT_ANTIBODY_LIVE_ENABLED=true
AGENT_ANTIBODY_LIVE_API_TOKEN=<loaded-from-Secret-Manager>
AGENT_ANTIBODY_LIVE_CACHE_SECONDS=600
AGENT_ANTIBODY_ATTACK_SUITE_ATTEMPTS=2
AGENT_ANTIBODY_SUITE_CONCURRENCY=3
AGENT_ANTIBODY_EXPORT_TRACES=true
```

`AGENT_ANTIBODY_SUITE_CONCURRENCY` must stay between 3 and 10 so the bounded
ten-case live campaign fits within the Cloud Run request timeout.

Recommended deployment shape:

- One dedicated Cloud Run service: `agent-antibody`.
- Dedicated runtime service account with `roles/aiplatform.user`.
- Dedicated source-build service account with Cloud Run build permissions.
- A Secret Manager secret named by the required GitHub variable
  `AGENT_ANTIBODY_LIVE_TOKEN_SECRET`; grant Secret Accessor to the runtime service
  account and to the WIF deploy service account that configures the secret mapping.
- Grant the WIF deploy service account Cloud Run deployment permission and Service
  Account User on both the runtime and source-build service accounts.
- `max-instances=1`, `concurrency=1`, a 900-second request timeout, bounded
  three-way in-process campaign execution, and live result caching for demo cost control.
- Structured trace export to Cloud Logging without raw Runbook bodies, Issue bodies,
  prompts, or credentials.
- Public API responses are explicit allowlists: they expose only state summaries,
  tool lifecycle metadata, and campaign outcomes, never raw fixtures or raw state.

The workflow in `.github/workflows/deploy.yml` verifies tests and generated
regressions, then deploys to Cloud Run through Workload Identity Federation. It
fails closed if the new source revision cannot be built or deployed. It then
retrieves the presenter token from Secret Manager and runs one protected,
ten-case SupportMate Gemini campaign as the release gate: one memory seed plus
nine held-out variants must move from `10/10` infected to `0/10`, and normal
utility must remain healthy. The deterministic UI is public; paid live Gemini
execution remains bearer-protected. It is ready for `trknhr/agent-antibody`
once the GitHub repository, WIF variables, and live-token Secret Manager secret
exist.

## Current Boundary

The generic boundary is `ExecutionCase`, `TargetRuntime`, `TargetAdapter`, the
signed capability contract, target-neutral trace/oracle, attack plan schema,
antibody proposal schema, and policy compiler. Each target implements that
boundary with a local runtime, deterministic replay, and a Google ADK adapter.
