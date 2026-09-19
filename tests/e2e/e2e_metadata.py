"""Typed per-test metadata for the e2e suite: what a test drives, and what it did.

Two halves, deliberately separated.

The DECLARED half is `Subject`: one frozen dataclass passed as the single
positional argument of `@meta(...)`. Every field is a closed enum (or a free
string for `model`), so a typo is a basedpyright error at the call site rather
than a silently dropped property. `dataclasses.asdict()` turns the whole thing
into <property> pairs with no per-field plumbing -- adding a scalar field later
needs zero serializer changes.

The RECORDED half is `steps`, and it is NOT a field of `Subject`. Steps are
appended at runtime by `@step`-decorated harness helpers, in call order, so the
list IS the test's user story and its last element is where a failing test died.
Putting it on the declarable dataclass would invite hand-writing it, which is
exactly what it replaces.

Stdlib-only on purpose. `LlmProviders` is imported under TYPE_CHECKING and
`from __future__ import annotations` keeps the annotation unevaluated, so
importing this module costs no litellm import (measured: 1.44s, 2474 modules).
tests/e2e has zero runtime litellm imports today; this module keeps it that way.
Only a test file that actually names a provider pays for it, and basedpyright
still checks the member.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import TYPE_CHECKING, Final, ParamSpec, TypeVar, cast

import pytest

if TYPE_CHECKING:
    from litellm.types.utils import LlmProviders


class Domain(str, Enum):
    """The OSS issue-label taxonomy, verbatim.

    Shared with GitHub issue labels so an issue and a test join on one string.
    Exactly one per test; `UNKNOWN` is the honest answer, not an omission.
    """

    LLM_TRANSLATION = "llm-translation"
    SPEND_BUDGETS = "spend-budgets"
    UI = "ui"
    MCP = "mcp"
    OBSERVABILITY = "observability"
    ROUTING = "routing"
    DEPLOY_OPS = "deploy-ops"
    COST_MAP = "cost-map"
    PROXY_AUTH = "proxy-auth"
    GUARDRAILS = "guardrails"
    MANAGEMENT = "management"
    SDK = "sdk"
    PASSTHROUGH = "passthrough"
    DB = "db"
    CACHING = "caching"
    DOCS = "docs"
    AGENTS_API = "agents-api"
    UNKNOWN = "unknown"


class Route(str, Enum):
    """The customer-facing HTTP surface the test DRIVES.

    Orthogonal to the suite directory: a reliability test in router/ and a
    logging test in logging/ both drive `CHAT_COMPLETIONS`, which is why this is
    per-test and not per-module.

    "route" here means endpoint, matching litellm's own `LiteLLMRoutes`
    (litellm/proxy/_types.py). The coverage registry's `LlmCell.route` uses the
    same word for PROVIDER; that is a different namespace and is left alone.

    Deliberately collapsed against the registry's `LlmEndpoint`:
    images_generations + images_edits -> IMAGES, audio_speech +
    audio_transcriptions -> AUDIO, bedrock_native + google_native ->
    PASSTHROUGH. Those splits are wire detail, not a customer-facing surface,
    and `model` + `capabilities` already carry them.
    """

    # Core: each is the subject of many e2e tests.
    CHAT_COMPLETIONS = "chat_completions"
    MESSAGES = "messages"
    RESPONSES = "responses"
    EMBEDDINGS = "embeddings"
    COMPLETIONS = "completions"
    FILES = "files"
    BATCHES = "batches"
    PASSTHROUGH = "passthrough"
    MCP = "mcp"
    GUARDRAILS = "guardrails"
    KEY_MANAGEMENT = "key_management"
    TEAM_MANAGEMENT = "team_management"
    SPEND_REPORTING = "spend_reporting"
    MODEL_MANAGEMENT = "model_management"

    # Long tail: real, but one or two test files each.
    IMAGES = "images"
    AUDIO = "audio"
    MODERATIONS = "moderations"
    RERANK = "rerank"
    OCR = "ocr"
    VECTOR_STORES = "vector_stores"
    REALTIME = "realtime"
    A2A = "a2a"
    USER_MANAGEMENT = "user_management"
    BUDGET_MANAGEMENT = "budget_management"

    # Ops surfaces. logging/, load/, other/ and ui/ have no LLM route of their
    # own and would otherwise have to lie.
    HEALTH = "health"
    METRICS = "metrics"
    PROXY_CONFIG = "proxy_config"
    ADMIN_UI = "admin_ui"


class Capability(str, Enum):
    """A MODEL feature the test depends on, anchored 1:1 to a `supports_*` key
    in model_prices_and_context_window.json.

    Not to be confused with the coverage registry's `LlmCell.capability`, which
    means the endpoint feature under test (`basic`, `multi_turn`, ...) and half
    of whose values have no `supports_*` key at all.

    Plural by necessity, never a single enum: `thinking_with_tool_use` and
    `tool_search_history` only exist in the registry because a single-enum field
    had nowhere to put a conjunction. Here they are
    (REASONING, FUNCTION_CALLING) and (TOOL_SEARCH,).

    `audio_output` is deliberately absent: litellm/utils.py reads
    `supports_audio_input` for it, so the value would silently alias
    AUDIO_INPUT. Add it once that bug is fixed upstream.
    """

    FUNCTION_CALLING = "function_calling"
    PARALLEL_FUNCTION_CALLING = "parallel_function_calling"
    TOOL_CHOICE = "tool_choice"
    TOOL_SEARCH = "tool_search"
    VISION = "vision"
    PDF_INPUT = "pdf_input"
    AUDIO_INPUT = "audio_input"
    REASONING = "reasoning"
    WEB_SEARCH = "web_search"
    PROMPT_CACHING = "prompt_caching"
    RESPONSE_SCHEMA = "response_schema"
    MID_CONVERSATION_SYSTEM = "mid_conversation_system"


class Mode(str, Enum):
    """How the route was driven. The registry's cell ids already carry this
    axis as a segment (221 `.nonstream.`, 37 `.stream.`)."""

    NONSTREAM = "nonstream"
    STREAM = "stream"
    BATCH = "batch"
    WEBSOCKET = "websocket"


@dataclass(frozen=True, slots=True)
class Subject:
    """What a test is about.

    Every field is optional in this first phase -- nothing is enforced, and the
    backfill of the existing ~908 tests comes later. Named `Subject` rather than
    `TestMeta` because pytest tries to collect any imported class named `Test*`
    and would warn in every one of the ~570 modules that import it.
    """

    domain: Domain | None = None
    route: Route | None = None
    provider: LlmProviders | None = None
    model: str | None = None
    capabilities: tuple[Capability, ...] = ()
    mode: Mode | None = None

    def __post_init__(self) -> None:
        """Canonicalize capabilities at declaration: deduped and sorted by
        value, so the committed run files diff cleanly however a test spelled
        the tuple, and the serializer stays field-agnostic."""
        object.__setattr__(
            self,
            "capabilities",
            tuple(sorted(dict.fromkeys(self.capabilities), key=lambda c: c.value)),
        )


def meta(subject: Subject) -> pytest.MarkDecorator:
    """Attach a `Subject` to a test: `@meta(Subject(route=Route.RESPONSES, ...))`.

    A typed wrapper around `pytest.mark.meta` (registered in conftest.py's
    `pytest_configure`, like `covers`) so passing the wrong thing is a type
    error rather than a property that silently never appears.

    Separate from `@pytest.mark.covers` on purpose: `covers` args are flattened
    by `dedupe_covers`, which drops non-strings silently, and by
    tests/integration/conftest.py, which has no such filter and would hard-fail
    collection. `covers` is untouched by this change.
    """
    return pytest.mark.meta(subject)


# --- The recorded half: steps -------------------------------------------------

_P = ParamSpec("_P")
_R = TypeVar("_R")

# A retrying helper (poll_cost_row) or a load test calling a decorated helper in
# a loop would otherwise emit thousands of <property> entries per testcase.
MAX_STEPS: Final = 50
MAX_STEP_CHARS: Final = 200


class _StepRecorder:
    """The ordered step log for the running test.

    A plain lock-guarded list rather than a ContextVar: ContextVars do not
    propagate into worker threads, and several e2e helpers call out from
    threads. Under xdist each worker is its own process, so there is no
    cross-test bleed beyond what the per-test reset already handles.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._steps: list[str] = []

    def reset(self) -> None:
        """Called by an autouse fixture at setup, so each test starts empty."""
        with self._lock:
            self._steps.clear()

    def record(self, label: str) -> None:
        cleaned = " ".join(label.split())[:MAX_STEP_CHARS]
        if not cleaned:
            return
        with self._lock:
            # A poll loop is one step in the story, not fifty.
            if self._steps and self._steps[-1] == cleaned:
                return
            if len(self._steps) >= MAX_STEPS:
                return
            self._steps.append(cleaned)

    def taken(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._steps)


STEPS: Final = _StepRecorder()


def step(label: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Record `label` on the running test whenever this helper is called.

    Goes on HARNESS helpers (client methods, fixtures), never on tests. The
    label is recorded BEFORE the wrapped call, so a helper that raises still
    leaves its own label as the last element -- which is the whole point: the
    last step is where the test died.
    """

    def decorate(fn: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(fn)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            STEPS.record(label)
            return fn(*args, **kwargs)

        return wrapper

    return decorate


# --- Serialization: asdict() -> <property> pairs, no per-field plumbing -------

# The only field-specific knowledge in the module: which fields are plural, and
# what their repeated <property> is called. A new SCALAR field needs no edit.
_REPEATED: Final[dict[str, str]] = {"capabilities": "capability"}


def _scalar(value: object) -> str:
    """`str(member)` on a (str, Enum) gives 'Route.RESPONSES', not 'responses'
    -- StrEnum would not, but it is 3.11+ and this repo floors at 3.10. So the
    value is read explicitly, once, for every enum field."""
    if isinstance(value, Enum):
        # `Enum.value` is `Any` by construction, for every enum there has ever
        # been; this is the one place in the module that reads it, and `str()`
        # lands it back in the type system immediately.
        return str(value.value)  # pyright: ignore[reportAny]  # Enum.value is Any for every enum
    return str(value)


def _members(value: object) -> tuple[object, ...]:
    """The elements of a plural field, whatever `asdict` rebuilt it as.

    `asdict` reconstructs a tuple field as a tuple of the same members, but hands
    it back inside an untyped dict, so the elements are re-declared as plain
    objects here and converted by `_scalar` like any other value.
    """
    return cast("tuple[object, ...]", value) if isinstance(value, tuple) else ()


def _declared_subject(args: tuple[object, ...]) -> Subject | None:
    """The `Subject` a `meta` marker carries, or None for anything else.

    `Mark.args` is `tuple[Any, ...]`; taking it as `tuple[object, ...]` is what
    keeps the Any from leaking past this line. A bare `@pytest.mark.meta` (no
    args) and a `@pytest.mark.meta("spend-budgets")` (wrong type) both land here,
    and neither may produce a property whose value is a repr.
    """
    first = args[0] if args else None
    return first if isinstance(first, Subject) else None


def subject_properties(item: pytest.Item) -> tuple[tuple[str, str], ...]:
    """The declared half, in dataclass field order. Empty fields emit nothing;
    the emitter is what guarantees every key exists in the JSON."""
    from dataclasses import asdict  # local: keeps module import trivially cheap

    marker = item.get_closest_marker("meta")
    if marker is None:
        return ()
    subject = _declared_subject(marker.args)
    if subject is None:
        return ()
    fields: dict[str, object] = asdict(subject)
    pairs: list[tuple[str, str]] = []
    for name, value in fields.items():
        repeated = _REPEATED.get(name)
        if repeated is not None:
            pairs.extend((repeated, _scalar(member)) for member in _members(value))
        elif value is not None and value != "":
            pairs.append((name, _scalar(value)))
    return tuple(pairs)


def step_properties() -> tuple[tuple[str, str], ...]:
    """The recorded half. Appended after the call phase, never at collection."""
    return tuple(("step", label) for label in STEPS.taken())
