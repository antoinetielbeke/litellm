"""Harness coverage for the custom JUnit properties.

No proxy and no ``e2e`` marker. Pins the two normalizations that have to agree
about where a suite file lives -- ``package_from_nodeid`` (strip the suite root)
and ``source_from_location`` (re-root at it) -- across both ways the suite is
launched, plus the one-based line offset and the refusal to emit a path that
escapes the suite. The consumers of these properties are the Loki/Grafana
rollups and, for ``source``, the status page's per-test links to GitHub.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
from e2e_metadata import (
    MAX_STEPS,
    STEPS,
    Capability,
    Domain,
    Mode,
    Route,
    Subject,
    meta,
    step,
    step_properties,
    subject_properties,
)
from junit_properties import (
    SUITE_ROOT,
    attach_result_properties,
    attach_step_properties,
    dedupe_covers,
    package_from_nodeid,
    result_properties,
    source_from_location,
    suite_parts,
)


def collected_item(request: pytest.FixtureRequest, name: str) -> pytest.Item:
    """The Item pytest collected for test ``name`` in this file: the real nodeid,
    location and marker machinery the collection hook reads, as pytest built it."""
    return next(item for item in request.session.items if item.path == request.path and item.name == name)


def repo_root() -> Path | None:
    """The litellm checkout above this file, or None when there isn't one."""
    return next((p for p in Path(__file__).resolve().parents if (p / ".git").exists()), None)


class TestSuiteParts:
    @pytest.mark.parametrize(
        "path",
        ["logging/test_x.py", "tests/e2e/logging/test_x.py", "./logging/test_x.py", "tests\\e2e\\logging\\test_x.py"],
    )
    def test_both_invocation_shapes_collapse_to_the_same_components(self, path: str) -> None:
        """A repo-root run and a suite-cwd run report the same file differently;
        every downstream signal has to see one spelling."""
        assert suite_parts(path) == ("logging", "test_x.py")

    def test_top_level_suite_file_keeps_its_single_component(self) -> None:
        assert suite_parts("tests/e2e/test_fixture_mode.py") == ("test_fixture_mode.py",)


class TestPackageFromNodeid:
    @pytest.mark.parametrize(
        ("nodeid", "expected"),
        [
            ("logging/test_x.py::TestFoo::test_bar", "logging"),
            ("tests/e2e/logging/test_x.py::TestFoo::test_bar", "logging"),
            ("quota_management/spend_tracking/test_x.py::test_bar", "quota_management"),
            ("test_fixture_mode.py::TestParseFixtureMode::test_known_values_normalize", "root"),
            ("tests/e2e/test_fixture_mode.py::test_bar", "root"),
        ],
    )
    def test_package_is_the_first_dir_under_the_suite_root(self, nodeid: str, expected: str) -> None:
        assert package_from_nodeid(nodeid) == expected


class TestSourceFromLocation:
    @pytest.mark.parametrize("path", ["a2a/test_a2a_agent_e2e.py", "tests/e2e/a2a/test_a2a_agent_e2e.py"])
    def test_path_is_repo_relative_however_pytest_was_started(self, path: str) -> None:
        assert source_from_location(path, 40) == "tests/e2e/a2a/test_a2a_agent_e2e.py:41"

    def test_line_is_emitted_one_based(self) -> None:
        """pytest.Item.location counts from 0; editors, tracebacks and GitHub's
        #L anchor all count from 1, and an off-by-one lands on the decorator."""
        assert source_from_location("a2a/test_x.py", 0) == "tests/e2e/a2a/test_x.py:1"

    def test_top_level_suite_file_sits_directly_under_the_suite_root(self) -> None:
        assert source_from_location("test_fixture_mode.py", 39) == "tests/e2e/test_fixture_mode.py:40"

    @pytest.mark.parametrize(
        ("path", "lineno"),
        [
            ("a2a/test_x.py", None),
            ("/app/e2e/a2a/test_x.py", 40),
            ("C:\\app\\e2e\\a2a\\test_x.py", 40),
            ("../conftest.py", 40),
            ("", 40),
        ],
    )
    def test_nothing_linkable_yields_empty_rather_than_a_guess(self, path: str, lineno: int | None) -> None:
        """A colon is rejected on two counts: it is how a Windows absolute path
        arrives, and `path:line` cannot represent one in the path half."""
        assert source_from_location(path, lineno) == ""


class TestResultProperties:
    def test_every_test_carries_package_covers_and_source(self, request: pytest.FixtureRequest) -> None:
        """Read off this test's own collected Item, so the nodeid and location are
        whatever pytest reports for the launch shape in use, and the marker is added
        at run time so the coverage registry's collect-only pass never sees it."""
        test = type(self).test_every_test_carries_package_covers_and_source
        request.applymarker(pytest.mark.covers("LOG-1", "LOG-2"))
        assert result_properties(collected_item(request, test.__name__)) == (
            ("package", "root"),
            ("covers", "LOG-1,LOG-2"),
            ("source", f"tests/e2e/test_junit_properties.py:{test.__code__.co_firstlineno}"),
        )

    def test_attach_is_idempotent(self, request: pytest.FixtureRequest) -> None:
        """Collection can run the hook more than once; a second pass must not
        double the <property> entries in the report."""
        item = collected_item(request, type(self).test_attach_is_idempotent.__name__)
        attach_result_properties(item)
        attach_result_properties(item)
        assert [name for name, _ in item.user_properties] == ["package", "covers", "source"]


class TestSuiteRoot:
    def test_suite_root_names_this_file_s_real_home(self) -> None:
        """SUITE_ROOT is hardcoded because the runner image has no repo to read it
        from. Where there IS a checkout, prove the constant still points at us --
        otherwise a moved tests/e2e/ ships links that 404."""
        root = repo_root()
        if root is None:
            pytest.skip("no checkout above this file (the runner image copies tests/e2e/ to /app/e2e)")
        assert (root / SUITE_ROOT / Path(__file__).name).resolve() == Path(__file__).resolve()


class TestDedupeCovers:
    def test_ids_are_unique_order_preserving_and_non_empty_strings(self) -> None:
        assert dedupe_covers([("A", "B"), ("B", ""), ("C", 7)]) == ("A", "B", "C")


def fixed_prefix(covers: str, lineno: int) -> tuple[tuple[str, str], ...]:
    """The three-tuple every testcase in this suite has carried since before the
    typed marker existed. Spelled out rather than derived, so a change to its
    shape or order fails a test instead of agreeing with itself."""
    return (
        ("package", "root"),
        ("covers", covers),
        ("source", f"tests/e2e/test_junit_properties.py:{lineno}"),
    )


class TestSubjectProperties:
    """The declared half: `@meta(Subject(...))` -> `<property>` pairs.

    Markers are applied at run time via `request.applymarker`, the idiom the
    `covers` tests above already use, so the coverage registry's collect-only pass
    never sees a marker that exists only to be serialized.
    """

    def test_every_declared_field_becomes_a_property_in_field_order(self, request: pytest.FixtureRequest) -> None:
        """One pass over `dataclasses.asdict`: declaration order is emission order,
        a (str, Enum) member is written as its `.value` and never `str(member)`,
        the plural field emits a repeated SINGULAR `capability`, and an unset field
        (`provider` here) emits nothing at all."""
        test = type(self).test_every_declared_field_becomes_a_property_in_field_order
        request.applymarker(
            meta(
                Subject(
                    domain=Domain.SPEND_BUDGETS,
                    route=Route.CHAT_COMPLETIONS,
                    model="gpt-5.5",
                    capabilities=(Capability.VISION, Capability.FUNCTION_CALLING, Capability.VISION),
                    mode=Mode.NONSTREAM,
                )
            )
        )
        assert subject_properties(collected_item(request, test.__name__)) == (
            ("domain", "spend-budgets"),
            ("route", "chat_completions"),
            ("model", "gpt-5.5"),
            ("capability", "function_calling"),
            ("capability", "vision"),
            ("mode", "nonstream"),
        )

    def test_scalar_property_names_are_the_dataclass_field_names(self, request: pytest.FixtureRequest) -> None:
        """The mapping is `asdict`, not a hand-written table: a scalar field added
        to `Subject` later serializes under its own name with no edit to the
        serializer. Proven by reading the field list back off the dataclass."""
        test = type(self).test_scalar_property_names_are_the_dataclass_field_names
        request.applymarker(meta(Subject(domain=Domain.UNKNOWN, route=Route.HEALTH, model="m", mode=Mode.STREAM)))
        declared = tuple(field.name for field in fields(Subject))
        emitted = tuple(name for name, _ in subject_properties(collected_item(request, test.__name__)))
        assert emitted == tuple(name for name in declared if name in {"domain", "route", "model", "mode"})

    def test_capabilities_are_deduped_and_sorted_at_declaration(self) -> None:
        """Canonicalized in `__post_init__`, so two tests that spelled the same set
        in different orders produce byte-identical properties and the committed run
        files diff cleanly."""
        assert Subject(capabilities=(Capability.VISION, Capability.REASONING, Capability.VISION)).capabilities == (
            Capability.REASONING,
            Capability.VISION,
        )

    def test_the_typed_marker_only_ever_appends_to_the_fixed_prefix(self, request: pytest.FixtureRequest) -> None:
        """Loki, Grafana and the status page read `package`/`covers`/`source`; the
        typed fields ride behind them and must not disturb them."""
        test = type(self).test_the_typed_marker_only_ever_appends_to_the_fixed_prefix
        request.applymarker(pytest.mark.covers("quota_management.budget.key.blocks_over_limit"))
        request.applymarker(meta(Subject(route=Route.SPEND_REPORTING)))
        assert result_properties(collected_item(request, test.__name__)) == fixed_prefix(
            "quota_management.budget.key.blocks_over_limit", test.__code__.co_firstlineno
        ) + (("route", "spend_reporting"),)

    def test_a_test_with_only_the_old_string_covers_is_unchanged(self, request: pytest.FixtureRequest) -> None:
        """The existing `@pytest.mark.covers("cell.id")` call sites keep emitting
        exactly what they emitted before the typed marker existed."""
        test = type(self).test_a_test_with_only_the_old_string_covers_is_unchanged
        request.applymarker(pytest.mark.covers("llm.responses.openai.tool_use.nonstream.works"))
        assert result_properties(collected_item(request, test.__name__)) == fixed_prefix(
            "llm.responses.openai.tool_use.nonstream.works", test.__code__.co_firstlineno
        )

    def test_a_test_with_neither_marker_carries_only_the_prefix(self, request: pytest.FixtureRequest) -> None:
        """Which is every test in the suite until the backfill lands: an empty
        `covers` and no typed properties at all, never five empty ones."""
        test = type(self).test_a_test_with_neither_marker_carries_only_the_prefix
        item = collected_item(request, test.__name__)
        assert subject_properties(item) == ()
        assert result_properties(item) == fixed_prefix("", test.__code__.co_firstlineno)

    def test_a_marker_carrying_something_other_than_a_subject_emits_nothing(
        self, request: pytest.FixtureRequest
    ) -> None:
        """`@meta` is typed, but `pytest.mark.meta` is not, and a bare
        `@pytest.mark.meta` carries no args at all. Neither may produce a property
        whose value is a repr."""
        test = type(self).test_a_marker_carrying_something_other_than_a_subject_emits_nothing
        request.applymarker(pytest.mark.meta("spend-budgets"))
        assert subject_properties(collected_item(request, test.__name__)) == ()


class TestStepRecording:
    """The recorded half: `@step`-decorated harness helpers append to the running
    test's story as they execute.

    Each test here starts from an empty log because conftest's autouse
    `_record_steps` fixture resets the recorder at setup -- the same reset the
    live suite relies on for per-test isolation.
    """

    def test_steps_land_in_call_order(self) -> None:
        @step("register deployment")
        def register() -> str:
            return "model-id"

        @step("generate virtual key")
        def generate() -> str:
            return "sk-x"

        _ = register()
        _ = generate()
        assert STEPS.taken() == ("register deployment", "generate virtual key")

    def test_a_decorated_helper_still_returns_exactly_what_it_did(self) -> None:
        """`@step` records, it does not intercept: arguments, return value and
        `__name__` all survive it, so decorating a live harness method cannot
        change what the test observes."""

        @step("POST /chat/completions")
        def chat(key: str, *, model: str) -> str:
            return f"{key}:{model}"

        assert chat("sk-x", model="gpt-5.5") == "sk-x:gpt-5.5"
        assert chat.__name__ == "chat"

    def test_a_helper_that_raises_leaves_its_own_label_last(self) -> None:
        """The whole point of the field. The label is recorded BEFORE the call, so
        a test that dies inside a helper keeps a partial story whose last element
        names the helper it died in."""

        @step("generate virtual key")
        def generate() -> str:
            return "sk-x"

        @step("POST /chat/completions")
        def chat() -> None:
            raise RuntimeError("502 from upstream")

        _ = generate()
        with pytest.raises(RuntimeError, match="502 from upstream"):
            chat()
        assert STEPS.taken() == ("generate virtual key", "POST /chat/completions")

    def test_a_poll_loop_is_one_step_in_the_story_not_fifty(self) -> None:
        @step("poll /spend/logs for the request id")
        def poll() -> None:
            return None

        for _ in range(20):
            poll()
        assert STEPS.taken() == ("poll /spend/logs for the request id",)

    def test_the_same_label_recorded_again_later_is_a_new_step(self) -> None:
        """Only CONSECUTIVE duplicates collapse; a helper called again after
        something else happened is a genuine second beat of the story."""
        STEPS.record("POST /chat/completions")
        STEPS.record("poll /spend/logs")
        STEPS.record("POST /chat/completions")
        assert STEPS.taken() == ("POST /chat/completions", "poll /spend/logs", "POST /chat/completions")

    def test_the_log_is_capped_so_a_load_test_cannot_bury_the_story(self) -> None:
        for index in range(MAX_STEPS * 2):
            STEPS.record(f"call {index}")
        taken = STEPS.taken()
        assert len(taken) == MAX_STEPS
        assert taken[0] == "call 0"

    def test_whitespace_is_normalized_and_an_empty_label_records_nothing(self) -> None:
        STEPS.record("  POST   /chat/completions\n  ")
        STEPS.record("   ")
        assert STEPS.taken() == ("POST /chat/completions",)

    def test_reset_empties_the_log_so_one_test_never_inherits_another_s(self) -> None:
        STEPS.record("register deployment")
        STEPS.reset()
        assert STEPS.taken() == ()
        assert step_properties() == ()

    def test_steps_serialize_as_repeated_properties_in_order(self) -> None:
        """Repeated rather than joined on a delimiter: the labels are free text, so
        no separator can be reserved, and a repeated property has none to corrupt."""
        STEPS.record('attach guardrail, comma & "quoted" <tag>')
        STEPS.record("POST /chat/completions")
        assert step_properties() == (
            ("step", 'attach guardrail, comma & "quoted" <tag>'),
            ("step", "POST /chat/completions"),
        )


class TestAttachStepProperties:
    def test_steps_are_appended_after_the_declared_properties(self, request: pytest.FixtureRequest) -> None:
        """Order inside `<properties>` is list order, so the story reads after the
        fixed prefix the collection hook already attached."""
        test = type(self).test_steps_are_appended_after_the_declared_properties
        item = collected_item(request, test.__name__)
        STEPS.record("register deployment")
        STEPS.record("POST /chat/completions")
        attach_step_properties(item)
        assert [name for name, _ in item.user_properties] == ["package", "covers", "source", "step", "step"]
        assert [value for name, value in item.user_properties if name == "step"] == [
            "register deployment",
            "POST /chat/completions",
        ]

    def test_a_rerun_replaces_the_story_rather_than_appending_a_second_one(
        self, request: pytest.FixtureRequest
    ) -> None:
        """The suite runs with `--reruns 1`. Without this the retry's steps would
        queue up behind the first attempt's and the report would read as one test
        that did everything twice."""
        test = type(self).test_a_rerun_replaces_the_story_rather_than_appending_a_second_one
        item = collected_item(request, test.__name__)
        STEPS.record("attempt one died here")
        attach_step_properties(item)
        STEPS.reset()
        STEPS.record("attempt two got further")
        attach_step_properties(item)
        assert [value for name, value in item.user_properties if name == "step"] == ["attempt two got further"]
