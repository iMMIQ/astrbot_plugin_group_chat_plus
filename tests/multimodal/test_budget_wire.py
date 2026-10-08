import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from multimodal.budget import BudgetEstimator, calibrated, profile_key
from multimodal.context import ContextLimit, ContextSelector
from multimodal.storage.assets import MediaStore
from multimodal.storage.sqlite import Journal
from multimodal.wire import WireObserver, request_metadata


def sample(prompt=5000, raw=10000, images=0, **changes):
    return {
        "estimate_version": 1,
        "status": "completed",
        "stage": "first",
        "input_tokens": prompt,
        "raw_text_tokens": raw,
        "images": images,
        "raw_image_tokens": images * 1600,
    } | changes


def response(prompt=5000, cached=3000):
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
            completion_tokens=20,
        )
    )


def test_calibration_warmup_margin_and_rare_underestimate():
    cfg = {"model": "glm-5.3-flash"}
    assert calibrated(cfg, []).text_scale == 0.75
    assert calibrated({"model": "other"}, []).text_scale == 1
    assert calibrated(cfg, [sample()] * 7).text_scale == 0.75
    estimate = calibrated(cfg, [sample()] * 8)
    assert estimate.text_scale == pytest.approx(0.6)
    assert estimate.text_samples == 8
    # A single large call increases the allowance immediately, even in warmup.
    assert calibrated(cfg, [sample(prompt=12000)]).text_scale == pytest.approx(1.44)
    assert calibrated(cfg, [sample()] * 95 + [sample(prompt=12000)]).text_scale == pytest.approx(1.44)


@pytest.mark.parametrize(
    "bad",
    [
        sample(status="error"),
        sample(status="aborted"),
        sample(stage="tool"),
        sample(input_tokens=None),
        sample(prompt=0),
        sample(raw=200),
        sample(estimate_version=0),
    ],
)
def test_invalid_or_continuation_usage_cannot_train(bad):
    assert calibrated({"model": "glm-5.3-flash"}, [bad] * 12).text_scale == 0.75


def test_images_never_reduce_reserve_and_do_not_train_text():
    cfg = {"model": "glm-5.3-flash"}
    estimate = calibrated(cfg, [sample(prompt=6000, images=2)] * 12)
    assert estimate.text_samples == 0
    assert estimate.text_scale == 0.75 and estimate.image(1600) == 1600
    expensive = calibrated(cfg, [sample(prompt=16000, images=2)])
    assert expensive.image_scale > 1
    assert expensive.image(16384) >= 16384
    text_then_image = calibrated(cfg, [sample()] * 8 + [sample(prompt=11000, images=2)])
    assert text_then_image.text_scale == pytest.approx(0.6)
    assert text_then_image.image_scale == pytest.approx(1.875)


def test_final_body_overrides_hashes_without_retaining_content():
    config = {"id": "fixture-provider", "api_base": "https://fixture.invalid"}
    kwargs = {
        "model": "fixture-model",
        "messages": [{"role": "user", "content": "PRIVATE_ORIGINAL"}],
        "tools": [],
        "extra_headers": {"Authorization": "PRIVATE_CREDENTIAL"},
        "extra_body": {
            "reasoning_effort": "low",
            "messages": [
                {"role": "system", "content": "PRIVATE_PERSONA"},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,PRIVATE_IMAGE"}}
                    ],
                },
            ],
        },
    }
    first = request_metadata(kwargs, config, 1600)
    wire = json.dumps(first)
    assert "PRIVATE" not in wire and "data:image" not in wire
    assert first["images"] == 1 and first["raw_image_tokens"] == 1600
    kwargs["extra_headers"]["Authorization"] = "ANOTHER_CREDENTIAL"
    assert request_metadata(kwargs, config, 1600) == first
    kwargs["extra_body"]["messages"][-1]["content"][0]["image_url"]["url"] += "CHANGED"
    second = request_metadata(kwargs, config, 1600)
    assert second["messages"] != first["messages"] and second["image_hashes"] != first["image_hashes"]
    assert second["raw_text_tokens"] == first["raw_text_tokens"]
    kwargs["extra_body"]["reasoning_effort"] = "high"
    assert request_metadata(kwargs, config, 1600)["parameters"] != second["parameters"]


async def test_attempts_fallback_and_tool_rounds_are_separate():
    records = []

    async def record(payload):
        records.append(payload)

    observer = WireObserver(record)

    async def fail(**kwargs):
        raise ValueError("PRIVATE_ERROR")

    async def ok(**kwargs):
        return response()

    messages = [{"role": "user", "content": "PRIVATE_QUESTION"}]
    with pytest.raises(ValueError, match="PRIVATE_ERROR"):
        await observer.wrap(fail, {"id": "primary"})(messages=messages, model="model-a")
    await observer.wrap(ok, {"id": "fallback"})(messages=messages, model="model-b")
    await observer.wrap(ok, {"id": "fallback"})(
        messages=messages
        + [{"role": "assistant", "tool_calls": []}, {"role": "tool", "content": "PRIVATE_RESULT"}],
        model="model-b",
    )
    assert [r["stage"] for r in records] == ["first", "first", "tool"]
    assert [r["attempt"] for r in records] == [1, 2, 3]
    assert [r["call"] for r in records] == [1, 1, 2]
    assert [r["status"] for r in records] == ["error", "completed", "completed"]
    assert records[1]["input_tokens"] == 5000 and records[2]["cached_tokens"] == 3000
    assert records[0]["profile"] != records[1]["profile"]
    assert "PRIVATE" not in json.dumps(records)


async def test_observation_failure_does_not_change_response_or_cancellation():
    async def broken(payload):
        raise OSError("fixture storage failure")

    result = response()

    async def ok(**kwargs):
        return result

    async def cancelled(**kwargs):
        raise asyncio.CancelledError

    observer = WireObserver(broken)
    assert await observer.wrap(ok, {})(messages=[]) is result
    with pytest.raises(asyncio.CancelledError):
        await observer.wrap(cancelled, {})(messages=[])


async def test_profiles_persist_isolate_models_and_expire_without_double_counting(tmp_path):
    journal = Journal(tmp_path)
    await journal.ready()
    cfg = {"id": "fixture", "api_base": "https://fixture.invalid"}
    first, second = profile_key(cfg, "model-a"), profile_key(cfg, "model-b")
    try:
        for index in range(8):
            await journal.diagnose("room-a", first, "wire_reply_first", sample() | {"anchor": index})
        await journal.diagnose("room-b", second, "wire_reply_first", sample(prompt=12000))
        await journal.diagnose("room-a", first, "wire_reply_tool", sample(stage="tool", prompt=40000))
        assert calibrated({}, await journal.budget_samples(first)).text_scale == pytest.approx(0.6)
        assert len(await journal.budget_samples(second)) == 1
        assert (await journal.status("room-a"))["trace"] == []
        await journal.reset("room-a")
        assert len(await journal.budget_samples(first)) == 8  # token estimates contain no memory
        await journal.close()
        journal = Journal(tmp_path)
        await journal.ready()
        assert len(await journal.budget_samples(first)) == 8
        await journal.prune_events(time.time() + 1)
        assert await journal.budget_samples(first) == []
    finally:
        await journal.close()


async def test_calibrated_chinese_context_retains_more_history_and_protects_anchor(tmp_path):
    journal = Journal(tmp_path)
    await journal.ready()
    media = MediaStore(journal, tmp_path / "media", {})
    selector = ContextSelector(journal, media, {"input_token_budget": 10000})
    try:
        for i in range(6):
            anchor, _ = await journal.add(
                "room", str(i), "alice", "Alice", [{"type": "text", "text": "中文旅行计划" * 180}]
            )
        conservative = await selector.candidates(anchor)
        await selector.assemble(conservative, freeze=False)
        revised = await selector.candidates(anchor)
        revised.estimator = BudgetEstimator(text_scale=0.6)
        await selector.assemble(revised, freeze=False)
        assert len(revised.chosen_events) > len(conservative.chosen_events)
        assert revised.tokens <= 10000
        assert anchor["seq"] in {e["seq"] for e in revised.chosen_events}
        with pytest.raises(ContextLimit):
            await selector.assemble(revised, total_limit=100, freeze=False)
    finally:
        await media.close()
        await journal.close()
