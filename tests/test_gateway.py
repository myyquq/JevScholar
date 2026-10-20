"""Gateway: pure request normalisation, then a real Gateway in front of a scripted fake upstream."""

from __future__ import annotations

import asyncio
import copy
import json
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import openai
import pytest
import requests
from aiohttp import web

from jevscholar import gateway, ledger
from jevscholar.config import Alias, Budget, ModelsConfig, Price, Upstream
from jevscholar.gateway import BadRequest, Gateway, normalize, repair_tools
from jevscholar.types import BENCHMARKS, LOCAL_KEY, make_tag

MSG = [{"role": "user", "content": "hi"}]
COST = (800 * 1.0 + 200 * 0.1 + 100 * 2.0) / 1e6  # list USD of one fake reply
TAG = make_tag("r1", "deepscholar", "sysA", "t1", "gen")
CELLS = [(b, s) for b in BENCHMARKS for s in ("sysA", "sysB", "sys:A")]  # the cells tests tag
BUSY = (503, {"error": {"message": "busy"}})


# --------------------------------------------------------------------------- normalisation
def alias(**kw: Any) -> Alias:
    return Alias(upstream="u", model="m", **kw)


TOOL = {"type": "function", "function": {"name": "f", "parameters": {}}}
STRICT = {**TOOL, "function": {**TOOL["function"], "strict": True}}
ON, OFF, DS = {"thinking": True}, {"thinking": False}, "deepseek"
RE, MCT = "reasoning_effort", "max_completion_tokens"
PTC, RF = "parallel_tool_calls", "response_format"
JO, DEF, FORCE = {"type": "json_object"}, {"default_max_tokens": 4096}, {"force_json_object": True}
SYS = {"role": "system", "content": "Be brief."}
JSYS = {"role": "system", "content": "Respond in JSON."}
CASES = [  # (id, alias fields, dialect, request fields, expected fields, absent fields)
    ("ds-off", OFF, DS, {RE: "medium", "temperature": 0.0}, {"temperature": 0.0}, [RE]),
    ("ds-off-explicit", OFF, DS, {}, {"model": "m", "thinking": {"type": "disabled"}}, []),
    ("ds-on", ON, DS, {RE: "medium", "temperature": 1.0}, {RE: "high"}, ["temperature"]),
    ("ds-on-explicit", ON, DS, {}, {"thinking": {"type": "enabled"}}, [RE]),
    ("ds-minimal", ON, DS, {RE: "minimal"}, {RE: "low"}, []),
    ("ds-ultra", ON, DS, {RE: "ultra"}, {RE: "max"}, []),
    ("ds-alias-effort", {**ON, "effort": "low"}, DS, {RE: "xhigh"}, {RE: "low"}, []),
    ("ds-whitelist", {}, DS, {"top_p": 0.9, "seed": 1, PTC: True}, {}, ["top_p", "seed", PTC]),
    ("ds-cap", {}, DS, {MCT: 50000}, {"max_tokens": 32768}, [MCT]),
    ("ds-default-tokens", DEF, DS, {}, {"max_tokens": 4096}, []),
    ("ds-client-tokens", DEF, DS, {"max_tokens": 9}, {"max_tokens": 9}, []),
    ("ds-json-word", {}, DS, {RF: JO}, {"messages": [JSYS, *MSG]}, []),
    ("ds-force", FORCE, DS, {"messages": [{"role": "user", "content": "JSON"}]}, {RF: JO}, []),
    ("ds-force-no-word", FORCE, DS, {}, {}, [RF]),
    ("ds-developer", {}, DS, {"messages": [{**SYS, "role": "developer"}]}, {"messages": [SYS]}, []),
    ("ds-strict", {}, DS, {"tools": [STRICT]}, {"tools": [TOOL]}, []),
    ("llama-think", OFF, "llama", {}, {"chat_template_kwargs": {"enable_thinking": False}}, []),
    ("llama-tokens", {}, "llama", {MCT: 9, "x": 1}, {"max_tokens": 9}, [MCT, "x", "thinking"]),
    ("openai", {}, "openai", {"x": 1, MCT: 9}, {"x": 1, MCT: 9}, ["thinking", "max_tokens"]),
    ("passthrough", {"passthrough": True}, DS, {"x": 1, MCT: 9}, {"x": 1, MCT: 9}, ["thinking"]),
]


@pytest.mark.parametrize(
    ("al", "dialect", "req", "want", "absent"), [c[1:] for c in CASES], ids=[c[0] for c in CASES]
)
def test_normalize(al: dict, dialect: str, req: dict, want: dict, absent: list) -> None:
    body = {"model": "alias", "messages": MSG, **req}
    frozen = copy.deepcopy(body)
    out, _, schema = normalize(body, alias(**al), Upstream(base_url="http://x", dialect=dialect))
    assert {k: out.get(k) for k in want} == want
    assert not set(absent) & out.keys()
    assert schema is None and body == frozen


def test_json_schema_rejections_and_orphan_repair() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    js = {"type": "json_schema", "json_schema": {"name": "S", "schema": schema, "strict": True}}
    body = {"model": "x", "messages": [{"role": "system", "content": "Judge."}, *MSG], "seed": 1}
    ds = Upstream(base_url="http://x", dialect="deepseek")
    out, notes, got = normalize(body | {"response_format": js}, alias(thinking=False), ds)
    assert got == schema and out["response_format"] == {"type": "json_object"}
    assert out["messages"][0]["content"] == (
        "Judge.\n\nReturn only a JSON object, no markdown, conforming to this JSON schema: "
        + json.dumps(schema)
    )
    assert notes == ["json_schema->json_object", "max_tokens:default->32768", "dropped:seed"]
    for bad in ({"stream": True}, {"n": 2}):
        with pytest.raises(BadRequest):
            normalize(body | bad, alias(), ds)

    def call(*ids: str) -> dict:
        fn = {"name": "f", "arguments": "{}"}
        calls = [{"id": i, "type": "function", "function": fn} for i in ids]
        return {"role": "assistant", "content": None, "tool_calls": calls}

    def tool(i: str, content: str = "out") -> dict:
        return {"role": "tool", "tool_call_id": i, "content": content}

    msgs = [
        MSG[0],
        tool("zz"),
        call("a", "b"),
        tool("b"),
        {"role": "user", "content": "go"},
        call("c"),
    ]
    out, notes = repair_tools(msgs)
    stub = "[output trimmed]"
    assert out == [MSG[0], msgs[2], tool("b"), tool("a", stub), msgs[4], msgs[5], tool("c", stub)]
    assert notes == ["orphan_repair:+2stub,-1tool"]
    answered = [*msgs[2:4], tool("a")]
    assert repair_tools(answered) == (answered, [])


# --------------------------------------------------------------------------- fake upstream
def reply(content: str = "ok", finish: str = "stop") -> dict:
    message = {"role": "assistant", "content": content}
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "total_tokens": 1100,
        "prompt_cache_hit_tokens": 200,
        "prompt_cache_miss_tokens": 800,
        "prompt_tokens_details": {"cached_tokens": 200},
    }
    choice = {"index": 0, "finish_reason": finish, "message": message}
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "flash",
        "choices": [choice],
        "usage": usage,
    }


class Fake:
    """Scripted upstream: pops (status, body) per request, else replies 'reply <n>'."""

    def __init__(self) -> None:
        self.script: list[tuple[int, dict]] = []
        self.seen: list[dict] = []
        self.auth: list[str | None] = []
        self.delay = 0.0
        self.url = ""

    async def handle(self, request: web.Request) -> web.Response:
        self.seen.append(await request.json())
        self.auth.append(request.headers.get("Authorization"))
        await asyncio.sleep(self.delay)
        default = (200, reply(f"reply {len(self.seen)}"))
        status, body = self.script.pop(0) if self.script else default
        return web.json_response(body, status=status)


@pytest.fixture
def fake() -> Iterator[Fake]:
    f, loop = Fake(), asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    async def up() -> web.AppRunner:
        app = web.Application()
        app.router.add_post("/chat/completions", f.handle)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        return runner

    runner = asyncio.run_coroutine_threadsafe(up(), loop).result()
    f.url = f"http://127.0.0.1:{runner.addresses[0][1]}"
    yield f
    asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result()
    loop.call_soon_threadsafe(loop.stop)
    thread.join()


@pytest.fixture
def make_gw(tmp_path: Path, fake: Fake, monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable]:
    monkeypatch.setattr(gateway, "BACKOFF_S", 0)
    monkeypatch.setattr(ledger, "is_peak", lambda _: False)
    price = Price(miss=1.0, hit=0.1, out=2.0)
    models = ModelsConfig(
        campaign_usd=10.0,
        upstreams={
            "ds": Upstream(
                base_url=fake.url, key_env="DS_KEY", dialect="deepseek", offpeak_discount=0.5
            ),
            "local": Upstream(base_url=fake.url, dialect="llama", billed=False),
        },
        aliases={
            "exec": Alias(upstream="ds", model="flash", thinking=False, cache=True, price=price),
            "judge": Alias(upstream="ds", model="flash", thinking=False, price=price),
            "local": Alias(upstream="local", model="qwen", price=price),
        },
        tools={"arxiv_search": 0.01, "tavily_search": 0.02},
        free_tools=("arxiv_search",),
    )
    started: list[Gateway] = []

    def make(budget: Budget | None = None, *, replay_only: bool = False) -> Gateway:
        budget = budget or Budget()
        gw = Gateway(
            models,
            tmp_path / "run",
            budget,
            dict.fromkeys(CELLS, 2.0 if budget.task_usd is None else budget.task_usd),
            secrets={"DS_KEY": "sk-test"},
            cache_dir=tmp_path / "cache",
            replay_only=replay_only,
        )
        gw.start()
        started.append(gw)
        return gw

    yield make
    for gw in started:
        gw.stop()


def chat(gw: Gateway, tag: str = TAG, timeout: float = 10, **body: Any) -> requests.Response:
    body = {"model": "exec", "messages": MSG, **body}
    return requests.post(f"{gw.url}/t/{tag}/v1/chat/completions", json=body, timeout=timeout)


def content(r: requests.Response) -> str:
    return r.json()["choices"][0]["message"]["content"]


def code(r: requests.Response) -> str:
    assert r.headers["x-should-retry"] == "false"
    return r.json()["error"]["code"]


def events(gw: Gateway) -> list[dict]:
    return [json.loads(line) for line in gw.ledger.path.read_text().splitlines()]


def usage(gw: Gateway, tag: str) -> dict:
    return requests.get(f"{gw.url}/t/{tag}/usage", timeout=10).json()


# --------------------------------------------------------------------------- gateway
def test_json_guard_reasks_until_valid(make_gw: Callable, fake: Fake) -> None:
    gw = make_gw()
    fake.script = [
        (200, reply("not json")),
        (200, reply('```json\n{"b": 1}\n```')),
        (200, reply('```json\n{"a": 1}\n```')),
    ]
    rf = {"type": "json_schema", "json_schema": {"name": "S", "schema": {"required": ["a"]}}}
    assert content(chat(gw, model="judge", response_format=rf)) == '{"a": 1}'
    assert [len(b["messages"]) for b in fake.seen] == [2, 4, 6]
    assert fake.seen[2]["messages"][-2]["content"] == '```json\n{"b": 1}\n```'
    assert all(b["response_format"] == {"type": "json_object"} for b in fake.seen)
    assert [e["notes"] for e in events(gw)][1:] == [["json_guard:1"], ["json_guard:2"]]


def test_cache_replays_by_occurrence(make_gw: Callable, fake: Fake) -> None:
    gw = make_gw()
    fake.script = [BUSY] * 5  # the first occurrence fails: it must not shift the replay index
    assert code(chat(gw)) == "upstream_unavailable"
    assert [content(chat(gw)) for _ in range(2)] == ["reply 6", "reply 7"]
    ablation = make_tag("r2", "deepscholar", "sysB", "t1", "gen")  # other run and system
    assert [content(chat(gw, ablation)) for _ in range(3)] == ["reply 6", "reply 7", "reply 8"]
    assert content(chat(gw, make_tag("r2", "deepscholar", "sysB", "t1@1", "gen"))) == "reply 9"
    judge = make_tag("r2", "deepscholar", "sysB", "t1", "judge.cite_p")  # cached alias, still fresh
    assert [content(chat(gw, judge)) for _ in range(2)] == ["reply 10", "reply 11"]
    rows = [e for e in events(gw) if e["system"] == "sysB" and e["task"] == "t1"]
    assert [(e["stage"], e["cached"], e["meta"].get("k")) for e in rows] == [
        ("gen", True, 0),
        ("gen", True, 1),
        ("gen", False, 2),
        ("judge.cite_p", False, None),
        ("judge.cite_p", False, None),
    ]
    assert rows[0]["cost_usd"] == pytest.approx(COST) and rows[0]["billed_usd"] == 0
    assert usage(gw, ablation)["spent_usd"] == pytest.approx(3 * COST)
    # a client that gives up mid-call: the upstream call completes and is metered, but only the
    # retry the client sends instead is stored, so a replay sees what the client saw
    slow, n, fake.delay = [{"role": "user", "content": "slow"}], len(events(gw)), 0.5
    with pytest.raises(requests.Timeout):
        chat(gw, messages=slow, timeout=0.2)
    for _ in range(100):
        if len(events(gw)) > n:
            break
        time.sleep(0.05)
    gone = events(gw)[-1]
    assert (gone["error"], gone["notes"][-1]) == (None, "client_gone")
    assert gone["cost_usd"] == pytest.approx(COST) and gone["billed_usd"] == pytest.approx(COST / 2)
    fake.delay = 0
    assert content(chat(gw, messages=slow)) == "reply 13"
    assert content(chat(gw, ablation, messages=slow)) == "reply 13" and len(fake.seen) == 13


def test_budgets_isolate_tasks_judges_and_runs(make_gw: Callable, fake: Fake) -> None:
    gw = make_gw(Budget(task_usd=0.002, judge_usd={"deepscholar": 0.001}))
    judge = make_tag("r1", "deepscholar", "sysA", "t1", "judge.organization")
    other = make_tag("r1", "deepscholar", "sysB", "t1", "gen")
    assert [chat(gw, model="judge").status_code for _ in range(3)] == [200, 200, 402]
    assert chat(gw, judge, model="judge").status_code == 200  # the task's latch spares its judge
    r = chat(gw, judge, model="judge")
    assert code(r) == "budget_exhausted" and "judge task budget" in r.json()["error"]["message"]
    assert chat(gw, other, model="judge").status_code == 200
    assert chat(gw, other, model="local").status_code == 200  # billed: false upstream
    a, j, b = usage(gw, TAG), usage(gw, judge), usage(gw, other)
    assert a["latched"] and a["spent_usd"] == pytest.approx(2 * COST) and a["remaining_usd"] == 0
    assert j["latched"] and (j["task_cap"], j["spent_usd"]) == pytest.approx((0.001, COST))
    assert (b["spent_usd"], b["billed_usd"]) == pytest.approx((2 * COST, COST / 2))
    st = gw.status()
    assert (st["run_billed_usd"], st["judge_billed_usd"]) == pytest.approx((3 * COST / 2, COST / 2))
    assert {e["session"] for e in events(gw)} == {st["session"]} and len(fake.seen) == 5
    # new gateways on the same run dir keep counting generation, judge and campaign spend apart
    gen, rubric = (make_tag("r1", "drb2", "sysA", "9", s) for s in ("gen", "judge.rubric"))
    again = make_gw(Budget(run_usd=0.0015))
    assert again.status()["campaign_billed_usd"] == pytest.approx(2 * COST)
    r = chat(again, gen)
    assert code(r) == "budget_exhausted" and r.json()["error"]["message"].startswith("run budget")
    assert chat(again, rubric, model="judge").status_code == 200
    judged = make_gw(Budget(judge_run_usd=0.0005))
    assert code(chat(judged, rubric, model="judge")) == "budget_exhausted"
    assert chat(judged, gen).status_code == 200


def test_retries_stop_once_a_budget_closes(
    make_gw: Callable, fake: Fake, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: retries went out after the call's own attempts had spent its budget, or after
    another call had tripped the breaker during the backoff."""
    gw = make_gw(Budget(task_usd=COST))
    fake.script = [(200, reply("cut", "insufficient_system_resource"))]  # billed, retryable
    assert code(chat(gw)) == "budget_exhausted" and len(fake.seen) == 1
    assert [(e["attempt"], e["http_status"]) for e in events(gw)] == [(1, 200), (0, 402)]
    monkeypatch.setattr(gateway, "BACKOFF_S", 1.0)
    fake.script = [BUSY] * 2
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(chat, gw, make_tag("r1", "deepscholar", "sysA", "t2", "gen"))
        while len(fake.seen) < 2:
            time.sleep(0.01)
        gw._trip("another call's upstream refused the key")
        assert code(pending.result()) == "circuit_open"
    assert len(fake.seen) == 2


def test_tools_admit_charge_storm_and_breaker(make_gw: Callable, fake: Fake) -> None:
    gw = make_gw(Budget(task_usd=0.05))

    def post(route: str, tag: str = TAG, **body: Any) -> requests.Response:
        body = {"tool": "arxiv_search", **body}
        return requests.post(f"{gw.url}/t/{tag}/{route}", json=body, timeout=10)

    assert post("charge", n=2).json() == {"latched": False}  # free tool: list price, $0 billed
    assert post("charge", tool="tavily_search", cached=True).json() == {"latched": False}
    assert post("admit", tool="tavily_search").json() == {"admitted": True}
    assert post("charge", tool="tavily_search").json() == {"latched": True}  # recorded in full
    r = post("admit", meta={"query": "q"})  # the next call never starts
    assert r.status_code == 402 and code(r) == "budget_exhausted"
    refused = events(gw)[-1]
    assert (refused["error"], refused["cost_usd"], refused["meta"]) == ("budget_exhausted", 0, {
        "query": "q"})  # fmt: skip
    u = usage(gw, TAG)
    assert (u["spent_usd"], u["billed_usd"]) == pytest.approx((0.06, 0.02))
    assert (u["task_cap"], u["remaining_usd"], u["latched"], u["calls"]) == (0.05, 0, True, 4)
    assert post("charge", tool="nope").status_code == post("admit", tool="nope").status_code == 400
    storm = make_tag("r1", "deepscholar", "sys:A", "t/2", "gen")  # quoted parts survive routing
    codes = [chat(gw, storm, model="judge").status_code for _ in range(9)]
    assert codes == [200] * 8 + [402] and code(chat(gw, storm, model="judge")) == "retry_storm"
    last = events(gw)[-1]
    assert (last["system"], last["task"], last["error"]) == ("sys:A", "t/2", "retry_storm")
    assert len(fake.seen) == 8
    fatal = post("charge", storm, ok=False, error="tavily 432", fatal=True)
    assert fatal.json() == {"latched": True} and "tavily 432" in gw.status()["tripped"]
    fresh = make_tag("r1", "drb2", "sysA", "1", "gen")
    assert code(chat(gw, fresh)) == code(post("admit", fresh)) == "circuit_open"
    assert len(fake.seen) == 8


async def test_openai_sdk_retries_and_upstream_breaker(make_gw: Callable, fake: Fake) -> None:
    gw = make_gw()
    async with openai.AsyncOpenAI(
        base_url=f"{gw.url}/t/{TAG}/v1", api_key=LOCAL_KEY, max_retries=2
    ) as client:
        create = client.chat.completions.create
        r = await create(model="exec", messages=MSG, max_completion_tokens=64)
        assert r.model == "exec" and r.choices[0].message.content == "reply 1"
        assert r.usage.prompt_tokens_details.cached_tokens == 200
        sent = {k: fake.seen[0][k] for k in ("model", "max_tokens", "thinking")}
        assert sent == {"model": "flash", "max_tokens": 64, "thinking": {"type": "disabled"}}
        assert fake.auth == ["Bearer sk-test"]
        fake.script = [
            (429, {"error": {"message": "slow down"}}),
            (200, reply("cut", "insufficient_system_resource")),
            (200, reply("fine")),
        ]
        assert (await create(model="judge", messages=MSG)).choices[0].message.content == "fine"
        assert [(e["attempt"], e["http_status"]) for e in events(gw)][1:] == [
            (1, 429),
            (2, 200),
            (3, 200),
        ]
        fake.script = [BUSY] * 5
        with pytest.raises(openai.InternalServerError):
            await create(model="judge", messages=MSG)
        assert len(fake.seen) == 9  # the gateway retried 5 times; the SDK added none
        fake.script = [(422, {"error": {"message": "bad param"}})]
        with pytest.raises(openai.BadRequestError, match="bad param"):
            await create(model="judge", messages=MSG)
        fake.script = [(402, {"error": {"message": "Insufficient Balance"}})]
        with pytest.raises(openai.APIStatusError) as e:
            await create(model="judge", messages=MSG)
        assert e.value.status_code == 402 and e.value.code == "circuit_open"
    assert "Insufficient Balance" in requests.get(f"{gw.url}/status", timeout=10).json()["tripped"]
    models = requests.get(f"{gw.url}/v1/models", timeout=10).json()["data"]
    assert [m["id"] for m in models] == ["exec", "judge", "local"]


def test_cache_never_replays_another_upstreams_reply(make_gw: Callable, fake: Fake) -> None:
    """Regression: a fake server's replies, cached under the same alias, leaked into real runs."""
    gw = make_gw()
    assert content(chat(gw)) == "reply 1"
    moved = gw.models.model_copy(deep=True)  # same alias and request, another endpoint
    ds = moved.upstreams["ds"]
    moved.upstreams["ds"] = ds.model_copy(
        update={"base_url": ds.base_url.replace("127.0.0.1", "localhost")}
    )
    run_dir = gw.ledger.path.parent
    cache = run_dir.parent / "cache"  # make_gw's cache dir
    other = Gateway(
        moved,
        run_dir,
        Budget(),
        dict.fromkeys(CELLS, 2.0),
        secrets={"DS_KEY": "sk-test"},
        cache_dir=cache,
    )
    other.start()
    try:
        assert content(chat(other)) == "reply 2" and len(fake.seen) == 2
    finally:
        other.stop()


def test_replay_only_refuses_what_the_cache_cannot_serve(make_gw, fake):
    first = make_gw()
    assert content(chat(first)) == "reply 1"
    replay = make_gw(replay_only=True)
    assert content(chat(replay)) == "reply 1" and len(fake.seen) == 1  # served from the cache
    r = chat(replay, messages=[{"role": "user", "content": "new"}])
    assert r.status_code == 402 and r.json()["error"]["code"] == "replay_only"
    judge = make_tag("r1", "deepscholar", "sysA", "t1", "judge.cite_p")
    assert chat(replay, judge, model="judge").status_code == 200  # judges are never replayed
    assert len(fake.seen) == 2


@pytest.mark.parametrize("first", ["over the cap", "a fatal error"])
def test_a_queued_call_is_refused_once_the_call_ahead_closes_the_latch(make_gw, fake, first):
    """Regression: a call queued for the upstream slot behind one that crosses the task cap, or
    trips the breaker, is refused when it gets the slot: the call ahead is metered (and the
    breaker updated) before the slot is freed, so the queued one never reaches the upstream."""
    if first == "over the cap":
        gw = make_gw(Budget(task_usd=0.0005))  # one reply costs $0.00102
    else:
        gw = make_gw()
        fake.script.append((401, {"error": {"message": "bad key"}}))
    gw._sems["ds"] = asyncio.Semaphore(1)  # one upstream slot
    fake.delay = 0.3
    other = [{"role": "user", "content": "another question"}]  # not a cache replay
    with ThreadPoolExecutor(2) as pool:
        ahead = pool.submit(chat, gw)
        time.sleep(0.1)  # queued behind it
        queued = pool.submit(chat, gw, messages=other).result()
        ahead.result()
    assert queued.status_code == 402 and len(fake.seen) == 1
    want = "budget_exhausted" if first == "over the cap" else "circuit_open"
    assert queued.json()["error"]["code"] == want


def test_a_departed_clients_fatal_error_trips_the_breaker_before_the_queue_moves(make_gw, fake):
    """Regression: a client that left before its upstream call ended had the call metered, but a
    fatal answer (403) did not trip the breaker, so the next queued call reached the upstream."""
    gw = make_gw()
    gw._sems["ds"] = asyncio.Semaphore(1)
    fake.script.append((403, {"error": {"message": "key revoked"}}))
    fake.delay = 0.5
    with ThreadPoolExecutor(2) as pool:
        gone = pool.submit(chat, gw, timeout=0.1)  # gives up while the upstream call runs
        time.sleep(0.2)
        queued = pool.submit(chat, gw, messages=[{"role": "user", "content": "next"}]).result()
        with pytest.raises(requests.exceptions.Timeout):
            gone.result()
    assert queued.status_code == 402 and queued.json()["error"]["code"] == "circuit_open"
    assert len(fake.seen) == 1 and gw.status()["tripped"]
