"""The LLM gateway: the only path from any client to any model, and the only budget authority.

Every client gets a tagged OpenAI-compatible base URL `/t/{tag}/v1`, the one hook they all share.
The gateway rewrites requests into what the upstream accepts (DeepSeek defaults thinking on and
rejects `json_schema`), retries transient failures itself while telling SDKs not to, meters every
attempt into the ledger, replays the executor cache, and refuses work once a budget is spent or a
fatal provider error has tripped the circuit breaker. A refusal is checked before every upstream
attempt, retries included, and before every external tool call (`/admit`, the tool's own `/charge`
following once it has ended), so no request leaves after a latch has closed. An upstream slot is
held until its attempt is metered, and a call that waited for a slot checks again once it has
one, so a queued call never starts on spend that is not yet on the books. Judge stages
(`judge.*`) get their own budget keys and caps, so judging and generation can never starve each
other, and are never replayed. LLM provider keys live only here (the Tavily and OpenAlex tools
hold their own keys). With `replay_only`, an executor call that the cache cannot serve is refused
instead of sent.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web
from dotenv import dotenv_values

from jevscholar.config import Alias, Budget, ModelsConfig, Upstream
from jevscholar.ledger import Event, Ledger, SpendStore, llm_costs, scope, usage_tokens
from jevscholar.paths import CACHE, SECRETS
from jevscholar.types import make_tag, parse_tag
from jevscholar.util import connect

log = logging.getLogger("jevscholar.gateway")

ATTEMPTS = 5  # upstream attempts per logical call
BACKOFF_S = 1.0  # first retry delay, doubled per attempt
STORM_LIMIT = 8  # identical requests allowed per tag
GUARD_REASKS = 2
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
RETRY_FINISH = frozenset({"insufficient_system_resource", "aborted"})
FATAL_STATUS = frozenset({401, 402, 403})
CACHEABLE = frozenset({"stop", "tool_calls", "length"})
NO_RETRY = {"x-should-retry": "false"}
EFFORT = {"minimal": "low", "medium": "high", "xhigh": "high", "ultra": "max"}  # DeepSeek's map
_BASE = "model messages max_tokens temperature stop response_format tools tool_choice logprobs"
KEYS = {  # request whitelist per rewritten dialect; everything else is dropped and noted
    "deepseek": frozenset(f"{_BASE} top_logprobs thinking reasoning_effort".split()),
    "llama": frozenset(f"{_BASE} top_logprobs top_p seed chat_template_kwargs".split()),
}
SCHEMA_NOTE = "Return only a JSON object, no markdown, conforming to this JSON schema: "
REASK = "Your previous reply was not a JSON object conforming to the schema. Reply with only that."
JUDGE_TASK_USD = 0.15  # judging cap per task of a benchmark the budget does not name
FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)


class BadRequest(ValueError):
    """The request cannot be served as sent (answered with 400)."""


def load_secrets(path: Path = SECRETS) -> dict[str, str]:
    """Provider keys: the dotenv file (outside every load_dotenv search path) wins; exported
    `*_API_KEY` variables fill the gaps. Workers never see either (runner.worker_env scrubs)."""
    env = {k: v for k, v in os.environ.items() if k.endswith("_API_KEY") and v}
    return env | {k: v for k, v in dotenv_values(path, interpolate=False).items() if v}


# --------------------------------------------------------------------------- normalisation
def normalize(
    body: dict[str, Any], alias: Alias, upstream: Upstream
) -> tuple[dict[str, Any], list[str], dict[str, Any] | None]:
    """Client body -> (upstream body, notes, JSON schema the reply must satisfy or None). Pure.

    OpenAI upstreams serve only the official judges, so their requests pass through unmodified."""
    if body.get("stream"):
        raise BadRequest("stream=true is not supported by the gateway")
    out = {**body, "model": alias.model}
    if alias.passthrough or upstream.dialect == "openai":
        return out, [], None
    if (body.get("n") or 1) > 1:
        raise BadRequest(f"n>1 is not supported by the {upstream.dialect} upstream")
    out["messages"], notes = repair_tools(body.get("messages") or [])
    schema = _deepseek(out, alias, notes) if upstream.dialect == "deepseek" else None
    cap = alias.max_tokens_cap
    asked = out.pop("max_completion_tokens", None) or out.get("max_tokens")
    out["max_tokens"] = min(asked or alias.default_max_tokens or cap, cap)
    if not asked:
        notes.append(f"max_tokens:default->{out['max_tokens']}")
    elif asked > cap:
        notes.append(f"max_tokens:{asked}->{cap}")
    if upstream.dialect == "llama" and alias.thinking is not None:
        kwargs = body.get("chat_template_kwargs") or {}
        out["chat_template_kwargs"] = {**kwargs, "enable_thinking": alias.thinking}
    if dropped := sorted(set(out) - KEYS[upstream.dialect]):
        notes.append("dropped:" + ",".join(dropped))
        out = {k: v for k, v in out.items() if k not in dropped}
    return out, notes, schema


def _deepseek(out: dict[str, Any], alias: Alias, notes: list[str]) -> dict[str, Any] | None:
    msgs = out["messages"]
    if any(m["role"] == "developer" for m in msgs):
        msgs = [m | {"role": "system"} if m["role"] == "developer" else m for m in msgs]
        out["messages"] = msgs
        notes.append("developer->system")
    on = alias.thinking is not False  # unset means DeepSeek's default (on), sent explicitly
    out["thinking"] = {"type": "enabled" if on else "disabled"}
    asked = out.pop("reasoning_effort", None)
    if on and (effort := alias.effort or EFFORT.get(asked, asked)) in ("low", "high", "max"):
        out["reasoning_effort"] = effort
    if asked and asked != out.get("reasoning_effort"):
        notes.append(f"effort:{asked}->{out.get('reasoning_effort', 'off')}")
    if on and out.pop("temperature", None) is not None:
        notes.append("dropped:temperature(thinking)")
    if any("strict" in t["function"] for t in out.get("tools") or ()):
        out["tools"] = [
            t | {"function": {k: v for k, v in t["function"].items() if k != "strict"}}
            for t in out["tools"]
        ]
        notes.append("tools:strict removed")
    rf = out.get("response_format") or {}
    prompt = " ".join(_text(m.get("content")) for m in msgs if m["role"] in ("system", "user"))
    if rf.get("type") == "json_schema":
        schema = rf["json_schema"].get("schema") or {}
        out["response_format"] = {"type": "json_object"}
        _instruct(out, SCHEMA_NOTE + json.dumps(schema, ensure_ascii=False))
        notes.append("json_schema->json_object")
        return schema
    if rf.get("type") == "json_object" and "json" not in prompt.lower():
        _instruct(out, "Respond in JSON.")
        notes.append("json_object:instruction added")
    elif not rf and alias.force_json_object and "json" in prompt.lower():
        out["response_format"] = {"type": "json_object"}
        notes.append("json_object:forced")
    return None


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return " ".join(p.get("text", "") for p in content or ())


def _instruct(out: dict[str, Any], line: str) -> None:
    """Append `line` to the first system message, inserting one if there is none."""
    msgs = list(out["messages"])
    i = next((i for i, m in enumerate(msgs) if m["role"] == "system"), None)
    if i is None:
        msgs.insert(0, {"role": "system", "content": line})
    else:
        msgs[i] = msgs[i] | {"content": f"{_text(msgs[i]['content'])}\n\n{line}"}
    out["messages"] = msgs


def repair_tools(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Answer every assistant tool call (stub '[output trimmed]') and drop tool messages without a
    preceding call. Character-count context trimmers orphan them and DeepSeek then returns 400."""
    out: list[dict[str, Any]] = []
    pending: list[str] = []
    added = dropped = 0
    for m in [*messages, None]:
        if m is not None and m.get("role") == "tool":
            if m.get("tool_call_id") in pending:
                pending.remove(m["tool_call_id"])
                out.append(m)
            else:
                dropped += 1
            continue
        out += [{"role": "tool", "tool_call_id": i, "content": "[output trimmed]"} for i in pending]
        added += len(pending)
        if m is not None:
            out.append(m)
            pending = [c["id"] for c in m.get("tool_calls") or ()]
    return out, [f"orphan_repair:+{added}stub,-{dropped}tool"] if added or dropped else []


def json_reply(content: str | None, schema: dict[str, Any]) -> str | None:
    """The reply as bare JSON text if it is an object with the schema's required keys, else None."""
    text = (content or "").strip()
    if m := FENCE.match(text):
        text = m.group(1)
    try:
        obj = json.loads(text)
    except ValueError:
        return None
    ok = isinstance(obj, dict) and set(schema.get("required", ())).issubset(obj)
    return text if ok else None


# --------------------------------------------------------------------------- replay cache
class ReplayCache:
    """Executor replay cache (sqlite), shared by every run and process.

    Entries are keyed by (bench, task, stage, request sha, k), leaving run and system out, so
    another run or system (the as-is ablation, a resumed task) replays the same trajectory. k is
    the occurrence index of an identical request under one tag and counts only occurrences that
    were replayed or stored, so a failed or abandoned call never shifts the sequence."""

    def __init__(self, path: Path) -> None:
        self.db = connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS llm (key TEXT PRIMARY KEY, value TEXT)")
        self._used: Counter[tuple[str, str]] = Counter()

    def get(self, tag: str, sha: str) -> tuple[int, dict[str, Any]] | None:
        """(k, stored value) of the next occurrence, or None (a miss consumes nothing)."""
        query = "SELECT value FROM llm WHERE key=?"
        row = self.db.execute(query, (self._key(tag, sha),)).fetchone()
        return (self._take(tag, sha), json.loads(row[0])) if row else None

    def put(self, tag: str, sha: str, value: dict[str, Any]) -> int:
        """Store the next occurrence (a concurrent writer's entry wins); returns its k."""
        entry = (self._key(tag, sha), json.dumps(value))
        self.db.execute("INSERT OR IGNORE INTO llm VALUES (?, ?)", entry)
        return self._take(tag, sha)

    def _key(self, tag: str, sha: str) -> str:
        _, bench, _, task, stage = parse_tag(tag)
        return json.dumps([bench, task, stage, sha, self._used[tag, sha]])

    def _take(self, tag: str, sha: str) -> int:
        self._used[tag, sha] += 1
        return self._used[tag, sha] - 1


# --------------------------------------------------------------------------- server
def _err(status: int, message: str, code: str) -> tuple[int, dict[str, Any]]:
    return status, {"error": {"message": message, "type": "jev_gateway", "code": code}}


def _respond(status: int, payload: dict[str, Any]) -> web.Response:
    return web.json_response(payload, status=status, headers=None if status == 200 else NO_RETRY)


def _tag(request: web.Request) -> str:
    """Canonical tag from the still-encoded path segment (match_info is already decoded)."""
    return make_tag(*parse_tag(request.rel_url.raw_parts[2]))  # ValueError -> 400


def _key(tag: str) -> tuple[str, ...]:
    """Budget key: (run, bench, system, task, scope), scope "gen" or "judge" (separate caps)."""
    parts = parse_tag(tag)
    return (*parts[:4], scope(parts[4]))


@web.middleware
async def _errors(request: web.Request, handler: Any) -> web.StreamResponse:
    """Every error answer carries x-should-retry: false, so SDKs never retry behind our back."""
    try:
        return await handler(request)
    except web.HTTPException:
        raise
    except ValueError as e:  # BadRequest, malformed tag or JSON body
        return _respond(*_err(400, str(e), "invalid_request"))
    except Exception as e:
        log.exception("gateway error on %s", request.path)
        return _respond(*_err(500, f"{type(e).__name__}: {e}", "gateway_error"))


@dataclass
class _Spend:
    spent: float = 0.0  # list USD, cache replays included at recorded cost
    billed: float = 0.0
    calls: int = 0


class Gateway:
    """aiohttp server on 127.0.0.1 in a daemon thread with its own event loop."""

    def __init__(
        self,
        models: ModelsConfig,
        run_dir: Path,
        budget: Budget,
        caps: dict[tuple[str, str], float],
        *,
        secrets: dict[str, str],
        cache_dir: Path = CACHE,
        replay_only: bool = False,
    ) -> None:
        """`caps`: the generation cap per task of each (benchmark, system) cell (list USD; none
        in a judging session, which makes no generation call)."""
        self.models, self.budget, self.caps, self.secrets = models, budget, caps, secrets
        self.replay_only = replay_only
        self.session = f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
        self.ledger = Ledger(run_dir / "ledger.jsonl", self.session)
        self.spend = SpendStore(cache_dir / "spend.sqlite")
        self.cache = ReplayCache(cache_dir / "llm.sqlite")
        self.billed = self.ledger.billed_totals()  # per scope; a resumed run keeps its spend
        self.campaign = self.spend.total()
        self.tripped: str | None = None
        self.url = ""
        self._usage: defaultdict[tuple[str, ...], _Spend] = defaultdict(_Spend)
        self._seen: Counter[tuple[str, str]] = Counter()  # (tag, request sha) -> arrivals
        self._inflight: set[asyncio.Future] = set()  # upstream posts, also of departed clients

    def start(self) -> str:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        port = asyncio.run_coroutine_threadsafe(self._startup(), self._loop).result()
        self.url = f"http://127.0.0.1:{port}"
        return self.url

    def stop(self) -> None:
        asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop).result()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()

    def status(self) -> dict[str, Any]:
        return {
            "session": self.session,
            "run_billed_usd": self.billed["gen"],
            "judge_billed_usd": self.billed["judge"],
            "campaign_billed_usd": self.campaign,
            "tripped": self.tripped,
        }

    async def _startup(self) -> int:
        ups = self.models.upstreams
        self._sems = {n: asyncio.Semaphore(u.concurrency) for n, u in ups.items()}
        self._clients = {n: self._client(u) for n, u in ups.items()}
        app = web.Application(client_max_size=64 * 2**20, middlewares=[_errors])
        app.router.add_post("/t/{tag}/v1/chat/completions", self._chat)
        app.router.add_post("/t/{tag}/admit", self._admit)
        app.router.add_post("/t/{tag}/charge", self._charge)
        app.router.add_get("/t/{tag}/usage", self._usage_view)
        app.router.add_get("/v1/models", self._models_view)
        app.router.add_get("/status", self._status_view)
        self._runner = web.AppRunner(app, handler_cancellation=True, access_log=None)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", 0).start()
        return self._runner.addresses[0][1]

    def _client(self, u: Upstream) -> aiohttp.ClientSession:
        key = self.secrets.get(u.key_env) if u.key_env else None
        auth = {"Authorization": f"Bearer {key}"} if key else {}
        return aiohttp.ClientSession(headers=auth, timeout=aiohttp.ClientTimeout(total=u.timeout_s))

    async def _shutdown(self) -> None:
        await self._runner.cleanup()
        await asyncio.gather(*self._inflight, return_exceptions=True)  # meter them before closing
        for client in self._clients.values():
            await client.close()

    # ------------------------------------------------------------------ accounting
    def _record(self, ev: Event) -> None:
        self.ledger.append(ev)
        key = _key(ev.tag)
        u = self._usage[key]
        u.spent, u.billed, u.calls = u.spent + ev.cost_usd, u.billed + ev.billed_usd, u.calls + 1
        if ev.billed_usd:
            self.billed[key[4]] += ev.billed_usd
            self.campaign = self.spend.add(key[0], ev.billed_usd)

    def _caps(self, key: tuple[str, ...]) -> list[tuple[str, float, float]]:
        """(name, used, cap) of every cap a call on budget key `key` must be under."""
        b, spent, run = self.budget, self._usage[key].spent, self.billed[key[4]]
        if key[4] == "judge":
            task = b.judge_usd.get(key[1], JUDGE_TASK_USD)
            caps = [("judge task", spent, task), ("judge run", run, b.judge_run_usd)]
        else:
            caps = [("task", spent, self.caps.get(key[1:3], 0.0)), ("run", run, b.run_usd)]
        return [*caps, ("campaign", self.campaign, self.models.campaign_usd)]

    def _refusal(self, tag: str) -> tuple[int, dict[str, Any]] | None:
        """Why a new call on `tag` may not start: the breaker, else the first exhausted cap."""
        if self.tripped:
            return _err(402, f"circuit breaker tripped: {self.tripped}", "circuit_open")
        for name, used, cap in self._caps(_key(tag)):
            if used >= cap:
                message = f"{name} budget exhausted (${used:.4f} >= ${cap:.4f})"
                return _err(402, message, "budget_exhausted")
        return None

    def _refused(
        self, event: Event, refusal: tuple[int, dict[str, Any]]
    ) -> tuple[int, dict[str, Any]]:
        """Record a refused call (it never started, so it costs nothing); returns the refusal."""
        self._record(replace(event, http_status=402, error=refusal[1]["error"]["code"]))
        return refusal

    def _trip(self, reason: str) -> None:
        log.error("circuit breaker tripped: %s", reason)
        self.tripped = self.tripped or reason

    # ------------------------------------------------------------------ routes
    async def _chat(self, request: web.Request) -> web.Response:
        tag, body = _tag(request), await request.json()
        if (name := body.get("model")) not in self.models.aliases:
            raise BadRequest(f"unknown model alias {name!r}")
        alias = self.models.aliases[name]
        up, notes, schema = normalize(body, alias, self.models.upstreams[alias.upstream])
        for i in range(1 + (GUARD_REASKS if schema is not None else 0)):
            status, resp = await self._call(tag, name, alias, up, notes)
            if status != 200 or schema is None:
                break
            msg = resp["choices"][0]["message"]
            if (fixed := json_reply(msg.get("content"), schema)) is not None:
                msg["content"] = fixed
                break
            bad = {"role": "assistant", "content": msg.get("content") or ""}
            up = up | {"messages": [*up["messages"], bad, {"role": "user", "content": REASK}]}
            notes = [f"json_guard:{i + 1}"]
        return _respond(status, resp)

    async def _call(
        self, tag: str, name: str, alias: Alias, body: dict[str, Any], notes: list[str]
    ) -> tuple[int, dict[str, Any]]:
        """One logical upstream call: refusals, storm guard, cache replay, retries. A retry, too,
        starts only while its budgets (spent by earlier attempts) and the breaker allow it."""
        upstream = self.models.upstreams[alias.upstream]
        # the endpoint is part of the identity: a reply from another server is never replayed
        ident = {"url": upstream.base_url, "body": body}
        sha = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()
        event = partial(Event, tag, "llm", name, alias.model, notes=notes, meta={"sha": sha[:16]})
        if not (refusal := self._refusal(tag)):
            self._seen[tag, sha] += 1
            if self._seen[tag, sha] > STORM_LIMIT:
                refusal = _err(402, f"more than {STORM_LIMIT} identical requests", "retry_storm")
        if refusal:
            return self._refused(event(), refusal)
        generation = _key(tag)[4] == "gen"
        slot = (tag, sha) if alias.cache and generation else None  # judges: never
        if slot and (hit := self.cache.get(*slot)):
            k, value = hit
            ev = event(http_status=200, cached=True, **value["event"])
            ev.meta["k"] = k
            self._record(ev)
            return 200, value["response"] | {"model": name}
        if self.replay_only and generation:
            return self._refused(event(), _err(402, "not in the replay cache", "replay_only"))
        sem = self._sems[alias.upstream]
        for attempt in range(1, ATTEMPTS + 1):
            if attempt > 1:
                await asyncio.sleep(BACKOFF_S * 2 ** (attempt - 2))
            await sem.acquire()
            handed_over = False  # to a departed client's metering callback
            try:
                if refusal := self._refusal(tag):  # also spend booked while this call queued
                    return self._refused(event(), refusal)
                post = asyncio.ensure_future(self._post(alias.upstream, upstream, body))
                self._inflight.add(post)
                post.add_done_callback(self._inflight.discard)
                try:
                    await asyncio.shield(post)  # a client that leaves does not cancel the call
                except asyncio.CancelledError:  # ... which the provider bills: meter it at its end
                    gone = partial(event, notes=[*notes, "client_gone"])
                    post.add_done_callback(partial(self._metered, sem, gone, alias, attempt))
                    handed_over = True
                    raise
                status, data, err = self._meter(event, alias, attempt, slot, post)
                if err is None:
                    return 200, data | {"model": name}
                msg = err["message"]
                if self._fatal(alias, status, err):
                    return _err(402, f"upstream HTTP {status}: {msg}", "circuit_open")
                if status in (400, 422):
                    return _err(400, f"upstream HTTP {status}: {msg}", "upstream_rejected")
                if status not in RETRY_STATUS and status not in (None, 200):
                    return _err(502, f"upstream HTTP {status}: {msg}", "upstream_error")
            finally:
                if not handed_over:
                    sem.release()
        return _err(502, f"failed after {ATTEMPTS} attempts: {msg}", "upstream_unavailable")

    def _metered(
        self,
        sem: asyncio.Semaphore,
        event: partial[Event],
        alias: Alias,
        attempt: int,
        post: asyncio.Future,
    ) -> None:
        """A departed client's attempt, metered once it ends (never stored: the retry the client
        sends instead takes its occurrence index) and a fatal error tripping the breaker; then its
        upstream slot is freed."""
        try:
            status, _, err = self._meter(event, alias, attempt, None, post)
            if err is not None:
                self._fatal(alias, status, err)
        finally:
            sem.release()

    def _fatal(self, alias: Alias, status: int | None, err: dict[str, Any]) -> bool:
        """Whether an upstream error can never succeed in this run; if so, trip the breaker."""
        fatal = status in FATAL_STATUS or "insufficient_quota" in (err.get("code"), err.get("type"))
        if fatal:
            self._trip(f"{alias.upstream} HTTP {status}: {err['message']}")
        return fatal

    def _meter(
        self,
        event: partial[Event],
        alias: Alias,
        attempt: int,
        slot: tuple[str, str] | None,
        post: asyncio.Future,
    ) -> tuple[int | None, dict[str, Any], dict[str, Any] | None]:
        """Record a finished upstream attempt, store it when cacheable: (status, data, error).

        Attempts whose client left are metered with slot None: never stored, so the retry the
        client sends instead takes their occurrence index and replays stay exact."""
        status, data, latency = post.result()
        upstream = self.models.upstreams[alias.upstream]
        finish = (data.get("choices") or [{}])[0].get("finish_reason")
        tok, at = usage_tokens(data.get("usage")), datetime.now(UTC)
        cost, billed = llm_costs(alias.price, upstream, tok, at)
        ok = status == 200 and bool(data.get("choices")) and finish not in RETRY_FINISH
        err = data.get("error")
        err = err if isinstance(err, dict) else {"message": err}
        msg = None if ok else str(err.get("message") or f"HTTP {status} finish={finish}")
        replay = {"finish_reason": finish, "tokens": tok, "cost_usd": cost}
        ev = event(
            attempt=attempt,
            http_status=status,
            latency_s=latency,
            error=msg,
            ts=at,
            billed_usd=billed,
            **replay,
        )
        if ok and slot and finish in CACHEABLE:
            ev.meta["k"] = self.cache.put(*slot, {"response": data, "event": replay})
        self._record(ev)
        return status, data, None if ok else err | {"message": msg}

    async def _post(
        self, name: str, upstream: Upstream, body: dict[str, Any]
    ) -> tuple[int | None, dict[str, Any], float]:
        """One upstream attempt (the caller holds its upstream slot)."""
        url, t0 = f"{upstream.base_url.rstrip('/')}/chat/completions", time.monotonic()
        try:
            async with self._clients[name].post(url, json=body) as r:
                status, raw = r.status, await r.read()
            data = json.loads(raw)  # tolerates DeepSeek's leading keep-alive blank lines
        except (aiohttp.ClientError, TimeoutError) as e:
            status, data = None, {"error": {"message": f"{type(e).__name__}: {e}"}}
        except ValueError:
            data = {"error": {"message": raw[:500].decode(errors="replace")}}
        return status, data, round(time.monotonic() - t0, 3)

    async def _admit(self, request: web.Request) -> web.Response:
        """Before an external tool call: refused like an LLM call, so it never reaches the tool."""
        tag, b = _tag(request), await request.json()
        tool = self._tool(b)
        if refusal := self._refusal(tag):
            event = Event(tag, "tool", tool, meta=b.get("meta") or {})
            return _respond(*self._refused(event, refusal))
        return web.json_response({"admitted": True})

    async def _charge(self, request: web.Request) -> web.Response:
        """After an admitted tool call has ended: recorded in full (the call that crosses a cap
        included); a fatal failure trips the breaker. Answers whether the latch is now closed."""
        tag, b = _tag(request), await request.json()
        tool = self._tool(b)
        n, cached = b.get("n", 1), bool(b.get("cached"))
        cost = self.models.tools[tool] * n
        billed = 0.0 if cached or tool in self.models.free_tools else cost
        error = None if b.get("ok", True) else str(b.get("error") or "failed")
        ev = Event(tag, "tool", tool, n=n, cost_usd=cost, billed_usd=billed, cached=cached)
        self._record(replace(ev, error=error, meta=b.get("meta") or {}))
        if b.get("fatal"):
            self._trip(f"{tool}: {error}")
        return web.json_response({"latched": self._refusal(tag) is not None})

    def _tool(self, body: dict[str, Any]) -> str:
        if (tool := body.get("tool")) not in self.models.tools:
            raise BadRequest(f"unknown tool {tool!r}")
        return tool

    async def _usage_view(self, request: web.Request) -> web.Response:
        tag = _tag(request)
        key = _key(tag)
        u, cap = self._usage[key], self._caps(key)[0][2]
        return web.json_response(
            {
                "spent_usd": u.spent,
                "billed_usd": u.billed,
                "task_cap": cap,
                "remaining_usd": max(0.0, cap - u.spent),
                "latched": self._refusal(tag) is not None,
                "calls": u.calls,
            }
        )

    async def _models_view(self, _: web.Request) -> web.Response:
        aliases = self.models.aliases.items()
        data = [{"id": a, "object": "model", "owned_by": v.upstream} for a, v in aliases]
        return web.json_response({"object": "list", "data": data})

    async def _status_view(self, _: web.Request) -> web.Response:
        return web.json_response(self.status())
