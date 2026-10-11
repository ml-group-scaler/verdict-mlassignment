import json
import os
import stat
import time

import pytest

from verdict.config import ModelCfg, ProviderCfg
from verdict.llm import CircuitBreaker
from verdict.providers import ProviderError
from verdict.providers.droid_exec import DroidExecProvider, flatten

CFG = ProviderCfg(name="droid", type="droid_exec", api_key_env="FAKE_FK", timeout_s=20, max_retries=1)
MODEL = ModelCfg(id="droid-glm-5.3", provider="droid", model="glm-5.3", family="glm")


def fake_droid(tmp_path, stdout: str, code: int = 0):
    """A stand-in `droid` binary that records its argv/env and prints `stdout`."""
    log = tmp_path / "calls.jsonl"
    script = tmp_path / "droid"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        f"open({str(log)!r}, 'a').write(json.dumps({{'argv': sys.argv[1:], 'key': os.environ.get('FACTORY_API_KEY'),"
        " 'prompt': open(sys.argv[sys.argv.index('-f') + 1]).read()}) + '\\n')\n"
        f"print({stdout!r})\n"
        f"sys.exit({code})\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script, log


def ok_json(result='{"rationale": "asks for OTP", "score": 1}'):
    return json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": result,
                       "usage": {"input_tokens": 3, "cache_creation_input_tokens": 8858, "output_tokens": 55,
                                 "thinking_tokens": 20, "factory_credits": 9081}})


def test_droid_exec_call_shape_and_parsing(tmp_path, monkeypatch):
    binary, log = fake_droid(tmp_path, ok_json())
    monkeypatch.setenv("DROID_BIN", str(binary))
    monkeypatch.setenv("FAKE_FK", "fk-test")
    r = DroidExecProvider(CFG).complete(MODEL, [{"role": "user", "content": "grade this"}])
    assert json.loads(r.text)["score"] == 1
    assert (r.input_tokens, r.output_tokens, r.credits) == (8861, 75, 9081)
    call = json.loads(log.read_text().splitlines()[0])
    assert call["key"] == "fk-test"                       # DROID_API_KEY is passed as FACTORY_API_KEY
    assert call["argv"][:6] == ["exec", "-o", "json", "-m", "glm-5.3", "--disable-builtin-skills"]
    assert "--auto" not in call["argv"] and "--skip-permissions-unsafe" not in call["argv"]  # read-only
    assert "grade this" in call["prompt"] and "Do not use any tools" in call["prompt"]


def test_droid_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_FK", "fk-test")
    monkeypatch.setattr(time, "sleep", lambda s: None)
    binary, _ = fake_droid(tmp_path, json.dumps({"is_error": True, "result": "quota exceeded"}))
    monkeypatch.setenv("DROID_BIN", str(binary))
    with pytest.raises(ProviderError, match="quota exceeded"):
        DroidExecProvider(CFG).complete(MODEL, [{"role": "user", "content": "x"}])
    binary, _ = fake_droid(tmp_path, "Invalid model: glm-5.3", code=1)
    monkeypatch.setenv("DROID_BIN", str(binary))
    with pytest.raises(ProviderError, match="Invalid model") as e:
        DroidExecProvider(CFG).complete(MODEL, [{"role": "user", "content": "x"}])
    assert not e.value.retryable
    monkeypatch.setenv("DROID_BIN", str(tmp_path / "missing-droid"))
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(ProviderError):
        DroidExecProvider(CFG).complete(MODEL, [{"role": "user", "content": "x"}])


def test_flatten_repair_transcript():
    t = flatten([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "bad"}, {"role": "user", "content": "fix"}])
    assert "### USER\nQ" in t and "### YOUR PREVIOUS REPLY\nbad" in t and t.rstrip().endswith("fix")


def test_repeated_throttle_exhaustion_opens_circuit(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    b = CircuitBreaker()
    b.record_throttled()
    assert b.allow()          # one exhausted 429 budget = congestion, keep trying
    b.record_throttled()
    assert not b.allow()      # twice in a row = exhausted quota → straight to fallback
    now[0] += 601
    assert b.allow()
