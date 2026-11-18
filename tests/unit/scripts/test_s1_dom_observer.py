"""Software-only page-clock/observer checks; never Windows/display evidence."""

import importlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.development.s1_screen_acceptance import DOM_STATE, assess_motion_display


def tool():
    assert Path("scripts/development/s1_dom_observer.py").is_file(), "DOM observer is missing"
    return importlib.import_module("scripts.development.s1_dom_observer")


# Real observer JavaScript executes in Node. Only DOM getters, mutation delivery
# and the page clock are controlled; the helper's buffering code is not mocked.
NODE = r"""
const readline=require('node:readline'), vm=require('node:vm');
let now=0, callbacks=[], timers=[], writes=0;
globalThis.fixture={phase:'准备',has_dash:false,has_region:false,has_halo:false,
route_lines:[],region_points:null,observed_frame_id:null,feedback_motion_frame_id:null};
globalThis.performance={now:()=>now,timeOrigin:1700000000000};
const protect=o=>new Proxy(o,{set(){writes++;throw Error('DOM write forbidden');}});
const polygon=protect({getAttribute:()=>fixture.region_points});
const svg=protect({querySelector:s=>s==='polygon'?(fixture.has_region?polygon:null):
s==='circle'?(fixture.has_halo?{}:null):(fixture.has_dash?{}:null),
querySelectorAll:()=>fixture.route_lines.map(points=>protect({getAttribute:()=>points})),
getAttribute:k=>String((k==='data-s1-feedback-motion'?fixture.feedback_motion_frame_id:
fixture.observed_frame_id)??'')});
const phase=protect({get textContent(){return fixture.phase;}});
const nodes={'route-overlay':svg,phase,'snapshot-note':protect({textContent:''}),
'route-frame':protect({naturalWidth:1920,naturalHeight:1080}),
'postgame-panel':protect({hidden:true}),'postgame-shot-list':protect({textContent:''}),
'postgame-player':protect({currentTime:0,duration:NaN,paused:true,ended:false,error:null})};
globalThis.document={getElementById:id=>nodes[id]||null,querySelectorAll:()=>[]};
globalThis.MutationObserver=class{
constructor(cb){this.cb=cb;this.active=true;callbacks.push(this);}
observe(){} disconnect(){this.active=false;}};
globalThis.setInterval=(fn,ms)=>{timers.push({fn,ms,active:true});return timers.length-1;};
globalThis.clearInterval=id=>{timers[id].active=false;};
readline.createInterface({input:process.stdin}).on('line',line=>{
try {const q=JSON.parse(line); let result;
if(q.expression)result=vm.runInThisContext(q.expression);
else {now=q.time??now;Object.assign(fixture,q.state||{});
if(q.origin)performance.timeOrigin=q.origin;
if(q.mutate)callbacks.filter(v=>v.active).forEach(v=>v.cb([]));
if(q.tick)timers.filter(v=>v.active).forEach(v=>v.fn());
result={writes,intervals:timers.map(v=>v.ms)};}
process.stdout.write(JSON.stringify({result})+'\n');
}catch(e){process.stdout.write(JSON.stringify({error:e.message})+'\n');}});
"""


class NodeBrowser:
    def __init__(self):
        node = shutil.which("node")
        assert node, "Node is required to exercise the actual observer JavaScript"
        self.process = subprocess.Popen(
            [node, "-e", NODE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.now, self.clock_calls = 0, 0

    def send(self, request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        reply = json.loads(self.process.stdout.readline())
        if "error" in reply:
            raise ValueError(reply["error"])
        return reply

    def evaluate(self, expression):
        return self.send(dict(expression=expression))

    def action(self, at, **values):
        self.now = at
        return self.send(dict(time=at, **values))["result"]

    def clock(self):
        self.clock_calls += 1
        return 100 + self.now / 1000 + (0.002 if self.clock_calls % 2 == 0 else 0)

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=3)
        self.process.stdout.close()
        self.process.stderr.close()


@pytest.fixture
def browser():
    value = NodeBrowser()
    try:
        yield value
    finally:
        value.close()


def test_observer_entry_exists():
    assert callable(tool().DOMObserver)


def test_actual_js_retains_complete_dom_fields_and_true_transition_predecessor(browser):
    observer = tool().DOMObserver(browser, DOM_STATE, clock=browser.clock)
    observer.start()
    browser.action(25, tick=True)
    browser.action(50, tick=True)
    browser.action(60, mutate=True, state=dict(phase="分析中"))
    browser.action(100, tick=True)
    result = observer.stop()
    dom = result["dom"]
    assert len(dom) == 4  # initial, true last pre-change sample, change, stop
    assert dom[1]["monotonic"] == pytest.approx(100.05)
    assert dom[2]["monotonic"] == pytest.approx(100.06)
    assert dom[2]["observed_at"] == pytest.approx(100.062)
    assert dom[2]["phase"] == "分析中"
    assert dom[2]["image_width"] == 1920 and dom[2]["route_lines"] == []
    assert {"region_points", "feedback_motion_frame_id", "observed_frame_id", "video_error"} <= dom[
        2
    ].keys()
    assert result["calibration"]["width_seconds"] == pytest.approx(0.002)
    assert len(result["calibration"]["samples"]) == 6
    assert browser.action(100)["writes"] == 0


def test_feedback_keeps_real_continuous_samples_and_os_delay_is_not_an_extra_gate(browser):
    observer = tool().DOMObserver(browser, DOM_STATE, clock=browser.clock)
    observer.start()
    browser.action(20, tick=True)
    browser.action(
        25,
        mutate=True,
        state=dict(
            phase="已进入建议区",
            feedback_motion_frame_id=9,
            observed_frame_id=20,
            has_region=True,
            region_points="1,1 2,1 2,2",
        ),
    )
    for at in (50, 75, 140, 165):
        browser.action(at, tick=True)
    browser.action(
        180, mutate=True, state=dict(phase="暂无可靠方案", feedback_motion_frame_id=None)
    )
    result = observer.stop()
    feedback = [row for row in result["dom"] if row["feedback_motion_frame_id"] == 9]
    assert len(feedback) == 5
    assert feedback[3]["monotonic"] - feedback[2]["monotonic"] == pytest.approx(0.065)
    assert browser.action(180)["intervals"] == [25]


def test_page_samples_can_prove_hiding_even_when_cli_return_is_late(browser):
    observer = tool().DOMObserver(browser, DOM_STATE, clock=browser.clock)
    observer.start()
    browser.action(100, tick=True)
    browser.action(200, mutate=True, state=dict(phase="等待停稳"))
    browser.action(2000, tick=True)
    result = observer.stop()  # CLI finishes much later than the real read at .2s.
    arrivals = [100 + n / 10 for n in range(31)]
    assert assess_motion_display(result["dom"], [(1, 29)], arrival_times=arrivals)[0] < 0.5


def test_late_actual_page_observation_still_fails_original_500ms_gate(browser):
    observer = tool().DOMObserver(browser, DOM_STATE, clock=browser.clock)
    observer.start()
    browser.action(700, mutate=True, state=dict(phase="等待停稳"))
    result = observer.stop()
    with pytest.raises(ValueError, match="500ms"):
        assess_motion_display(
            result["dom"], [(1, 29)], arrival_times=[100 + n / 10 for n in range(31)]
        )


def test_real_hidden_ticks_prove_early_hiding_despite_clock_offset_uncertainty(browser):
    calls = 0

    def clock():
        nonlocal calls
        calls += 1
        return 100 + browser.now / 1000 + (0.08 if calls % 2 == 0 else 0)

    observer = tool().DOMObserver(browser, DOM_STATE, clock=clock)
    observer.start()
    browser.action(50, tick=True)
    # The true page-to-host offset may be 100.04s. Hidden at page .090s
    # then means host 100.130s, only 30ms after motion starts at 100.100s.
    # Its conservative bracket [100.090,100.170] crosses the motion start.
    browser.action(90, mutate=True, state=dict(phase="等待停稳"))
    for at in range(115, 716, 25):
        browser.action(at, tick=True)
    result = observer.stop()
    hidden = [row for row in result["dom"] if row["phase"] == "等待停稳"]
    assert hidden[0]["monotonic"] < 100.1 < hidden[0]["observed_at"]
    delays = assess_motion_display(
        result["dom"], [(1, 29)], arrival_times=[100 + n / 10 for n in range(31)]
    )
    assert delays[0] == pytest.approx(0.095)
    assert len(result["dom"]) < 1024


@pytest.mark.parametrize("overflow", ["events", "bytes"])
def test_buffer_overflow_is_a_failure_not_a_truncated_success(browser, overflow):
    observer = tool().DOMObserver(browser, DOM_STATE, clock=browser.clock)
    observer.start()
    if overflow == "bytes":
        browser.action(25, mutate=True, state=dict(phase="x" * (2 * 1024**2)))
    else:
        for index in range(1025):
            browser.action(index + 1, mutate=True, state=dict(phase=str(index)))
    with pytest.raises(ValueError, match="budget"):
        observer.stop()


def test_one_observer_per_page_and_no_repeated_stop(browser):
    observer = tool().DOMObserver(browser, DOM_STATE, clock=browser.clock)
    observer.start()
    with pytest.raises(ValueError, match="already"):
        tool().DOMObserver(browser, DOM_STATE, clock=browser.clock).start()
    observer.stop()
    with pytest.raises(ValueError, match="stopped"):
        observer.stop()


def test_navigation_invalidates_page_clock_identity(browser):
    observer = tool().DOMObserver(browser, DOM_STATE, clock=browser.clock)
    observer.start()
    browser.action(25, origin=1700000001000)
    with pytest.raises(ValueError, match="clock|identity"):
        observer.stop()


@pytest.mark.parametrize("corruption", ["missing", "state", "order", "identity"])
def test_missing_or_invalid_stop_receipt_never_returns_mapped_evidence(browser, corruption):
    original = browser.evaluate

    def evaluate(expression):
        result = original(expression)
        value = result["result"]
        if "events" in value:
            if corruption == "missing":
                value["events"] = []
            elif corruption == "state":
                value["events"][0]["state"].pop("has_halo")
            elif corruption == "order":
                value["events"][-1]["page_before"] = -1
            else:
                value["token"] = "wrong-page"
        return result

    browser.evaluate = evaluate
    observer = tool().DOMObserver(browser, DOM_STATE, clock=browser.clock)
    observer.start()
    browser.action(25, tick=True)
    with pytest.raises(ValueError, match="missing|invalid|identity"):
        observer.stop()


@pytest.mark.parametrize(
    "ticks",
    [
        [100, 101] * 6,  # Accurate page clock, but transport cannot bound offset.
        [100, 100.002] * 3 + [101, 101.002] * 3,  # No common before/after offset.
    ],
)
def test_unbounded_or_disjoint_clock_offsets_are_rejected(browser, ticks):
    clock = iter(ticks)
    observer = tool().DOMObserver(browser, DOM_STATE, clock=lambda: next(clock))
    observer.start()
    with pytest.raises(ValueError, match="clock|offset"):
        observer.stop()
