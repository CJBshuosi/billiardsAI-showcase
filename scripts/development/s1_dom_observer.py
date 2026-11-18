"""Development-only DOM reads with conservative page-to-parent clock bounds."""

import json
import math
import time
from uuid import uuid4

MAX_EVENTS, MAX_BYTES = 1024, 2 * 1024**2
CLOCK = (
    "(()=>{const before=performance.now();return {before,after:performance.now(),"
    "time_origin:performance.timeOrigin};})()"
)
INSTALL = r"""(()=>{
const key='__billiards_s1_dom_observer_v1__',token=__TOKEN__,read=()=> (__EXPRESSION__);
if(globalThis[key])throw Error('DOM observer already installed');
const overlay=document.getElementById('route-overlay'),phase=document.getElementById('phase');
if(!overlay||!phase)throw Error('DOM observer elements missing');
const origin=performance.timeOrigin,events=[];
let bytes=2,previous=null,lastKey=null,error=null,stopped=false;
const append=row=>{
 if(events[events.length-1]===row)return;
 const size=new TextEncoder().encode(JSON.stringify(row)).length+(events.length?1:0);
 if(events.length>=1024||bytes+size>2093056){error='DOM observer budget exhausted';return;}
 events.push(row);bytes+=size;
};
const sample=(force=false)=>{
 if(error||stopped)return;
 try{
  if(performance.timeOrigin!==origin||document.getElementById('route-overlay')!==overlay||
     document.getElementById('phase')!==phase)throw Error('identity');
  const before=performance.now(),state=read(),after=performance.now();
  const row={page_before:before,page_after:after,state:JSON.parse(JSON.stringify(state))};
  const current=JSON.stringify({...row.state,video_time:undefined});
  const continuous=row.state.feedback_motion_frame_id!=null||
   String(row.state.phase||'').includes('等待停稳');
  if(force||current!==lastKey||continuous){
   if(previous&&current!==lastKey)append(previous);
   append(row);
  }
  previous=row;lastKey=current;
 }catch(e){error='DOM observer read or identity failed';}
};
const observer=new MutationObserver(()=>sample());
for(const node of [overlay,phase,document.getElementById('snapshot-note')]){
 if(node)observer.observe(node,{subtree:true,childList:true,attributes:true,characterData:true});
}
// 25ms is a scheduling target. Actual times/gaps remain evidence, not a new gate.
const timer=setInterval(()=>sample(),25);
globalThis[key]={stop:expected=>{
 if(expected!==token||stopped)throw Error('DOM observer identity or stopped');
 sample(true);stopped=true;observer.disconnect();clearInterval(timer);
 return {token,time_origin:origin,error,events};
}};
sample(true);return {token,time_origin:origin};
})()"""


def _finite(value):
    return type(value) in {int, float} and math.isfinite(value)


class DOMObserver:
    def __init__(self, browser, expression, *, clock=time.perf_counter):
        self.browser, self.expression, self.clock = browser, expression, clock
        self.token, self.origin = uuid4().hex, None
        self.started = self.stopped = False
        self.low, self.high, self.samples = -math.inf, math.inf, []

    def _evaluate(self, expression):
        reply = self.browser.evaluate(expression)
        if not isinstance(reply, dict) or not isinstance(reply.get("result"), dict):
            raise ValueError("DOM observer result missing")
        return reply["result"]

    def _calibrate(self):
        for _ in range(3):
            sent = self.clock()
            value = self._evaluate(CLOCK)
            received = self.clock()
            before, after, origin = (value.get(k) for k in ("before", "after", "time_origin"))
            if not all(_finite(v) for v in (sent, received, before, after, origin)) or not (
                sent <= received and 0 <= before <= after and origin > 0
            ):
                raise ValueError("invalid DOM clock calibration")
            if self.origin is not None and self.origin != origin:
                raise ValueError("DOM clock identity changed")
            self.origin = origin
            self.low = max(self.low, sent - after / 1000)
            self.high = min(self.high, received - before / 1000)
            self.samples.append(dict(host_before=sent, host_after=received, **value))
            if self.low > self.high:
                raise ValueError("DOM clock offset intervals do not intersect")

    def start(self):
        if self.started or self.stopped:
            raise ValueError("DOM observer already started or stopped")
        self._calibrate()
        script = INSTALL.replace("__TOKEN__", json.dumps(self.token)).replace(
            "__EXPRESSION__", self.expression
        )
        value = self._evaluate(script)
        if value.get("token") != self.token or value.get("time_origin") != self.origin:
            raise ValueError("DOM observer page identity changed")
        self.started = True

    def stop(self):
        if not self.started or self.stopped:
            raise ValueError("DOM observer not started or already stopped")
        self.stopped = True
        value = self._evaluate(
            "(()=>{const api=globalThis.__billiards_s1_dom_observer_v1__;"
            "if(!api)throw Error('DOM observer missing');return api.stop("
            + json.dumps(self.token)
            + ");})()"
        )
        if value.get("token") != self.token or value.get("time_origin") != self.origin:
            raise ValueError("DOM observer page identity changed")
        if value.get("error"):
            raise ValueError(str(value["error"]))
        events = value.get("events")
        if not isinstance(events, list) or not 1 <= len(events) <= MAX_EVENTS:
            raise ValueError("DOM observer records missing or exceed budget")
        self._calibrate()
        if self.high - self.low > 0.2:
            raise ValueError("DOM clock offset uncertainty exceeds 0.2 seconds")
        dom, previous = [], -math.inf
        required = {
            "phase",
            "route_lines",
            "has_dash",
            "has_region",
            "has_halo",
            "region_points",
            "observed_frame_id",
            "feedback_motion_frame_id",
        }
        for event in events:
            before, after, state = (event.get(k) for k in ("page_before", "page_after", "state"))
            if not (
                _finite(before)
                and _finite(after)
                and 0 <= before <= after
                and previous <= before
                and isinstance(state, dict)
                and required <= state.keys()
                and not {"monotonic", "observed_at"} & state.keys()
            ):
                raise ValueError("DOM observer record or ordering invalid")
            dom.append(
                dict(
                    state,
                    page_before_ms=before,
                    page_after_ms=after,
                    monotonic=before / 1000 + self.low,
                    observed_at=after / 1000 + self.high,
                )
            )
            previous = after
        result = dict(
            dom=dom,
            calibration=dict(
                time_origin=self.origin,
                offset_low=self.low,
                offset_high=self.high,
                width_seconds=self.high - self.low,
                samples=self.samples,
            ),
        )
        if len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode()) > MAX_BYTES:
            raise ValueError("DOM observer result budget exhausted")
        return result
