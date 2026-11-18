import json
import shutil
import subprocess
from html.parser import HTMLParser

import pytest

from billiards_ai.lab.field_assistance.pages import render_screen_page, render_tablet_page


class _PageElements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.elements[attrs["id"]] = (tag, attrs)


def test_postgame_controls_start_hidden_and_use_native_accessible_elements():
    parser = _PageElements()
    parser.feed(render_screen_page())
    assert "postgame-panel" in parser.elements, "postgame review panel is missing"
    tag, attrs = parser.elements["postgame-panel"]
    assert tag == "section" and "hidden" in attrs
    assert attrs["aria-labelledby"] == "postgame-title"
    tag, attrs = parser.elements["postgame-player"]
    assert tag == "video" and "controls" in attrs and "hidden" in attrs
    assert "autoplay" not in attrs and "src" not in attrs
    assert attrs["aria-describedby"] == "postgame-player-label"
    assert parser.elements["postgame-status"][1]["role"] == "status"
    for name in ("postgame-finish", "postgame-retry"):
        assert parser.elements[name][0] == "button"
    assert "完成并删除未保存片段" in render_screen_page()
    assert "已保存片段会保留" in render_screen_page()


_REVIEW_HARNESS = r"""
const vm=require('node:vm'), assert=require('node:assert/strict');
const elements={}, requests=[], intervals=[], timeouts=[], events={}, documentEvents={};
class Element {
  constructor(tag='div') {this.tagName=tag;this.children=[];this.attributes={};this.dataset={};
    this.hidden=false;this.disabled=false;this.textContent='';this.srcWrites=0;this.pauses=0;this.loads=0;
    this.style={};this.classList={add(){},remove(){},toggle(){}};}
  appendChild(child){this.children.push(child);child.parent=this;return child;}
  replaceChildren(...children){this.children=[];children.forEach(c=>this.appendChild(c));}
  remove(){if(this.parent)this.parent.children=this.parent.children.filter(c=>c!==this);}
  contains(child){return child===this||this.children.some(c=>c.contains(child));}
  setAttribute(key,value){this.attributes[key]=String(value);}
  getAttribute(key){return key==='src'?this._src??null:this.attributes[key]??null;}
  removeAttribute(key){delete this.attributes[key];if(key==='src')delete this._src;}
  set src(value){this._src=value;this.srcWrites++;} get src(){return this._src||'';}
  pause(){this.pauses++;} load(){this.loads++;} focus(){document.activeElement=this;}
  click(){if(!this.disabled&&!this.hidden&&this.onclick)return this.onclick();}
}
const el=id=>elements[id]??=(new Element());
const document={getElementById:el,createElement:tag=>new Element(tag),activeElement:null,
  visibilityState:'visible',addEventListener:(kind,cb)=>documentEvents[kind]=cb};
['title','session','progress','status','shot-list','player-label','player','retry','finish'].forEach(
  name=>el('postgame-panel').appendChild(el('postgame-'+name)));
const context={document,URL,Uint8Array,Map,Set,Date,AbortController,
  location:{origin:'http://127.0.0.1:8765'},
  crypto:{getRandomValues(a){a.fill(requests.length+1);return a;}},
  setInterval(cb,ms){intervals.push({cb,ms});},
  setTimeout(cb,ms){const timer={cb,ms};timeouts.push(timer);return timer;},
  clearTimeout(timer){timer.cleared=true;},
  window:{addEventListener:(kind,cb)=>events[kind]=cb},
  fetch:(url,options={})=>new Promise((resolve,reject)=>
    requests.push({url,options,resolve,reject})),
  localStorage:{setItem(){throw Error('storage forbidden')}},
  sessionStorage:{setItem(){throw Error('storage forbidden')}},
  console:{log(){throw Error('logging forbidden')},error(){throw Error('logging forbidden')}}};
vm.createContext(context);vm.runInContext(SCRIPT,context);
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const empty={phase:'REVIEW_COMPLETE',review_session_id:null,postgame_manifest_sha256:null,
  analysis_status:null,expected_count:0,available_count:0,clips:[]};
const digest='a'.repeat(64),clip='clip-'+'b'.repeat(64),nonce='PRIVATE-KIOSK-NONCE';
const media='/media/postgame/'+clip+'.mp4?manifest='+digest;
function projection(overrides={}){return {phase:'POSTGAME_REVIEW',review_session_id:'ended-1',
  postgame_manifest_sha256:digest,analysis_status:'PARTIAL',expected_count:2,available_count:1,
  clips:[{shot_id:'shot-2',clip_id:null,save_status:'NOT_SAVED',media_url:null},
    {shot_id:'shot-1',clip_id:clip,save_status:'NOT_SAVED',media_url:media}],kiosk_nonce:nonce,can_complete:false,...overrides};}
function readyProjection(overrides={}){return projection({analysis_status:'AVAILABLE',
  available_count:2,can_complete:true,...overrides});}
function response(req,data,status=200){
 if(Array.isArray(data?.clips))data={...data,clips:data.clips.map((c,i)=>(
   {sequence:i+1,actor_role:'red',can_save:true,...c}))};
 req.resolve({ok:status===200,status,json:async()=>data});}

const poll=()=>intervals.find(i=>i.ms===1000).cb();
const rows=()=>el('postgame-shot-list').children;
const play=i=>rows()[i].children[1];
const save=i=>rows()[i].children[2];
const rowStatus=i=>rows()[i].children[3].textContent;
function privacy(){function inspect(e){assert(!e.textContent.includes(nonce));
  assert(!JSON.stringify(e.attributes).includes(nonce));assert(!JSON.stringify(e.dataset).includes(nonce));
  assert(!e.src.includes(nonce));e.children.forEach(inspect);}Object.values(elements).forEach(inspect);
  requests.forEach(r=>{assert(!r.url.includes(nonce));assert(!(r.options.body||'').includes(nonce));});}
(async()=>{SCENARIO})().catch(error=>{process.stderr.write(error.stack);process.exitCode=1});
"""


def _run_review(scenario):
    page = render_screen_page()
    assert '<script id="postgame-review-script">' in page, "postgame review script is missing"
    script = page.split('<script id="postgame-review-script">', 1)[1].split("</script>")[0]
    harness = _REVIEW_HARNESS.replace("SCRIPT", json.dumps(script)).replace("SCENARIO", scenario)
    result = subprocess.run([shutil.which("node"), "-e", harness], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="JavaScript runtime unavailable")
def test_recovery_diagnostic_is_visible_without_foreign_identity_or_kiosk_authority():
    _run_review(r"""
response(requests[0],{...empty,phase:'RECOVERY_REQUIRED',can_complete:false,
  recovery_diagnostics:{'own-pending':'POSTGAME_REVIEW_QUEUE_RECOVERY_REQUIRED'}});await tick();
assert(!el('postgame-panel').hidden);assert(el('postgame-finish').disabled);
assert(el('postgame-status').textContent.includes('现场恢复'));
assert(!el('postgame-session').textContent.includes('null'));assert.equal(rows().length,0);
el('postgame-finish').click();assert.equal(requests.length,1);privacy();
poll();response(requests[1],projection({recovery_diagnostics:{
  'old-own':'POSTGAME_ORIGINAL_CONTEXT_UNAVAILABLE'}}));await tick();
assert(el('postgame-status').textContent.includes('现场恢复'));
poll();response(requests[2],{...empty,recovery_diagnostics:{
  'old-own':'LOCAL_MEDIA_CLEANUP_ORIGINAL_CONTEXT_UNAVAILABLE'}});await tick();
assert(!el('postgame-recovery').hidden);
assert(el('postgame-recovery').textContent.includes('现场恢复'));privacy();
""")


@pytest.mark.skipif(shutil.which("node") is None, reason="JavaScript runtime unavailable")
@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(
            r"""
assert.equal(requests[0].url,'/api/postgame-review');
response(requests[0],empty);await tick();
assert(el('postgame-panel').hidden);assert(el('postgame-player').hidden);
assert.equal(el('postgame-player').src,'');assert(el('postgame-finish').disabled);
assert.equal(requests.length,1);
""",
            id="initial-open-empty-review-has-no-controls",
        ),
        pytest.param(
            r"""
response(requests[0],projection({phase:'POSTGAME_PROCESSING',analysis_status:'PROCESSING',can_complete:false}));
await tick();
assert(!el('postgame-panel').hidden);assert(el('postgame-session').textContent.includes('ended-1'));
assert(el('postgame-progress').textContent.includes('1 / 2'));assert(play(0).disabled);
assert.equal(el('postgame-player').src,'');assert(el('postgame-finish').disabled);
save(0).click();save(0).click();assert.equal(requests.length,2);
const sent=requests[1],body=JSON.parse(sent.options.body);
assert.equal(sent.url,'/api/postgame-review/save');assert.equal(sent.options.headers['X-Kiosk-Nonce'],nonce);
assert.deepEqual(Object.keys(body).sort(),['postgame_manifest_sha256','request_id','session_id','shot_id']);
assert.equal(body.session_id,'ended-1');assert.equal(body.shot_id,'shot-2');assert.equal(body.postgame_manifest_sha256,digest);
assert(body.request_id);assert(save(0).disabled);assert(el('postgame-finish').disabled);
response(sent,{...body,status:'SAVE_PENDING',clip_id:null});await tick();
response(requests[2],projection({clips:[
  {shot_id:'shot-2',clip_id:null,save_status:'SAVE_PENDING',media_url:null}]}));await tick();
assert(rowStatus(0).includes('生成中'));assert(save(0).disabled);privacy();
""",
            id="processing-save-exact-shot-guard-and-durable-pending",
        ),
        pytest.param(
            r"""
el('route-frame').src='/media/live.mjpg?epoch=new-open';
response(requests[0],projection());await tick();
assert.equal(rows()[0].dataset.shotId,'shot-2');assert.equal(rows()[1].dataset.shotId,'shot-1');
play(1).click();assert.equal(el('postgame-player').src,media);assert(!el('postgame-player').hidden);
const button=play(1),player=el('postgame-player'),writes=player.srcWrites;
player.currentTime=12;button.focus();
poll();response(requests[1],projection());await tick();
assert.equal(play(1),button);assert.equal(document.activeElement,button);
assert.equal(player.currentTime,12);
assert.equal(player.srcWrites,writes);assert.equal(el('route-frame').src,'/media/live.mjpg?epoch=new-open');privacy();
""",
            id="partial-review-order-playback-stable-poll-and-live-rollover",
        ),
        pytest.param(
            r"""
response(requests[0],projection());await tick();poll();const oldPoll=requests[1];
save(1).click();const sent=requests[2],body=JSON.parse(sent.options.body);
response(sent,{...body,status:'SAVE_PENDING',clip_id:null});await tick();
response(requests[3],projection({clips:[
  {shot_id:'shot-1',clip_id:clip,media_url:media,save_status:'SAVED'}]}));await tick();
response(oldPoll,projection());await tick();
assert(rowStatus(0).includes('已保存'));assert(!rowStatus(0).includes('生成中'));assert(save(0).disabled);
assert.equal(save(0).textContent,'已保存');
""",
            id="historical-receipt-and-stale-poll-cannot-downgrade-save",
        ),
        pytest.param(
            r"""
response(requests[0],projection());await tick();play(1).click();
save(1).click();const first=requests[1];first.reject(Error('lost response'));await tick();
assert(save(1).disabled);assert(el('postgame-finish').disabled);assert.equal(el('postgame-player').src,media);
response(requests[2],projection({kiosk_nonce:'rotated'}));await tick();save(1).click();
assert.equal(requests[3].options.body,first.options.body);assert.equal(requests[3].options.headers['X-Kiosk-Nonce'],'rotated');
response(requests[3],{},403);await tick();assert(save(1).disabled);
response(requests[4],projection({review_session_id:'ended-2',kiosk_nonce:'next'}));await tick();
assert.equal(el('postgame-player').src,'');assert.equal(requests.length,5);privacy();
""",
            id="uncertain-save-reuses-request-and-clears-authority-on-403",
        ),
        pytest.param(
            r"""
response(requests[0],projection({kiosk_nonce:undefined}));await tick();
assert(save(0).disabled);assert(save(1).disabled);assert(el('postgame-finish').disabled);
play(1).click();assert.equal(el('postgame-player').src,media);
assert(el('postgame-status').textContent.includes('本机大屏'));assert.equal(requests.length,1);privacy();
""",
            id="lan-read-only-keeps-playback",
        ),
        pytest.param(
            r"""
response(requests[0],projection({phase:'FAILED_RETRYING',analysis_status:'FAILED_RETRYING',
  clips:[{shot_id:'shot-1',clip_id:clip,media_url:media,save_status:'SAVE_FAILED_RETRYING'}]}));
await tick();
assert(el('postgame-status').textContent.includes('分析'));assert(rowStatus(0).includes('同步重试'));
assert(save(0).disabled);assert(el('postgame-finish').disabled);assert(!play(0).disabled);
el('postgame-retry').click();assert.equal(requests[1].url,'/api/postgame-review');
assert.equal(requests[1].options.method,undefined);
""",
            id="analysis-retry-and-sync-retry-distinct-no-invented-write",
        ),
        pytest.param(
            r"""
response(requests[0],projection({clips:[
  {shot_id:'shot-1',clip_id:clip,media_url:media,save_status:'SAVE_BLOCKED'}]}));
await tick();
assert(rowStatus(0).includes('无法同步'));assert(!rowStatus(0).includes('重试'));
assert.equal(save(0).textContent,'保存失败');assert(save(0).disabled);
save(0).click();await tick();assert.equal(requests.length,1);
assert(el('postgame-finish').disabled);assert(!play(0).disabled);
""",
            id="terminal-save-failure-is-honest-and-cannot-create-duplicate-save",
        ),
        pytest.param(
            r"""
response(requests[0],readyProjection());await tick();play(1).click();play(1).focus();
poll();const old=requests[1];el('postgame-finish').click();el('postgame-finish').click();
assert.equal(requests.length,3);const sent=requests[2];
assert.equal(sent.url,'/api/postgame-review/complete');
assert.deepEqual(JSON.parse(sent.options.body),{session_id:'ended-1',postgame_manifest_sha256:digest});
assert.equal(sent.options.headers['X-Kiosk-Nonce'],nonce);
response(sent,readyProjection({review_session_id:'ended-2',kiosk_nonce:'next',clips:[],
  expected_count:0,available_count:0}));await tick();
assert(el('postgame-session').textContent.includes('ended-2'));assert.equal(el('postgame-player').src,'');
assert(el('postgame-player').pauses>0);assert.equal(document.activeElement,el('postgame-title'));
response(old,projection());await tick();
assert(el('postgame-session').textContent.includes('ended-2'));
el('postgame-finish').click();assert.equal(requests[3].options.headers['X-Kiosk-Nonce'],'next');
response(requests[3],empty);await tick();assert(el('postgame-panel').hidden);
assert(el('postgame-finish').disabled);privacy();
""",
            id="finish-exact-scope-atomic-advance-unload-and-stale-poll",
        ),
        pytest.param(
            r"""
response(requests[0],projection({phase:'FAILED_RETRYING',clips:[],expected_count:0,available_count:0,can_complete:true}));
await tick();
assert(!el('postgame-finish').disabled);el('postgame-finish').click();
requests[1].reject(Error('lost complete'));
await tick();assert(el('postgame-finish').disabled);
response(requests[2],projection({review_session_id:'ended-2',kiosk_nonce:'next'}));await tick();
assert.equal(requests.length,3);assert(el('postgame-session').textContent.includes('ended-2'));
""",
            id="zero-clips-can-finish-lost-ack-never-completes-next",
        ),
        pytest.param(
            r"""
el('route-frame').src='/media/live.mjpg?epoch=keep';response(requests[0],projection());
await tick();play(1).click();poll();requests[1].reject(Error('offline'));await tick();
assert(save(1).disabled);assert(el('postgame-finish').disabled);
assert.equal(el('postgame-player').src,media);
assert.equal(el('route-frame').src,'/media/live.mjpg?epoch=keep');
events.online();response(requests[2],projection({kiosk_nonce:'new'}));await tick();
assert(!save(1).disabled);
el('postgame-player').onerror();assert(el('postgame-status').textContent.includes('回放'));
assert.equal(el('route-frame').src,'/media/live.mjpg?epoch=keep');
documentEvents.visibilitychange();response(requests[3],projection());await tick();privacy();
""",
            id="reconnect-and-media-error-preserve-live-stream",
        ),
        pytest.param(
            r"""
response(requests[0],projection({clips:[{shot_id:'<img onerror=evil>',clip_id:clip,
  save_status:'NOT_SAVED',media_url:'https://evil.test/'+media}]}));await tick();
assert(play(0).disabled);assert.equal(rows()[0].children[0].textContent,'红方 · 第 1 杆');
assert.equal(el('postgame-player').src,'');
poll();response(requests[1],projection({clips:[{shot_id:'shot-1',clip_id:clip,
  save_status:'NOT_SAVED',media_url:media.replace(digest,'c'.repeat(64))}]}));await tick();
assert(play(0).disabled);privacy();
""",
            id="untrusted-text-and-foreign-or-wrong-manifest-media-rejected",
        ),
        pytest.param(
            r"""
response(requests[0],projection());await tick();save(0).click();response(requests[1],{},409);
await tick();assert(save(0).disabled);
response(requests[2],{phase:'POSTGAME_REVIEW',clips:'invalid'});
await tick();
assert(save(0).disabled);assert(el('postgame-finish').disabled);
poll();response(requests[3],projection({kiosk_nonce:'fresh'}));await tick();
assert(!save(0).disabled);
""",
            id="conflict-invalid-projection-fail-closed-and-recover",
        ),
        pytest.param(
            r"""
response(requests[0],projection());await tick();poll();
const stalled=requests[1],timeout=timeouts.find(timer=>!timer.cleared);
assert(timeout,'review GET needs a bounded timeout');timeout.cb();await tick();
assert(stalled.options.signal.aborted);assert(save(1).disabled);assert(el('postgame-finish').disabled);
poll();response(requests[2],projection({review_session_id:'ended-2',kiosk_nonce:'next'}));
await tick();response(stalled,projection());await tick();
assert(el('postgame-session').textContent.includes('ended-2'));
save(1).click();const write=requests[3],body=write.options.body;
timeouts.find(timer=>!timer.cleared).cb();await tick();assert(write.options.signal.aborted);
response(requests[4],projection({review_session_id:'ended-2',kiosk_nonce:'new'}));await tick();
save(1).click();assert.equal(requests[5].options.body,body);
""",
            id="stalled-read-and-write-timeout-fail-closed-with-safe-retry",
        ),
        pytest.param(
            r"""
response(requests[0],readyProjection());await tick();play(1).click();
el('postgame-finish').focus();el('postgame-finish').click();
response(requests[1],empty);await tick();assert(!el('postgame-completion').hidden);
poll();response(requests[2],empty);await tick();assert(!el('postgame-completion').hidden);
assert.equal(document.activeElement,el('postgame-completion'));
""",
            id="completed-status-keeps-focus-on-unchanged-empty-poll",
        ),
        pytest.param(
            r"""
response(requests[0],readyProjection());await tick();el('postgame-finish').focus();
el('postgame-finish').click();response(requests[1],empty);await tick();
assert.equal(document.activeElement,el('postgame-completion'));
poll();response(requests[2],projection({review_session_id:'ended-2'}));await tick();
assert(el('postgame-completion').hidden);assert(!el('postgame-panel').hidden);
assert.equal(document.activeElement,el('postgame-title'));
""",
            id="completed-status-transfers-focus-to-later-review",
        ),
        pytest.param(
            r"""
response(requests[0],projection());await tick();play(1).click();
const video=el('postgame-player');video.onerror();
assert.equal(typeof video.onloadeddata,'function');
play(1).click();assert(el('postgame-status').textContent.includes('回放暂不可用'));
video.currentSrc='http://127.0.0.1:8765/stale.mp4';video.readyState=2;
video.onloadeddata();assert(el('postgame-status').textContent.includes('回放暂不可用'));
video.currentSrc='http://127.0.0.1:8765'+media;video.readyState=1;
video.onloadeddata();assert(el('postgame-status').textContent.includes('回放暂不可用'));
video.readyState=2;video.error={code:3};
video.onloadeddata();assert(el('postgame-status').textContent.includes('回放暂不可用'));
video.error=null;video.onloadeddata();
assert(!el('postgame-status').textContent.includes('回放暂不可用'));
""",
            id="current-media-readiness-clears-error-after-success-only",
        ),
        pytest.param(
            r"""
const otherClip='clip-'+'c'.repeat(64);
const otherMedia='/media/postgame/'+otherClip+'.mp4?manifest='+digest;
const twoClips=[{shot_id:'shot-2',clip_id:otherClip,save_status:'NOT_SAVED',media_url:otherMedia},
  {shot_id:'shot-1',clip_id:clip,save_status:'NOT_SAVED',media_url:media}];
response(requests[0],projection({clips:twoClips}));await tick();play(1).click();
const video=el('postgame-player'),oldReady=video.onloadeddata;
assert.equal(typeof oldReady,'function');video.onerror();play(0).click();
video.currentSrc='http://127.0.0.1:8765'+otherMedia;video.readyState=2;video.error=null;
oldReady();assert(el('postgame-status').textContent.includes('回放暂不可用'));
const priorScopeReady=video.onloadeddata;priorScopeReady();
assert(!el('postgame-status').textContent.includes('回放暂不可用'));
poll();response(requests[1],projection({review_session_id:'ended-2',clips:twoClips}));await tick();
play(0).click();video.onerror();priorScopeReady();
assert(el('postgame-status').textContent.includes('回放暂不可用'));
video.onloadeddata();assert(!el('postgame-status').textContent.includes('回放暂不可用'));
""",
            id="stale-media-readiness-cannot-clear-new-shot-or-scope-error",
        ),
        pytest.param(
            r"""
response(requests[0],projection());await tick();
const oldSave=save(0),oldPlay=play(1);
poll();response(requests[1],projection({review_session_id:'ended-2',kiosk_nonce:'next'}));
await tick();
oldSave.click();oldPlay.click();assert.equal(requests.length,2);assert.equal(el('postgame-player').src,'');
save(0).click();const sent=requests[2],body=JSON.parse(sent.options.body);
response(sent,{...body,session_id:'wrong-ended-session',status:'SAVED'});await tick();
assert(save(0).disabled);assert(el('postgame-finish').disabled);assert.equal(requests[3].url,'/api/postgame-review');
""",
            id="detached-old-buttons-and-mismatched-save-receipt-cannot-retarget",
        ),
        pytest.param(
            r"""
response(requests[0],projection());await tick();const priorPlay=play(1);
poll();response(requests[1],projection({clips:[
  {shot_id:'shot-1',clip_id:clip,save_status:'NOT_SAVED',media_url:media},
  {shot_id:'shot-3',clip_id:null,save_status:'NOT_SAVED',media_url:null}]}));await tick();
assert.equal(play(0),priorPlay);
assert.notEqual(rows()[0].children[3].id,rows()[1].children[3].id);
assert.equal(save(0).getAttribute('aria-describedby'),rows()[0].children[3].id);
assert.equal(save(1).getAttribute('aria-describedby'),rows()[1].children[3].id);
""",
            id="row-reconciliation-preserves-unique-accessible-status-targets",
        ),
    ],
)
def test_postgame_review_browser_contract(scenario):
    _run_review(scenario)


@pytest.mark.skipif(shutil.which("node") is None, reason="JavaScript runtime unavailable")
def test_review_scope_survives_real_live_script_polling_a_new_open_session():
    live_script = render_screen_page().split("<script>")[1].split("</script>")[0]
    _run_review(
        r"""
response(requests[0],projection());await tick();play(1).click();
vm.runInContext(LIVE_SCRIPT,context);
const state={session:{session_id:'new-open-session',status:'open'},
  frame:{frame_id:2,media_frame_id:40,source_kind:'usb_live',source_epoch_id:'epoch-new',
    guidance_visible:false},display:{phase:'WAITING_FOR_SHOT'},
  match:{revision:1},join:{qr_available:false}};
assert.equal(requests[1].url,'/api/screen');response(requests[1],state);await tick();
assert.equal(el('route-frame').src,'/media/live.mjpg?epoch=epoch-new');
assert.equal(el('postgame-player').src,media);assert(!el('postgame-panel').hidden);
save(1).click();assert.equal(JSON.parse(requests[2].options.body).session_id,'ended-1');
""".replace("LIVE_SCRIPT", json.dumps(live_script))
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="JavaScript runtime unavailable")
def test_lan_tablet_sends_displayed_exact_reference_without_secure_context_uuid():
    page = render_tablet_page(live=True)
    script = page.rsplit("<script>", 1)[1].split("</script>")[0]
    reference = {
        "session_id": "session-1",
        "shot_id": "shot-0001",
        "evidence_digest": "a" * 64,
        "match_revision": 7,
    }
    harness = """
const vm=require('node:vm');
const sent=[];
const context={renderState(){},command(){},status:{},document:{querySelectorAll(){return []}},
  token:'secret', crypto:{getRandomValues(a){a.fill(1);return a;}},
  api:async(url, options)=>{sent.push({url,...options});},refresh:async()=>{}};
vm.createContext(context);
vm.runInContext(SCRIPT,context);
vm.runInContext('renderState('+STATE+');',context);
vm.runInContext("command({dataset:{command:'continue_turn'}})",context).then(()=>{
  if(sent.length!==1)throw Error('no authenticated confirmation was sent');
  console.log(JSON.stringify(sent[0]));
}).catch(e=>{console.error(e);process.exitCode=1});
""".replace("SCRIPT", json.dumps(script)).replace(
        "STATE",
        json.dumps(
            json.dumps(
                {
                    "session": {"status": "open"},
                    "rule_status": "RULES_PENDING",
                    "confirmation_ref": reference,
                }
            )
        ),
    )
    result = subprocess.run([shutil.which("node"), "-e", harness], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    sent = json.loads(result.stdout)
    assert sent["url"] == "/api/control"
    assert sent["headers"]["X-Control-Token"] == "secret"
    payload = json.loads(sent["body"])
    assert payload["confirmation_ref"] == reference
    assert payload["command"] == "continue_turn"
    assert payload["command_id"]


@pytest.mark.skipif(shutil.which("node") is None, reason="JavaScript runtime unavailable")
def test_screen_disconnect_removes_old_route_frame_and_reloads_same_frame_on_recovery():
    script = render_screen_page().split("<script>")[1].split("</script>")[0]
    harness = """
const vm=require('node:vm'), elements={};
function element(id){return elements[id]??=(
 {style:{},classList:{add(){},remove(){},toggle(){}},removeAttribute(name){delete this[name]}});}
let failed=false;
const state={frame:{frame_id:7,guidance_visible:true},display:{phase:'READY'},
 match:{revision:1},join:{qr_available:false}};
const context={document:{getElementById:element},setInterval(){},
 fetch:async()=>{if(failed)throw Error('disconnected');return {ok:true,json:async()=>state}}};
vm.createContext(context);vm.runInContext(SCRIPT,context);
(async()=>{
 await vm.runInContext('sync()',context);
 if(!elements['route-frame'].src)throw Error('initial route missing');
 failed=true;await vm.runInContext('sync()',context);
 if(elements['route-frame'].src)throw Error('disconnected screen retained old route');
 failed=false;await vm.runInContext('sync()',context);
 if(elements['route-frame'].src!=='/media/frame/7.jpg')throw Error('recovery did not reload frame');
})().catch(e=>{console.error(e);process.exitCode=1});
""".replace("SCRIPT", json.dumps(script))
    result = subprocess.run([shutil.which("node"), "-e", harness], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="JavaScript runtime unavailable")
def test_live_screen_uses_one_continuous_mjpeg_connection_instead_of_frame_polling():
    script = render_screen_page().split("<script>")[1].split("</script>")[0]
    harness = """
const vm=require('node:vm'), elements={};
function element(id){return elements[id]??=(
 {style:{},classList:{add(){},remove(){},toggle(){}},removeAttribute(name){delete this[name]}});}
let mediaFrameId=40;
const context={document:{getElementById:element},setInterval(){},
 fetch:async()=>({ok:true,json:async()=>({
  frame:{frame_id:2,media_frame_id:mediaFrameId,source_kind:'usb_live',source_epoch_id:'epoch-1',guidance_visible:false},
  display:{phase:'WAITING_FOR_SHOT'},match:{revision:1},join:{qr_available:false}})})};
vm.createContext(context);vm.runInContext(SCRIPT,context);
(async()=>{
 await vm.runInContext('sync()',context);
 if(elements['route-frame'].src!=='/media/live.mjpg?epoch=epoch-1')
   throw Error('continuous live stream missing');
 mediaFrameId=41;await vm.runInContext('sync()',context);
 if(elements['route-frame'].src!=='/media/live.mjpg?epoch=epoch-1')
   throw Error('live polling replaced the continuous stream');
})().catch(e=>{console.error(e);process.exitCode=1});
""".replace("SCRIPT", json.dumps(script))
    result = subprocess.run([shutil.which("node"), "-e", harness], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="JavaScript runtime unavailable")
def test_live_screen_keeps_mjpeg_when_state_poll_temporarily_fails():
    script = render_screen_page().split("<script>")[1].split("</script>")[0]
    harness = """
const vm=require('node:vm'), elements={};
function element(id){return elements[id]??=(
 {style:{},classList:{add(){},remove(){},toggle(){}},removeAttribute(name){delete this[name]}});}
let failed=false;
const state={frame:{frame_id:2,media_frame_id:40,source_kind:'usb_live',
 source_epoch_id:'epoch-1',guidance_visible:false},
 display:{phase:'WAITING_FOR_SHOT'},match:{revision:1},join:{qr_available:false}};
const context={document:{getElementById:element},setInterval(){},
 fetch:async()=>{if(failed)throw Error('state timeout');return {ok:true,json:async()=>state}}};
vm.createContext(context);vm.runInContext(SCRIPT,context);
(async()=>{
 await vm.runInContext('sync()',context);
 if(elements['route-frame'].src!=='/media/live.mjpg?epoch=epoch-1')
   throw Error('continuous live stream missing');
 failed=true;await vm.runInContext('sync()',context);
 if(elements['route-frame'].src!=='/media/live.mjpg?epoch=epoch-1')
   throw Error('state failure removed healthy MJPEG stream');
})().catch(e=>{console.error(e);process.exitCode=1});
""".replace("SCRIPT", json.dumps(script))
    result = subprocess.run([shutil.which("node"), "-e", harness], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="JavaScript runtime unavailable")
def test_live_screen_reconnects_mjpeg_after_stream_error():
    script = render_screen_page().split("<script>")[1].split("</script>")[0]
    harness = """
const vm=require('node:vm'), elements={};
function element(id){return elements[id]??=(
 {style:{},classList:{add(){},remove(){},toggle(){}},removeAttribute(name){delete this[name]}});}
const state={frame:{frame_id:2,media_frame_id:40,source_kind:'usb_live',
 source_epoch_id:'epoch-1',guidance_visible:false},
 display:{phase:'WAITING_FOR_SHOT'},match:{revision:1},join:{qr_available:false}};
const context={document:{getElementById:element},setInterval(){},Date,
 fetch:async()=>({ok:true,json:async()=>state})};
vm.createContext(context);vm.runInContext(SCRIPT,context);
(async()=>{
 await vm.runInContext('sync()',context);
 if(typeof elements['route-frame'].onerror!=='function')
   throw Error('stream reconnect handler missing');
 elements['route-frame'].onerror();
 await vm.runInContext('sync()',context);
 if(elements['route-frame'].src!=='/media/live.mjpg?epoch=epoch-1&retry=1')
   throw Error('failed MJPEG stream was not reconnected');
})().catch(e=>{console.error(e);process.exitCode=1});
""".replace("SCRIPT", json.dumps(script))
    result = subprocess.run([shutil.which("node"), "-e", harness], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="JavaScript runtime unavailable")
def test_live_screen_periodically_reconnects_a_silently_stalled_mjpeg_stream():
    script = render_screen_page().split("<script>")[1].split("</script>")[0]
    harness = """
const vm=require('node:vm'), elements={};let now=1000;
function element(id){return elements[id]??=(
 {style:{},classList:{add(){},remove(){},toggle(){}},removeAttribute(name){delete this[name]}});}
const state={frame:{frame_id:2,media_frame_id:40,source_kind:'usb_live',
 source_epoch_id:'epoch-1',guidance_visible:false},
 display:{phase:'WAITING_FOR_SHOT'},match:{revision:1},join:{qr_available:false}};
const context={document:{getElementById:element},setInterval(){},Date:{now:()=>now},
 fetch:async()=>({ok:true,json:async()=>state})};
vm.createContext(context);vm.runInContext(SCRIPT,context);
(async()=>{
 await vm.runInContext('sync()',context);
 now=12000;state.frame.media_frame_id=700;
 await vm.runInContext('sync()',context);
 if(elements['route-frame'].src!=='/media/live.mjpg?epoch=epoch-1&retry=1')
   throw Error('silent MJPEG stall was not bounded by a reconnect');
})().catch(e=>{console.error(e);process.exitCode=1});
""".replace("SCRIPT", json.dumps(script))
    result = subprocess.run([shutil.which("node"), "-e", harness], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_screen_state_poll_does_not_contend_with_sixty_fps_capture() -> None:
    page = render_screen_page()

    assert "setInterval(sync,200)" in page
    assert "setInterval(sync,50)" not in page


def test_screen_labels_frozen_assistance_and_non_assistance_modes() -> None:
    page = render_screen_page()

    assert "辅助模式" in page
    assert "非辅助模式" in page


@pytest.mark.skipif(shutil.which("node") is None, reason="JavaScript runtime unavailable")
def test_join_qr_remains_visible_while_a_player_is_missing_during_join_degradation():
    script = render_screen_page().split("<script>")[1].split("</script>")[0]
    harness = """
const vm=require('node:vm'), elements={};
function element(id){return elements[id]??=(
 {style:{},classList:{values:new Set(),add(){},remove(){},toggle(name,on){
   if(on)this.values.add(name);else this.values.delete(name)}},
   removeAttribute(name){delete this[name]}});}
const state={frame:{frame_id:null,guidance_visible:false},
 display:{phase:'DEGRADED',join_prompt:'微信扫码加入本桌',red_slot:'红方待认领',blue_slot:'蓝方待认领'},
 match:{revision:1,red_user_id:null,blue_user_id:null},
 join:{qr_available:true,error:'join polling temporarily unavailable'}};
const context={document:{getElementById:element},setInterval(){},
 fetch:async()=>({ok:true,json:async()=>state})};
vm.createContext(context);vm.runInContext(SCRIPT,context);
(async()=>{
 await vm.runInContext('sync()',context);
 if(!elements['join-card'].classList.values.has('visible'))throw Error('QR was hidden');
 if(!elements['join-prompt'].textContent.includes('入局服务暂时不可用'))
   throw Error('join degradation was not shown beside QR');
})().catch(e=>{console.error(e);process.exitCode=1});
""".replace("SCRIPT", json.dumps(script))
    result = subprocess.run([shutil.which("node"), "-e", harness], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="JavaScript runtime unavailable")
def test_failed_analysis_allows_finishing_selection_of_recorded_originals():
    _run_review(r"""
response(requests[0],projection({phase:'FAILED_RETRYING',analysis_status:'FAILED_RETRYING',
 can_complete:true,clips:[{shot_id:'one',clip_id:clip,media_url:media,
 can_save:false,save_status:'NOT_SAVED'}]}));await tick();
assert(!el('postgame-finish').disabled);assert(!play(0).disabled);assert(save(0).disabled);
assert(el('postgame-status').textContent.includes('分析'));
assert(el('postgame-status').textContent.includes('未保存片段将删除'));
assert(!el('postgame-status').textContent.includes('全部片段就绪后'));
el('postgame-finish').click();assert.equal(requests[1].url,'/api/postgame-review/complete');
response(requests[1],empty);await tick();assert(el('postgame-panel').hidden);
""")


@pytest.mark.skipif(shutil.which("node") is None, reason="JavaScript runtime unavailable")
def test_private_shot_labels_destination_and_unknown_save_guard():
    _run_review(r"""
response(requests[0],projection({clips:[
 {shot_id:'one',sequence:1,actor_role:'red',can_save:true,save_status:'NOT_SAVED'},
 {shot_id:'two',sequence:2,actor_role:'blue',can_save:true,save_status:'NOT_SAVED'},
 {shot_id:'three',sequence:3,actor_role:null,can_save:false,save_status:'NOT_SAVED',clip_id:clip,media_url:media}
]}));await tick();
assert.equal(rows()[0].children[0].textContent,'红方 · 第 1 杆');
assert.equal(rows()[1].children[0].textContent,'蓝方 · 第 2 杆');
assert.equal(save(0).textContent,'保存至红方的小程序');
assert.equal(save(1).textContent,'保存至蓝方的小程序');
assert(rowStatus(2).includes('归属未确认，无法保存'));assert(save(2).disabled);
assert(!play(2).disabled);save(2).onclick();await tick();assert.equal(requests.length,1);
""")


@pytest.mark.skipif(shutil.which("node") is None, reason="JavaScript runtime unavailable")
@pytest.mark.parametrize("owner_status", ["pending", "completed"])
def test_owner_deleted_projection_stops_player_and_disables_replay_and_save(owner_status):
    _run_review(
        r"""
response(requests[0],projection());await tick();play(1).click();
const player=el('postgame-player'),priorLoaded=player.onloadeddata;
assert.equal(player.src,media);
poll();response(requests[1],projection({clips:[
  {shot_id:'shot-2',clip_id:null,save_status:'NOT_SAVED',media_url:null},
  {shot_id:'shot-1',clip_id:clip,save_status:'SAVE_BLOCKED',media_url:media,
   owner_delete_status:OWNER_STATUS,can_save:true}]}));await tick();
assert.equal(player.src,'');assert(player.hidden);assert(player.pauses>0);
assert(play(1).disabled);assert(save(1).disabled);
assert.equal(rowStatus(1),'已由所属用户删除');
assert(!el('postgame-progress').textContent.includes('可回看'));
assert(el('postgame-progress').textContent.includes('1 / 2'));
assert(!save(0).disabled);play(1).click();save(1).click();
assert.equal(player.src,'');assert.equal(requests.length,2);
priorLoaded();assert.equal(player.src,'');privacy();
""".replace("OWNER_STATUS", json.dumps(owner_status))
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="JavaScript runtime unavailable")
def test_recording_end_browser_status_does_not_claim_processing_complete():
    _run_review(r"""
response(requests[0],projection({phase:'RECORDING_ENDED',analysis_status:'NOT_REQUESTED',
  encoding_status:'NOT_SCHEDULED',expected_count:2,available_count:0,clips:[],
  kiosk_nonce:undefined,can_complete:false}));await tick();
assert(!el('postgame-panel').hidden);assert(el('postgame-finish').disabled);
assert(el('postgame-status').textContent.includes('未请求分析'));
assert(el('postgame-status').textContent.includes('尚未调度编码'));
assert(!el('postgame-status').textContent.includes('处理已完成'));
assert.equal(rows().length,0);
el('postgame-finish').click();assert.equal(requests.length,1);
""")


@pytest.mark.skipif(shutil.which("node") is None, reason="JavaScript runtime unavailable")
def test_recording_terminal_failure_is_not_presented_as_generation_or_retry():
    _run_review(r"""
response(requests[0],projection({phase:'RECORDING_ENDED',analysis_status:'NOT_REQUESTED',
  encoding_status:'FAILED',expected_count:1,available_count:0,can_complete:false,
  clips:[{shot_id:'shot-1',clip_id:null,media_url:null,save_status:'NOT_SAVED',
    can_save:false,
    ownership_unavailable_reason:'RECORDING_ONLY_READ_ONLY',encoding_status:'FAILED',
    terminal_failure:true,error_code:'INCOMPLETE_RECORDING_WINDOW'}]}));await tick();
const feedback=rows()[0].children[3].textContent;
assert(feedback.includes('录制区间不完整'));assert(feedback.includes('不会自动重试'));
assert(!feedback.includes('正在生成'));assert(!el('postgame-status').textContent.includes('将按策略重试'));
assert(play(0).disabled);assert(save(0).disabled);
""")


@pytest.mark.skipif(shutil.which("node") is None, reason="JavaScript runtime unavailable")
def test_started_live_match_hides_and_unloads_old_review_without_claiming_completion():
    _run_review(r"""
response(requests[0],readyProjection());await tick();play(1).click();
assert(!el('postgame-player').hidden);
poll();response(requests[1],{...empty,phase:'LIVE_MATCH'});await tick();
assert(el('postgame-panel').hidden);assert(el('postgame-player').hidden);
assert.equal(el('postgame-player').getAttribute('src'),null);
assert(el('postgame-completion').hidden);
poll();response(requests[2],readyProjection());await tick();
assert(!el('postgame-panel').hidden);assert(el('postgame-player').hidden);
""")


@pytest.mark.skipif(shutil.which("node") is None, reason="JavaScript runtime unavailable")
def test_recording_end_button_sends_displayed_original_session():
    script = render_tablet_page(live=True).rsplit("<script>", 1)[1].split("</script>")[0]
    harness = r"""
const vm=require('node:vm'),assert=require('node:assert/strict');
const sent=[],button={dataset:{command:'end_session'}};
const context={renderState(){},command(){},status:{textContent:''},
 document:{querySelectorAll(s){return s.includes('end_session')?[button]:[];}},
 token:'secret',crypto:{getRandomValues(a){a.fill(1);return a;}},
 api:async(url,options)=>{sent.push({url,...options});},refresh:async()=>{}};
vm.createContext(context);vm.runInContext(SCRIPT,context);
vm.runInContext("renderState({session_policy:'recording-v1',session:{session_id:'original',status:'open'}})",context);
assert.equal(button.textContent,'结束本场录制');
context.button=button;
vm.runInContext('command(button)',context).then(()=>{
 const body=JSON.parse(sent[0].body);assert.equal(body.session_id,'original');
 assert.equal(body.command,'end_session');assert.equal(sent[0].headers['X-Control-Token'],'secret');
}).catch(e=>{console.error(e);process.exitCode=1});
""".replace("SCRIPT", json.dumps(script))
    result = subprocess.run([shutil.which("node"), "-e", harness], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="JavaScript runtime unavailable")
def test_growing_recording_shows_progress_and_keeps_selected_playback_on_refresh():
    _run_review(r"""
const growing=projection({phase:'RECORDING_PROCESSING',analysis_status:'NOT_REQUESTED',
 expected_count:null,discovered_count:2,available_count:1,can_complete:false,
 processing:{status:'PROCESSING',tail_status:'SAVED',remaining_frames:201},
 clips:[{shot_id:'one',sequence:1,clip_id:clip,media_url:media,can_save:false,
   save_status:'NOT_SAVED'},
  {shot_id:'two',sequence:2,clip_id:null,media_url:null,can_save:false,save_status:'NOT_SAVED'}]});
response(requests[0],growing);await tick();
assert(!el('postgame-panel').hidden);assert(el('postgame-finish').disabled);
assert(el('postgame-progress').textContent.includes('201'));
assert(el('postgame-progress').textContent.includes('已发现 2'));
assert(!el('postgame-progress').textContent.includes('null'));
assert(!el('postgame-status').textContent.includes('分析已完成'));
assert(!play(0).disabled);assert(save(0).disabled);play(0).click();
const player=el('postgame-player'),before=player.pauses;
poll();response(requests[1],{...growing,discovered_count:3,
 processing:{status:'PROCESSING',tail_status:'SAVED',remaining_frames:120},
 clips:[...growing.clips,{shot_id:'three',sequence:3,clip_id:null,media_url:null,
 can_save:false,save_status:'NOT_SAVED'}]});await tick();
assert.equal(player.src,media);assert.equal(player.pauses,before);assert(!player.hidden);
assert.equal(rows().length,3);assert(el('postgame-finish').disabled);
assert(el('postgame-progress').textContent.includes('120'));
el('postgame-finish').click();assert.equal(requests.length,2);
""")


@pytest.mark.skipif(shutil.which("node") is None, reason="JavaScript runtime unavailable")
def test_live_generated_progress_is_visible_without_review_or_media_controls():
    _run_review(r"""
const live={...empty,phase:'LIVE_MATCH',expected_count:null,analysis_status:'NOT_REQUESTED',
 progress_status:'READY',discovered_count:3,generated_count:1,pending_count:1,encoding_count:1,
 failed_count:0,retrying_count:0};
response(requests[0],live);await tick();
assert(el('postgame-panel').hidden);assert(el('postgame-finish').disabled);
assert(!el('recording-progress').hidden);
assert(el('recording-progress').textContent.includes('已生成 1'));
assert(el('recording-progress').textContent.includes('结束后'));
assert.equal(el('postgame-player').srcWrites,0);assert.equal(rows().length,0);privacy();
poll();response(requests[1],{...live,generated_count:2,pending_count:0});await tick();
assert(el('recording-progress').textContent.includes('已生成 2'));
poll();response(requests[2],{...live,progress_status:'UNAVAILABLE'});await tick();
assert(el('recording-progress').textContent.includes('暂不可用'));
poll();response(requests[3],empty);await tick();
assert(el('recording-progress').hidden);privacy();
""")


def test_interrupted_recording_page_retains_playback_but_disables_deletion():
    _run_review(r"""
response(requests[0],projection({phase:'RECORDING_INTERRUPTED',can_complete:false,
  recording_capture:{end_reason:'PROCESS_INTERRUPTED',tail_completeness:'UNKNOWN'}}));await tick();
assert(!el('postgame-panel').hidden);
assert(el('postgame-status').textContent.includes('未提交尾部未知'));
assert(el('postgame-status').textContent.includes('原片继续保留'));
assert(el('postgame-finish').disabled);
await el('postgame-finish').click();await tick();
assert(!requests.some(r=>r.url.endsWith('/complete')));
""")
