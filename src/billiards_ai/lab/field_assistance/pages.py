# ruff: noqa: E501
"""Dependency-free pages for the local screen and tablet surfaces."""

from __future__ import annotations

from billiards_ai.lab.field_assistance.snapshot_page import (
    SNAPSHOT_PANEL,
    SNAPSHOT_SCRIPT,
    SNAPSHOT_STYLE,
)


def render_screen_page() -> str:
    return (
        _SCREEN_PAGE.replace("</style>", _POSTGAME_STYLE + SNAPSHOT_STYLE + "</style>")
        .replace("<main>", '<main id="screen-stage">')
        .replace("</main>", _POSTGAME_PANEL + SNAPSHOT_PANEL + "</main>")
        .replace("</body>", SNAPSHOT_SCRIPT + _POSTGAME_SCRIPT + "</body>")
        .replace(
            "joinCard.classList.toggle('visible',state.join.qr_available&&(!state.match.red_user_id||!state.match.blue_user_id))",
            "joinCard.classList.toggle('visible',state.join.qr_available&&(!state.match.red_user_id||!state.match.blue_user_id));"
            "if(typeof window!=='undefined'&&typeof window.renderSnapshotGuidance==='function')"
            "window.renderSnapshotGuidance(state.frame.snapshot_guidance);",
        )
    )


def render_tablet_page(*, live: bool = False) -> str:
    if live:
        page = _TABLET_PAGE.replace(
            '<section class="panel"><h2>开发回放控制</h2><div class="row"><button class="primary" data-command="start">开始</button><button data-command="pause">暂停</button><button data-command="reset">复位</button></div></section>',
            '<section class="panel"><h2>管理员故障诊断</h2><p>仅用于诊断时临时覆盖回合状态，不代表自动规则结果。现场正常产品流程不依赖此入口。</p><div class="row"><button data-command="continue_turn">诊断：保持击球方</button><button data-command="switch_turn">诊断：切换击球方</button><button data-command="end_session">诊断：结束本场</button></div></section>',
        )
        return page.replace("</body>", _LIVE_TABLET_SCRIPT + "</body>")
    return _TABLET_PAGE


_LIVE_TABLET_SCRIPT = r"""<script>
let liveState=null;
const baseRenderState=renderState,baseCommand=command;
renderState=function(state){liveState=state;baseRenderState(state);status.textContent+='\n规则状态：'+state.rule_status;const ordinary=state.session_policy==='recording-v1';document.querySelectorAll('button[data-command="end_session"]').forEach(endButton=>{endButton.textContent=ordinary?'结束本场录制':'诊断：结束本场';endButton.disabled=state.session.status==='ended';});document.querySelectorAll('button[data-command="continue_turn"],button[data-command="switch_turn"]').forEach(button=>button.disabled=!state.confirmation_ref);if(state.session.status==='ended')status.textContent=ordinary?'本场采集已停止；片段处理状态请在赛后大屏查看':'本场已结束，USB 记录已保存，等待分析与发布';};
command=async function(button){const kind=button.dataset.command;if(!['continue_turn','switch_turn','end_session'].includes(kind))return baseCommand(button);const commandId=Array.from(crypto.getRandomValues(new Uint8Array(16)),n=>n.toString(16).padStart(2,'0')).join('');const payload={command:kind,command_id:commandId};if(kind==='end_session'&&liveState&&liveState.session_policy==='recording-v1')payload.session_id=liveState.session.session_id;if(kind!=='end_session'){if(!liveState||!liveState.confirmation_ref)return;payload.confirmation_ref=liveState.confirmation_ref;}try{await api('/api/control',{method:'POST',headers:{'Content-Type':'application/json','X-Control-Token':token},body:JSON.stringify(payload)});await refresh();}catch(error){status.textContent='操作失败：'+error.message;}};
</script>"""


_POSTGAME_STYLE = r"""
[hidden]{display:none!important}
#postgame-panel{position:absolute;left:24px;bottom:24px;width:min(760px,calc(100vw - 48px));max-height:calc(100vh - 170px);overflow:auto;padding:20px;border:1px solid #52645b;border-radius:16px;background:rgba(5,12,9,.97);z-index:2}
#postgame-panel h2{margin:0 0 8px;font-size:24px}#postgame-panel p{margin:8px 0;overflow-wrap:anywhere}
#postgame-shot-list{max-height:25vh;overflow:auto;padding-left:28px}#postgame-shot-list li{padding:8px 0;display:flex;gap:10px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #34443b}
#postgame-panel button{min-height:48px;padding:10px 16px;border:1px solid #6b8a79;border-radius:8px;background:#1d3e2c;color:#fff;font:inherit;cursor:pointer}
#postgame-panel button:disabled{opacity:.5;cursor:not-allowed}#postgame-panel :focus-visible,#postgame-completion:focus-visible{outline:3px solid #f6d977;outline-offset:3px}
#postgame-player{width:100%;max-height:30vh;background:#000}#postgame-status{color:#c7e9d6}#postgame-completion{position:absolute;left:24px;bottom:24px;background:#10271c;padding:12px;z-index:2}
#recording-progress{position:absolute;left:24px;bottom:24px;max-width:760px;background:#10271c;color:#c7e9d6;padding:12px;z-index:2}
#postgame-recovery{position:absolute;left:24px;bottom:64px;max-width:760px;background:#10271c;color:#f6d977;padding:12px;z-index:2}
"""


_POSTGAME_PANEL = r"""
<section id="postgame-panel" hidden aria-labelledby="postgame-title">
<h2 id="postgame-title" tabindex="-1">已结束场次 · 逐杆回看</h2>
<p id="postgame-session"></p><p id="postgame-progress"></p>
<p id="postgame-status" role="status" aria-live="polite" aria-atomic="true"></p>
<ol id="postgame-shot-list" tabindex="0" aria-label="赛后逐杆列表"></ol>
<p id="postgame-player-label"></p>
<video id="postgame-player" hidden controls playsinline preload="metadata" aria-label="赛后逐杆回放" aria-describedby="postgame-player-label"></video>
<button id="postgame-retry" type="button">刷新回看连接</button>
<p>可先保存多个片段。已保存片段会保留；完成选择后，未保存片段将从本机删除。</p>
<button id="postgame-finish" type="button" disabled>完成并删除未保存片段</button>
</section><p id="postgame-completion" hidden tabindex="-1" role="status">本场回看已结束</p>
<p id="postgame-recovery" hidden role="status" aria-live="polite"></p>
<p id="recording-progress" hidden role="status" aria-live="polite"></p>
"""


_POSTGAME_SCRIPT = r"""<script id="postgame-review-script">
(()=>{
  'use strict';
  const byId=id=>document.getElementById(id),panel=byId('postgame-panel'),title=byId('postgame-title'),
    session=byId('postgame-session'),progress=byId('postgame-progress'),status=byId('postgame-status'),
    list=byId('postgame-shot-list'),video=byId('postgame-player'),label=byId('postgame-player-label'),
    retry=byId('postgame-retry'),finish=byId('postgame-finish'),completion=byId('postgame-completion'),recovery=byId('postgame-recovery'),liveProgress=byId('recording-progress');
  // Review authority is independent of the automatically rolled-over live match.
  // Secrets and uncertain request identities exist only inside this closure.
  let current=null,scope='',nonce='',selected='',epoch=0,getFlight=null,busy=null,notice='',mediaError=false,rowNumber=0;
  const rows=new Map(),attempts=new Map();
  const savedStates=new Set(['SAVE_PENDING','SAVED','SAVE_FAILED_RETRYING','SAVE_BLOCKED']);
  const receiptStates=new Set(['SAVE_PENDING','SAVED']);
  const text=(element,value)=>{if(element.textContent!==value)element.textContent=value;};
  const recoveryText='历史场次仍保留，原始资源或回看队列需要现场恢复；未保存片段不会自动删除。';
  const identity=value=>['REVIEW_COMPLETE','LIVE_MATCH'].includes(value.phase)?'':value.phase==='RECOVERY_REQUIRED'?'recovery':JSON.stringify([value.review_session_id,value.postgame_manifest_sha256]);
  function validate(value){
    if(!value||!['REVIEW_COMPLETE','LIVE_MATCH','RECOVERY_REQUIRED','RECORDING_ENDED','RECORDING_INTERRUPTED','RECORDING_PROCESSING','POSTGAME_PROCESSING','POSTGAME_REVIEW','FAILED_RETRYING'].includes(value.phase)||
       !Array.isArray(value.clips)||!Number.isInteger(value.available_count)||value.available_count<0)throw Error('invalid projection');
    if(value.phase==='LIVE_MATCH'&&value.expected_count===null){
      if(value.progress_status!=='UNAVAILABLE'&&(!['discovered_count','generated_count','pending_count','encoding_count','failed_count','retrying_count'].every(key=>Number.isSafeInteger(value[key])&&value[key]>=0)||value.generated_count+value.failed_count>value.discovered_count))throw Error('invalid live progress');
    }else if(value.phase==='RECORDING_PROCESSING'){
      if(value.expected_count!==null||!Number.isInteger(value.discovered_count)||value.discovered_count!==value.clips.length||
         value.available_count>value.discovered_count||value.can_complete!==false)throw Error('invalid growing projection');
    }else if(!Number.isInteger(value.expected_count)||value.expected_count<0||value.available_count>value.expected_count)throw Error('invalid final count');
    if(['REVIEW_COMPLETE','LIVE_MATCH','RECOVERY_REQUIRED'].includes(value.phase)){
      if(value.review_session_id!==null||value.postgame_manifest_sha256!==null||value.clips.length||value.expected_count||value.available_count)throw Error('invalid empty projection');
      return {...value,kiosk_nonce:undefined,can_complete:false};
    }
    if(value.phase==='RECORDING_INTERRUPTED'&&(value.can_complete!==false||!value.recording_capture||value.recording_capture.end_reason!=='PROCESS_INTERRUPTED'||value.recording_capture.tail_completeness!=='UNKNOWN'))throw Error('invalid interruption');
    if(typeof value.review_session_id!=='string'||!value.review_session_id||value.review_session_id.length>512||
       !/^[a-f0-9]{64}$/.test(value.postgame_manifest_sha256))throw Error('invalid identity');
    const ids=new Set();
    for(const clip of value.clips){
      if(!clip||typeof clip.shot_id!=='string'||!clip.shot_id||clip.shot_id.length>512||ids.has(clip.shot_id)||
         !['NOT_SAVED','SAVE_PENDING','SAVED','SAVE_FAILED_RETRYING','SAVE_BLOCKED'].includes(clip.save_status)||
         clip.playback_clip_id!=null&&!/^clip-[a-f0-9]{64}$/.test(clip.playback_clip_id))throw Error('invalid clip');
      ids.add(clip.shot_id);
    }
    return value;
  }
  function mediaUrl(clip){
    if(clip&&clip.owner_delete_status!=null)return '';
    const playbackId=clip&&(clip.playback_clip_id||clip.clip_id);
    if(!current||!clip||!/^clip-[a-f0-9]{64}$/.test(playbackId)||typeof clip.media_url!=='string')return '';
    const expected='/media/postgame/'+playbackId+'.mp4?manifest='+current.postgame_manifest_sha256;
    if(clip.media_url!==expected)return '';
    const url=new URL(clip.media_url,location.origin);
    return url.origin===location.origin?expected:'';
  }
  function unload(){video.onloadeddata=null;video.pause();video.removeAttribute('src');video.load();video.hidden=true;selected='';mediaError=false;text(label,'');}
  const canSave=clip=>clip.owner_delete_status==null&&clip.can_save===true&&['red','blue'].includes(clip.actor_role)&&Number.isSafeInteger(clip.sequence)&&clip.sequence>0;
  const actorLabel=clip=>clip.actor_role==='red'?'红方':clip.actor_role==='blue'?'蓝方':'';
  function rowStatus(clip){
    if(clip.owner_delete_status!=null)return '已由所属用户删除';
    if(clip.ownership_unavailable_reason==='RECORDING_ONLY_READ_ONLY'){
      if(mediaUrl(clip))return '原始录制可在本机大屏回看；不可保存或同步';
      const reason={INCOMPLETE_RECORDING_WINDOW:'录制区间不完整，无法生成回看',MEDIA_H264_ENCODER_UNAVAILABLE:'本机编码器不可用，无法生成回看',MEDIA_SOURCE_INVALID:'原始媒体校验失败，无法生成回看',MEDIA_SOURCE_REVOKED:'此片段已不可用'}[clip.error_code]||'片段生成失败';
      if(clip.terminal_failure===true||clip.encoding_status==='FAILED')return reason+'；不会自动重试';
      if(clip.encoding_status==='FAILED_RETRYING')return reason+'；等待后台重试';
      return clip.encoding_status==='NOT_SCHEDULED'?'尚未调度本机回看':'原始录制正在生成本机回看';
    }
    if(!canSave(clip)&&clip.save_status==='NOT_SAVED')return '归属未确认，无法保存';
    if(busy&&busy.shot_id===clip.shot_id)return '正在保存…';
    return {NOT_SAVED:'尚未保存',SAVE_PENDING:'已选择保存，片段生成中…',SAVED:'已保存',SAVE_FAILED_RETRYING:'已保存，同步重试中…',SAVE_BLOCKED:'保存失败，片段无法同步；请联系现场支持。'}[clip.save_status];
  }
  function render(){
    const active=Boolean(current&&scope);panel.hidden=!active;finish.disabled=!active||!nonce||Boolean(busy)||current.can_complete!==true;
    const needsRecovery=current&&(current.phase==='RECOVERY_REQUIRED'||current.recovery_required);
    recovery.hidden=active||!needsRecovery;text(recovery,needsRecovery?recoveryText:'');
    const live=Boolean(current&&current.phase==='LIVE_MATCH'&&current.expected_count===null);
    liveProgress.hidden=!live;
    text(liveProgress,live?(notice||current.progress_status==='UNAVAILABLE'?'片段生成进度暂不可用，请检查连接。':'录制进行中：已发现 '+current.discovered_count+' 段，已生成 '+current.generated_count+' 段，待生成 '+current.pending_count+' 段，生成中 '+current.encoding_count+' 段，失败 '+current.failed_count+' 段；整局结束后开放回看与保存。'):'');
    if(!active)return;
    panel.dataset.phase=current.phase;
    text(session,current.interrupted?'上次录制中断 · 已提交片段恢复':current.phase==='RECOVERY_REQUIRED'?'历史场次回看等待恢复':'回看已结束场次：'+current.review_session_id);
    const growing=current.phase==='RECORDING_PROCESSING',remaining=current.processing&&current.processing.remaining_frames;
    text(progress,growing?'已发现 '+current.discovered_count+' 段，'+current.available_count+' 段可回看；'+(Number.isSafeInteger(remaining)?'剩余 '+remaining+' 帧待处理':'处理进度暂不可用'):
      (['RECORDING_ENDED','RECORDING_INTERRUPTED'].includes(current.phase)?'回看片段进度：':'分析处理进度：')+current.available_count+' / '+current.expected_count);
    const phaseText=current.interrupted?'上次录制中断；仅恢复已完整提交的部分，未提交尾部未知。原片继续保留，不能完成选择或清理。':growing?(current.processing&&current.processing.status==='FAILED'?'本场采集已停止；后台处理尚未完成，请查看故障状态，原片继续保留。':current.processing&&current.processing.tail_status==='PENDING'?'本场采集已停止，正在保存尾段；已完成的片段可先回看。':'本场采集已停止，后台继续处理；已完成的片段可先回看，最终数量尚未确定。'):current.phase==='RECORDING_ENDED'?(current.available_count===current.expected_count?'本场录制已结束；原始片段可在本机大屏回看，不可保存或同步。':current.encoding_status==='NOT_SCHEDULED'?'本场录制已结束；未请求分析，尚未调度编码，本机回看暂不可用。':current.encoding_status==='FAILED'?'本场录制已结束；部分原始片段无法生成回看，请查看逐杆状态。':current.encoding_status==='FAILED_RETRYING'?'本场录制已结束；部分原始片段生成失败，后台将按策略重试。':'本场录制已结束；正在生成本机大屏原始回看，不可保存或同步。'):current.phase==='FAILED_RETRYING'?'赛后分析暂未完成，后台正在重试；已录片段可回看或选择保存。':
      current.phase==='POSTGAME_PROCESSING'?'正在进行赛后分析；已录片段可回看或选择保存。':
      current.analysis_status==='PARTIAL'?'部分赛后分析已完成。':'赛后分析已完成。';
    text(status,[needsRecovery?recoveryText:phaseText,current.can_complete===true?'可完成选择；未保存片段将删除。':'',notice,mediaError?'回放暂不可用，请刷新回看连接。':'',!nonce&&!busy&&current.phase!=='RECOVERY_REQUIRED'&&current.phase!=='RECORDING_ENDED'&&current.phase!=='RECORDING_INTERRUPTED'&&current.phase!=='RECORDING_PROCESSING'?'本机大屏可保存；连接恢复前仅可回看。':''].filter(Boolean).join(' '));
    const ordered=[];
    current.clips.forEach(clip=>{
      let row=rows.get(clip.shot_id);
      if(!row){
        const li=document.createElement('li'),name=document.createElement('span'),play=document.createElement('button'),save=document.createElement('button'),feedback=document.createElement('span');
        const rowScope=scope,shotId=clip.shot_id;
        li.dataset.shotId=shotId;play.type=save.type='button';
        play.onclick=()=>playShot(rowScope,shotId);save.onclick=()=>saveShot(rowScope,shotId);
        feedback.id='postgame-shot-status-'+rowNumber++;save.setAttribute('aria-describedby',feedback.id);
        li.appendChild(name);li.appendChild(play);li.appendChild(save);li.appendChild(feedback);
        row={li,name,play,save,feedback};rows.set(shotId,row);
      }
      const name=actorLabel(clip)&&Number.isSafeInteger(clip.sequence)&&clip.sequence>0?actorLabel(clip)+' · 第 '+clip.sequence+' 杆':clip.shot_id;
      text(row.name,name);text(row.play,'回看 '+name);
      text(row.save,clip.owner_delete_status!=null?'已由所属用户删除':clip.ownership_unavailable_reason==='RECORDING_ONLY_READ_ONLY'?'仅本机回看':clip.save_status==='SAVE_BLOCKED'?'保存失败':savedStates.has(clip.save_status)?(clip.save_status==='SAVE_PENDING'?'已选择保存':'已保存'):attempts.has(clip.shot_id)?'重试保存此杆':canSave(clip)?'保存至'+actorLabel(clip)+'的小程序':'归属未确认，无法保存');
      row.play.disabled=!mediaUrl(clip);row.play.setAttribute('aria-pressed',selected===clip.shot_id?'true':'false');
      row.save.disabled=!nonce||Boolean(busy)||!canSave(clip)||savedStates.has(clip.save_status);text(row.feedback,rowStatus(clip));
      if(clip.guidance_review){const review=clip.guidance_review,region=review.recommendation.region,actual=review.actual_cue;const desired=region?'击球前建议区中心：('+region.center.x_mm+','+region.center.y_mm+')，半径 '+region.radius_mm+' mm':'击球前未提供停球区';const measured=actual?'实际白球终点：('+actual.x_mm+','+actual.y_mm+') mm':'实际白球终点：未可靠测得';text(row.feedback,rowStatus(clip)+'；'+desired+'；'+measured+'；'+({inside:'已进入建议区',outside:'未进入建议区，已按新局面重算',uncertain:'区域边界附近，结果不确定',unknown:'尚无到区结论'}[review.region_result]||'尚无到区结论')+'。现场精度待验。');}
      row.li.dataset.saveStatus=clip.save_status;ordered.push(row.li);
    });
    const changed=list.children.length!==ordered.length||ordered.some((row,index)=>list.children[index]!==row);
    if(changed){
      const hadFocus=list.contains(document.activeElement);
      list.replaceChildren(...ordered);
      if(hadFocus&&!list.contains(document.activeElement))title.focus();
    }
    for(const [id,row] of rows){if(!current.clips.some(clip=>clip.shot_id===id))rows.delete(id);}
    if(selected){
      const clip=current.clips.find(item=>item.shot_id===selected);
      if(!clip||mediaUrl(clip)!==video.getAttribute('src'))unload();
    }
  }
  function apply(value){
    const next=identity(value),changed=next!==scope,
      hadFocus=panel.contains(document.activeElement)||!completion.hidden&&completion.contains(document.activeElement);
    if(changed){unload();rows.clear();list.replaceChildren();attempts.clear();}
    scope=next;
    nonce=next&&typeof value.kiosk_nonce==='string'?value.kiosk_nonce:'';
    // Retain public projection data only; the nonce never enters presentation state.
    current={interrupted:Boolean(value.recording_capture&&value.recording_capture.end_reason==='PROCESS_INTERRUPTED'&&value.recording_capture.tail_completeness==='UNKNOWN'),phase:value.phase,review_session_id:value.review_session_id,postgame_manifest_sha256:value.postgame_manifest_sha256,
      progress_status:value.progress_status,generated_count:value.generated_count,pending_count:value.pending_count,encoding_count:value.encoding_count,failed_count:value.failed_count,retrying_count:value.retrying_count,
      analysis_status:value.analysis_status,encoding_status:value.encoding_status,expected_count:value.expected_count,discovered_count:value.discovered_count,processing:value.processing,available_count:value.available_count,can_complete:value.can_complete===true,clips:value.clips,
      recovery_required:Boolean(value.recovery_diagnostics&&typeof value.recovery_diagnostics==='object'&&Object.keys(value.recovery_diagnostics).length)};
    notice='';if(next||value.phase==='LIVE_MATCH')completion.hidden=true;else if(changed)completion.hidden=false;render();
    if(changed&&hadFocus&&value.phase!=='LIVE_MATCH')(next?title:completion).focus();
  }
  function failClosed(message){nonce='';notice=message;render();}
  async function readJson(url,options={}){
    const controller=new AbortController();let timer;
    try{
      return await Promise.race([
        (async()=>{const response=await fetch(url,{...options,signal:controller.signal});
          if(!response.ok)throw Error('review request rejected');return await response.json();})(),
        new Promise((resolve,reject)=>{timer=setTimeout(()=>{controller.abort();reject(Error('review timeout'));},8000);})
      ]);
    }finally{clearTimeout(timer);}
  }
  async function refresh(){
    if(busy||getFlight&&getFlight.epoch===epoch)return;
    const flight={epoch};getFlight=flight;
    try{
      const value=validate(await readJson('/api/postgame-review',{cache:'no-store'}));
      if(flight.epoch===epoch&&!busy)apply(value);
    }catch(error){if(flight.epoch===epoch&&!busy)failClosed('回看连接中断，请刷新连接。');}
    finally{if(getFlight===flight)getFlight=null;}
  }
  function playShot(rowScope,shotId){
    if(rowScope!==scope)return;
    const clip=current.clips.find(item=>item.shot_id===shotId),url=mediaUrl(clip);if(!url)return;
    if(video.getAttribute('src')!==url){video.pause();video.src=url;}
    selected=shotId;video.hidden=false;text(label,'当前回看：'+shotId);
    video.onloadeddata=()=>{
      const activeClip=current&&current.clips.find(item=>item.shot_id===shotId);
      if(rowScope!==scope||selected!==shotId||mediaUrl(activeClip)!==url||video.getAttribute('src')!==url||
         video.currentSrc!==new URL(url,location.origin).href||video.readyState<2||video.error)return;
      mediaError=false;render();
    };
    render();
    // Selecting a row never starts playback automatically; native controls do that.
  }
  function requestId(){return Array.from(crypto.getRandomValues(new Uint8Array(16)),n=>n.toString(16).padStart(2,'0')).join('');}
  async function saveShot(rowScope,shotId){
    if(rowScope!==scope||!nonce||busy)return;
    const clip=current.clips.find(item=>item.shot_id===shotId);if(!clip||!canSave(clip)||savedStates.has(clip.save_status))return;
    let body=attempts.get(shotId);
    if(!body){body={request_id:requestId(),session_id:current.review_session_id,shot_id:shotId,postgame_manifest_sha256:current.postgame_manifest_sha256};attempts.set(shotId,body);}
    await mutate('/api/postgame-review/save',body,false);
  }
  async function mutate(url,body,complete){
    const snapshot={scope,nonce,epoch:++epoch,...body};busy=snapshot;notice=complete?'正在结束回看…':'正在保存选择…';render();
    try{
      const value=await readJson(url,{method:'POST',headers:{'Content-Type':'application/json','X-Kiosk-Nonce':snapshot.nonce},body:JSON.stringify(body)});
      if(snapshot.scope!==scope||snapshot.epoch!==epoch)return;
      if(complete){const next=validate(value);busy=null;epoch++;apply(next);return;}
      if(value.request_id!==body.request_id||value.session_id!==body.session_id||value.shot_id!==body.shot_id||
         value.postgame_manifest_sha256!==body.postgame_manifest_sha256||!receiptStates.has(value.status))throw Error('invalid receipt');
      // A receipt is historical. Only the following GET may update saved status.
      notice='保存请求已接收，正在确认当前状态…';
    }catch(error){notice=complete?'结束结果待确认，正在刷新；请核对场次后重试。':'保存结果待确认，正在刷新；未确认时可重试同一杆。';}
    finally{
      if(busy===snapshot){busy=null;epoch++;nonce='';render();refresh();}
    }
  }
  finish.onclick=()=>{
    if(!current||!scope||!nonce||busy||current.can_complete!==true)return;
    return mutate('/api/postgame-review/complete',{session_id:current.review_session_id,postgame_manifest_sha256:current.postgame_manifest_sha256},true);
  };
  video.onerror=()=>{if(selected){mediaError=true;render();}};
  retry.onclick=()=>{if(mediaError&&selected){const clip=current.clips.find(item=>item.shot_id===selected);if(mediaUrl(clip))video.load();}return refresh();};
  window.addEventListener('online',refresh);
  document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='visible')refresh();});
  panel.hidden=true;video.hidden=true;finish.disabled=true;
  setInterval(refresh,1000);refresh();
})();
</script>"""


_SCREEN_PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>billiardsAI 现场路线</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#050806;color:#fff;font-family:system-ui,-apple-system,sans-serif;overflow:hidden}
main{width:100vw;height:100vh;position:relative;display:grid;place-items:center;background:#000}#route-frame{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#000}#route-overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}#video-status{position:absolute;color:#a9bbb1;font-size:22px}.camera-ready+#route-overlay+#video-status{display:none}
.hud{position:absolute;left:24px;right:24px;top:20px;display:flex;justify-content:space-between;gap:18px;padding:14px 18px;border-radius:14px;background:rgba(5,12,9,.78);backdrop-filter:blur(8px);font-size:18px}
#phase{font-weight:750;color:#8ff0bd}#player{color:#f6d977}.offline{color:#ff9a9a!important}.join-card{position:absolute;right:28px;bottom:28px;display:none;align-items:center;gap:16px;padding:16px;border-radius:16px;background:rgba(5,12,9,.92);min-width:470px}#table-code{width:180px;height:180px;background:#fff;border-radius:10px}.join-copy{max-width:260px;font-size:19px;font-weight:700;line-height:1.4}.join-slots{display:flex;gap:8px;margin-top:12px;font-size:14px}.join-slot{padding:7px 10px;border:1px solid #52645b;border-radius:999px}.join-slot.red{border-color:#ea5b63}.join-slot.blue{border-color:#4a9cff}.join-card.visible{display:flex}
.match-strip{position:absolute;left:24px;right:24px;top:86px;display:none;grid-template-columns:1fr auto 1fr;align-items:center;gap:20px;padding:12px 18px;border-radius:14px;background:rgba(5,12,9,.86);backdrop-filter:blur(8px)}.match-strip.visible{display:grid}.side{display:flex;gap:12px;align-items:center}.side.blue{justify-content:flex-end;text-align:right}.player-name{font-weight:800}.group{padding:5px 9px;border-radius:999px;background:#23332b;color:#c7d6ce}.side.red .player-name{color:#ff8a86}.side.blue .player-name{color:#81b5ff}#last-action{color:#8ff0bd;font-weight:750;white-space:nowrap}
</style></head><body><main><img id="route-frame" alt="实时摄像头画面"><svg id="route-overlay" preserveAspectRatio="xMidYMid meet"></svg><div id="video-status">实时画面连接中…</div><div class="hud"><span id="phase">等待现场状态</span><span id="player"></span></div><div id="match-strip" class="match-strip"><div class="side red"><span id="red-player" class="player-name">红方待认领</span><span id="red-group" class="group">未分组</span></div><span id="last-action">现场状态已同步</span><div class="side blue"><span id="blue-group" class="group">未分组</span><span id="blue-player" class="player-name">蓝方待认领</span></div></div><div id="join-card" class="join-card"><img id="table-code" src="/api/table-code.png" alt="球桌小程序码"><div class="join-copy"><div id="join-prompt">微信扫码加入本桌</div><div class="join-slots"><span id="red-slot" class="join-slot red">红方待认领</span><span id="blue-slot" class="join-slot blue">蓝方待认领</span></div></div></div></main>
<script>
const frame=document.getElementById('route-frame'),overlay=document.getElementById('route-overlay'),videoStatus=document.getElementById('video-status'),phase=document.getElementById('phase'),player=document.getElementById('player'),joinCard=document.getElementById('join-card'),joinPrompt=document.getElementById('join-prompt'),redSlot=document.getElementById('red-slot'),blueSlot=document.getElementById('blue-slot'),matchStrip=document.getElementById('match-strip'),redPlayer=document.getElementById('red-player'),bluePlayer=document.getElementById('blue-player'),redGroup=document.getElementById('red-group'),blueGroup=document.getElementById('blue-group'),lastAction=document.getElementById('last-action');let shown='',liveEpoch='',liveRetry=0,liveConnectedAt=0;
frame.onerror=()=>{if(shown.startsWith('live:')){shown='';liveRetry+=1;frame.classList.remove('camera-ready');videoStatus.textContent='实时画面正在重连…'}};
function groupLabel(value){return value==='stripe'?'花色':value==='solid'?'全色':'未分组'}
function routePolyline(points,color){if(!points||points.length<2)return;const line=document.createElementNS('http://www.w3.org/2000/svg','polyline');line.setAttribute('points',points.map(point=>point.join(',')).join(' '));line.setAttribute('fill','none');line.setAttribute('stroke',color);line.setAttribute('stroke-width','4');line.setAttribute('vector-effect','non-scaling-stroke');overlay.appendChild(line)}
function renderOverlay(value){overlay.innerHTML='';if(!value)return;overlay.setAttribute('viewBox',`0 0 ${value.width} ${value.height}`);value.routes.forEach((route,index)=>{const color=['#00ffff','#00c8ff','#ff7878'][Math.min(index,2)];routePolyline(route.cue_path,color);routePolyline(route.object_path,'#00c8ff')})}
async function sync(){try{const response=await fetch('/api/screen',{cache:'no-store'}),state=await response.json();if(!response.ok)throw Error(state.error||response.statusText);const live=state.frame.source_kind==='usb_live',mediaId=live?state.frame.media_frame_id:state.frame.frame_id,key=mediaId==null?'':live?`live:${state.frame.source_epoch_id}`:`replay:${mediaId}`,stale=live&&key===shown&&Date.now()-liveConnectedAt>=10000;if(key&&(key!==shown||stale)){if(live){if(state.frame.source_epoch_id!==liveEpoch){liveEpoch=state.frame.source_epoch_id;liveRetry=0}else if(stale){liveRetry+=1}liveConnectedAt=Date.now()}shown=key;frame.style.visibility='';frame.classList.add('camera-ready');frame.src=live?'/media/live.mjpg?epoch='+encodeURIComponent(state.frame.source_epoch_id)+(liveRetry?'&retry='+liveRetry:''):'/media/frame/'+mediaId+'.jpg'}else if(!key){frame.removeAttribute('src');frame.style.visibility='hidden';frame.classList.remove('camera-ready');shown=''}renderOverlay(state.frame.route_overlay);const modeLabel=state.frame.match_mode==='assistance'?'辅助模式':state.frame.match_mode==='observation'?'非辅助模式':'模式待确认';phase.textContent=state.display.phase+' · '+modeLabel+' · '+(state.frame.guidance_visible?'路线显示中':'路线已隐藏');phase.classList.remove('offline');player.textContent=state.match.active_role?`${state.match.active_role.toUpperCase()} 击球`:'尚未选择击球方';joinPrompt.textContent=state.display.join_prompt+(state.join.error?' · 入局服务暂时不可用，二维码仍可扫码':'');redSlot.textContent=state.display.red_slot;blueSlot.textContent=state.display.blue_slot;redPlayer.textContent=state.display.red_slot;bluePlayer.textContent=state.display.blue_slot;redGroup.textContent=groupLabel(state.match.red_group);blueGroup.textContent=groupLabel(state.match.blue_group);lastAction.textContent=state.display.last_action||`现场状态已同步 · 版本 ${state.match.revision}`;matchStrip.classList.toggle('visible',Boolean(state.match.red_user_id||state.match.blue_user_id));joinCard.classList.toggle('visible',state.join.qr_available&&(!state.match.red_user_id||!state.match.blue_user_id))}catch(error){if(!shown.startsWith('live:')){frame.removeAttribute('src');frame.style.visibility='hidden';frame.classList.remove('camera-ready');shown=''}overlay.innerHTML='';videoStatus.textContent='实时画面状态暂不可用';phase.textContent='现场状态服务连接中断';phase.classList.add('offline');player.textContent='';joinCard.classList.remove('visible')}}
setInterval(sync,200);sync();
</script></body></html>"""


_TABLET_PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>billiardsAI 管理员应急</title><style>
:root{color-scheme:dark;--bg:#0d1210;--panel:#18201c;--line:#34443b;--green:#35d58b;--red:#ea5b63;--blue:#4a9cff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#f5f7f6;font:16px system-ui,-apple-system,sans-serif}main{max-width:820px;margin:auto;padding:18px;display:grid;gap:14px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px}h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:0 0 12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.row{display:flex;gap:10px;flex-wrap:wrap}input,select,button{min-height:48px;border-radius:9px;border:1px solid #53665b;background:#243029;color:#fff;padding:10px 12px;font-size:16px}input,select{width:100%}button{flex:1;min-width:120px;font-weight:700;cursor:pointer}button:disabled{cursor:not-allowed;opacity:.45}button.primary{background:#137c55;border-color:var(--green)}button.red{border-color:var(--red)}button.blue{border-color:var(--blue)}#status{white-space:pre-wrap;color:#b8c8bf;line-height:1.55}.ended{color:#8ff0bd!important;font-weight:750}@media(max-width:620px){.grid{grid-template-columns:1fr}}
</style></head><body><main><section><h1>管理员应急入口</h1><div id="status">读取状态中…</div></section>
<section class="panel"><h2>开发回放控制</h2><div class="row"><button class="primary" data-command="start">开始</button><button data-command="pause">暂停</button><button data-command="reset">复位</button></div></section>
<section class="panel"><h2>管理员应急修正（仅首杆前）</h2><input id="emergency-reason" placeholder="必填：应急修正原因（至少 8 个字符）"><div class="grid" style="margin-top:10px"><div><input id="red-user" placeholder="红方临时参与者 ID"><button class="red" data-command="emergency_set_participant" data-role="red">应急设置红方</button></div><div><input id="blue-user" placeholder="蓝方临时参与者 ID"><button class="blue" data-command="emergency_set_participant" data-role="blue">应急设置蓝方</button></div></div><div class="row" style="margin-top:10px"><button class="red" data-command="emergency_set_breaker" data-role="red">应急设红方开球</button><button class="blue" data-command="emergency_set_breaker" data-role="blue">应急设蓝方开球</button></div><div class="row" style="margin-top:10px"><select id="red-group"><option value="stripe">红色打花色</option><option value="solid">红色打全色</option></select><button data-command="emergency_assign_groups">应急确认球组</button></div></section></main>
<script>
const token=new URLSearchParams(location.search).get('token')||'',status=document.getElementById('status');
async function api(url,options={}){const response=await fetch(url,options),data=await response.json();if(!response.ok)throw Error(data.error||response.statusText);return data}
function renderState(state){const ended=state.session.status==='ended';status.classList.toggle('ended',ended);status.textContent=ended?`本场已结束，事实报告已生成\n场次：${state.session.session_id}`:`阶段：${state.frame.phase}\n手机控制：${state.cloud_control.status}\n当前：${state.match.active_role||'未选择'}\n红色：${state.match.red_user_id||'未绑定'} / ${state.match.red_group||'未分组'}\n蓝色：${state.match.blue_user_id||'未绑定'} / ${state.match.blue_group||'未分组'}`;document.querySelectorAll('button[data-command]').forEach(button=>button.disabled=ended)}
async function refresh(){try{renderState(await api('/api/tablet'))}catch(error){status.textContent='连接失败：'+error.message}}
async function command(button){const kind=button.dataset.command,payload={command:kind};if(kind.startsWith('emergency_'))payload.reason=document.getElementById('emergency-reason').value;if(kind==='emergency_set_participant'){payload.role=button.dataset.role;payload.user_id=document.getElementById(button.dataset.role+'-user').value}else if(kind==='emergency_set_breaker'){payload.role=button.dataset.role}else if(kind==='emergency_assign_groups'){payload.red_group=document.getElementById('red-group').value}try{await api('/api/control',{method:'POST',headers:{'Content-Type':'application/json','X-Control-Token':token},body:JSON.stringify(payload)});await refresh()}catch(error){status.textContent='操作失败：'+error.message}}
document.querySelectorAll('button[data-command]').forEach(button=>button.onclick=()=>command(button));setInterval(refresh,500);refresh();
</script></body></html>"""
