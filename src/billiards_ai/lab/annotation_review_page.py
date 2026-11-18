# ruff: noqa: E501
"""Dependency-free browser UI for the local annotation review queue."""

from __future__ import annotations


def render_annotation_page() -> str:
    return _PAGE


_PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>台球标注审查</title><style>
:root{color-scheme:dark;--bg:#11151a;--panel:#1b222a;--line:#33404c;--accent:#4ee1a0;--warn:#ffca5c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#edf2f7;font:15px system-ui,-apple-system,sans-serif}
header{height:58px;display:flex;align-items:center;gap:18px;padding:0 22px;background:#171d23;border-bottom:1px solid var(--line)}
header strong{font-size:18px}#progress{color:#9eb0bf}.status{margin-left:auto;color:var(--accent)}
main{height:calc(100vh - 58px);display:grid;grid-template-columns:minmax(0,1fr) 310px;gap:12px;padding:12px}
.stage{min-width:0;display:flex;align-items:center;justify-content:center;background:#080b0e;border:1px solid var(--line);border-radius:10px;overflow:hidden}
#review-canvas{max-width:100%;max-height:100%;cursor:crosshair}.side{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px;overflow:auto}
h2{font-size:16px;margin:0 0 8px}.meta{color:#9eb0bf;font-size:13px;line-height:1.6;word-break:break-all}.hint{font-size:13px;line-height:1.6;color:#bac7d1}
button{width:100%;padding:11px;margin:5px 0;border:1px solid #455462;border-radius:7px;background:#26313b;color:#fff;font-weight:650;cursor:pointer}
button.primary{background:#137c57;border-color:#24b77e}button.active{outline:2px solid var(--warn)}button:disabled{opacity:.45;cursor:not-allowed}
label{display:flex;align-items:flex-start;gap:9px;margin:13px 0;line-height:1.45}input{margin-top:3px}.row{display:grid;grid-template-columns:1fr 1fr;gap:8px}.danger{color:#ff9f9f}
@media(max-width:850px){main{grid-template-columns:1fr;grid-template-rows:minmax(420px,65vh) auto;height:auto}.side{min-height:320px}}
</style></head><body>
<header><strong>台球球体标注审查</strong><span id="progress">载入中…</span><span class="status" id="save-state">尚未修改</span></header>
<main><section class="stage"><canvas id="review-canvas"></canvas></section><aside class="side">
<h2 id="frame-title">帧信息</h2><div class="meta" id="frame-meta"></div>
<p class="hint">绿色圈是球。拖动圈可移动；选中后滚轮调整大小；Delete 删除。点击“新增漏检球”后，再点图中球心。</p>
<button id="add-ball">＋ 新增漏检球</button><button id="delete-ball" class="danger">删除选中球</button>
<label><input type="checkbox" id="complete-frame">这一帧所有可见球都已标完</label>
<label><input type="checkbox" id="difficult-frame">困难帧（暂不进入训练集）</label>
<div class="row"><button id="previous">← 上一张</button><button id="next">下一张 →</button></div>
<button class="primary" id="save-next">保存并下一张</button><button id="export-labels">导出已完成标签</button>
<div class="meta" id="message"></div></aside></main>
<script>
const $=id=>document.getElementById(id), canvas=$('review-canvas'), ctx=canvas.getContext('2d');
let session, index=0, draft, image=new Image(), selected=-1, dragging=false, addMode=false, dirty=false, saveTimer;
const api=async(url,opts={})=>{const r=await fetch(url,opts);const data=await r.json();if(!r.ok)throw Error(data.error||r.statusText);return data};
function frame(){return session.frames[index]} function scale(){return canvas.width/frame().width_px}
function setDirty(){dirty=true;$('save-state').textContent='未保存';clearTimeout(saveTimer);saveTimer=setTimeout(()=>save(false),700)}
function redraw(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0,canvas.width,canvas.height);const s=scale();draft.annotations.forEach((a,i)=>{ctx.beginPath();ctx.arc(a.center_x_px*s,a.center_y_px*s,a.radius_px*s,0,Math.PI*2);ctx.strokeStyle=i===selected?'#ffca5c':'#4ee1a0';ctx.lineWidth=i===selected?4:2;ctx.stroke();ctx.fillStyle='rgba(78,225,160,.12)';ctx.fill()})}
async function load(i){if(dirty)await save(false);index=Math.max(0,Math.min(session.frames.length-1,i));const f=frame();draft=await api('/api/frame/'+f.frame_id);selected=-1;addMode=false;$('add-ball').classList.remove('active');$('complete-frame').checked=draft.whole_frame_complete;$('difficult-frame').checked=draft.difficult;$('progress').textContent=`${index+1} / ${session.frames.length} · 已完成 ${session.completed_count}`;$('frame-title').textContent=f.video_id;const reason=f.queue_reason==='high_confidence'?'高置信候选':f.queue_reason==='low_confidence'?'低置信候选':'漏检检查';$('frame-meta').textContent=`帧 ${f.frame_index} · ${reason} · ${f.width_px}×${f.height_px}`;$('previous').disabled=index===0;$('next').disabled=index===session.frames.length-1;image.onload=()=>{const box=document.querySelector('.stage').getBoundingClientRect(), ratio=f.width_px/f.height_px;let w=box.width-4,h=w/ratio;if(h>box.height-4){h=box.height-4;w=h*ratio}canvas.width=Math.round(w);canvas.height=Math.round(h);redraw()};image.src='/media/'+f.frame_id+'.jpg?'+f.source_sha256;dirty=false;$('save-state').textContent='已保存'}
function point(e){const r=canvas.getBoundingClientRect(),s=frame().width_px/r.width;return{x:(e.clientX-r.left)*s,y:(e.clientY-r.top)*s}}
function hit(p){let best=-1,dist=Infinity;draft.annotations.forEach((a,i)=>{const d=Math.hypot(p.x-a.center_x_px,p.y-a.center_y_px);if(d<=Math.max(a.radius_px,12)&&d<dist){best=i;dist=d}});return best}
canvas.onmousedown=e=>{const p=point(e);if(addMode){const r=Math.max(5,Math.min(frame().width_px,frame().height_px)*.009);draft.annotations.push({annotation_id:'human-'+crypto.randomUUID(),center_x_px:p.x,center_y_px:p.y,radius_px:r,origin:'human_added',source_detection_id:null});selected=draft.annotations.length-1;addMode=false;$('add-ball').classList.remove('active');setDirty();redraw();return}selected=hit(p);dragging=selected>=0;redraw()};
canvas.onmousemove=e=>{if(!dragging||selected<0)return;const p=point(e),a=draft.annotations[selected];a.center_x_px=Math.max(a.radius_px,Math.min(frame().width_px-a.radius_px,p.x));a.center_y_px=Math.max(a.radius_px,Math.min(frame().height_px-a.radius_px,p.y));if(a.origin==='teacher')a.origin='teacher_modified';setDirty();redraw()};
window.onmouseup=()=>dragging=false;canvas.onwheel=e=>{if(selected<0)return;e.preventDefault();const a=draft.annotations[selected],max=Math.min(a.center_x_px,a.center_y_px,frame().width_px-a.center_x_px,frame().height_px-a.center_y_px);a.radius_px=Math.max(3,Math.min(max,a.radius_px+(e.deltaY<0?1:-1)));if(a.origin==='teacher')a.origin='teacher_modified';setDirty();redraw()};
function remove(){if(selected<0)return;draft.annotations.splice(selected,1);selected=-1;setDirty();redraw()}window.onkeydown=e=>{if(e.key==='Delete'||e.key==='Backspace')remove()};
$('add-ball').onclick=()=>{addMode=!addMode;$('add-ball').classList.toggle('active',addMode)};$('delete-ball').onclick=remove;
$('complete-frame').onchange=e=>{draft.whole_frame_complete=e.target.checked;if(e.target.checked){draft.difficult=false;$('difficult-frame').checked=false}setDirty()};
$('difficult-frame').onchange=e=>{draft.difficult=e.target.checked;if(e.target.checked){draft.whole_frame_complete=false;$('complete-frame').checked=false}setDirty()};
async function save(goNext){clearTimeout(saveTimer);$('save-state').textContent='保存中…';try{draft=await api('/api/frame/'+frame().frame_id,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({expected_revision:draft.revision,draft})});dirty=false;$('save-state').textContent='已保存';session.completed_count=session.frames.filter((_,i)=>i===index?draft.whole_frame_complete:false).length||session.completed_count;if(goNext&&index<session.frames.length-1)await load(index+1)}catch(e){$('save-state').textContent='保存失败';$('message').textContent=e.message;throw e}}
$('previous').onclick=()=>load(index-1);$('next').onclick=()=>load(index+1);$('save-next').onclick=()=>save(true);
$('export-labels').onclick=async()=>{try{if(dirty)await save(false);const x=await api('/api/export',{method:'POST'});$('message').textContent=`已导出 ${x.included_frame_count} 帧、${x.annotation_count} 个球：${x.path}`}catch(e){$('message').textContent=e.message}};
window.onresize=()=>{if(image.complete)image.onload()};(async()=>{try{session=await api('/api/session');await load(0)}catch(e){$('message').textContent='启动失败：'+e.message}})();
</script></body></html>"""
