# ruff: noqa: E501
"""Single-page local UI for group-first ball-identity review."""

from __future__ import annotations


def render_identity_review_page() -> str:
    group_labels = {
        "cue": "母球",
        "solid": "全色球",
        "eight": "黑八",
        "stripe": "花色球",
        "unknown": "无法判断",
    }
    buttons = "".join(
        f'<button class="label" type="button" data-group="{group}">{label}</button>'
        for group, label in group_labels.items()
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>球号身份复核</title>
<style>
:root{{--bg:#0b1220;--panel:#111b2e;--line:#30415e;--text:#eef5ff;--muted:#9eb0c9;--ok:#25c579;--accent:#38a8ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,sans-serif}}
header{{height:62px;padding:0 22px;display:flex;align-items:center;gap:18px;border-bottom:1px solid var(--line);background:#0e1727}}
header strong{{font-size:20px}}#progress{{color:var(--muted)}}main{{display:grid;grid-template-columns:minmax(420px,1fr) 390px;gap:16px;padding:16px;height:calc(100vh - 62px)}}
.stage{{display:flex;align-items:center;justify-content:center;border:1px solid var(--line);border-radius:12px;background:#05080e;overflow:hidden}}
#crop{{width:min(72vh,80%);max-width:720px;image-rendering:auto;border:1px solid #52647e;border-radius:10px}}
.panel{{border:1px solid var(--line);border-radius:12px;background:var(--panel);padding:18px;overflow:auto}}
.meta{{color:var(--muted);font-size:13px;word-break:break-all}}h2{{font-size:16px;margin:18px 0 8px}}.labels{{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}}
button,select,textarea,input{{font:inherit}}button{{border:1px solid #435776;border-radius:8px;background:#1b2a43;color:var(--text);padding:10px;cursor:pointer}}
button.selected{{outline:2px solid #ffca5c;background:#31496f}}button.primary{{width:100%;margin-top:16px;background:#117a50;border-color:var(--ok);font-weight:700}}
button.export{{width:100%;margin-top:9px;background:#17679e;border-color:var(--accent)}}.row{{display:grid;grid-template-columns:1fr 1fr;gap:9px}}
select,textarea,input{{width:100%;border:1px solid #435776;border-radius:8px;background:#0c1525;color:var(--text);padding:9px}}textarea{{min-height:76px}}.navigation{{margin-top:9px}}.jump{{display:grid;grid-template-columns:1fr 110px;gap:9px;margin-top:9px}}.hint{{margin-top:6px;color:var(--muted);font-size:12px}}#status{{margin-top:12px;color:#ffce68}}
@media(max-width:900px){{main{{grid-template-columns:1fr;height:auto}}.stage{{min-height:52vh}}}}
</style>
</head>
<body>
<header><strong>球号身份复核</strong><span id="progress">加载中…</span></header>
<main>
  <section class="stage"><img id="crop" alt="待复核球体裁图"></section>
  <aside class="panel">
    <div class="meta" id="crop-id"></div>
    <div class="meta" id="model-context"></div>
    <h2>球组（必选）</h2>
    <div class="labels">{buttons}</div>
    <h2>精确号码（可选）</h2>
    <select id="exact-identity"><option value="">看不清 / 不填写</option></select>
    <div class="hint">只有号码清楚、跨帧确认或已知球组映射时才填写；不要仅凭颜色猜号码。</div>
    <h2>精确号码证据</h2>
    <select id="exact-evidence"><option value="">无精确号码</option><option value="number_visible">画面中号码可见</option><option value="track_confirmed">跨帧已确认</option><option value="known_set_mapping">已知球组映射</option><option value="group_intrinsic">母球/黑八固有身份</option></select>
    <h2>画面质量</h2>
    <div class="row">
      <select id="visibility"><option value="visible">清晰可见</option><option value="partial">部分遮挡</option><option value="occluded">严重遮挡</option><option value="unknown">无法判断</option></select>
      <select id="orientation"><option value="number_visible">号码可见</option><option value="number_hidden">号码不可见</option><option value="ambiguous">方向/号码模糊</option></select>
    </div>
    <h2>备注（可选）</h2><textarea id="note" maxlength="500"></textarea>
    <button id="save" class="primary" type="button">保存并查看下一颗</button>
    <div class="row navigation"><button id="previous" type="button">上一颗</button><button id="next" type="button">下一颗</button></div>
    <div class="jump"><input id="jump" type="number" min="1" inputmode="numeric" aria-label="跳转序号"><button id="jump-button" type="button">跳转</button></div>
    <div class="hint">上一颗、下一颗和跳转只切换图片；修改标签后请点击“保存并查看下一颗”。</div>
    <button id="export" class="export" type="button">导出已完成球号标签</button>
    <div id="status"></div>
  </aside>
</main>
<script>
const $=id=>document.getElementById(id);let session,index=0,selectedGroup=null;
async function api(path,options={{}}){{const response=await fetch(path,options);const payload=await response.json();if(!response.ok)throw new Error(payload.error||'请求失败');return payload}}
function preloadNext(){{for(let step=1;step<=2;step++){{const id=session.crop_ids[index+step];if(id){{const image=new Image();image.src=`/media/${{id}}.png`}}}}}}
function configureExact(preferred=null){{const select=$('exact-identity');let values=[];if(selectedGroup==='cue')values=['cue'];else if(selectedGroup==='eight')values=['8'];else if(selectedGroup==='solid')values=['',...'1234567'];else if(selectedGroup==='stripe')values=['',...'9,10,11,12,13,14,15'.split(',')];else values=[''];select.innerHTML=values.map(value=>`<option value="${{value}}">${{value||'看不清 / 不填写'}}</option>`).join('');select.value=values.includes(preferred)?preferred:values[0];const intrinsic=selectedGroup==='cue'||selectedGroup==='eight';select.disabled=intrinsic||selectedGroup==='unknown'||!selectedGroup;if(intrinsic)$('exact-evidence').value='group_intrinsic';else if(!select.value)$('exact-evidence').value=''}}
function show(){{const id=session.crop_ids[index],prior=session.decisions[id]||session.suggestions?.[id],context=session.contexts?.[id];$('crop').src=`/media/${{id}}.png`;$('crop-id').textContent=id;$('model-context').textContent=context?`原标签：${{context.truth}} · 模型预测：${{context.predicted}} · 置信度：${{Number(context.confidence).toFixed(3)}} · 间隔：${{Number(context.margin).toFixed(3)}}`:'';selectedGroup=prior?.group||null;$('visibility').value=prior?.visibility||'visible';$('orientation').value=prior?.orientation||'number_visible';$('note').value=prior?.note||'';$('jump').value=index+1;document.querySelectorAll('[data-group]').forEach(button=>button.classList.toggle('selected',button.dataset.group===selectedGroup));configureExact(prior?.exact_identity||null);$('exact-evidence').value=prior?.exact_evidence||($('exact-identity').value?(selectedGroup==='cue'||selectedGroup==='eight'?'group_intrinsic':''):'');$('progress').textContent=`${{index+1}} / ${{session.total_count}} · 已复核 ${{session.reviewed_count}}`;preloadNext()}}
function goTo(target){{index=Math.max(0,Math.min(target,session.total_count-1));$('status').textContent='已切换；当前未保存的修改不会生效';show()}}
document.querySelectorAll('[data-group]').forEach(button=>button.onclick=()=>{{selectedGroup=button.dataset.group;document.querySelectorAll('[data-group]').forEach(item=>item.classList.toggle('selected',item===button));configureExact()}});
$('exact-identity').onchange=()=>{{if(!$('exact-identity').value)$('exact-evidence').value=''}};
$('save').onclick=async()=>{{if(!selectedGroup){{$('status').textContent='请先选择球组';return}}const exact=$('exact-identity').value||null,evidence=$('exact-evidence').value||null;if(exact&&!evidence){{$('status').textContent='填写精确号码时必须选择证据';return}}const id=session.crop_ids[index];session=await api('/api/decision',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{crop_id:id,group:selectedGroup,exact_identity:exact,exact_evidence:evidence,visibility:$('visibility').value,orientation:$('orientation').value,note:$('note').value}})}});index=Math.min(index+1,session.total_count-1);$('status').textContent='已保存';show()}};
$('previous').onclick=()=>goTo(index-1);$('next').onclick=()=>goTo(index+1);$('jump-button').onclick=()=>{{const requested=Number.parseInt($('jump').value,10);if(!Number.isInteger(requested)||requested<1||requested>session.total_count){{$('status').textContent=`请输入 1 到 ${{session.total_count}} 之间的序号`;return}}goTo(requested-1)}};
$('export').onclick=async()=>{{try{{const result=await api('/api/export',{{method:'POST'}});$('status').textContent=`导出成功：${{result.path}}`}}catch(error){{$('status').textContent=error.message}}}};
(async()=>{{session=await api('/api/session');const first=session.crop_ids.findIndex(id=>!session.decisions[id]);index=first<0?0:first;show()}})().catch(error=>$('status').textContent=error.message);
</script>
</body></html>"""
