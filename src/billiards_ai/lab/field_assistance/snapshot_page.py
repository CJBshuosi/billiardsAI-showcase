"""Presentation-only snapshot layers; no recommendations or shape generation."""

SNAPSHOT_PANEL = '<div id="snapshot-note" style="display:none" role="status"></div>'

SNAPSHOT_STYLE = """
main.s1-screen{background:radial-gradient(ellipse at 50% 40%,#232827,#111615)}
main.s1-screen #route-frame,main.s1-screen #route-overlay{top:64px;bottom:46px;
height:calc(100% - 110px);background:transparent}
main.s1-screen .hud{top:0;left:0;right:0;height:54px;padding:17px 30px;
border-radius:0;background:transparent;border-bottom:1px solid #ffffff14;
backdrop-filter:none;font-size:14px;align-items:center}
main.s1-screen #phase{color:#dae1dc;font-weight:500;letter-spacing:.04em}
main.s1-screen #player{font-size:12px;color:#a8b6ae}
main.s1-screen #match-strip{display:none}
main.s1-screen #snapshot-note{position:absolute;right:30px;bottom:14px;
max-width:40vw;font-size:12px;color:#a8bbb2}
main.s1-screen #recording-progress{left:30px;bottom:10px;max-width:55vw;
padding:0;background:transparent;font-size:12px;color:#95a89d;
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
"""

SNAPSHOT_SCRIPT = r"""<script>
window.renderSnapshotGuidance=function(value){
  const note=document.getElementById('snapshot-note');
  const svg=document.getElementById('route-overlay');
  const stage=document.getElementById('screen-stage');
  svg.setAttribute('data-s1-feedback-motion','');
  svg.setAttribute('data-s1-observed-frame','');
  if(stage)stage.classList.toggle('s1-screen',Boolean(value));
  if(!value){note.style.display='none';return;}
  svg.innerHTML='';
  const phase=document.getElementById('phase');
  const status={preparing:'智能建议准备中',moving:'等待停稳',analyzing:'分析中',
    failed:'建议暂不可用',
    ended:'本场已结束',waiting_for_context:'等待球组',abstained:'暂无可靠方案'};
  const feedback=value.phase==='reviewing'?value.feedback:null;
  const outcome={inside:'已进入建议区',outside:'未进入建议区',
    uncertain:'区域边界附近',unknown:'落点未可靠确认'};
  if(feedback)status.reviewing=outcome[feedback.region_result]||outcome.unknown;
  if(phase)phase.textContent='系统自动推荐'+(status[value.phase]?' · '+status[value.phase]:'');
  const rec=value.recommendation,draw=value.overlay;
  svg.setAttribute('data-s1-feedback-motion',feedback?.motion_frame_id??'');
  svg.setAttribute('data-s1-observed-frame',value.observed_frame_id??rec?.source_frame_id??'');
  if((!rec&&!feedback)||!draw||value.phase==='moving'){
    note.style.display='block';
    note.textContent=value.phase==='moving'?'球在运动，录像继续':status[value.phase]||'等待稳定画面';
    return;
  }
  note.style.display='none';
  svg.setAttribute('viewBox','0 0 '+draw.width+' '+draw.height);
  const scale=draw.width/640;
  const element=(tag,attributes,parent=svg)=>{
    const node=document.createElementNS('http://www.w3.org/2000/svg',tag);
    Object.entries(attributes).forEach(([key,value])=>node.setAttribute(key,value));
    parent.appendChild(node);return node;
  };
  if(draw.region){
    const outside=feedback&&feedback.region_result==='outside';
    const uncertain=feedback&&['uncertain','unknown'].includes(feedback.region_result);
    element('polygon',{points:draw.region.map(p=>p.join(',')).join(' '),
      fill:outside?'#e8bd7322':'#80d9ca22',
      stroke:outside?'#e8bd73':uncertain?'#bec8c1':'#a6e5da','stroke-width':scale,
      'stroke-linejoin':'round'});
  }
  if(feedback)return;
  const defs=element('defs',{});
  const gradient=element('radialGradient',{id:'s1-target-halo'},defs);
  [[0,0],[.3,0],[.48,.38],[.7,.15],[1,0]].forEach(([offset,opacity])=>{
    element('stop',{offset:offset*100+'%','stop-color':'#efc86e',
      'stop-opacity':opacity},gradient);
  });
  const target=draw.object_path[0];
  element('circle',{cx:target[0],cy:target[1],r:21*scale,
    fill:'url(#s1-target-halo)'});
  element('polyline',{points:draw.cue_path.map(p=>p.join(',')).join(' '),
    fill:'none',stroke:'#f5f6eb','stroke-width':1.15*scale,
    'stroke-dasharray':(5*scale)+' '+(5*scale),'stroke-linecap':'round',opacity:.9});
  element('polyline',{points:draw.object_path.map(p=>p.join(',')).join(' '),
    fill:'none',stroke:'#f5f6eb','stroke-width':1.15*scale,
    'stroke-dasharray':(5*scale)+' '+(5*scale),'stroke-linecap':'round',opacity:.9});
};
</script>"""
