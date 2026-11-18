"""Execute actual SVG advice rendering; ball and endpoint must be identifiable."""

import json
import subprocess

from billiards_ai.lab.field_assistance.snapshot_page import SNAPSHOT_SCRIPT


def test_approved_dashed_line_halo_and_unlabelled_region_clear_on_motion():
    script = SNAPSHOT_SCRIPT.removeprefix("<script>").removesuffix("</script>")
    harness = r"""
const vm=require('node:vm'),assert=require('node:assert/strict');
class Element{
  constructor(tag){this.tag=tag;this.children=[];this.attrs={};this.style={};this.textContent='';
    this.classList={toggle(){}};}
  setAttribute(k,v){this.attrs[k]=String(v);}
  appendChild(child){this.children.push(child);return child;}
  set innerHTML(value){this.children=[];}
}
const note=new Element('div'),svg=new Element('svg'),stage=new Element('main');
const phase=new Element('span');
const document={getElementById:id=>({'snapshot-note':note,'route-overlay':svg,
  'screen-stage':stage,'phase':phase})[id],
  createElementNS:(_,tag)=>new Element(tag)};
const context={document,window:{}};vm.createContext(context);
vm.runInContext(JSON.parse(process.argv[1]),context);
const draw={width:640,height:360,cue_path:[[320,270],[320,160]],
  object_path:[[320,140],[320,30]],region:[[300,150],[340,150],[340,190],[300,190]]};
const render=context.window.renderSnapshotGuidance;
render({phase:'preparing',recommendation:null});
assert.equal(phase.textContent,'系统自动推荐 · 智能建议准备中');
assert.equal(svg.children.length,0);
render({phase:'ready',recommendation:{region:{center:{x_mm:1270,y_mm:743}}},overlay:draw});
assert(!svg.children.some(x=>x.tag==='text'),'approved display has no callout labels');
const aim=svg.children.find(x=>x.tag==='polyline');
assert(aim,'cue-to-target line must exist');
assert.equal(aim.attrs.points,'320,270 320,160');
assert(aim.attrs['stroke-dasharray'],'aim line must be dashed');
const pot=svg.children.find(x=>x.tag==='polyline'&&x.attrs.points==='320,140 320,30');
assert(pot,'target-to-selected-pocket line must exist');
assert(pot.attrs['stroke-dasharray'],'pot route must be dashed');
assert(svg.children.some(x=>x.tag==='circle'&&Number(x.attrs.cx)===320&&Number(x.attrs.cy)===140),
  'halo must identify the actual recommended target');
assert(svg.children.find(x=>x.tag==='circle').attrs.fill.startsWith('url('));
assert.equal(phase.textContent,'系统自动推荐');
assert.equal(note.style.display,'none','no explanatory card while advice is visible');
const trapezoid=[[300,150],[340,150],[355,190],[285,190]];
svg.innerHTML='';render({phase:'ready',recommendation:{region:{}},
  overlay:{...draw,region:trapezoid}});
assert.equal(svg.children.find(x=>x.tag==='polygon').attrs.points,
  '300,150 340,150 355,190 285,190','render supplied polygon without inventing its shape');
svg.innerHTML='';render({recommendation:{region:null},overlay:{...draw,region:null}});
assert(!svg.children.some(x=>x.tag==='polygon'),'pot-only must not paint a region');
svg.innerHTML='';render({phase:'moving',recommendation:null});
assert.equal(svg.children.length,0);assert(note.textContent.includes('球在运动'));
for(const [result,label] of [['inside','已进入建议区'],['outside','未进入建议区'],
  ['uncertain','区域边界附近'],['unknown','落点未可靠确认']]){
  render({phase:'reviewing',recommendation:null,feedback:{region_result:result},
    overlay:{...draw,cue_path:[],object_path:[]}});
  assert.equal(svg.children.find(x=>x.tag==='polygon')?.attrs.points,
    draw.region.map(p=>p.join(',')).join(' '),
    'preserve previous region over actual settled pixels');
  assert(!svg.children.some(x=>x.tag==='polyline'||x.tag==='circle'),
    'previous route and next target must not appear during endpoint comparison');
  assert(phase.textContent.includes(label),'honest outcome must be visible without callout labels');
  assert(!svg.children.some(x=>x.tag==='text'));
}
render({phase:'moving',recommendation:null,feedback:{region_result:'inside'},overlay:draw});
assert.equal(svg.children.length,0,'motion revokes even a supplied prior feedback region');
render({phase:'ready',recommendation:{region:null},overlay:{...draw,region:null}});
assert.equal(svg.children.filter(x=>x.tag==='polyline').length,2);
assert(!svg.children.some(x=>x.tag==='polygon'),'next pot-only advice clears prior feedback');
render({phase:'ready',observed_frame_id:20,recommendation:{source_frame_id:20},overlay:draw});
assert.equal(svg.attrs['data-s1-observed-frame'],'20');
render(null);
assert.equal(svg.attrs['data-s1-observed-frame'],'');
assert.equal(svg.attrs['data-s1-feedback-motion'],'');
"""
    result = subprocess.run(
        ["node", "-e", harness, json.dumps(script)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
