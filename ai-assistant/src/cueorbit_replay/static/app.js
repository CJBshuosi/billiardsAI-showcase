"use strict";
const $ = id => document.getElementById(id);
let version = 0, sessionVersion = 0, controller = null, ready = true;
let selectedShot = null, nextOffset = null, clipsLoading = false;

const examples = ["What happened in this session?", "打8号球需要指袋吗？", "What happened in this shot?", "Can this clip prove a foul?"];
function node(tag, text, cls) {
  const n = document.createElement(tag);
  if (text !== undefined) n.textContent = text;
  if (cls) n.className = cls;
  return n;
}
function clearResponse() {
  $("response").replaceChildren(node("div", "Choose a question to explore this session. Missing evidence will stay visible.", "empty"));
  $("response").setAttribute("aria-busy", "false");
}
function cancel() {
  version++;
  if (controller) controller.abort();
  controller = null;
  $("ask").disabled = !ready;
  $("ask").textContent = "Ask ↗";
  clearResponse();
}
async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const zh=/[\u4e00-\u9fff]/.test($("question").value);
    if (response.status === 503) throw new Error(zh ? "模型或数据服务暂时不可用，请检查本地配置。" : "The model or data service is unavailable. Check the local configuration, then try again.");
    throw new Error(zh ? "请求未完成，请检查输入或稍后重试。" : "The requested record is unavailable. Please try again.");
  }
  return response.json();
}
function stopPlayer() {
  $("replay-player").querySelectorAll("video").forEach(video => { video.pause(); video.removeAttribute("src"); video.load(); });
  $("replay-player").replaceChildren();
}
function showSelection(shot) {
  selectedShot = shot;
  stopPlayer();
  $("clear-selection").hidden = !shot;
  $("selection-context").textContent = shot
    ? "Selected clip / 当前片段 · " + (shot.start_ms/1000).toFixed(1) + "–" + (shot.end_ms/1000).toFixed(1) + "s · Ask about this shot / 可直接问这一杆"
    : "Whole session / 整场 · Click a clip to ask about this shot.";
  $("clip-list").querySelectorAll("button[data-shot-id]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.shotId === shot?.shot_id)));
  if (!shot) return;
  if (!shot.media_id) { $("replay-player").append(node("p","No replay media for this record. You can still ask about its observations.","small")); return; }
  const video = document.createElement("video"), start = shot.start_ms/1000, end = shot.end_ms/1000;
  video.className = "media";
  video.controls = true;
  video.preload = "metadata";
  video.setAttribute("aria-label", "Synthetic illustrative clip for shot " + shot.sequence);
  video.src = "/api/media/" + encodeURIComponent(shot.media_id) + "#t=" + start + "," + end;
  video.addEventListener("loadedmetadata", () => { video.currentTime = start; video.play().catch(() => {}); }, {once:true});
  video.addEventListener("seeking", () => { if (video.currentTime < start || video.currentTime > end) video.currentTime = Math.max(start, Math.min(end, video.currentTime)); });
  video.addEventListener("play", () => { if (video.currentTime >= end) video.currentTime = start; });
  video.addEventListener("timeupdate", () => { if (video.currentTime >= end) { video.pause(); if (video.currentTime > end) video.currentTime = end; } });
  video.addEventListener("error", () => { if (selectedShot?.shot_id === shot.shot_id) $("replay-player").append(node("p","Replay unavailable. The selected record can still be queried.","small")); });
  $("replay-player").append(video);
}
function chooseShot(shot) { cancel(); showSelection(shot); }
async function loadClips(generation = sessionVersion, offset = 0) {
  if (clipsLoading) return;
  clipsLoading = true;
  $("clip-status").textContent = "Loading clips…";
  $("more-clips").disabled = true;
  $("retry-clips").hidden = true;
  try {
    const result = await api("/api/sessions/" + encodeURIComponent($("session").value) + "/shots?offset=" + offset);
    if (generation !== sessionVersion) return;
    if (!offset) $("clip-list").replaceChildren();
    for (const shot of result.data.shots) {
      const button = node("button",undefined,"clip-choice"); button.type = "button";
      button.dataset.shotId = shot.shot_id;
      button.setAttribute("aria-pressed",String(selectedShot?.shot_id === shot.shot_id));
      button.append(node("strong",(shot.start_ms/1000).toFixed(1) + "–" + (shot.end_ms/1000).toFixed(1) + "s"),node("span",shot.outcome.replaceAll("_"," ") + (shot.media_id ? " · Play ↗" : " · Record only")));
      button.addEventListener("click",()=>chooseShot(shot));
      $("clip-list").append(button);
    }
    nextOffset = result.data.next_offset;
    $("more-clips").hidden = nextOffset === null;
    const count = $("clip-list").children.length;
    $("clip-status").textContent = count ? count + " clips shown" + (result.status === "partial" ? " · Partial observations / 观测不完整" : "") : "No recorded clips in this session.";
  } catch(error) {
    if (generation !== sessionVersion) return;
    $("clip-status").textContent = "Clips unavailable. You can still ask about the session.";
    $("retry-clips").hidden = false;
  } finally {
    if (generation === sessionVersion) { clipsLoading = false; $("more-clips").disabled = false; }
  }
}
async function selectSession() {
  cancel();
  ready = true;
  $("ask").disabled = false;
  $("stats").replaceChildren();
  const current = ++sessionVersion, sessionId = $("session").value;
  showSelection(null);
  clipsLoading = false;
  nextOffset = null;
  $("clip-list").replaceChildren();
  $("clip-status").textContent = "Loading clips…";
  $("more-clips").hidden = true;
  $("retry-clips").hidden = true;
  if (!sessionId) {
    $("session-state").textContent = "No replay session selected. You can ask about rules.";
    $("clip-status").textContent = "Choose an optional session to browse its clips.";
    return;
  }
  $("session-state").textContent = "Loading the selected session…";
  try {
    const s = await api("/api/sessions/" + encodeURIComponent(sessionId));
    if (current !== sessionVersion) return;
    for (const [title, value] of [["Recorded shots",s.recorded_shot_count],["Pocket observed",s.pocket_observed_count],["No pocket observed",s.no_pocket_observed_count],["Unknown",s.unknown_count]]) {
      const item = node("div", undefined, "stat");
      item.append(node("strong",String(value)),node("span",title));
      $("stats").append(item);
    }
    $("session-state").textContent = s.recorded_shot_count === 0 ? "Empty session · no shot records." : s.unknown_count || s.limitations.includes("incomplete_observations") ? "Partial observations · revision " + s.revision : "Complete observations · revision " + s.revision;
    ready = true;
    $("ask").disabled = controller !== null;
    void loadClips(current);
  } catch(error) {
    if (current !== sessionVersion) return;
    $("session-state").textContent = error.message;
    $("clip-status").textContent = "Select an available session to browse clips.";
  }
}
function clip(shot, parent) {
  const row = node("div",undefined,"shot");
  row.append(node("span","Shot " + shot.sequence + " · " + shot.outcome.replaceAll("_"," ")));
  if (shot.media_id) {
    const button = node("button","Replay " + (shot.start_ms/1000).toFixed(1) + "s ↗");
    button.type = "button";
    button.addEventListener("click", () => {
      chooseShot(shot);
      $("replay-player").scrollIntoView({block:"nearest"});
    });
    row.append(button);
  }
  parent.append(row);
}
function render(response, areaId = "response") {
  const area = $(areaId);
  area.replaceChildren();
  const head = node("div",undefined,"answer-head");
  head.append(node("span",response.status.replaceAll("_"," ").toUpperCase(),"badge"),node("span",response.mode === "offline" ? "SCRIPTED FIXTURE" : "LIVE MODEL","small"));
  area.append(head,node("p",response.answer,"answer-text"));
  if (response.limitations.length) area.append(node("p","Limitations: " + response.limitations.join(" · ").replaceAll("_"," "),"limitations"));
  const facts = node("div",undefined,"facts");
  for (const card of response.replay_fact_cards || response.fact_cards || []) {
    const box = node("div",undefined,"fact"), d = card.data;
    box.append(node("h3",card.name.replaceAll("_"," ").toUpperCase() + " / " + card.status));
    if ("recorded_shot_count" in d) {
      box.append(node("p",d.recorded_shot_count + " recorded shots · " + d.pocket_observed_count + " pocket observed · " + d.no_pocket_observed_count + " no pocket observed · " + d.unknown_count + " unknown"));
      const inspect = node("button","Inspect recorded shots ↗"); inspect.type = "button";
      inspect.addEventListener("click",()=>{ $("question").value="Show recorded shots."; $("ask-form").requestSubmit(); });
      box.append(inspect);
    }
    if (d.shot_id) clip(d,box);
    for (const shot of d.shots || []) clip(shot,box);
    if (d.next_offset != null) box.append(node("p","More records are available after offset " + d.next_offset + ".","small"));
    for (const ref of card.evidence_refs) box.append(node("span",ref,"ref"));
    facts.append(box);
  }
  area.append(facts);
  for (const source of response.rule_citations || []) {
    const card = node("article",undefined,"fact rule-source");
    card.append(node("h3",source.rule_number + " / " + source.version),node("blockquote",source.quote));
    const link = node("a","Official PDF · pp. " + source.page_start + (source.page_end === source.page_start ? "" : "–" + source.page_end));
    if (source.source_url === "https://wpapool.com/wp-content/uploads/2025/08/250816-Rules-of-Heyball.pdf") {
      link.href = source.source_url + "#page=" + source.page_start; link.target="_blank"; link.rel="noopener noreferrer"; card.append(link);
    }
    area.append(card);
  }
  for (const note of response.source_annotations || []) {
    const box=node("aside",undefined,"fact source-warning");
    box.append(node("h3","EDITORIAL AUDIT / 非官方原文修订"),node("p",note.fragment),node("p",note.note),node("p",note.rule_number+" · PDF pp. "+note.page_start+" · "+note.registry_version,"small"));
    area.append(box);
  }
  if (response.adjudication_supported === false) area.append(node("p","Rule explanation and recorded observations only. Automatic adjudication is not supported.","small"));
  const details = node("details");
  details.append(node("summary","Execution record · " + (response.model_rounds ?? "unknown") + " model rounds · " + response.tool_trace.length + " tool calls"));
  details.append(node("pre",response.tool_trace.map(t => t.name + " " + JSON.stringify(t.arguments) + " → " + t.status + " / " + t.elapsed_ms.toFixed(1) + " ms" + (t.executed_query ? "\nExecuted query: " + t.executed_query : "") + (t.returned_chunk_ids ? "\nReturned sources: " + t.returned_chunk_ids.join(", ") : "")).join("\n") || "No tools called."));
  details.append(node("pre",(response.execution_events || []).map(e=>"Model round " + e.round + " · " + e.status + " / " + e.elapsed_ms.toFixed(1) + " ms" + (e.error_code ? " · " + e.error_code : "") + ((e.validation_errors || []).length ? " · " + e.validation_errors.map(d=>d.path + ":" + d.type).join(", ") : "")).join("\n")));
  details.append(node("p","One model round can request several tools. Tool calls and model rounds are counted separately; hidden reasoning is not displayed.","small"));
  const usage = response.usage;
  details.append(node("p","Tokens: input " + (usage.input_tokens ?? "unknown") + ", output " + (usage.output_tokens ?? "unknown") + ". Request " + response.request_id,"small"));
  area.append(details);
}
$("session").addEventListener("change",selectSession);
$("clear-selection").addEventListener("click",()=>chooseShot(null));
$("more-clips").addEventListener("click",()=>{ if (nextOffset !== null) void loadClips(sessionVersion,nextOffset); });
$("retry-clips").addEventListener("click",()=>void loadClips(sessionVersion,nextOffset ?? 0));
$("mode").addEventListener("change",()=>{
  cancel();
  $("mode-note").textContent = $("mode").value === "offline" ? "OFFLINE FIXTURE MODE · Scripted responses for development. This is not a real model evaluation." : "LIVE MODEL MODE · Sends your question, synthetic observations and retrieved rule excerpts to the configured model.";
});
$("examples").querySelectorAll("button").forEach((button,i) => button.addEventListener("click",()=>{ $("question").value = examples[i]; if (!$("ask").disabled) $("ask-form").requestSubmit(); }));
$("ask-form").addEventListener("submit",async event => {
  event.preventDefault();
  if ($("ask").disabled || !$("question").value.trim()) return;
  controller = new AbortController();
  const current = ++version, submittedQuestion = $("question").value;
  $("ask").disabled = true;
  $("ask").textContent = "Reviewing…";
  $("response").replaceChildren(node("div","Reading the record and checking evidence…","empty"));
  $("response").setAttribute("aria-busy","true");
  try {
    const response = await api("/api/assistant/ask",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({...($("session").value ? {session_id:$("session").value} : {}),question:$("question").value,mode:$("mode").value,...(selectedShot ? {selected_shot_id:selectedShot.shot_id} : {})}),signal:controller.signal});
    if (current !== version) return;
    render(response);
  } catch(error) {
    if (current !== version || error.name === "AbortError") return;
    $("response").replaceChildren(node("p",/[\u4e00-\u9fff]/.test(submittedQuestion) ? "请求未完成，模型或网络暂时不可用，请稍后重试。" : "The request could not be completed. Check the local service or connection.","error"));
  } finally {
    if (current === version) {
      $("ask").disabled = !ready;
      $("ask").textContent = "Ask ↗";
      $("response").setAttribute("aria-busy","false");
      controller = null;
    }
  }
});
async function boot() {
  $("ask").disabled = controller !== null;
  void loadCapabilities();
  try {
    const [sessions, health] = await Promise.all([api("/api/sessions"),api("/health")]);
    $("session").replaceChildren(Object.assign(node("option","No replay session / 不选场次"),{value:""}),...sessions.map(s=>{ const o=node("option",s.title); o.value=s.session_id; return o; }));
    $("session").disabled = false;
    if (health.live_ready) { const option=$("mode").options[1]; option.disabled=false; option.textContent="Live model"; }
    if (!$("session").value) {
      $("session-state").textContent = "No replay session selected. You can ask about rules.";
      $("clip-status").textContent = "Choose an optional session to browse its clips.";
    }
  } catch(error) { $("session-state").textContent=error.message; }
}
boot();
async function loadCapabilities() {
  try {
    const capability = await api("/api/capabilities"), rules=capability.rules;
    $("rules-edition").textContent = rules.rulesets[0]?.label || "Rule edition unavailable / 规则库未就绪";
    $("rules-capability").textContent = rules.ready ? "Rules and synthetic replay records are separate sources. Ask directly; evidence will show below." : "Rules unavailable · " + rules.reason + ". Replay records remain available.";
  } catch (_) { $("rules-capability").textContent = "Rule capabilities unavailable. Replay remains available."; }
}
