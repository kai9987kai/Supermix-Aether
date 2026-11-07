'use strict';

(() => {
  const $ = id => document.getElementById(id);
  const colors = { teal: '#79ddcc', amber: '#e6b879', muted: '#94aab8', line: '#29424e', ink: '#e4edf1' };
  const actions = ['+X', '−X', '+Y', '−Y', '+Z', '−Z', 'HOLD'];
  const actionDescriptions = ['Positive X', 'Negative X', 'Positive Y', 'Negative Y', 'Positive Z', 'Negative Z', 'Hold position'];
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let frame = null;
  let busy = false;
  let connected = false;
  let animation = 0;
  let catalogSignature = '';
  let linkSignature = '';
  let eventSource;
  let initialConnectionError = false;

  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const number = (value, digits = 0) => finite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits }) : '—';
  const signed = (value, digits = 2) => finite(value) ? `${value > 0 ? '+' : ''}${number(value, digits)}` : '—';
  const percent = value => finite(value) ? `${number(value * 100, 1)}%` : '—';
  const actionName = value => Number.isInteger(value) && actions[value] ? actions[value] : '—';
  const clamp = value => Math.max(0, Math.min(1, finite(value) ? value : 0));
  const trainingBusy = () => ['running', 'evaluating'].includes(frame?.training?.status);
  const candidateAllowed = () => frame?.training?.candidate?.can_apply === true || frame?.training?.can_apply === true;
  const text = (id, value) => { $(id).textContent = value; };

  function setConnection(state) {
    connected = state === 'connected';
    const labels = { connected: 'Live connection', reconnecting: 'Reconnecting', disconnected: 'Offline', connecting: 'Connecting' };
    const indicator = $('connection');
    indicator.className = `connection ${state === 'connected' ? 'connected' : state === 'disconnected' ? 'disconnected' : ''}`;
    indicator.replaceChildren(Object.assign(document.createElement('i'), { ariaHidden: 'true' }), document.createTextNode(labels[state] || state));
    syncControls();
  }

  function showError(message) {
    text('error-message', message || 'The local laboratory could not complete this request.');
    $('error').hidden = false;
  }

  async function request(payload) {
    if (busy) return;
    busy = true;
    $('error').hidden = true;
    syncControls();
    try {
      const response = await fetch('/api/control', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
      });
      let result;
      try { result = await response.json(); } catch { throw new Error(`The laboratory returned an unreadable response (${response.status}).`); }
      if (!response.ok || result.error) throw new Error(result.error || `Request failed (${response.status}).`);
      acceptFrame(result);
    } catch (error) {
      showError(error.message);
      // Restore controls from the last accepted server state, including rejected settings.
      if (frame) renderSettings(true);
    } finally {
      busy = false;
      syncControls();
    }
  }

  function syncControls() {
    const unavailable = !frame || !connected || busy;
    const training = trainingBusy();
    document.querySelectorAll('[data-mutation]').forEach(element => { element.disabled = unavailable || training; });
    if (!frame) return;
    $('play').disabled = unavailable || training;
    $('step').disabled = unavailable || training || frame.playing === true;
    $('train').disabled = unavailable || training;
    $('apply').disabled = unavailable || frame.training?.status !== 'ready' || !candidateAllowed();
    $('rollback').disabled = unavailable || training || !(frame.can_rollback || frame.training?.can_rollback);
    $('lesion').disabled = unavailable || training || frame.model?.protocol === 'scalar' || !frame.topology?.links?.length;
    $('rewire').disabled = unavailable || training || !frame.plasticity?.can_rewire;
    $('play').innerHTML = frame.playing ? '<span aria-hidden="true">Ⅱ</span> Pause' : '<span aria-hidden="true">▶</span> Play';
    $('play').setAttribute('aria-label', frame.playing ? 'Pause live execution' : 'Play live execution');
  }

  function inputValue(id, value, force = false) {
    if (value === undefined || value === null) return;
    if (force || document.activeElement !== $(id)) $(id).value = String(value);
  }

  function renderSettings(force = false) {
    const settings = frame.settings || {};
    inputValue('seed', frame.seed, force);
    inputValue('speed', frame.speed || 1, force);
    inputValue('checkpoint', typeof frame.selected_checkpoint === 'object' ? frame.selected_checkpoint.id : frame.selected_checkpoint, force);
    inputValue('intervention', settings.intervention || 'none', force);
    inputValue('task-family', settings.task_family || 'straight', force);
    inputValue('delay', settings.delay || 3, force);
    ['loss', 'noise', 'latency', 'bandwidth'].forEach(id => {
      inputValue(id, settings[$(id).dataset.setting], force);
      renderSlider(id);
    });
  }

  function renderSlider(id) {
    const value = Number($(id).value);
    text(`${id}-value`, id === 'loss' || id === 'bandwidth' ? `${Math.round(value * 100)}%` : id === 'latency' ? `${value} ${value === 1 ? 'tick' : 'ticks'}` : number(value, 2));
  }

  function renderCatalog() {
    const entries = Array.isArray(frame.checkpoints) ? frame.checkpoints : [];
    const signature = JSON.stringify(entries);
    if (signature === catalogSignature) return;
    catalogSignature = signature;
    const select = $('checkpoint');
    select.replaceChildren();
    if (!entries.length) select.add(new Option('No checkpoints available', ''));
    const groups = new Map();
    entries.forEach(entry => {
      const protocol = entry.protocol || 'Available checkpoints';
      if (!groups.has(protocol)) {
        const group = document.createElement('optgroup');
        group.label = protocol;
        groups.set(protocol, group);
        select.append(group);
      }
      const option = new Option(entry.label || entry.id, entry.id);
      groups.get(protocol).append(option);
    });
  }

  function renderHeader() {
    const model = frame.model || {};
    const provenance = frame.provenance || {};
    const source = String(provenance.kind || provenance.source || provenance.type || 'recorded').toLowerCase();
    const sourceLabel = source.includes('structural') ? 'Structurally adapted model' : source.includes('adapt') || source.includes('candidate') || source.includes('session') ? 'Session-adapted weights' : source.includes('fresh') || source.includes('random') ? 'Fresh weights' : 'Recorded checkpoint';
    text('protocol', String(model.protocol || '—').toUpperCase());
    text('provenance', sourceLabel);
    text('checkpoint-note', provenance.label || `${sourceLabel} · ${frame.checkpoints?.length || 0} available`);
    text('fabric', `${number(model.neurons)} neurons`);
    text('fabric-detail', `${number(Array.isArray(model.regions) ? model.regions.length : model.regions)} regions / ${number(model.modes)} message modes`);
    const flops = model.step_flops;
    text('compute-flops', finite(flops) ? (flops >= 1e6 ? `${(flops / 1e6).toFixed(2)}M` : `${number(flops)}`) : '—');
    text('compute-detail', `estimated FLOPs / step · ${number(model.parameter_count)} params`);
    const timing = frame.execution_profile;
    text('execution-note', finite(timing?.last_paired_tick_us) ? `Paired execution: ${number(timing.last_paired_tick_us / 1000, 2)} ms last / ${number(timing.mean_paired_tick_us / 1000, 2)} ms mean · ${number(timing.samples)} ticks · includes telemetry` : 'Step to measure paired execution time.');
    text('task-note', `${frame.observation_boundary || ''}${frame.task?.family === 'waypoint' ? ` Active waypoint ${frame.candidate?.task?.stage || 1} of 2.` : ''}`);
    text('episode', finite(frame.episode) ? String(frame.episode).padStart(3, '0') : '—');
    text('seed-summary', `seed ${number(frame.seed)}`);
    text('tick', finite(frame.tick) ? String(frame.tick).padStart(2, '0') : '—');
    const waiting = finite(frame.tick) && frame.tick < (frame.settings?.delay || 3);
    text('episode-state', frame.done ? 'Episode complete' : waiting ? 'Cue / movement locked' : 'Navigation enabled');
    text('movement-state', frame.done ? 'EPISODE COMPLETE' : waiting ? 'MOVEMENT LOCKED' : 'MOVEMENT ENABLED');
    const runState = $('run-state');
    runState.className = `status-pill${frame.playing ? ' live' : trainingBusy() ? ' warning' : ''}`;
    runState.textContent = trainingBusy() ? 'Training paused world' : frame.playing ? '● Running' : frame.done ? 'Complete / paused' : 'Paused';
    text('completed', frame.stats?.episodes === undefined ? '—' : number(frame.stats.episodes));
    text('successes', `Observed successes ${frame.stats?.successes === undefined ? '—' : number(frame.stats.successes)}`);
    const comparison = frame.comparison || {};
    text('effective-diffs', number(comparison.effective_action_differences));
    text('waiting-diffs', `Waiting ticks: ${number(comparison.waiting_action_differences)} differences`);
    text('separation', number(comparison.position_distance, 2));
    text('return-delta', signed(comparison.return_delta, 2));
    const outcome = branch => branch?.success ? 'goal reached' : branch?.done ? 'horizon reached' : 'in progress';
    const results = $('branch-results');
    const clean = document.createElement('span');
    clean.textContent = `Clean control: ${outcome(frame.control)} · return ${signed(frame.control?.return)}`;
    results.replaceChildren(document.createTextNode(`Intervened: ${outcome(frame.candidate)} · return ${signed(frame.candidate?.return)}`), clean);
    $('world-empty').hidden = Boolean(frame.candidate?.position && frame.control?.position);
    $('world').setAttribute('aria-label', `Projected navigation world. Intervened position ${coordinate(frame.candidate?.position)}; clean position ${coordinate(frame.control?.position)}; spectator goal ${coordinate(frame.candidate?.goal)}. ${waiting ? 'Movement locked.' : ''}`);
  }

  function coordinate(position) { return Array.isArray(position) ? position.map(value => number(value)).join(', ') : 'unavailable'; }

  function renderActions() {
    const candidate = frame.candidate || {};
    const control = frame.control || {};
    text('action-current', actionName(candidate.action));
    actions.forEach((label, index) => {
      const value = candidate.probabilities?.[index];
      const cleanValue = control.probabilities?.[index];
      const row = $(`action-${index}`);
      row.classList.toggle('active', candidate.action === index);
      row.querySelector('.fill').style.width = `${clamp(value) * 100}%`;
      row.querySelector('.control-marker').style.left = `${clamp(cleanValue) * 100}%`;
      row.querySelector('.control-marker').hidden = !finite(cleanValue);
      row.querySelector('output').textContent = percent(value);
      row.setAttribute('aria-label', `${actionDescriptions[index]}: intervened ${percent(value)}, clean ${percent(cleanValue)}`);
    });
    const history = Array.isArray(frame.history) ? frame.history.slice(-24) : [];
    const timeline = $('timeline');
    timeline.replaceChildren();
    if (!history.length) {
      const empty = document.createElement('p');
      empty.className = 'empty-copy';
      empty.textContent = 'Step or play to record actions.';
      timeline.append(empty);
      return;
    }
    history.forEach(entry => {
      const cell = document.createElement('div');
      cell.className = `tick-cell${entry.movement_enabled === false ? ' locked' : ''}`;
      const chosen = document.createElement('span');
      chosen.textContent = actionName(entry.action);
      const clean = document.createElement('em');
      clean.textContent = actionName(entry.control_action);
      const tick = document.createElement('small');
      tick.textContent = `t${entry.tick}`;
      cell.append(chosen, clean, tick);
      cell.setAttribute('role', 'img');
      cell.setAttribute('aria-label', `Tick ${entry.tick}: intervened ${actionName(entry.action)}, clean ${actionName(entry.control_action)}${entry.movement_enabled === false ? ', movement locked' : ''}`);
      cell.title = cell.getAttribute('aria-label');
      timeline.append(cell);
    });
    timeline.scrollLeft = timeline.scrollWidth;
  }

  function renderModes() {
    const energies = Array.isArray(frame.candidate?.mode_energy) ? frame.candidate.mode_energy : [];
    const container = $('mode-bars');
    container.replaceChildren();
    const max = Math.max(...energies.filter(finite), 0);
    if (!energies.length) {
      const empty = document.createElement('p'); empty.className = 'empty-copy'; empty.textContent = 'No mode telemetry available.'; container.append(empty);
    }
    energies.forEach((energy, index) => {
      const row = document.createElement('div'); row.className = 'mode-row';
      const label = document.createElement('label');
      const name = document.createElement('span'); name.textContent = `M${String(index).padStart(2, '0')}`;
      const value = document.createElement('span'); value.textContent = finite(energy) ? energy.toExponential(2) : '—';
      label.append(name, value);
      const track = document.createElement('div'); track.className = 'track';
      const fill = document.createElement('div'); fill.className = 'fill'; fill.style.width = `${max > 0 ? clamp(energy / max) * 100 : 0}%`;
      track.append(fill); row.append(label, track); container.append(row);
    });
    const bytes = frame.candidate?.payload_bytes;
    text('payload', finite(bytes) ? bytes >= 1024 ? `${number(bytes / 1024, 2)} KiB` : `${number(bytes)} B` : '—');
    const delivered = frame.candidate?.delivered_payload_bytes;
    text('delivered-payload', `Delivered ${finite(delivered) ? delivered >= 1024 ? `${number(delivered / 1024, 2)} KiB` : `${number(delivered)} B` : '—'}`);
    text('link-count', `${number(frame.topology?.links?.length)} directed links`);
    const links = frame.topology?.links || [];
    const signature = JSON.stringify(links);
    if (signature !== linkSignature) {
      const selected = $('lesion-link').value;
      $('lesion-link').replaceChildren(...links.map(([source, target]) => new Option(`Region ${source} → ${target}`, `${source}:${target}`)));
      if (links.some(([source, target]) => `${source}:${target}` === selected)) $('lesion-link').value = selected;
      linkSignature = signature;
    }
    const p = frame.plasticity || {};
    text('plasticity-note', `${number(p.observed_ticks_available)} executed activity ticks available.${p.operation ? ` Last ${p.operation}: ${number(p.region_links_pruned)} removed, ${number(p.region_links_added)} added.${p.activity_source ? ` Used ${number(p.observed_ticks)} measured ticks.` : ''}` : ''}`);
  }

  function lossPoints() {
    return (Array.isArray(frame?.training?.curve) ? frame.training.curve : []).map((point, index) => typeof point === 'number' ? { update: index + 1, loss: point } : point).filter(point => point && finite(point.loss));
  }

  function renderTraining() {
    const training = frame.training || {};
    const status = training.status || 'idle';
    const names = { idle: 'Idle', running: 'Training', evaluating: 'Evaluating', ready: 'Candidate ready', cancelled: 'Cancelled', error: 'Error', applied: 'Candidate active' };
    text('training-status', names[status] || status);
    $('training-status').className = `status-pill${trainingBusy() ? ' live' : status === 'error' ? ' warning' : ''}`;
    const total = finite(training.total) ? training.total : 0;
    const progress = finite(training.progress) ? training.progress : 0;
    $('training-meter').max = Math.max(total, 1);
    $('training-meter').value = progress;
    text('training-total', `${number(progress)} / ${number(total)}`);
    text('training-progress', status === 'evaluating' ? 'Checking held-out + longer-delay worlds' : status === 'running' ? 'Executing cloned-model updates' : status === 'ready' ? 'Held-out evaluation complete' : status === 'error' ? 'Training stopped' : status === 'cancelled' ? 'Job cancelled' : 'Awaiting candidate');
    const points = lossPoints();
    text('loss-current', points.length ? number(points[points.length - 1].loss, 4) : '—');
    $('loss-empty').hidden = points.length > 0;
    text('loss-empty', trainingBusy() ? 'Waiting for the first measured update' : 'No training run yet');
    $('loss-chart').setAttribute('aria-label', points.length ? `Imitation loss from ${number(points[0].loss, 4)} to ${number(points[points.length - 1].loss, 4)} across ${points.length} measured updates.` : 'Training loss: no measured updates yet.');
    let note = 'Up to 100 updates. Live execution pauses during training.';
    if (training.error) note = String(training.error);
    else if (status === 'evaluating') note = 'Updates complete. Evaluating the candidate on fixed, disjoint worlds.';
    else if (status === 'running') note = 'Training a separate copy. The active model is unchanged.';
    else if (status === 'ready') note = candidateAllowed() ? 'Retention check passed. Use candidate to apply these weights and restart both worlds.' : 'Application blocked: retention failed or the active model / task no longer matches this candidate.';
    if (frame.can_rollback && status !== 'running' && status !== 'evaluating') note += ' Rollback is available.';
    text('candidate-note', note);
    const comparison = $('training-comparison'); comparison.replaceChildren();
    const before = training.comparison?.before;
    const after = training.comparison?.after;
    if (before && after) {
      [['Held-out success', 'query_success'], ['Longer-delay success', 'retention_success']].forEach(([label, key]) => {
        const item = document.createElement('div'); item.textContent = label;
        const values = document.createElement('strong'); values.textContent = `${percent(before[key])} → ${percent(after[key])}`;
        item.append(values); comparison.append(item);
      });
    }
    const matrix = $('transfer-matrix');
    matrix.replaceChildren();
    const rows = Object.entries(training.comparison?.task_matrix || {});
    if (rows.length) {
      const table = document.createElement('table');
      const caption = document.createElement('caption');
      caption.textContent = 'Before → after success on fixed paired seeds. Longer-delay retention margin: 5 percentage points.';
      table.append(caption);
      const header = table.createTHead().insertRow();
      ['Task family', 'Query', 'Longer delay', 'Retention gate'].forEach(value => { const th = document.createElement('th'); th.scope = 'col'; th.textContent = value; header.append(th); });
      const body = table.createTBody();
      rows.forEach(([family, values]) => {
        const row = body.insertRow();
        [family, `${percent(values.before?.query_success)} → ${percent(values.after?.query_success)}`, `${percent(values.before?.retention_success)} → ${percent(values.after?.retention_success)}`, values.after ? values.retention_passed ? 'Passed' : 'Blocked' : 'Evaluating'].forEach(value => { row.insertCell().textContent = value; });
      });
      matrix.append(table);
    } else {
      const empty = document.createElement('p'); empty.className = 'empty-copy'; empty.textContent = trainingBusy() ? 'Evaluating all three task families…' : 'Train a candidate to compare all three task families.'; matrix.append(empty);
    }
  }

  // CSS fixes the display size; only the backing store follows device pixel ratio.
  function surface(id) {
    const canvas = $(id);
    const bounds = canvas.getBoundingClientRect();
    const width = bounds.width, height = bounds.height;
    if (width < 1 || height < 1) return null;
    const dpr = window.devicePixelRatio || 1;
    const backingWidth = Math.round(width * dpr), backingHeight = Math.round(height * dpr);
    if (canvas.width !== backingWidth || canvas.height !== backingHeight) {
      canvas.width = backingWidth; canvas.height = backingHeight;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    return { ctx, width, height };
  }

  function line(ctx, from, to, color, width = 1, dash = []) {
    ctx.beginPath(); ctx.moveTo(from[0], from[1]); ctx.lineTo(to[0], to[1]); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.stroke(); ctx.setLineDash([]);
  }

  function drawWorld() {
    const canvas = surface('world'); if (!canvas) return;
    const { ctx, width, height } = canvas;
    const r = frame?.task?.radius || 3;
    const scale = Math.min(width / (r * 4.1), (height - 90) / (r * 2.94));
    const origin = [width * .5, height * .53];
    const project = p => [origin[0] + (p[0] - p[1]) * scale * .87, origin[1] + (p[0] + p[1]) * scale * .39 - p[2] * scale * .92];
    // Fixed navigation boundaries, independent of outcomes or movement.
    for (let value = -r; value <= r; value++) {
      line(ctx, project([-r, value, -r]), project([r, value, -r]), value === 0 ? '#2b4652' : '#1e3541');
      line(ctx, project([value, -r, -r]), project([value, r, -r]), value === 0 ? '#2b4652' : '#1e3541');
    }
    const corners = [[-r,-r,-r],[r,-r,-r],[r,r,-r],[-r,r,-r],[-r,-r,r],[r,-r,r],[r,r,r],[-r,r,r]];
    [[0,1],[1,2],[2,3],[3,0],[0,4],[1,5],[2,6],[3,7],[4,5],[5,6],[6,7],[7,4]].forEach(([a,b]) => line(ctx,project(corners[a]),project(corners[b]),'#29424e88',1,[3,5]));
    ctx.font = '9px Consolas, monospace'; ctx.fillStyle = '#7494a2';
    [['+X',[r+.6,0,-r]],['+Y',[0,r+.6,-r]],['+Z',[-r,-r,r+.5]]].forEach(([name,p]) => { const xy=project(p); ctx.fillText(name,xy[0]-5,xy[1]); });
    if (!frame) return;
    const waypoints = frame.candidate?.task?.waypoints || [];
    if (waypoints.length > 1) {
      line(ctx, project(waypoints[0]), project(waypoints[1]), '#7b9aa7', 1, [4, 5]);
      waypoints.forEach((waypoint, index) => {
        const point = project(waypoint);
        ctx.beginPath(); ctx.arc(point[0], point[1], 16, 0, Math.PI * 2); ctx.strokeStyle = '#7b9aa7'; ctx.stroke();
        ctx.fillStyle = '#b6cbd2'; ctx.fillText(`W${index + 1}`, point[0] - 6, point[1] - 21);
      });
    }
    const goal = frame.candidate?.goal;
    if (Array.isArray(goal)) {
      const base = project([goal[0],goal[1],-r]), p = project(goal);
      line(ctx,base,p,'#91b5c04d',1,[3,4]);
      ctx.strokeStyle='#b6cbd2';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(p[0],p[1]-11);ctx.lineTo(p[0]+11,p[1]);ctx.lineTo(p[0],p[1]+11);ctx.lineTo(p[0]-11,p[1]);ctx.closePath();ctx.stroke();
      line(ctx,[p[0]-16,p[1]],[p[0]-5,p[1]],'#789aa7');line(ctx,[p[0]+5,p[1]],[p[0]+16,p[1]],'#789aa7');
      ctx.font='8px Consolas, monospace';ctx.fillStyle='#b6cbd2';ctx.fillText('GOAL',p[0]+17,p[1]-8);
    }
    function drawBranch(branch, color, dashed) {
      if (!branch || !Array.isArray(branch.position)) return;
      const trail = Array.isArray(branch.trajectory) ? branch.trajectory : [];
      if (trail.length) {
        ctx.beginPath();trail.forEach((point,index)=>{const p=project(point);index===0?ctx.moveTo(p[0],p[1]):ctx.lineTo(p[0],p[1]);});
        ctx.strokeStyle=color;ctx.lineWidth=dashed?3:2.5;ctx.setLineDash(dashed?[5,5]:[]);ctx.lineJoin='round';ctx.stroke();ctx.setLineDash([]);
        trail.forEach(point=>{const p=project(point);ctx.beginPath();ctx.arc(p[0],p[1],2,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();});
      }
      const p=project(branch.position),base=project([branch.position[0],branch.position[1],-r]);
      line(ctx,base,p,`${color}40`,1,[2,4]);
      ctx.beginPath();ctx.ellipse(base[0],base[1],9,4,0,0,Math.PI*2);ctx.strokeStyle=`${color}55`;ctx.stroke();
      ctx.beginPath();ctx.arc(p[0],p[1],dashed?9:5.5,0,Math.PI*2);ctx.fillStyle=dashed?'#0f2029':color;ctx.fill();ctx.lineWidth=1.7;ctx.strokeStyle=color;ctx.stroke();
      if(branch.success){ctx.beginPath();ctx.arc(p[0],p[1],15,0,Math.PI*2);ctx.strokeStyle=`${color}88`;ctx.lineWidth=1;ctx.stroke();}
    }
    drawBranch(frame.control,colors.amber,true);
    drawBranch(frame.candidate,colors.teal,false);
    const originPoint = project([0,0,0]);
    ctx.fillStyle='#78919f';ctx.font='8px Consolas, monospace';ctx.fillText('ORIGIN',originPoint[0]-18,originPoint[1]+24);
  }

  function drawRegions(time = 0) {
    const canvas = surface('regions'); if (!canvas) return;
    const { ctx, width, height } = canvas;
    const raw = Array.isArray(frame?.candidate?.regions) ? frame.candidate.regions : [];
    const count = Array.isArray(frame?.model?.regions) ? frame.model.regions.length : (frame?.model?.regions || 16);
    const nodes = Array.from({length:count},(_,index)=>raw.find(node=>node.id===index)||{id:index,activity:0});
    const cx=width/2,cy=height/2,r=Math.min(width*.37,height*.37);
    const positions=nodes.map((node,index)=>{
      const outer = index<10;
      const angle=(outer?index/Math.min(count,10):(index-10)/Math.max(count-10,1))*Math.PI*2-Math.PI/2;
      const radius=outer?r:r*.51;
      return [cx+Math.cos(angle)*radius,cy+Math.sin(angle)*radius];
    });
    ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);ctx.strokeStyle='#213844';ctx.setLineDash([2,5]);ctx.stroke();ctx.setLineDash([]);
    const links=Array.isArray(frame?.candidate?.links)?frame.candidate.links:[];
    const measured=new Map(links.map(link=>[`${link.source}:${link.target}`,link.energy]));
    const topology=Array.isArray(frame?.topology?.links)?frame.topology.links:[];
    const maximum=Math.max(...links.map(link=>finite(link.energy)?link.energy:0),0);
    topology.forEach(([source,target],index)=>{
      const from=positions[source],to=positions[target];if(!from||!to)return;
      const energy=measured.get(`${source}:${target}`)||0;
      const activity=maximum>0?clamp(energy/maximum):0;
      line(ctx,from,to,activity>0?`rgba(121,221,204,${.07+activity*.43})`:'#263f4b55',activity>.65?1.3:.7);
      if(frame?.playing && !reducedMotion.matches && energy>0 && activity>.3){
        const fraction=((time/1600)+index*.127)%1;
        ctx.beginPath();ctx.arc(from[0]+(to[0]-from[0])*fraction,from[1]+(to[1]-from[1])*fraction,1.2,0,Math.PI*2);ctx.fillStyle=`rgba(161,240,222,${activity*.75})`;ctx.fill();
      }
    });
    const maxActivity=Math.max(...nodes.map(node=>finite(node.activity)?node.activity:0),0);
    nodes.forEach((node,index)=>{
      const [x,y]=positions[index];const activity=maxActivity>0?clamp(node.activity/maxActivity):0;
      if(activity>0){ctx.beginPath();ctx.arc(x,y,9+activity*4,0,Math.PI*2);ctx.fillStyle=`rgba(121,221,204,${activity*.07})`;ctx.fill();}
      ctx.beginPath();ctx.arc(x,y,5+activity*3,0,Math.PI*2);ctx.fillStyle=activity>0?`rgba(121,221,204,${.2+activity*.7})`:'#203943';ctx.fill();ctx.strokeStyle=index===0||index===15?colors.amber:'#68b9af';ctx.lineWidth=index===0||index===15?1.4:1;ctx.stroke();
      ctx.font='8px Consolas, monospace';ctx.textAlign='center';ctx.fillStyle='#b5c9d1';ctx.fillText(String(node.id).padStart(2,'0'),x,y-12);
    });
    ctx.textAlign='left';ctx.font='8px Consolas, monospace';ctx.fillStyle=colors.muted;ctx.fillText('00 SENSORY',12,17);ctx.textAlign='right';ctx.fillText(`${String(Math.max(count-1,0)).padStart(2,'0')} ACTION`,width-12,17);ctx.textAlign='left';
    $('regions').setAttribute('aria-label', `${count} neural regions. ${topology.length} directed links. ${raw.length ? raw.map(node=>`Region ${node.id} activity ${finite(node.activity)?node.activity.toPrecision(3):'unavailable'}`).join('; ') : 'Awaiting measured regional activity.'}`);
  }

  function drawLoss() {
    const canvas=surface('loss-chart');if(!canvas)return;
    const {ctx,width,height}=canvas;
    const left=28,right=12,top=12,bottom=22;
    for(let index=0;index<3;index++){const y=top+(height-top-bottom)*index/2;line(ctx,[left,y],[width-right,y],'#29404a');}
    const points=lossPoints();if(!points.length)return;
    const values=points.map(point=>point.loss),low=Math.min(...values),high=Math.max(...values);
    const padding=(high-low)*.15||Math.max(high*.1,.01),minimum=Math.max(0,low-padding),maximum=high+padding;
    const xy=points.map((point,index)=>[left+(points.length===1?0:index/(points.length-1))*(width-left-right),top+(1-(point.loss-minimum)/(maximum-minimum))*(height-top-bottom)]);
    const gradient=ctx.createLinearGradient(0,top,0,height-bottom);gradient.addColorStop(0,'#79ddcc30');gradient.addColorStop(1,'#79ddcc00');
    ctx.beginPath();ctx.moveTo(xy[0][0],height-bottom);xy.forEach(p=>ctx.lineTo(p[0],p[1]));ctx.lineTo(xy[xy.length-1][0],height-bottom);ctx.closePath();ctx.fillStyle=gradient;ctx.fill();
    ctx.beginPath();xy.forEach((p,index)=>index===0?ctx.moveTo(p[0],p[1]):ctx.lineTo(p[0],p[1]));ctx.strokeStyle=colors.teal;ctx.lineWidth=1.6;ctx.lineJoin='round';ctx.stroke();
    const last=xy[xy.length-1];ctx.beginPath();ctx.arc(last[0],last[1],3,0,Math.PI*2);ctx.fillStyle=colors.teal;ctx.fill();
    ctx.fillStyle=colors.muted;ctx.font='8px Consolas, monospace';ctx.fillText(high.toFixed(2),0,top+3);ctx.fillText(low.toFixed(2),0,height-bottom+3);ctx.fillText(`Update ${points[0].update ?? 1}`,left,height-5);ctx.textAlign='right';ctx.fillText(`Update ${points[points.length-1].update ?? points.length}`,width-right,height-5);ctx.textAlign='left';
  }

  function redraw() { drawWorld(); drawRegions(performance.now()); drawLoss(); }

  function syncAnimation() {
    if (animation) { cancelAnimationFrame(animation); animation=0; }
    if (!frame?.playing || reducedMotion.matches || !connected || document.hidden) return;
    const animate=time=>{
      if (!frame?.playing || reducedMotion.matches || !connected || document.hidden) { animation=0; return; }
      drawRegions(time);
      animation=requestAnimationFrame(animate);
    };
    animation=requestAnimationFrame(animate);
  }

  function acceptFrame(next) {
    if (!next || typeof next !== 'object' || !next.model || !next.settings) throw new Error('The laboratory returned an incomplete state.');
    frame=next;
    if (next.runtime_error) showError(String(next.runtime_error));
    renderCatalog();renderSettings();renderHeader();renderActions();renderModes();renderTraining();syncControls();redraw();syncAnimation();
  }

  function configure(settings) { return request({action:'configure',settings}); }
  function integerInput(id, minimum, maximum) {
    const value=Number($(id).value);
    if (!Number.isInteger(value)||value<minimum||value>maximum) { showError(`Enter a whole number from ${minimum} to ${maximum}.`); return null; }
    return value;
  }

  $('dismiss-error').addEventListener('click',()=>{$('error').hidden=true;});
  $('play').addEventListener('click',()=>request({action:frame?.playing?'pause':'play'}));
  $('step').addEventListener('click',()=>request({action:'step'}));
  $('reset').addEventListener('click',()=>request({action:'reset'}));
  $('speed').addEventListener('change',()=>request({action:'speed',value:Number($('speed').value)}));
  $('checkpoint').addEventListener('change',()=>request({action:'checkpoint',id:$('checkpoint').value}));
  $('intervention').addEventListener('change',()=>configure({intervention:$('intervention').value}));
  $('task-family').addEventListener('change',()=>configure({task_family:$('task-family').value}));
  $('delay').addEventListener('change',()=>configure({delay:Number($('delay').value)}));
  $('seed').addEventListener('change',()=>{const seed=integerInput('seed',0,4294967295);if(seed!==null)request({action:'reset',seed});});
  $('new-seed').addEventListener('click',()=>{const values=new Uint32Array(1);crypto.getRandomValues(values);request({action:'reset',seed:values[0]});});
  ['loss','noise','latency','bandwidth'].forEach(id=>{
    $(id).addEventListener('input',()=>renderSlider(id));
    $(id).addEventListener('change',()=>configure({[$(id).dataset.setting]:Number($(id).value)}));
  });
  $('train').addEventListener('click',()=>{const updates=integerInput('updates',1,100);if(updates!==null)request({action:'train',updates});});
  $('apply').addEventListener('click',()=>request({action:'apply'}));
  $('rollback').addEventListener('click',()=>request({action:'rollback'}));
  $('lesion').addEventListener('click',()=>request({action:'lesion',region_pair:$('lesion-link').value.split(':').map(Number)}));
  $('rewire').addEventListener('click',()=>request({action:'rewire'}));
  reducedMotion.addEventListener('change',()=>{redraw();syncAnimation();});
  document.addEventListener('visibilitychange',syncAnimation);
  window.addEventListener('resize',redraw);
  const observer=new ResizeObserver(redraw);
  ['world','regions','loss-chart'].forEach(id=>observer.observe($(id)));
  for(let index=1;index<=12;index++)$('delay').add(new Option(`${index} ${index===1?'tick':'ticks'}`,String(index)));
  actions.forEach((label,index)=>{
    const row=document.createElement('div');row.className='action-row';row.id=`action-${index}`;
    const name=document.createElement('span');name.textContent=label;
    const track=document.createElement('div');track.className='track';
    const fill=document.createElement('div');fill.className='fill';
    const marker=document.createElement('div');marker.className='control-marker';marker.hidden=true;
    track.append(fill,marker);const value=document.createElement('output');value.textContent='—';
    row.append(name,track,value);$('action-bars').append(row);
  });

  async function start() {
    setConnection('connecting');
    try {
      const response=await fetch('/api/state',{cache:'no-store'});
      const initial=await response.json();
      if(!response.ok)throw new Error(initial.error||`State unavailable (${response.status}).`);
      acceptFrame(initial);
    } catch(error) {
      initialConnectionError = true;
      showError(error.message);text('world-empty','Local state unavailable. The connection will retry automatically.');
    }
    // EventSource reconnects itself and supplies a fresh server snapshot on recovery.
    eventSource=new EventSource('/api/events');
    eventSource.onopen=()=>setConnection('connected');
    eventSource.addEventListener('frame',event=>{
      try {
        const next = JSON.parse(event.data);
        if (initialConnectionError) { $('error').hidden=true; initialConnectionError=false; }
        acceptFrame(next);setConnection('connected');
      }
      catch(error) { showError(`Could not read the live frame: ${error.message}`); }
    });
    eventSource.onerror=()=>{setConnection(eventSource.readyState===EventSource.CLOSED?'disconnected':'reconnecting');syncAnimation();};
  }
  redraw();
  start();
})();
