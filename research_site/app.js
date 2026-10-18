import {readSelection, selectionHash, resolveEvent, squaresFromFen, resolveRun, traceTurns, turnSteps, navigateTurn, recordedOutcome, turnEvidence, recordedMoves, lastMovesByColor, benchmarkRun, resultCounts} from './model.js?v=28';

const $ = id => document.getElementById(id);
const cache = new Map();
async function json(path) {
  if (!cache.has(path)) cache.set(path, fetch(path, {cache:'no-store'}).then(r => {
    if (!r.ok) throw new Error(`Cannot load ${path} (${r.status})`);
    return r.json();
  }).catch(error => {cache.delete(path); throw error;}));
  return cache.get(path);
}
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = String(text);
  return node;
}
const fmt = value => value == null ? 'Unavailable' : Number(value).toLocaleString('en-US');
const pretty = value => typeof value === 'string' ? (() => {
  try {return JSON.stringify(JSON.parse(value), null, 2);} catch {return value;}
})() : JSON.stringify(value, null, 2);
function option(value, text) {
  const node = el('option', '', text);
  node.value = value;
  return node;
}
const modeLabels = {harness:'LLM + Tools', search:'Tools Only'};
const outcome = game => game.completed === true ? ({'1-0':'White win','0-1':'White loss','1/2-1/2':'Draw'}[game.outcome] ?? 'Complete · outcome unavailable') : game.completed === false ? 'Incomplete' : 'Completion unavailable';

let catalog, run, game, gameSummary, selected, base, payload;
const benchmarks = new Map();
let traceMode = 'react', flipped = false, routeGeneration = 0, traceGeneration = 0;
let playing = false, playTimer;
let recordedPlies = [], boardState = null, boardAnimationGeneration = 0, boardAnimations = [];
const expandedTurns = new Set();
const tags = {context:'CTX',llm:'LLM',tool_call:'REQ',tool_result:'TOOL',verification:'VER',replan:'PLAN',commit:'MOVE',opponent:'OPP',error:'ERR',move:'PLY',state:'FEN'};
function status(text) {$('status-message').textContent = text;}
function currentLink(event) {return selectionHash(run.id, game.id, event.id);}
function setPlaying(value) {
  playing = value;
  clearTimeout(playTimer);
  $('play-turns').textContent = playing ? 'Ⅱ Pause' : '▶ Play';
  $('play-turns').setAttribute('aria-label', playing ? 'Pause playback' : 'Play turns');
  $('play-turns').setAttribute('aria-pressed', String(playing));
  if (playing) navigate(1, true);
}
function scheduleNextTurn() {
  clearTimeout(playTimer);
  playTimer = setTimeout(() => navigate(1, true), 2000);
}
function selectEvent(event, automatic = false) {
  if (!automatic) setPlaying(false);
  if (event) location.hash = currentLink(event);
}
function navigate(offset, automatic = false) {
  selectEvent(navigateTurn(game.events, selected, offset), automatic);
}
function filteredEvents() {
  const type = $('event-filter').value;
  return game.events.filter(e => !type || e.type === type);
}
function renderTimeline(center = false) {
  const all = filteredEvents();
  const matching = new Set(all.map(e => e.id));
  const nodes = [];
  const turns = traceTurns(game.events);
  for (const turn of turns) {
    const events = turn.events.filter(e => matching.has(e.id));
    if (!events.length) continue;
    const active = turn.turn === selected?.turn;
    const section = el('section', `timeline-turn${active ? ' selected-turn' : ''}`);
    section.dataset.turn = String(turn.turn);
    const button = el('button', 'turn-select');
    button.setAttribute('aria-current', String(active));
    button.append(el('strong', '', `Turn ${turn.turn ?? 'unknown'}`));
    const moves = turn.events.filter(e => ['commit','opponent','move'].includes(e.type) && e.move).map(e => e.move);
    if (moves.length) button.append(el('span', 'turn-moves', moves.join(' · ')));
    button.append(el('small', '', active ? selected.title : `${run.configuration.policy === 'search' ? modeLabels.search : `${turn.calls.length} calls`} · ${turn.events.length} events`));
    button.addEventListener('click', () => selectEvent(turn === turns.at(-1) && events.length === turn.events.length ? turn.events.at(-1) : events[0]));
    const details = el('details', 'turn-events');
    const key = `${base}:${turn.turn}`;
    details.open = expandedTurns.has(key);
    details.append(el('summary', '', `${events.length} ${events.length === 1 ? 'event' : 'events'}${events.length !== turn.events.length ? ' matching' : ''}`));
    let loaded = false;
    const expand = () => {
      if (details.open) expandedTurns.add(key); else expandedTurns.delete(key);
      if (!details.open || loaded) return;
      loaded = true;
      for (const event of events) {
        const child = el('button', `timeline-event${event.id === selected?.id ? ' selected' : ''}`);
        child.dataset.type = event.type;child.dataset.eventId = event.id;
        child.setAttribute('aria-current', String(event.id === selected?.id));
        child.append(el('span', 'event-dot', tags[event.type] ?? 'LOG'));
        const text = el('span', 'event-text');
        text.append(el('strong', '', event.title), el('small', '', `${event.source.file}:${event.source.location}`));
        child.append(text);child.addEventListener('click', () => selectEvent(event));details.append(child);
      }
    };
    details.addEventListener('toggle', expand);expand();
    section.append(button, details);nodes.push(section);
  }
  if (!all.length) nodes.push(el('p', 'muted', 'No events match this filter.'));
  $('timeline').replaceChildren(...nodes);
  $('filter-status').textContent = `${all.length} matching events · ${nodes.filter(n => n.dataset.turn != null).length} turns`;
  if (center) ($('timeline').querySelector('.timeline-event.selected') ?? $('timeline').querySelector('.turn-select[aria-current=true]'))?.scrollIntoView({block:'nearest'});
}

const glyphs = {p:'♟',n:'♞',b:'♝',r:'♜',q:'♛',k:'♚'};
const names = {p:'pawn',n:'knight',b:'bishop',r:'rook',q:'queen',k:'king'};
function drawPosition(fen, index) {
  let squares = squaresFromFen(fen);
  if (flipped) squares = squares.toReversed();
  const recent = lastMovesByColor(recordedPlies,index);
  const moves = Object.values(recent).filter(Boolean);
  const moveSquares = moves.flatMap(move => [move.from,move.to]);
  $('board').replaceChildren(...squares.map(({square, piece, dark}, i) => {
    const node = el('div', `square${dark ? ' dark' : ''}${moveSquares.includes(square) ? ' highlight' : ''}`);
    node.dataset.square = square;
    if (moves.some(move => square === move.to)) node.classList.add('move-destination');
    for (const [color,move] of Object.entries(recent)) {
      if (move && [move.from,move.to].includes(square)) node.classList.add(`${color}-move`);
    }
    if (piece) node.append(el('span', `piece${piece === piece.toUpperCase() ? ' white-piece' : ''}`, glyphs[piece.toLowerCase()]));
    if (i % 8 === 0) node.append(el('span', 'coordinate rank-coordinate', square[1]));
    if (i >= 56) node.append(el('span', 'coordinate file-coordinate', square[0]));
    return node;
  }));
  if (!squares.length) $('board').append(el('p', 'board-empty', 'Position unavailable in the selected evidence.'));
  return squares;
}
async function animateMoves(steps, generation) {
  for (const step of steps) {
    if (generation !== boardAnimationGeneration) return;
    drawPosition(step.afterFen, step.index);
    boardAnimations = [];
    for (const movement of [step,step.rook].filter(Boolean)) {
      const from = $('board').querySelector(`[data-square="${movement.from}"]`).getBoundingClientRect();
      const target = $('board').querySelector(`[data-square="${movement.to}"]`);
      const to = target.getBoundingClientRect();
      const piece = target.querySelector('.piece');
      if (piece) {
        piece.classList.add('moving-piece');
        boardAnimations.push(piece.animate([{transform:`translate(${from.left-to.left}px,${from.top-to.top}px)`},{transform:'translate(0,0)'}],{duration:260,easing:'ease-in-out'}));
      }
    }
    if (step.captured) {
      const ghost = el('span', `piece captured-ghost${step.captured === step.captured.toUpperCase() ? ' white-piece' : ''}`, glyphs[step.captured.toLowerCase()]);
      $('board').querySelector(`[data-square="${step.captureSquare}"]`).append(ghost);
      boardAnimations.push(ghost.animate([{opacity:1},{opacity:0,transform:'scale(.6)'}],{duration:240}));
    }
    await Promise.allSettled(boardAnimations.map(animation => animation.finished));
    if (generation !== boardAnimationGeneration) return;
    $('board').querySelectorAll('.captured-ghost').forEach(node => node.remove());
    $('board').querySelectorAll('.moving-piece').forEach(node => node.classList.remove('moving-piece'));
  }
}
function renderCaptured(id, pieces, color) {
  const node = $(id);
  const ordered = pieces.toSorted((a,b) => 'qrbnp'.indexOf(a.toLowerCase())-'qrbnp'.indexOf(b.toLowerCase()));
  node.replaceChildren(el('small', '', 'Captured'), ...ordered.map(piece => {
    const glyph = el('span', `captured-piece${piece === piece.toUpperCase() ? ' captured-white' : ''}`, glyphs[piece.toLowerCase()]);
    glyph.setAttribute('aria-hidden','true');return glyph;
  }));
  if (!pieces.length) node.append(el('small', '', '—'));
  node.setAttribute('aria-label', `${color} captured: ${pieces.length ? pieces.map(p=>names[p.toLowerCase()]).join(', ') : 'none in recorded play'}`);
  node.title = 'Captures in recorded moves up to this position';
}
function renderBoard() {
  const index = game.events.indexOf(selected);
  const history = recordedPlies.filter(move => move.index <= index);
  const generation = ++boardAnimationGeneration;
  boardAnimations.forEach(animation => animation.cancel());
  $('board').setAttribute('aria-busy','false');
  const squares = drawPosition(selected?.fen, index);
  const previous = boardState;
  boardState = {base,index,fen:selected?.fen,flipped};
  const steps = previous?.base === base && previous.flipped === flipped && previous.index < index
    ? recordedPlies.filter(move => move.index > previous.index && move.index <= index) : [];
  // shortcut: animate up to four recorded forward plies; longer jumps and rewinds fade.
  if (!matchMedia('(prefers-reduced-motion: reduce)').matches && previous?.base === base && previous.flipped === flipped && previous.fen !== selected?.fen) {
    if (steps.length && steps.length <= 4 && steps[0].beforeFen === previous.fen && steps.at(-1).afterFen === selected?.fen && steps.every((step,i) => !i || step.beforeFen === steps[i-1].afterFen)) {
      $('board').setAttribute('aria-busy','true');
      animateMoves(steps,generation).catch(() => {if (generation === boardAnimationGeneration) drawPosition(selected?.fen,index);})
        .finally(() => {if (generation === boardAnimationGeneration) $('board').setAttribute('aria-busy','false');});
    } else boardAnimations = [$('board').animate([{opacity:.55},{opacity:1}],{duration:160})];
  }
  renderCaptured('captured-by-white',history.filter(move=>move.piece === move.piece.toUpperCase() && move.captured).map(move=>move.captured),'White');
  renderCaptured('captured-by-black',history.filter(move=>move.piece !== move.piece.toUpperCase() && move.captured).map(move=>move.captured),'Black');
  const description = squares.filter(s => s.piece).map(s => `${s.piece === s.piece.toUpperCase() ? 'white' : 'black'} ${names[s.piece.toLowerCase()]} on ${s.square}`).join(', ');
  $('board').setAttribute('aria-label', selected?.fen ? `${selected.board_phase}. ${description}` : 'Chess position unavailable');
  $('fen-value').textContent = selected?.fen ?? 'Unavailable';
  const {verification, search, chosenMove} = turnEvidence(game.events, selected?.turn);
  const budgetEvent = run.configuration.policy === 'search' ? search : verification;
  const phase = run.configuration.policy === 'search' ? 'search' : 'verification';
  const b = budgetEvent?.budget ?? {};
  $('budget-phase').textContent = `AFTER ${phase.toUpperCase()}`;
  const used = b.nodes_used ?? (b.max_nodes != null && b.remaining_nodes != null ? b.max_nodes - b.remaining_nodes : null);
  $('budget-value').replaceChildren(document.createTextNode(used == null ? 'Unavailable' : fmt(used)), el('small', '', b.max_nodes != null ? ` / ${fmt(b.max_nodes)} nodes` : ''));
  $('budget-fill').style.width = b.max_nodes && used != null ? `${Math.max(0, Math.min(100, used/b.max_nodes*100))}%` : '0%';
  const remaining = b.remaining_nodes ?? (b.max_nodes != null && used != null ? b.max_nodes-used : null);
  $('budget-note').textContent = budgetEvent ? `After ${phase} · ${remaining != null ? `${fmt(remaining)} nodes remaining` : 'Remaining nodes unavailable'} · ${budgetEvent.source.file}:${budgetEvent.source.location}` : `No ${phase} feedback recorded for this turn.`;
  const candidates = search?.candidates ?? [];
  $('candidates').replaceChildren(...candidates.map(c => {
    const chosen = c.move === chosenMove;
    const row = el('div', `candidate-row${chosen ? ' chosen-candidate' : ''}`);
    row.dataset.move = c.move;
    const move = el('strong', '', c.move);
    if (chosen) move.append(el('span', 'chosen-badge', 'Chosen'));
    row.append(move, el('span', '', `score ${c.primary_score ?? '?'} · ${c.bound ?? 'bound unavailable'}`));
    const details = el('details');details.append(el('summary', '', 'Principal variation · hypothetical'), el('code', '', (c.pv ?? []).join(' → ') || 'Unavailable'));
    row.append(details);return row;
  }));
  if (search) {
    const a = search.analysis ?? {};
    $('candidates').prepend(el('p', '', `Search feedback · completed depth ${a.completed_depth ?? 'unavailable'} · scope ${Array.isArray(a.candidate_scope) ? a.candidate_scope.join(', ') : a.candidate_scope ?? a.score_scope ?? 'unavailable'} · ${search.source.file}:${search.source.location}`));
  } else $('candidates').append(el('p', '', 'No search candidate table recorded for this turn.'));
  if (chosenMove && !candidates.some(c => c.move === chosenMove)) {
    $('candidates').prepend(el('p', 'chosen-note', `Chosen move: ${chosenMove} · ${search ? 'not listed in this search shortlist' : 'search evidence unavailable'}`));
  }
  const turns = traceTurns(game.events);
  const i = turns.findIndex(turn => turn.turn === selected?.turn);
  const finalTurn = i >= 0 && i === turns.length - 1;
  $('event-position').textContent = i < 0 ? 'No turns' : `Turn ${selected.turn ?? i+1} / ${turns.length}`;
  $('first-event').disabled = $('previous-event').disabled = i <= 0;
  $('last-event').disabled = $('next-event').disabled = i < 0 || finalTurn;
  $('play-turns').disabled = i < 0 || finalTurn;
  if (finalTurn || i < 0) setPlaying(false);
  const result = recordedOutcome(gameSummary, finalTurn);
  $('board-result').hidden = result == null;
  $('board-result').dataset.result = result ?? '';
  $('board-result').replaceChildren(...(result ? [el('strong', '', result), el('span', '', `${modeLabels[run.configuration.policy] ?? 'LLM agent'} · White · ${gameSummary.outcome}`)] : []));

}

function block(title, value) {
  const details = el('details', 'trace-block');
  details.append(el('summary', '', title));
  let loaded = false;
  function expand() {
    if (details.open && !loaded) {details.append(el('pre', '', pretty(value) ?? 'Unavailable'));loaded = true;}
  }
  details.addEventListener('toggle', expand);expand();return details;
}
async function sourceEvidence(event, resourceBase) {
  const data = await json(`${resourceBase}/${event.detail}`);
  const record = data.raw_record ? await json(`${resourceBase}/${data.raw_record}`) : null;
  return {data, record};
}

function compactTurnRows(results, turn, current) {
  const records = results.map(result => result.status === 'fulfilled' ? result.value.record : null);
  const steps = turnSteps(records);
  const nodes = [];
  for (const [i, result] of results.entries()) {
    const call = turn.calls[i];
    const active = call.events.some(event => event.id === current.id);
    const callSteps = steps.filter(step => step.callIndex === i);
    if (result.status === 'rejected' || !records[i]) callSteps.push({type:'error', label:'Source evidence unavailable', value:result.status === 'rejected' ? result.reason.message : result.value.data.evidence});
    for (const step of callSteps) {
      let label = step.label;
      if (step.type === 'thinking') label = 'Thinking · provider recorded';
      if (step.type === 'rationale') label = 'Reasoning';
      if (step.type === 'tool_call') label += ` #${i+1}`;
      if (step.type === 'status') label = `Status · ${step.value.status ?? 'not recorded'}`;
      const row = block(label, step.value);
      row.className = `trace-block turn-trace-row row-${step.type}${active ? ' selected-call' : ''}`;
      row.dataset.callLocation = call.location;
      const summary = row.querySelector('summary');
      summary.title = `Call ${i+1} · ${call.file}:${call.location}`;
      const bytes = new TextEncoder().encode(pretty(step.value) ?? '').length;
      summary.append(el('span', 'row-size', `${bytes < 1024 ? `${bytes} B` : `${(bytes/1024).toFixed(1)} KB`}`));
      const source = el('div', 'trace-row-source');
      const jump = el('button', 'row-source-link', `Call ${i+1} · ${call.file}:${call.location} ↗`);
      jump.title = 'Select this call in the timeline';
      jump.addEventListener('click', () => selectEvent(call.events.find(event => event.type === step.type) ?? call.events.find(event => event.type === 'llm') ?? call.events[0]));
      source.append(jump);row.append(source);nodes.push(row);
    }
  }
  const raw = results.map((result, i) => ({source:{file:turn.calls[i].file,location:turn.calls[i].location}, ...(result.status === 'fulfilled' ? {record:result.value.record,evidence:result.value.record ? undefined : result.value.data.evidence} : {error:result.reason.message})}));
  const transcript = block('Raw transcript · this turn', raw);
  transcript.classList.add('turn-trace-row', 'row-raw');
  transcript.querySelector('summary').append(el('span', 'row-size', `${(new TextEncoder().encode(pretty(raw)).length/1024).toFixed(1)} KB`));
  nodes.push(transcript);
  return nodes;
}

async function renderTurn(container, turn, current, resourceBase, generation) {
  container.replaceChildren(el('p', 'muted', 'Loading recorded calls…'));
  if (run.configuration.policy === 'search') {
    const results = await Promise.allSettled(turn.events.filter(event => ['tool_result','verification'].includes(event.type)).map(event => sourceEvidence(event, resourceBase)));
    if (generation !== traceGeneration) return;
    container.replaceChildren(...results.map(result => result.status === 'fulfilled' ? block('Search result', result.value.data.evidence) : el('p','muted',result.reason.message)));
    if (!results.length) container.append(el('p','muted','No search feedback recorded for this turn. See Event JSON for the recorded decision.'));
    return;
  }
  const results = await Promise.allSettled(turn.calls.map(call => sourceEvidence(call.events[0], resourceBase)));
  if (generation !== traceGeneration) return;
  const nodes = compactTurnRows(results, turn, current);
  if (!turn.calls.length) nodes.push(el('p', 'muted', 'Individual model calls were not recorded for this turn. Available evidence remains accessible in Event JSON.'));
  container.replaceChildren(...nodes);
}

async function renderTrace() {
  const generation = ++traceGeneration;
  const current = selected;
  const resourceBase = base;
  const mode = traceMode;
  $('trace-type').textContent = mode === 'react' && current ? 'Turn' : current?.type.replaceAll('_', ' ') ?? 'Unavailable';
  $('trace-title').textContent = mode === 'react' && current ? `Turn ${current.turn ?? 'unknown'}` : current?.title ?? 'No events available';
  const time = current?.timestamp != null ? `provider-created ${new Date(current.timestamp*1000).toISOString()} (recorded)` : 'event timestamp unavailable';
  $('trace-source').textContent = current ? `${current.source.file} · location ${current.source.location}\nTurn ${current.turn ?? 'unavailable'} · ${time}` : '';
  document.querySelectorAll('[data-trace-mode]').forEach(tab => {
    tab.setAttribute('aria-selected', String(tab.dataset.traceMode === mode));
    tab.tabIndex = tab.dataset.traceMode === mode ? 0 : -1;
  });
  $('trace-body').setAttribute('aria-labelledby', document.querySelector(`[data-trace-mode="${mode}"]`).id);
  $('trace-body').replaceChildren(el('p', 'muted', 'Loading recorded evidence…'));
  payload = null;
  if (!current) { $('trace-body').replaceChildren(el('p', 'muted', 'No event evidence available.'));return; }
  try {
    const data = await json(`${resourceBase}/${current.detail}`);
    if (generation !== traceGeneration) return;
    payload = data;
    if (mode === 'json') {
      const record = data.raw_record ? await json(`${resourceBase}/${data.raw_record}`) : null;
      if (generation !== traceGeneration) return;
      $('trace-body').replaceChildren(el('p', 'trace-caption', 'SELECTED EVENT + SANITIZED SOURCE EVIDENCE · A source call can span several timeline events. Additional fields are preserved.'), el('pre', '', pretty({event:current,evidence:data.evidence,source_record:record})));
      $('trace-body').scrollTop = 0;
      return;
    }
    const turns = traceTurns(game.events);
    const turn = turns.find(t => t.turn === current.turn);
    const body = el('div', 'react-turn-flow');
    $('trace-body').replaceChildren(el('p', 'trace-caption', run.configuration.policy === 'search' ? 'TOOLS ONLY · Recorded candidate scores, budget and decision.' : `${turn?.calls.length ?? 0} RECORDED CALLS · Unchanged prompts reused. Updated prompts shown in sequence. Selected call marked at left.`), body);
    if (turn) await renderTurn(body, turn, current, resourceBase, generation);
    if (generation === traceGeneration) $('trace-body').scrollTop = 0;
  } catch (error) {
    if (generation === traceGeneration) $('trace-body').replaceChildren(el('p', 'muted', error.message));
  }
}

async function route() {
  const generation = ++routeGeneration;
  const selection = readSelection(location.hash);
  const runEntry = resolveRun(catalog, selection.run);
  if (!runEntry) throw new Error('No imported runs. See research_site/README.md for the offline import command.');
  const loadedRun = await json(`data/${runEntry.path}`);
  const summary = loadedRun.games.find(g => g.id === selection.game) ?? loadedRun.games[0];
  if (!summary) throw new Error('This run has no games.');
  const loadedGame = await json(`data/${summary.path}`);
  if (generation !== routeGeneration) return;
  const changed = game !== loadedGame;
  if (changed) setPlaying(false);
  run = loadedRun;game = loadedGame;gameSummary = summary;
  if (changed) {recordedPlies = recordedMoves(game.events);document.querySelector('.board-scroll').scrollTop = 0;}
  base = `data/${summary.path.replace(/\/index\.json$/, '')}`;
  selected = resolveEvent(game.events, selection.event);
  status(game.warnings.join('; '));
  if (selection.event && selected?.id !== selection.event) status('Requested event unavailable; showing the first recorded event.');
  if (selected) history.replaceState(null, '', currentLink(selected));
  $('mode-select').value = run.policy ?? '';
  $('elo-select').value = String(run.elo ?? summary.opponent?.elo ?? 1500);
  renderComparison();
  $('game-select').replaceChildren(...run.games.map(g => option(g.id, `${g.id.replace('_', ' ')} · ${outcome(g)}`)));
  $('game-select').value = game.id;
  $('event-count').textContent = `${game.events.length} EVENTS`;
  $('opponent-name').textContent = summary.opponent?.name ?? 'Opponent unavailable';
  $('opponent-setting').textContent = `Elo setting ${summary.opponent?.elo ?? 'unavailable'} · ${summary.opponent?.move_time ?? '?'} s / move`;
  $('model-name').textContent = run.configuration.policy === 'search' ? modeLabels.search : summary.model ?? 'Model unavailable';
  $('strategy-description').textContent = run.configuration.policy === 'search' ? 'Search tool selects the move automatically' : 'LLM directs analysis and chooses the move';
  if (changed) {
    $('event-filter').replaceChildren(option('', 'All events'), ...[...new Set(game.events.map(e => e.type))].map(t => option(t, t.replaceAll('_', ' '))));
  }
  renderTimeline(true);renderBoard();await renderTrace();
  if (playing && generation === routeGeneration) scheduleNextTurn();
}

function renderComparison() {
  const nodes = [];
  for (const policy of ['harness','search']) {
    const entry = benchmarkRun(catalog, policy, $('elo-select').value);
    const data = benchmarks.get(entry?.id);
    if (!data) continue;
    const counts = resultCounts(data.games);
    const button = el('button', 'comparison-choice', `${modeLabels[policy]} · ${counts.wins}W / ${counts.draws}D / ${counts.losses + counts.incomplete}L${counts.unknown ? ` / ${counts.unknown} unknown` : ''}`);
    button.setAttribute('aria-pressed', String(entry.id === run.id));
    button.title = `${data.games.length} games · White results · ${entry.elo} Elo`;
    button.addEventListener('click', () => switchBenchmark(policy, entry.elo));
    nodes.push(button);
  }
  $('comparison').replaceChildren(...nodes);
}
function switchBenchmark(policy, elo) {
  const entry = benchmarkRun(catalog, policy, elo);
  if (!entry) return;
  setPlaying(false);
  location.hash = selectionHash(entry.id, game?.id ?? '', '');
}
function setupViewer() {
  const helpTrigger = $('budget-help-trigger'), help = $('budget-help');
  let helpHideTimer;
  function hideBudgetHelp() {
    clearTimeout(helpHideTimer);
    if (help.matches(':popover-open')) help.hidePopover();
  }
  function showBudgetHelp() {
    clearTimeout(helpHideTimer);
    if (!help.matches(':popover-open')) help.showPopover();
    const anchor = helpTrigger.getBoundingClientRect(), rect = help.getBoundingClientRect();
    help.style.left = `${Math.max(12, Math.min(innerWidth-rect.width-12, anchor.left))}px`;
    const top = anchor.bottom+8+rect.height <= innerHeight-12 ? anchor.bottom+8 : anchor.top-rect.height-8;
    help.style.top = `${Math.max(12, Math.min(innerHeight-rect.height-12, top))}px`;
  }
  function leaveBudgetHelp() {
    clearTimeout(helpHideTimer);
    if (document.activeElement !== helpTrigger) helpHideTimer = setTimeout(hideBudgetHelp,150);
  }
  helpTrigger.addEventListener('mouseenter',showBudgetHelp);
  helpTrigger.addEventListener('focus',showBudgetHelp);
  helpTrigger.addEventListener('click',showBudgetHelp);
  helpTrigger.addEventListener('mouseleave',leaveBudgetHelp);
  helpTrigger.addEventListener('blur',leaveBudgetHelp);
  help.addEventListener('mouseenter',() => clearTimeout(helpHideTimer));
  help.addEventListener('mouseleave',leaveBudgetHelp);
  help.addEventListener('toggle',() => helpTrigger.setAttribute('aria-expanded',String(help.matches(':popover-open'))));
  document.querySelector('.board-scroll').addEventListener('scroll',hideBudgetHelp);
  window.addEventListener('resize',hideBudgetHelp);
  $('mode-select').disabled = $('elo-select').disabled = !benchmarks.size;
  $('mode-select').addEventListener('change', () => switchBenchmark($('mode-select').value, $('elo-select').value));
  $('elo-select').addEventListener('change', () => switchBenchmark($('mode-select').value || 'harness', $('elo-select').value));
  $('game-select').addEventListener('change', () => {setPlaying(false);location.hash = selectionHash(run.id, $('game-select').value, '');});
  $('event-filter').addEventListener('change', () => renderTimeline());
  $('first-event').addEventListener('click', () => navigate(-game.events.length));
  $('last-event').addEventListener('click', () => navigate(game.events.length));
  $('previous-event').addEventListener('click', () => navigate(-1));
  $('next-event').addEventListener('click', () => navigate(1));
  $('play-turns').addEventListener('click', () => setPlaying(!playing));
  window.addEventListener('pagehide', () => setPlaying(false));
  $('flip-board').addEventListener('click', () => {
    flipped = !flipped;
    const opponent = document.querySelector('.player-strip:not(.white-player)');
    const agent = document.querySelector('.white-player');
    $('board-stage').before(flipped ? agent : opponent);
    $('board-stage').after(flipped ? opponent : agent);
    renderBoard();
  });
  const traceTabs = [...document.querySelectorAll('[data-trace-mode]')];
  traceTabs.forEach((tab, i) => {
    tab.addEventListener('click', () => {traceMode = tab.dataset.traceMode;renderTrace();});
    tab.addEventListener('keydown', event => {
      if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? traceTabs.length-1 : (i + (event.key === 'ArrowRight' ? 1 : -1) + traceTabs.length) % traceTabs.length;
      traceTabs[next].focus();traceTabs[next].click();
    });
  });
  $('download-event').addEventListener('click', async () => {
    if (!payload || !selected) return;
    const current = selected;
    const raw = payload.raw_record ? await json(`${base}/${payload.raw_record}`) : payload.evidence;
    const url = URL.createObjectURL(new Blob([JSON.stringify({event:current, sanitized_source:raw}, null, 2)], {type:'application/json'}));
    const a = el('a');a.href = url;a.download = `${current.id}.json`;a.click();setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  document.querySelectorAll('[data-panel]').forEach(button => button.addEventListener('click', () => {
    $('workspace').dataset.mobile = button.dataset.panel;
    document.querySelectorAll('[data-panel]').forEach(tab => tab.setAttribute('aria-selected', String(tab === button)));
  }));
  document.addEventListener('keydown', event => {
    if (!game || event.target.closest('input,select,textarea,button,summary,[role=separator]')) return;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {event.preventDefault();navigate(event.key === 'ArrowRight' ? 1 : -1);}
  });
  document.querySelectorAll('.resizer').forEach(handle => {
    const side = handle.dataset.side;
    function resize(clientX) {
      const rect = $('workspace').getBoundingClientRect();
      const other = document.querySelector(side === 'left' ? '.trace-panel' : '.timeline-panel').getBoundingClientRect().width;
      const width = Math.max(side === 'left' ? 190 : 280, Math.min(rect.width-other-314, side === 'left' ? clientX-rect.left : rect.right-clientX));
      $('workspace').style.setProperty(`--${side}`, `${width}px`);
      handle.setAttribute('aria-valuenow', String(Math.round(width)));
    }
    handle.setAttribute('aria-valuemin', side === 'left' ? '190' : '280');
    handle.setAttribute('aria-valuemax', '900');
    handle.addEventListener('pointerdown', event => {
      handle.setPointerCapture(event.pointerId);
      const move = e => resize(e.clientX);
      const up = () => {handle.removeEventListener('pointermove', move);handle.removeEventListener('pointerup', up);handle.removeEventListener('pointercancel', up);};
      handle.addEventListener('pointermove', move);handle.addEventListener('pointerup', up);handle.addEventListener('pointercancel', up);
    });
    handle.addEventListener('keydown', event => {
      if (!['ArrowLeft','ArrowRight'].includes(event.key)) return;
      event.preventDefault();const rect = handle.getBoundingClientRect();resize(rect.left + (event.key === 'ArrowRight' ? 20 : -20));
    });
  });
  window.addEventListener('hashchange', () => route().catch(error => {setPlaying(false);status(error.message);}));
}

try {
  catalog = await json('data/manifest.json');
  if (catalog.schema_version !== 1) throw new Error('Unsupported dataset version.');
  await Promise.all(catalog.runs.filter(r => r.collection === 'benchmark-report').map(async r => benchmarks.set(r.id, await json(`data/${r.path}`))));
  setupViewer();await route();
} catch (error) {
  status(error.message);
}
