export function readSelection(hash) {
  const p = new URLSearchParams(hash.replace(/^#/, ''));
  return {run: p.get('run'), game: p.get('game'), event: p.get('event')};
}

export function selectionHash(run, game, event) {
  return '#' + new URLSearchParams({run, game, event}).toString();
}

export function resolveEvent(events, id) {
  return events.find(e => e.id === id) ?? events[0] ?? null;
}

export function resolveRun(catalog, id) {
  return catalog.runs.find(r => r.id === id) ?? catalog.runs.find(r => r.id === catalog.default_run) ?? catalog.runs[0];
}

export function benchmarkRun(catalog, policy, elo) {
  return catalog.runs.find(r => r.collection === 'benchmark-report' && r.policy === policy && r.elo === Number(elo));
}

export function resultCounts(games) {
  const counts = {wins:0, draws:0, losses:0, incomplete:0, unknown:0};
  for (const game of games) {
    const key = game.completed === true ? ({'1-0':'wins','0-1':'losses','1/2-1/2':'draws'}[game.outcome] ?? 'unknown') : game.completed === false ? 'incomplete' : 'unknown';
    counts[key]++;
  }
  return counts;
}

export function traceTurns(events) {
  const turns = new Map();
  for (const event of events) {
    if (!turns.has(event.turn)) turns.set(event.turn, {turn:event.turn, events:[], calls:[], outcomes:[]});
    const turn = turns.get(event.turn);
    turn.events.push(event);
    if (event.source.file === 'harness-trace.jsonl' || event.source.location.includes('.harness_records.')) {
      const location = event.source.location.replace(/\.(replan|committed_move)$/, '');
      let call = turn.calls.find(c => c.location === location && c.file === event.source.file);
      if (!call) {call = {file:event.source.file, location, events:[]};turn.calls.push(call);}
      call.events.push(event);
    } else turn.outcomes.push(event);
  }
  return [...turns.values()];
}

export function navigateTurn(events, current, offset) {
  const turns = traceTurns(events);
  if (!turns.length) return null;
  const i = Math.max(0, turns.findIndex(turn => turn.turn === current?.turn));
  const next = Math.max(0, Math.min(turns.length - 1, i + offset));
  return next === turns.length - 1 ? turns[next].events.at(-1) : turns[next].events[0];
}

export function recordedOutcome(summary, finalTurn) {
  if (!finalTurn || summary.completed !== true) return null;
  return {'1-0':'Win','0-1':'Loss','1/2-1/2':'Draw'}[summary.outcome] ?? null;
}

export function turnEvidence(events, turn) {
  const entries = events.filter(event => event.turn === turn);
  return {
    verification: entries.findLast(event => event.type === 'verification') ?? null,
    search: entries.findLast(event => event.type === 'tool_result' && (event.candidates?.length || event.title === 'Search result')) ?? null,
    chosenMove: entries.findLast(event => event.type === 'commit' && event.move)?.move ?? null,
  };
}

export function recordedMoves(events) {
  const moves = [];
  let previousFen = null;
  for (const [index, event] of events.entries()) {
    if (['commit','opponent','move'].includes(event.type) && /^[a-h][1-8][a-h][1-8][qrbn]?$/.test(event.move ?? '') && previousFen && event.fen) {
      const before = new Map(squaresFromFen(previousFen).map(s => [s.square,s.piece]));
      const after = new Map(squaresFromFen(event.fen).map(s => [s.square,s.piece]));
      const from = event.move.slice(0,2), to = event.move.slice(2,4);
      const piece = before.get(from);
      const promotion = event.move[4] ?? null;
      const expected = promotion ? piece === piece?.toUpperCase() ? promotion.toUpperCase() : promotion : piece;
      if (piece && !after.get(from) && after.get(to) === expected) {
        let captureSquare = to;
        if (piece.toLowerCase() === 'p' && from[0] !== to[0] && !before.get(to)) captureSquare = to[0] + from[1];
        const victim = before.get(captureSquare);
        const captured = victim && (victim === victim.toUpperCase()) !== (piece === piece.toUpperCase()) && (captureSquare === to || !after.get(captureSquare)) ? victim : null;
        const rook = piece.toLowerCase() === 'k' && Math.abs(from.charCodeAt(0)-to.charCodeAt(0)) === 2
          ? {from:(to[0] === 'g' ? 'h' : 'a')+from[1],to:(to[0] === 'g' ? 'f' : 'd')+from[1]} : null;
        moves.push({index, move:event.move, from, to, piece, captured, captureSquare, rook, promotion, beforeFen:previousFen, afterFen:event.fen});
      }
    }
    if (event.fen) previousFen = event.fen;
  }
  return moves;
}

export function lastMovesByColor(moves, index) {
  const history = moves.filter(move => move.index <= index);
  return {
    white: history.findLast(move => move.piece === move.piece.toUpperCase()) ?? null,
    black: history.findLast(move => move.piece !== move.piece.toUpperCase()) ?? null,
  };
}

export function recordSteps(record) {
  const steps = [];
  for (const choice of record.response?.choices ?? []) {
    const message = choice.message ?? {};
    for (const key of ['reasoning_content', 'reasoning']) {
      if (message[key]) steps.push({type:'thinking', label:'Provider-returned thinking · recorded', value:message[key]});
    }
    let output;
    try {output = JSON.parse(message.content);} catch {output = null;}
    const rationale = record.action?.rationale ?? output?.rationale;
    if (rationale) steps.push({type:'rationale', label:'Recorded rationale · assistant output', value:rationale});
    if (!rationale && message.content != null) steps.push({type:'llm', label:'Assistant output · recorded', value:message.content});
    if (message.tool_calls) steps.push({type:'tool_call', label:'Provider tool calls · recorded', value:message.tool_calls});
  }
  if (!(record.response?.choices?.length) && record.action?.rationale) {
    steps.push({type:'rationale', label:'Recorded rationale · validated action', value:record.action.rationale});
  }
  if (record.action) steps.push({type:'tool_call', label:`Tool call · ${record.action.action ?? 'action'}`, value:record.action});
  if (record.execution_feedback != null) steps.push({type:'tool_result', label:'Tool result', value:record.execution_feedback});
  if (record.committed_move) steps.push({type:'commit', label:'Tool result', value:{committed_move:record.committed_move}});
  if (record.error != null || record.status === 'rejected') steps.push({type:'error', label:'Action error / rejection', value:{status:record.status,error:record.error ?? 'Error detail unavailable'}});
  if (steps.length || record.status != null) steps.push({type:'status', label:'Status', value:{status:record.status ?? null, recovered_after_transport_error:record.recovered_after_transport_error ?? null}});
  return steps;
}

export function turnSteps(records) {
  const steps = [];
  let previousRequest = [];
  for (const [callIndex, record] of records.entries()) {
    if (!record) {previousRequest = [];continue;}
    for (const [i, message] of (record.request ?? []).entries()) {
      if (JSON.stringify(message) !== JSON.stringify(previousRequest[i])) {
        steps.push({callIndex, type:'context', label:`${message.role ?? 'Unknown role'} prompt`, value:message.content ?? message});
      }
    }
    previousRequest = record.request ?? [];
    const response = recordSteps(record);
    steps.push(...response.map(step => ({...step, callIndex})));
  }
  return steps;
}

export function squaresFromFen(fen) {
  if (!fen) return [];
  const ranks = fen.split(' ')[0].split('/');
  if (ranks.length !== 8) return [];
  return ranks.flatMap((rank, r) => {
    const pieces = [...rank].flatMap(c => /[1-8]/.test(c) ? Array(Number(c)).fill('') : [c]);
    if (pieces.length !== 8) throw new Error('Invalid FEN rank');
    return pieces.map((piece, f) => ({square: 'abcdefgh'[f] + (8-r), piece, dark: (f+r)%2 === 1}));
  });
}
