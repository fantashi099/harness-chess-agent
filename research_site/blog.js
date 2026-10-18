import {benchmarkRun, resultCounts, selectionHash, squaresFromFen, recordedMoves} from './model.js?v=28';

export function gameOutcome(game) {
  if (game.completed === false) return {key:'incomplete', label:'Chưa hoàn tất'};
  return game.completed === true ? ({'1-0':{key:'win',label:'Thắng'},'0-1':{key:'loss',label:'Thua'},'1/2-1/2':{key:'draw',label:'Hòa'}}[game.outcome] ?? {key:'unknown',label:'Chưa rõ'}) : {key:'unknown',label:'Chưa rõ'};
}
export function viewerLink(run, game, event = '') {
  return `viewer.html${selectionHash(run, game, event)}`;
}
export function scoreLabel(games) {
  const counts = resultCounts(games);
  return `${(counts.wins + counts.draws / 2).toLocaleString('vi-VN')}/${games.length} điểm`;
}
export function lastBoardEvent(events) {
  return events.findLast(event => {
    if (!event.fen || !event.id) return false;
    try {return squaresFromFen(event.fen).length === 64;} catch {return false;}
  });
}
export function playbackFrames(events) {
  const final = lastBoardEvent(events);
  if (!final) return [];
  const moves = recordedMoves(events);
  const first = events.find(event => event.id && event.fen === moves[0]?.beforeFen) ?? final;
  const frames = [{event:first}, ...moves.map(move => ({event:events[move.index],move}))];
  if (frames.at(-1).event.fen === final.fen) frames.at(-1).event = final;
  else frames.push({event:final});
  return frames;
}
const pieces = {K:'♚',Q:'♛',R:'♜',B:'♝',N:'♞',P:'♟',k:'♚',q:'♛',r:'♜',b:'♝',n:'♞',p:'♟'};
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}
async function json(path) {
  const response = await fetch(path, {cache:'no-store'});
  if (!response.ok) throw new Error(`Không tải được ${path} (${response.status}).`);
  return response.json();
}
function showGames(container, run) {
  const section = element('div', 'run-library');
  const header = element('div', 'run-library-header');
  header.append(element('h3', '', run.policy === 'harness' ? 'LLM + Tools' : 'Tools Only'), element('span', '', run.elo === 2000 && run.policy === 'harness' ? '10 case study · tuyển chọn hồi cứu' : `${run.games.length} ván · ${scoreLabel(run.games)}`));
  const grid = element('div', 'game-grid');
  for (const game of run.games) {
    const outcome = gameOutcome(game);
    const link = element('a', 'game-tile');
    link.href = viewerLink(run.id, game.id);
    link.dataset.outcome = outcome.key;
    link.setAttribute('aria-label', `${run.policy === 'harness' ? 'LLM + Tools' : 'Tools Only'} ${run.elo}, ${game.id.replace('_',' ')}, ${outcome.label}`);
    const top = element('span', 'game-tile-top');
    top.append(element('span', '', game.id.replace('game_', 'GAME ').replace(/\b(\d)\b/, '0$1')), element('span', '', '↗'));
    link.append(top, element('span', 'game-tile-outcome', outcome.label), element('small', '', `${game.event_count.toLocaleString('vi-VN')} events`));
    grid.append(link);
  }
  section.append(header, grid);
  container.append(section);
}
function showBoard(event, board = document.getElementById('hero-board'), label = 'Thế cờ sau 25.Bxc5#, game 9 trước Stockfish 1500. Trắng thắng.') {
  const nodes = squaresFromFen(event.fen).map(square => {
    const node = element('span', `blog-square${square.dark ? ' dark' : ''}${[event.move?.slice(0,2),event.move?.slice(2,4)].includes(square.square) ? ' marked' : ''}`);
    node.setAttribute('aria-hidden', 'true');
    node.dataset.square = square.square;
    if (square.piece) node.append(element('span', `blog-piece${square.piece === square.piece.toUpperCase() ? ' white' : ''}`, pieces[square.piece]));
    if (square.square[1] === '1') node.append(element('span', 'blog-coordinate', square.square[0]));
    return node;
  });
  if (nodes.length !== 64) throw new Error('Thế cờ đã lưu không hợp lệ.');
  board.replaceChildren(...nodes);
  board.setAttribute('aria-label', label);
}
function setupPreview(catalog, manifests) {
  const preview = document.getElementById('episode-preview');
  if (!preview) return;
  const mode = document.getElementById('preview-mode');
  const elo = document.getElementById('preview-elo');
  const games = document.getElementById('preview-game');
  const board = document.getElementById('preview-board');
  const status = document.getElementById('preview-status');
  const note = document.getElementById('preview-note');
  const link = document.getElementById('preview-link');
  const play = document.getElementById('preview-play');
  const reset = document.getElementById('preview-reset');
  const progress = document.getElementById('preview-progress');
  let generation = 0;
  let frames = [], frame = 0, timer = null, animations = [];
  let activeRun, activeGame;
  function stop() {
    clearTimeout(timer);
    timer = null;
    play.textContent = '▶ Play';
    play.setAttribute('aria-label', 'Phát lại ván đấu');
    play.setAttribute('aria-pressed', 'false');
  }
  function cancelAnimations() {
    animations.forEach(animation => animation.cancel());
    animations = [];
  }
  function paint(animate = false) {
    cancelAnimations();
    const {event,move} = frames[frame];
    const finalFrame = frame === frames.length - 1;
    const label = `${activeGame.id.replace('_',' ')} · ${activeRun.elo} Elo · ${gameOutcome(activeGame).label}`;
    showBoard(event, board, `${label}. Vị trí ${frame}/${frames.length - 1} trong bản ghi.`);
    progress.textContent = `${frame}/${frames.length - 1} nước${move ? ` · ${move.from} → ${move.to}` : ''}`;
    reset.disabled = frame === 0;
    note.textContent = `${activeRun.policy === 'harness' ? 'LLM + Tools' : 'Tools Only'} · ${finalFrame ? activeGame.completed === false ? 'Ván dừng trước khi hoàn tất; đây là vị trí cuối được ghi lại.' : 'Vị trí cuối của ván đấu.' : 'Đang xem lại các nước đã ghi.'}${activeRun.elo === 2000 && activeRun.policy === 'harness' ? ' Case study tuyển chọn hồi cứu; không phải mẫu ngẫu nhiên.' : ''}`;
    link.href = viewerLink(activeRun.id, activeGame.id, event.id);
    if (!animate || !move || frames[frame - 1]?.event.fen !== move.beforeFen || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    for (const movement of [move,move.rook].filter(Boolean)) {
      const from = board.querySelector(`[data-square="${movement.from}"]`).getBoundingClientRect();
      const target = board.querySelector(`[data-square="${movement.to}"]`);
      const to = target.getBoundingClientRect();
      const piece = target.querySelector('.blog-piece');
      if (piece?.animate) {
        piece.classList.add('preview-moving');
        animations.push(piece.animate([{transform:`translate(${from.left-to.left}px,${from.top-to.top}px)`},{transform:'translate(0,0)'}],{duration:260,easing:'ease-in-out'}));
      }
    }
  }
  function advance() {
    frame += 1;
    paint(true);
    if (frame === frames.length - 1) stop();
    else timer = setTimeout(advance, 900);
  }
  play.addEventListener('click', () => {
    if (timer !== null) {stop();return;}
    if (frames.length < 2) return;
    if (frame === frames.length - 1) {frame = 0;paint();}
    play.textContent = 'Ⅱ Pause';
    play.setAttribute('aria-label', 'Tạm dừng phát lại');
    play.setAttribute('aria-pressed', 'true');
    timer = setTimeout(advance, 900);
  });
  reset.addEventListener('click', () => {stop();frame = 0;paint();});
  link.addEventListener('click', stop);
  document.addEventListener('visibilitychange', () => {if (document.hidden) stop();});
  window.addEventListener('pagehide', stop);
  let initialGame = 'game_9';
  try {
    const saved = JSON.parse(sessionStorage.getItem('chess-blog-preview'));
    const run = manifests.get(benchmarkRun(catalog, saved?.policy, Number(saved?.elo))?.id);
    if (['harness','search'].includes(saved?.policy) && [1500,2000].includes(Number(saved?.elo)) && run?.games.some(game => game.id === saved.game)) {
      mode.value = saved.policy;
      elo.value = String(saved.elo);
      initialGame = saved.game;
    }
  } catch { /* Use the default recorded game when storage is blocked or stale. */ }
  const currentRun = () => manifests.get(benchmarkRun(catalog, mode.value, Number(elo.value))?.id);
  async function render() {
    const request = ++generation;
    stop();
    cancelAnimations();
    frames = [];
    play.disabled = reset.disabled = true;
    progress.textContent = '—';
    const run = currentRun();
    const game = run?.games.find(game => game.id === games.value);
    if (game) {
      try {sessionStorage.setItem('chess-blog-preview', JSON.stringify({policy:mode.value,elo:Number(elo.value),game:game.id}));} catch { /* Selections still work without storage. */ }
    }
    board.replaceChildren(element('p','', 'Đang tải thế cờ…'));
    board.setAttribute('aria-label', 'Đang tải thế cờ đã ghi lại');
    link.removeAttribute('href');
    link.setAttribute('aria-disabled', 'true');
    status.textContent = 'Đang tải ván đấu…';
    status.removeAttribute('data-outcome');
    note.textContent = '';
    try {
      if (!game) throw new Error('Nhóm ván này chưa có trong dataset đã nhập.');
      const index = await json(`data/${game.path}`);
      if (request !== generation) return;
      frames = playbackFrames(index.events);
      if (!frames.length) throw new Error('Không có thế cờ đã ghi để hiển thị.');
      activeRun = run;
      activeGame = game;
      frame = 0;
      const outcome = gameOutcome(game);
      status.textContent = `${game.id.replace('_',' ')} · ${run.elo} Elo · ${outcome.label}`;
      status.dataset.outcome = outcome.key;
      paint();
      play.disabled = frames.length < 2;
      link.removeAttribute('aria-disabled');
    } catch (error) {
      if (request !== generation) return;
      board.replaceChildren(element('p','', 'Không tải được thế cờ.'));
      board.setAttribute('aria-label', 'Thế cờ không khả dụng');
      status.textContent = 'Chưa hiển thị được ván đấu';
      note.textContent = error.message;
    }
  }
  function selectRun(initial = false) {
    const run = currentRun();
    const previous = initial ? initialGame : games.value;
    games.replaceChildren(...(run?.games ?? []).map(game => {
      const option = element('option','', `${game.id.replace('_',' ')} · ${gameOutcome(game).label}`);
      option.value = game.id;
      return option;
    }));
    if (run?.games.some(game => game.id === previous)) games.value = previous;
    games.disabled = !run?.games.length;
    render();
  }
  mode.disabled = false;
  elo.disabled = false;
  mode.addEventListener('change', () => selectRun());
  elo.addEventListener('change', () => selectRun());
  games.addEventListener('change', render);
  selectRun(true);
}
async function loadEvidence() {
  const catalog = await json('data/manifest.json');
  if (catalog.schema_version !== 1) throw new Error('Dataset chưa được hỗ trợ.');
  const manifests = new Map();
  await Promise.all([...document.querySelectorAll('[data-game-library]')].map(async container => {
    container.replaceChildren();
    for (const policy of ['harness','search']) {
      const entry = benchmarkRun(catalog, policy, Number(container.dataset.gameLibrary));
      if (!entry) {container.append(element('p','dataset-status',`Chưa nhập dataset ${policy} ${container.dataset.gameLibrary}.`));continue;}
      try {
        const run = await json(`data/${entry.path}`);
        manifests.set(entry.id, run);
        showGames(container, run);
      } catch (error) {container.append(element('p','dataset-status dataset-error',error.message));}
    }
  }));
  setupPreview(catalog, manifests);
  const run = manifests.get(benchmarkRun(catalog, 'harness', 1500)?.id);
  if (!run) throw new Error('Chưa nhập batch harness 1500 để mở các ví dụ trong bài.');
  await Promise.all([['repair','game_1'],['win','game_9'],['incomplete','game_4']].map(async ([name,gameId]) => {
    const game = run.games.find(game => game.id === gameId);
    if (!game) return;
    const index = await json(`data/${game.path}`);
    const event = name === 'repair' ? index.events.find(event => event.type === 'error') : index.events.at(-1);
    if (!event) return;
    for (const link of document.querySelectorAll(`[data-episode="${name}"]`)) link.href = viewerLink(run.id, game.id, event.id);
    if (name === 'win') showBoard(event);
  }));
}
function watchSections() {
  if (!('IntersectionObserver' in window)) return;
  const links = [...document.querySelectorAll('.article-toc a')];
  const observer = new IntersectionObserver(entries => {
    const entry = entries.filter(entry => entry.isIntersecting).sort((a,b) => a.boundingClientRect.top-b.boundingClientRect.top)[0];
    if (!entry) return;
    for (const link of links) {
      if (link.hash === `#${entry.target.id}`) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    }
  }, {rootMargin:'-10% 0px -60% 0px'});
  document.querySelectorAll('.article-body>section').forEach(section => observer.observe(section));
}
if (typeof document !== 'undefined') {
  watchSections();
  loadEvidence().catch(error => {
    document.querySelectorAll('.dataset-status').forEach(node => {node.classList.add('dataset-error');node.textContent = `${error.message} Bài viết vẫn có thể đọc; mở Trajectories để kiểm tra dataset.`;});
    const board = document.getElementById('hero-board');
    if (board.querySelector('p')) board.replaceChildren(element('p','', 'Chưa tải được thế cờ đã ghi lại.'));
    const previewStatus = document.getElementById('preview-status');
    if (previewStatus && document.getElementById('preview-mode').disabled) {
      previewStatus.textContent = 'Chưa tải được dataset ván đấu.';
      document.getElementById('preview-board').replaceChildren(element('p','', 'Thế cờ chưa khả dụng.'));
      document.getElementById('preview-note').textContent = error.message;
    }
  });
}
