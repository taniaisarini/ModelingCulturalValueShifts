/* LLM baseline view: how often did the model predict the direction of change correctly?
   Data: window.BASELINE (baseline_results.js). Mounted into #baseline-view by index.html. */
(function () {
  const D = window.BASELINE;
  if (!D) return;
  /* one row set per model: the base file (gpt-4.1-mini) holds the shared prompt/question text; other models
     (baseline_models.js) only override the fields that differ, matched by country|waves|question code */
  const MODELS = { 'gpt-4.1-mini': { meta: Object.assign({ label: 'gpt-4.1-mini' }, D.meta), rows: D.rows } };
  const rid = r => `${r.country}|${r.wa}|${r.wb}|${r.code}`;
  for (const [name, m] of Object.entries(window.BASELINE_MODELS || {})) {
    MODELS[name] = { meta: m.meta, rows: D.rows.map(r => Object.assign({}, r, m.rows[rid(r)] || {})) };
  }
  const MODEL_NAMES = Object.keys(MODELS);
  let cur = 'gpt-4.1-mini';
  let ROWS = MODELS[cur].rows;

  /* ---------- labels, dimensions ---------- */
  const FAM_LABEL = { ordered: 'Scales & counts', share: 'Yes/no & select-many', option: 'Pick-one options' };
  const FMT_LABEL = { ordinal_scale: 'Scale', count: 'Count', binary: 'Yes/no', mention_flag: 'Select-many item',
    knowledge: 'Knowledge', categorical: 'Categorical', forced_choice: 'Forced choice', ranked_choice: 'Ranked choice' };
  const wkey = r => `Waves ${r.wa}\u2192${r.wb}`;
  const bucket = x => x < 1.5 ? '1.0\u20131.5\u00d7 the minimum' : x < 2 ? '1.5\u20132\u00d7' : x < 3 ? '2\u20133\u00d7' : '3\u00d7 or more';
  const dirLabel = r => r.tdir === 'up' ? 'Toward last-listed answer' : r.tdir === 'down' ? 'Toward first-listed answer' : 'No direction (pick-one)';

  const DIMS = {
    overall:  { label: 'Overall', key: () => 'All questions in view' },
    country:  { label: 'Country', key: r => r.name },
    waves:    { label: 'Wave pair', key: wkey },
    cw:       { label: 'Country \u00d7 wave pair', key: r => `${r.name}, ${r.wa}\u2192${r.wb}` },
    fam:      { label: 'Question type', key: r => FAM_LABEL[r.fam] },
    fmt:      { label: 'Question format', key: r => FMT_LABEL[r.fmt] },
    group:    { label: 'Question group', key: r => r.group },
    tdir:     { label: 'True direction', key: dirLabel },
    size:     { label: 'Size of the true change', key: r => bucket(r.ratio) },
    question: { label: 'Individual question', key: r => `${r.code} \u00b7 ${r.label}` },
  };
  // what clicking a bar in each breakdown turns into (filters set on other dimensions)
  const APPLY = {
    overall: () => ({}),
    country: r => ({ country: r.name }),
    waves: r => ({ waves: wkey(r) }),
    cw: r => ({ country: r.name, waves: wkey(r) }),
    fam: r => ({ fam: FAM_LABEL[r.fam] }),
    fmt: r => ({ fmt: FMT_LABEL[r.fmt] }),
    group: r => ({ group: r.group }),
    tdir: r => ({ tdir: dirLabel(r) }),
    size: r => ({ size: bucket(r.ratio) }),
    question: r => ({ question: `${r.code} \u00b7 ${r.label}` }),
  };
  const FLOW = ['country', 'waves', 'fam', 'group', 'question'];   // natural drill-down order
  const CHIP_DIMS = ['country', 'waves', 'fam'];
  const FILTER_KEY = { country: DIMS.country.key, waves: wkey, fam: DIMS.fam.key, fmt: DIMS.fmt.key,
    group: DIMS.group.key, tdir: dirLabel, size: r => bucket(r.ratio), question: DIMS.question.key };

  /* ---------- state ---------- */
  const S = { filters: {}, outcome: 'all', q: '', by: 'country', sort: 'acc', view: 'chart', showAll: false,
    tsort: { key: 'country', dir: 1 }, tlimit: 50, open: new Set() };
  let root = null;

  /* ---------- helpers ---------- */
  function h(tag, props, ...kids) {
    const e = document.createElement(tag);
    if (props) for (const [k, v] of Object.entries(props)) {
      if (v == null || v === false) continue;
      if (k === 'class') e.className = v;
      else if (k === 'text') e.textContent = v;
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids.flat()) { if (kid == null || kid === false) continue; e.append(kid.nodeType ? kid : document.createTextNode(String(kid))); }
    return e;
  }
  const pct = (x, d = 1) => (x * 100).toFixed(d) + '%';
  function wilson(k, n) {
    if (!n) return [0, 0];
    const z = 1.96, p = k / n, d = 1 + z * z / n, c = p + z * z / (2 * n), a = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n));
    return [Math.max(0, (c - a) / d), Math.min(1, (c + a) / d)];
  }
  function stats(rows) {
    const n = rows.length, k = rows.filter(r => r.correct).length;
    // base = accuracy of always giving the most common answer (scales, yes/no: the most common true direction;
    // pick-one: a random option). rand = accuracy of a coin flip (50%) or a random pick among the listed options.
    let base = 0, rand = 0;
    const byFam = {};
    for (const r of rows) (byFam[r.fam] = byFam[r.fam] || []).push(r);
    for (const f in byFam) {
      const g = byFam[f];
      if (f === 'option') { const t = g.reduce((s, r) => s + 1 / r.nopt, 0); base += t; rand += t; }
      else { const c = {}; g.forEach(r => { c[r.tdir] = (c[r.tdir] || 0) + 1; }); base += Math.max(...Object.values(c)); rand += g.length / 2; }
    }
    return { n, k, acc: n ? k / n : 0, ci: wilson(k, n), base: n ? base / n : 0, rand: n ? rand / n : 0, rows };
  }
  function passes(r) {
    for (const dim in S.filters) { const set = S.filters[dim]; if (set.size && !set.has(FILTER_KEY[dim](r))) return false; }
    if (S.outcome === 'correct' && !r.correct) return false;
    if (S.outcome === 'wrong' && r.correct) return false;
    if (S.q) { const q = S.q.toLowerCase(); if (!(r.code.toLowerCase().includes(q) || r.label.toLowerCase().includes(q) || r.question.toLowerCase().includes(q))) return false; }
    return true;
  }
  function setFilter(dim, value, replace) {
    const cur = S.filters[dim] || new Set();
    if (replace) S.filters[dim] = new Set([value]);
    else { cur.has(value) ? cur.delete(value) : cur.add(value); S.filters[dim] = cur; }
  }

  /* ---------- tooltip ---------- */
  let tip = null;
  function showTip(evt, lines) {
    if (!tip) { tip = h('div', { class: 'bl-tip', role: 'tooltip' }); document.body.append(tip); }
    tip.replaceChildren(...lines.map((l, i) => h('div', { class: i === 0 ? 'bl-tip-h' : 'bl-tip-l' }, l)));
    tip.hidden = false;
    const r = evt.currentTarget.getBoundingClientRect();
    const x = evt.clientX && evt.type.startsWith('pointer') ? evt.clientX : r.left + 40;
    const w = tip.offsetWidth, hgt = tip.offsetHeight;
    tip.style.left = Math.max(8, Math.min(window.innerWidth - w - 8, x + 12)) + 'px';
    tip.style.top = Math.max(8, r.top - hgt - 6 < 8 ? r.bottom + 6 : r.top - hgt - 6) + 'px';
  }
  const hideTip = () => { if (tip) tip.hidden = true; };

  /* ---------- rendering ---------- */
  function setModel(name) {
    cur = name; ROWS = MODELS[name].rows; S.open.clear();
    const m = MODELS[name].meta, intro = document.getElementById('bl-intro');
    if (intro) intro.textContent = `For every question where a country's answers really changed between two survey waves, ${m.label} was told only the country, the two survey years and the question, and asked which way the answers moved. ` +
      `${m.n} questions, five countries, waves 5\u21926 and 6\u21927, ${m.params.temperature === 0 ? 'temperature 0' : m.params.temperature}. The prompts never say where the questions come from.`;
  }
  function render() {
    if (!root) return;
    const rows = ROWS.filter(passes);
    const parts = [renderModels(), renderFilters(), renderKpis(rows), renderBreakdown(rows), renderTable(rows)];
    root.replaceChildren(...parts);
  }

  function chip(label, active, onclick) {
    return h('button', { class: 'bl-chip' + (active ? ' on' : ''), type: 'button', 'aria-pressed': active ? 'true' : 'false', onclick }, label);
  }
  function renderModels() {
    if (MODEL_NAMES.length < 2) return null;
    // the same filters are applied to every model (except the outcome filter), so the bars are comparable
    const saveOutcome = S.outcome; S.outcome = 'all';
    const list = MODEL_NAMES.map(n => Object.assign(stats(MODELS[n].rows.filter(passes)), { name: MODELS[n].meta.label, id: n }));
    S.outcome = saveOutcome;
    const pick = g => { setModel(g.id); render(); };
    const body = list[0].n ? breakdownChart(list, list.length, list.length, pick) : h('div', { class: 'bl-empty', text: 'Nothing to chart.' });
    return h('section', { class: 'bl-card', 'aria-label': 'Models' },
      h('div', { class: 'bl-head' }, h('h2', { class: 'bl-h2', text: 'Models' }),
        h('p', { class: 'bl-sub', text: `Same questions and prompts for every model, with the filters below applied. Click a model to see its individual answers (showing: ${MODELS[cur].meta.label}). ` +
          'gpt-5 models use their own settings: default temperature, larger output cap, reasoning effort minimal or medium.' })),
      body);
  }
  function renderFilters() {
    const groupOf = dim => {
      const values = [...new Set(ROWS.map(FILTER_KEY[dim]))].sort();
      const set = S.filters[dim] || new Set();
      return h('div', { class: 'bl-fgroup' }, h('span', { class: 'bl-flabel', text: dim === 'waves' ? 'Wave pair' : dim === 'fam' ? 'Question type' : 'Country' }),
        h('div', { class: 'bl-chips' }, values.map(v => chip(v, set.has(v), () => { setFilter(dim, v); render(); }))));
    };
    const outcomes = [['all', 'All'], ['correct', 'Correct'], ['wrong', 'Wrong']];
    const pills = Object.entries(S.filters).filter(([dim, set]) => !CHIP_DIMS.includes(dim) && set.size)
      .flatMap(([dim, set]) => [...set].map(v => h('button', { class: 'bl-pill', type: 'button', onclick: () => { set.delete(v); render(); }, 'aria-label': `Remove filter ${v}` },
        `${DIMS[dim].label}: ${v} \u00d7`)));
    const anyFilter = Object.values(S.filters).some(s => s.size) || S.outcome !== 'all' || S.q;
    return h('section', { class: 'bl-filters', 'aria-label': 'Filters' },
      h('div', { class: 'bl-frow' },
        CHIP_DIMS.map(groupOf),
        h('div', { class: 'bl-fgroup' }, h('span', { class: 'bl-flabel', text: 'Outcome' }),
          h('div', { class: 'bl-chips' }, outcomes.map(([v, l]) => chip(l, S.outcome === v, () => { S.outcome = v; render(); })))),
        h('div', { class: 'bl-fgroup grow' }, h('label', { class: 'bl-flabel', for: 'bl-search', text: 'Search' }),
          h('input', { id: 'bl-search', class: 'bl-input', type: 'search', placeholder: 'Question code, wording\u2026', value: S.q,
            oninput: e => { S.q = e.target.value.trim(); clearTimeout(render._t); render._t = setTimeout(() => { render(); const el = document.getElementById('bl-search'); if (el) { el.focus(); el.setSelectionRange(el.value.length, el.value.length); } }, 200); } }))),
      (pills.length || anyFilter) && h('div', { class: 'bl-frow' }, pills,
        anyFilter && h('button', { class: 'bl-link', type: 'button', onclick: () => { S.filters = {}; S.outcome = 'all'; S.q = ''; render(); } }, 'Clear all filters')));
  }

  function kpi(label, value, note, hero) {
    return h('div', { class: 'bl-kpi' + (hero ? ' hero' : '') }, h('div', { class: 'bl-kpi-l', text: label }),
      h('div', { class: 'bl-kpi-v', text: value }), note && h('div', { class: 'bl-kpi-n', text: note }));
  }
  function renderKpis(rows) {
    const s = stats(rows);
    if (!rows.length) return h('div', { class: 'bl-empty', text: 'No questions match these filters.' });
    const dirRows = rows.filter(r => r.tdir && r.pdir);
    const truthUp = dirRows.length ? dirRows.filter(r => r.tdir === 'up').length / dirRows.length : null;
    const predUp = dirRows.length ? dirRows.filter(r => r.pdir === 'up').length / dirRows.length : null;
    const gap = s.acc - s.base;
    const conf = (() => {
      const c = { up: { up: 0, down: 0 }, down: { up: 0, down: 0 } };
      dirRows.forEach(r => { c[r.tdir][r.pdir]++; });
      return c;
    })();
    const cell = (v, ok) => h('td', { class: ok ? 'ok' : 'bad' }, String(v));
    return h('section', { class: 'bl-kpis', 'aria-label': 'Summary of the questions in view' },
      kpi('Model accuracy', pct(s.acc), `${s.k} of ${s.n} questions \u00b7 95% interval ${pct(s.ci[0], 0)}\u2013${pct(s.ci[1], 0)}`, true),
      kpi('Guessing baselines', `${pct(s.rand, 0)} random \u00b7 ${pct(s.base, 0)} common`, `random guess ${pct(s.rand)} \u00b7 always the most common answer ${pct(s.base)}. Model is ${(Math.abs(gap) * 100).toFixed(1)} points ${gap >= 0 ? 'above' : 'below'} the second.`),
      kpi('Direction of predictions', predUp == null ? 'n/a' : `${pct(predUp, 0)} toward last-listed`, truthUp == null ? 'no scale or yes/no questions in view' : `in the real data ${pct(truthUp, 0)} went that way`),
      dirRows.length ? h('div', { class: 'bl-kpi conf' }, h('div', { class: 'bl-kpi-l', text: 'Real change vs model (scales & yes/no)' }),
        h('table', { class: 'bl-conf' },
          h('thead', null, h('tr', null, h('th'), h('th', { text: 'Model: last-listed' }), h('th', { text: 'Model: first-listed' }))),
          h('tbody', null,
            h('tr', null, h('th', { text: 'Real: last-listed' }), cell(conf.up.up, true), cell(conf.up.down, false)),
            h('tr', null, h('th', { text: 'Real: first-listed' }), cell(conf.down.up, false), cell(conf.down.down, true))))) : null);
  }

  function renderBreakdown(rows) {
    const dim = DIMS[S.by];
    const groups = new Map();
    for (const r of rows) { const k = dim.key(r); (groups.has(k) ? groups.get(k) : groups.set(k, []).get(k)).push(r); }
    let list = [...groups.entries()].map(([name, g]) => Object.assign(stats(g), { name }));
    const sorters = { acc: (a, b) => b.acc - a.acc || b.n - a.n, n: (a, b) => b.n - a.n, name: (a, b) => a.name.localeCompare(b.name),
      gap: (a, b) => (b.acc - b.base) - (a.acc - a.base) };
    list.sort(sorters[S.sort]);
    const total = list.length, cap = 40;
    const shown = S.view === 'chart' && !S.showAll ? list.slice(0, cap) : list;

    const controls = h('div', { class: 'bl-controls' },
      h('div', { class: 'bl-fgroup' }, h('label', { class: 'bl-flabel', for: 'bl-by', text: 'Break down by' }),
        h('select', { id: 'bl-by', class: 'bl-select', onchange: e => { S.by = e.target.value; S.showAll = false; render(); } },
          Object.entries(DIMS).map(([k, d]) => h('option', { value: k, selected: k === S.by }, d.label)))),
      h('div', { class: 'bl-fgroup' }, h('label', { class: 'bl-flabel', for: 'bl-sort', text: 'Sort by' }),
        h('select', { id: 'bl-sort', class: 'bl-select', onchange: e => { S.sort = e.target.value; render(); } },
          [['acc', 'Accuracy'], ['gap', 'Accuracy above most common answer'], ['n', 'Number of questions'], ['name', 'Name']].map(([k, l]) => h('option', { value: k, selected: k === S.sort }, l)))),
      h('div', { class: 'bl-seg', role: 'group', 'aria-label': 'Chart or table' },
        ['chart', 'table'].map(v => h('button', { class: 'bl-segbtn' + (S.view === v ? ' on' : ''), type: 'button', 'aria-pressed': S.view === v ? 'true' : 'false',
          onclick: () => { S.view = v; render(); } }, v === 'chart' ? 'Chart' : 'Table'))));

    const head = h('div', { class: 'bl-head' },
      h('h2', { class: 'bl-h2', text: `Accuracy by ${dim.label.toLowerCase()}` }),
      h('p', { class: 'bl-sub', text: 'Bars start at 0% and end at 100%. Click a bar to zoom into that group.' }));

    let body;
    if (!rows.length) body = h('div', { class: 'bl-empty', text: 'Nothing to chart.' });
    else if (S.view === 'table') body = breakdownTable(shown);
    else body = breakdownChart(shown, total, cap);
    return h('section', { class: 'bl-card', 'aria-label': 'Accuracy breakdown' }, head, controls, body);
  }

  function tipLines(g) {
    const gap = (g.acc - g.base) * 100;
    return [g.name, `${pct(g.acc)} correct \u00b7 ${g.k} of ${g.n} questions`, `95% interval ${pct(g.ci[0])}\u2013${pct(g.ci[1])}`,
      `Random guess: ${pct(g.rand)}`, `Always the most common answer: ${pct(g.base)} (${gap >= 0 ? '+' : '\u2212'}${Math.abs(gap).toFixed(1)} points for the model)`];
  }

  function breakdownChart(list, total, cap, onPick) {
    const axis = h('div', { class: 'bl-axis' }, h('div'), h('div', { class: 'bl-axis-t' },
      [0, 25, 50, 75, 100].map(v => h('span', { style: `left:${v}%`, text: v + '%' }))));
    const rowsEl = list.map(g => {
      const small = g.n < 10;
      const above = g.acc > g.base;
      const sample = g.rows[0];
      const onEnter = e => showTip(e, tipLines(g));
      const click = onPick ? () => { hideTip(); onPick(g); } : () => {
        const set = APPLY[S.by](sample);
        for (const [d, v] of Object.entries(set)) setFilter(d, v, true);
        if (S.by !== 'overall') {
          const idx = FLOW.indexOf(S.by);
          const next = FLOW.find((d, i) => i > idx && !(S.filters[d] && S.filters[d].size)) || FLOW.find(d => !(S.filters[d] && S.filters[d].size)) || 'question';
          S.by = next;
        }
        S.showAll = false; hideTip(); render(); window.scrollTo({ top: root.offsetTop - 8, behavior: 'smooth' });
      };
      const lo = g.ci[0] * 100, hi = g.ci[1] * 100;
      return h('div', { class: 'bl-row', tabindex: '0', role: 'button', 'aria-label': `${g.name}: ${pct(g.acc)} correct of ${g.n} questions. Activate to ${onPick ? 'show this model' : 'filter to this group'}.`,
        onpointermove: onEnter, onpointerleave: hideTip, onfocus: onEnter, onblur: hideTip, onclick: click,
        onkeydown: e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); click(); } } },
        h('div', { class: 'bl-lab' }, h('span', { class: 'bl-name' + (onPick && g.id === cur ? ' sel' : ''), text: g.name }), h('span', { class: 'bl-n', text: `n=${g.n}${small ? ' (few)' : ''}` })),
        h('div', { class: 'bl-plot' },
          h('div', { class: 'bl-bar' + (above ? '' : ' below') + (small ? ' few' : ''), style: `width:${g.acc * 100}%` }),
          h('div', { class: 'bl-ci', style: `left:${lo}%;width:${Math.max(hi - lo, 0.4)}%` }),
          h('div', { class: 'bl-base', style: `left:${g.base * 100}%` }),
          h('div', { class: 'bl-rand', style: `left:${g.rand * 100}%` }),
          h('span', { class: 'bl-val', style: `left:calc(${Math.max(g.acc, g.ci[1], g.base, g.rand) * 100}% + 10px)`, text: pct(g.acc) })));
    });
    const legend = h('div', { class: 'bl-legend', 'aria-label': 'How to read the chart' },
      h('span', null, h('i', { class: 'sw above' }), 'above always-the-most-common-answer'),
      h('span', null, h('i', { class: 'sw below' }), 'at or below it'),
      h('span', null, h('i', { class: 'sw tick' }), 'always the most common answer'),
      h('span', null, h('i', { class: 'sw rand' }), 'random guess'),
      h('span', null, h('i', { class: 'sw ci' }), '95% interval'));
    const more = total > cap && h('div', { class: 'bl-more' }, S.showAll ? `Showing all ${total} groups. ` : `Showing ${cap} of ${total} groups. `,
      h('button', { class: 'bl-link', type: 'button', onclick: () => { S.showAll = !S.showAll; render(); } }, S.showAll ? 'Show fewer' : 'Show all'));
    return h('div', null, legend, axis, h('div', { class: 'bl-rows' }, rowsEl), more);
  }

  function breakdownTable(list) {
    return h('div', { class: 'bl-scroll' }, h('table', { class: 'bl-table' },
      h('thead', null, h('tr', null, ['Group', 'Questions', 'Correct', 'Accuracy', '95% interval', 'Random guess', 'Always the most common answer', 'Model minus most common'].map(t => h('th', { text: t })))),
      h('tbody', null, list.map(g => h('tr', null, h('td', { text: g.name }), h('td', { class: 'num', text: g.n }), h('td', { class: 'num', text: g.k }),
        h('td', { class: 'num', text: pct(g.acc) }), h('td', { class: 'num', text: `${pct(g.ci[0], 0)}\u2013${pct(g.ci[1], 0)}` }),
        h('td', { class: 'num', text: pct(g.rand) }), h('td', { class: 'num', text: pct(g.base) }), h('td', { class: 'num', text: `${g.acc >= g.base ? '+' : '\u2212'}${Math.abs((g.acc - g.base) * 100).toFixed(1)} pts` }))))));
  }

  function renderTable(rows) {
    const cols = [['correct', ''], ['country', 'Country'], ['wave', 'Waves'], ['code', 'Question'], ['truth', 'What actually happened'], ['pred', 'What the model said']];
    const val = { correct: r => r.correct ? 1 : 0, country: r => r.name, wave: r => `${r.wa}${r.wb}`, code: r => r.code, truth: r => r.truth_text, pred: r => r.pred_text };
    const sorted = rows.slice().sort((a, b) => {
      const f = val[S.tsort.key]; const x = f(a), y = f(b);
      return (x < y ? -1 : x > y ? 1 : 0) * S.tsort.dir || a.code.localeCompare(b.code);
    });
    const shown = sorted.slice(0, S.tlimit);
    const th = ([k, l]) => h('th', { 'aria-sort': S.tsort.key === k ? (S.tsort.dir > 0 ? 'ascending' : 'descending') : 'none' },
      h('button', { class: 'bl-thbtn', type: 'button', onclick: () => { S.tsort = { key: k, dir: S.tsort.key === k ? -S.tsort.dir : 1 }; render(); } },
        k === 'correct' ? 'Result' : l, S.tsort.key === k ? (S.tsort.dir > 0 ? ' \u25b2' : ' \u25bc') : ''));
    const body = [];
    shown.forEach((r, i) => {
      const id = `${r.country}${r.wa}${r.wb}${r.code}`;
      const open = S.open.has(id);
      body.push(h('tr', { class: 'bl-qrow' + (open ? ' open' : ''), tabindex: '0', 'aria-expanded': open ? 'true' : 'false',
        onclick: () => { open ? S.open.delete(id) : S.open.add(id); render(); },
        onkeydown: e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open ? S.open.delete(id) : S.open.add(id); render(); } } },
        h('td', null, h('span', { class: 'bl-res ' + (r.correct ? 'ok' : 'bad') }, r.correct ? '\u2713 correct' : '\u2717 wrong')),
        h('td', { text: r.name }), h('td', { text: `${r.wa}\u2192${r.wb}` }),
        h('td', null, h('span', { class: 'bl-code', text: r.code }), ' ', r.label),
        h('td', { text: r.truth_text }), h('td', { text: r.pred_text })));
      if (open) body.push(h('tr', { class: 'bl-detail' }, h('td', { colspan: '6' },
        h('div', { class: 'bl-dgrid' },
          h('div', null, h('div', { class: 'bl-dl', text: 'Question as asked' }), h('div', { class: 'bl-dt', text: r.question }),
            h('div', { class: 'bl-dl', text: 'Years' }), h('div', { class: 'bl-dt', text: `${r.ya} \u2192 ${r.yb}` }),
            h('div', { class: 'bl-dl', text: 'Size of the true change' }), h('div', { class: 'bl-dt', text: `${r.ratio.toFixed(1)}\u00d7 the minimum that counts as a change` }),
            h('div', { class: 'bl-dl', text: 'Cache key' }), h('div', { class: 'bl-dt mono', text: r.key })),
          h('div', null, h('div', { class: 'bl-dl', text: 'Model\u2019s full reply' }), h('pre', { class: 'bl-pre', text: r.response })),
          h('div', null, h('div', { class: 'bl-dl', text: 'Prompt sent (user message)' }), h('pre', { class: 'bl-pre', text: r.prompt }))))));
    });
    const more = sorted.length > S.tlimit && h('div', { class: 'bl-more' }, `Showing ${S.tlimit} of ${sorted.length} questions. `,
      h('button', { class: 'bl-link', type: 'button', onclick: () => { S.tlimit += 50; render(); } }, 'Show 50 more'),
      ' \u00b7 ', h('button', { class: 'bl-link', type: 'button', onclick: () => { S.tlimit = sorted.length; render(); } }, 'Show all'));
    return h('section', { class: 'bl-card', 'aria-label': 'Individual questions' },
      h('div', { class: 'bl-head' }, h('h2', { class: 'bl-h2', text: 'Individual questions' }),
        h('p', { class: 'bl-sub', text: 'Click a row to read the exact prompt and the model\u2019s full reply.' })),
      h('div', { class: 'bl-scroll' }, h('table', { class: 'bl-table qs' }, h('thead', null, h('tr', null, cols.map(th))), h('tbody', null, body))), more);
  }

  /* ---------- styles (tokens come from index.html) ---------- */
  const CSS = `
  .bl-filters{display:flex;flex-direction:column;gap:10px;margin-bottom:16px}
  .bl-frow{display:flex;flex-wrap:wrap;gap:10px 22px;align-items:flex-end}
  .bl-fgroup{display:flex;flex-direction:column;gap:5px}.bl-fgroup.grow{flex:1;min-width:200px;max-width:340px}
  .bl-flabel{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
  .bl-chips{display:flex;flex-wrap:wrap;gap:6px}
  .bl-chip{font:inherit;font-size:12.5px;padding:5px 11px;border:1px solid var(--border);border-radius:999px;background:var(--surface);color:var(--ink);cursor:pointer}
  .bl-chip:hover{background:var(--accent-soft)}.bl-chip.on{background:var(--accent);border-color:var(--accent);color:var(--surface)}
  .bl-chip:focus-visible,.bl-pill:focus-visible,.bl-link:focus-visible,.bl-segbtn:focus-visible,.bl-thbtn:focus-visible,.bl-row:focus-visible,.bl-qrow:focus-visible,.bl-input:focus-visible,.bl-select:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
  .bl-input,.bl-select{font:inherit;font-size:13px;padding:7px 10px;border:1px solid var(--border);border-radius:8px;background:var(--surface);color:var(--ink);width:100%}
  .bl-pill{font:inherit;font-size:12px;padding:3px 10px;border-radius:999px;border:1px solid var(--accent);background:var(--accent-soft);color:var(--ink);cursor:pointer}
  .bl-link{font:inherit;font-size:12.5px;background:none;border:none;color:var(--accent);cursor:pointer;text-decoration:underline;padding:2px}
  .bl-kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:16px}
  .bl-kpi{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px;display:flex;flex-direction:column;gap:4px}
  .bl-kpi-l{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
  .bl-kpi-v{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}
  .bl-kpi.hero .bl-kpi-v{font-family:'Fraunces',Georgia,serif;font-size:52px;line-height:1.05;font-weight:600}
  .bl-kpi-n{font-size:12px;color:var(--muted)}
  .bl-conf{border-collapse:collapse;font-size:12px;min-width:0;width:100%}.bl-conf th,.bl-conf td{padding:4px 6px;text-align:center;border:none;font-weight:400;color:var(--muted);text-transform:none;letter-spacing:0;position:static}
  .bl-conf tbody th{text-align:left}.bl-conf td{font-variant-numeric:tabular-nums;font-weight:600;color:var(--ink)}.bl-conf td.ok{background:var(--up-soft)}.bl-conf td.bad{background:var(--down-soft)}
  .bl-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:18px 18px 14px;margin-bottom:16px}
  .bl-h2{font-size:18px;font-weight:600}.bl-sub{margin:4px 0 0;color:var(--muted);font-size:12.5px}
  .bl-controls{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:flex-end;margin:14px 0 12px}
  .bl-controls .bl-fgroup{min-width:170px}
  .bl-seg{display:flex;border:1px solid var(--border);border-radius:8px;overflow:hidden}
  .bl-segbtn{font:inherit;font-size:12.5px;padding:7px 14px;border:none;background:var(--surface);color:var(--muted);cursor:pointer}.bl-segbtn.on{background:var(--accent);color:var(--surface)}
  .bl-legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12px;color:var(--muted);margin-bottom:8px}
  .bl-legend span{display:inline-flex;align-items:center;gap:6px}
  .bl-legend .sw{display:inline-block}.sw.above,.sw.below{width:14px;height:10px;border-radius:2px;background:var(--accent)}.sw.below{background:var(--muted);opacity:.5}
  .sw.tick{width:2px;height:14px;background:var(--ink)}.sw.ci{width:22px;height:0;border-top:2px solid var(--ink);opacity:.7}
  .bl-axis{display:grid;grid-template-columns:var(--lab) 1fr;gap:0}.bl-axis-t{position:relative;height:18px;margin-right:52px}
  .bl-axis-t span{position:absolute;transform:translateX(-50%);font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums}
  .bl-axis-t span:first-child{transform:none}.bl-axis-t span:last-child{transform:translateX(-100%)}
  .bl-card{--lab:clamp(120px,30%,240px)}
  .bl-rows{display:flex;flex-direction:column}
  .bl-row{display:grid;grid-template-columns:var(--lab) 1fr;align-items:center;min-height:36px;cursor:pointer;border-radius:6px}
  .bl-row:hover{background:var(--accent-soft)}
  .bl-lab{display:flex;flex-direction:column;padding-right:10px;min-width:0}
  .bl-name{font-size:13px;overflow-wrap:anywhere;line-height:1.25}.bl-name.sel{font-weight:600;color:var(--accent)}.bl-n{font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums}
  .bl-plot{position:relative;height:36px;margin-right:52px;background-image:linear-gradient(to right,var(--border) 1px,transparent 1px);background-size:25% 100%;border-right:1px solid var(--border)}
  .bl-bar{position:absolute;left:0;top:10px;height:16px;background:var(--accent);border-radius:0 4px 4px 0;min-width:2px}
  .bl-bar.below{background:var(--muted);opacity:.5}.bl-bar.few{opacity:.55}
  .bl-ci{position:absolute;top:17px;height:0;border-top:2px solid var(--ink);opacity:.7}
  .bl-ci::before,.bl-ci::after{content:'';position:absolute;top:-6px;height:10px;border-left:2px solid var(--ink)}.bl-ci::before{left:0}.bl-ci::after{right:0}
  .bl-base{position:absolute;top:5px;width:2px;height:26px;background:var(--ink);transform:translateX(-1px);box-shadow:0 0 0 2px var(--surface)}
  .bl-rand{position:absolute;top:27px;width:9px;height:9px;border:2px solid var(--ink);background:var(--surface);transform:translateX(-50%) rotate(45deg)}
  .sw.rand{width:8px;height:8px;border:2px solid var(--ink);background:var(--surface);transform:rotate(45deg)}
  .bl-val{position:absolute;top:8px;font-size:12px;font-variant-numeric:tabular-nums;color:var(--ink);white-space:nowrap}
  .bl-more{margin-top:8px;font-size:12.5px;color:var(--muted)}
  .bl-empty{padding:40px 16px;text-align:center;color:var(--muted)}
  .bl-scroll{overflow:auto;max-height:min(75vh,720px)}
  .bl-table{width:100%;border-collapse:collapse;min-width:640px;font-size:13px}
  .bl-table th{position:sticky;top:0;z-index:2;background:var(--surface);box-shadow:0 1px 0 var(--border);text-align:left;padding:8px 12px;font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:500;border-bottom:1px solid var(--border);white-space:nowrap}
  .bl-table td{padding:8px 12px;border-bottom:1px solid var(--border);vertical-align:top}.bl-table td.num{text-align:right;font-variant-numeric:tabular-nums}
  .bl-thbtn{font:inherit;text-transform:inherit;letter-spacing:inherit;background:none;border:none;color:inherit;cursor:pointer;padding:0}
  .bl-qrow{cursor:pointer}.bl-qrow:hover,.bl-qrow.open{background:var(--accent-soft)}
  .bl-res{font-size:12px;font-weight:500;white-space:nowrap}.bl-res.ok{color:var(--up)}.bl-res.bad{color:var(--down)}
  .bl-code{font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--muted)}
  .bl-detail td{background:var(--bg);padding:14px 16px}
  .bl-dgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}
  .bl-dl{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-top:8px}.bl-dl:first-child{margin-top:0}
  .bl-dt{font-size:13px;margin-top:2px}.mono{font-family:'IBM Plex Mono',monospace;font-size:12px}
  .bl-pre{margin:4px 0 0;white-space:pre-wrap;overflow-wrap:anywhere;font-family:'IBM Plex Mono',monospace;font-size:12px;line-height:1.5;max-height:320px;overflow:auto;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px}
  .bl-tip{position:fixed;z-index:50;max-width:min(360px,calc(100vw - 16px));background:var(--ink);color:var(--surface);border-radius:8px;padding:9px 12px;pointer-events:none;font-size:12.5px;line-height:1.45;box-shadow:0 2px 10px rgba(0,0,0,.25)}
  .bl-tip-h{font-weight:600;margin-bottom:3px}.bl-tip-l{font-variant-numeric:tabular-nums}
  @media (max-width:640px){.bl-card{--lab:110px;padding:14px 12px}.bl-kpi.hero .bl-kpi-v{font-size:44px}}
  @media (prefers-reduced-motion:no-preference){.bl-bar{transition:width .25s ease}}
  `;

  window.BaselineView = {
    ensureStyle() { if (!document.getElementById('bl-style')) { const s = h('style', { id: 'bl-style' }); s.textContent = CSS; document.head.append(s); } },
    mount(el) {
      root = el;
      this.ensureStyle();
      render();
    },
    refresh: render,
    get meta() { return MODELS[cur].meta; },
  };
})();
