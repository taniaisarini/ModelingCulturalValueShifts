/* Reasoning view: what did each model talk about when it explained its answer?
   Data: window.REASONING (reasoning_data.js); question labels and the true change come from window.BASELINE (baseline_results.js).
   Mounted into #reasoning-view by index.html. */
(function () {
  const D = window.REASONING;
  if (!D) return;
  const MODELS = D.models, THEMES = D.themes, COUNTRIES = D.countries;
  const SHORT = m => m.replace(' (reasoning off)', ' · off').replace(' (reasoning on)', ' · on');
  const CNAME = { EGY: 'Egypt', HKG: 'Hong Kong', THA: 'Thailand', UKR: 'Ukraine', URY: 'Uruguay' };
  const cname = c => CNAME[c] || c;
  const LABELS = {}, TRUTH = {};
  for (const r of (window.BASELINE ? window.BASELINE.rows : [])) {
    const k = `${r.country}|${r.wa}->${r.wb}|${r.code}`;
    LABELS[k] = r.label; TRUTH[k] = r.truth_text;
  }
  const R = D.replies.map(a => ({ m: a[0], c: a[1], w: a[2], code: a[3], ok: !!a[4], g: { tfidf: a[5], embed: a[6] }, t: a[7], ct: a[8], pt: a[9], th: a[10] }));

  const S = { method: 'tfidf', country: 'all', sort: 'n', sel: null, dm: new Set(), outcome: 'all', limit: 12,
    qc: COUNTRIES[0], theme: null, openQ: new Set() };
  let root = null;

  const NOTE = {
    tfidf: 'Each reply is put in the topic that fits it best (TF-IDF + NMF on the reasoning text, with country names and years removed). ' +
      'Cells show what share of that model’s replies fall in the topic, so every column adds up to 100%.',
    country: 'These topics were fitted on this country’s replies only (all five model runs pooled, every question), so they pick up what is specific to the country. ' +
      'Cells show what share of that model’s replies for this country fall in the topic.',
    embed: 'Each reply is embedded (text-embedding-3-small) and grouped with k-means. These groups follow the country and its history more than the way a model reasons, ' +
      'which is why every model looks almost the same here. Use the topics view to compare models.',
  };

  function h(tag, props, ...kids) {
    const e = document.createElement(tag);
    if (props) for (const [k, v] of Object.entries(props)) {
      if (v == null || v === false) continue;
      if (k === 'class') e.className = v;
      else if (k === 'text') e.textContent = v;
      else if (k === 'style') e.setAttribute('style', v);
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
  const seg = (items, cur, on) => h('div', { class: 'bl-seg', role: 'group' }, items.map(([v, l]) =>
    h('button', { class: 'bl-segbtn' + (cur === v ? ' on' : ''), type: 'button', 'aria-pressed': cur === v ? 'true' : 'false', onclick: () => on(v) }, l)));
  const chip = (label, on, click) => h('button', { class: 'bl-chip' + (on ? ' on' : ''), type: 'button', 'aria-pressed': on ? 'true' : 'false', onclick: click }, label);
  function cellStyle(v, max) {
    const t = Math.min(1, v / max), mix = Math.round(t * 85);
    return `background:color-mix(in srgb, var(--accent) ${mix}%, var(--surface));color:${t > 0.55 ? 'var(--surface)' : 'var(--ink)'}`;
  }

  /* ---------- topic grid (all countries, or one country with its own topics) ---------- */
  const inScope = r => S.country === 'all' || r.c === S.country;
  const method = () => (S.country === 'all' ? S.method : 'country');
  const keyOf = r => (S.country === 'all' ? r.g[S.method] : r.ct);
  const groupDefs = () => (S.country === 'all' ? D.methods[S.method].groups : D.country_topics[S.country]);

  function compute() {
    const scope = R.filter(inScope);
    const perModel = MODELS.map(() => 0);
    scope.forEach(r => perModel[r.m]++);
    return groupDefs().map(g => {
      const rs = scope.filter(r => keyOf(r) === g.id);
      const share = MODELS.map((_, m) => (perModel[m] ? rs.filter(r => r.m === m).length / perModel[m] : 0));
      const accM = MODELS.map((_, m) => { const x = rs.filter(r => r.m === m); return { n: x.length, k: x.filter(r => r.ok).length }; });
      const k = rs.filter(r => r.ok).length;
      return Object.assign({}, g, { n: rs.length, k, acc: rs.length ? k / rs.length : 0, ci: wilson(k, rs.length), share, accM,
        spread: Math.max(...share) - Math.min(...share) });
    });
  }

  function render() {
    if (!root) return;
    const rows = compute();
    const sorters = { n: (a, b) => b.n - a.n, spread: (a, b) => b.spread - a.spread, acc: (a, b) => b.acc - a.acc };
    rows.sort(sorters[S.sort]);
    if (S.sel != null && !rows.some(r => r.id === S.sel)) S.sel = null;
    root.replaceChildren(renderGrid(rows), S.sel != null ? renderDetail(rows.find(r => r.id === S.sel)) : null, renderRelated());
  }

  function renderGrid(rows) {
    const max = Math.max(0.01, ...rows.flatMap(r => r.share));
    const reset = () => { S.sel = null; S.dm = new Set(); S.outcome = 'all'; S.limit = 12; };
    const controls = h('div', { class: 'bl-controls' },
      h('div', { class: 'bl-fgroup' }, h('span', { class: 'bl-flabel', text: 'Country' }),
        h('div', { class: 'bl-chips' }, [['all', 'All countries'], ...COUNTRIES.map(c => [c, cname(c)])].map(([v, l]) =>
          chip(l, S.country === v, () => { S.country = v; if (v !== 'all') S.qc = v; reset(); render(); })))),
      S.country === 'all' && h('div', { class: 'bl-fgroup' }, h('span', { class: 'bl-flabel', text: 'Grouping' }),
        seg(Object.entries(D.methods).map(([k, m]) => [k, m.label]), S.method, v => { S.method = v; reset(); render(); })),
      h('div', { class: 'bl-fgroup' }, h('label', { class: 'bl-flabel', for: 'rs-sort', text: 'Sort by' }),
        h('select', { id: 'rs-sort', class: 'bl-select', onchange: e => { S.sort = e.target.value; render(); } },
          [['n', 'Most replies'], ['spread', 'Biggest difference between models'], ['acc', 'Accuracy']].map(([k, l]) => h('option', { value: k, selected: k === S.sort }, l)))));

    const head = h('tr', null, h('th', { text: 'Topic' }),
      MODELS.map(m => h('th', { class: 'rs-mh', title: m }, SHORT(m))),
      h('th', { text: 'Replies' }), h('th', { text: 'Accuracy on these' }));
    const body = rows.map(r => {
      const lo = r.ci[0] * 100, hi = r.ci[1] * 100;
      const open = () => { const was = S.sel === r.id; reset(); S.sel = was ? null : r.id; render(); };
      return h('tr', { class: 'rs-row' + (S.sel === r.id ? ' sel' : ''), tabindex: '0', role: 'button', 'aria-pressed': S.sel === r.id ? 'true' : 'false',
        'aria-label': `${r.name}: ${r.n} replies, ${pct(r.acc, 0)} correct. Activate to read examples.`, onclick: open,
        onkeydown: e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } } },
        h('td', { class: 'rs-name' }, h('span', { text: r.name }), h('span', { class: 'rs-words', text: r.words.split(', ').slice(0, 5).join(', ') })),
        r.share.map((v, i) => h('td', { class: 'rs-cell', style: cellStyle(v, max), title: `${MODELS[i]}: ${r.accM[i].n} replies (${pct(v)} of its replies)` }, pct(v, 0))),
        h('td', { class: 'num', text: r.n }),
        h('td', { class: 'rs-acc' },
          h('div', { class: 'rs-track' }, h('div', { class: 'rs-fill', style: `width:${r.acc * 100}%` }), h('div', { class: 'bl-ci', style: `left:${lo}%;width:${Math.max(hi - lo, 0.4)}%`, title: `95% interval ${pct(r.ci[0], 0)}–${pct(r.ci[1], 0)}` }),
            h('div', { class: 'rs-half' })),
          h('span', { class: 'rs-accv', text: pct(r.acc, 0) })));
    });
    const title = S.country === 'all' ? 'What the models talked about' : `What the models said about ${cname(S.country)}`;
    return h('section', { class: 'bl-card', 'aria-label': 'Topics by model' },
      h('div', { class: 'bl-head' }, h('h2', { class: 'bl-h2', text: title }),
        h('p', { class: 'bl-sub', text: NOTE[method()] + ' Darker means more of that model’s replies. The accuracy bar starts at 0%, the dashed line marks 50%, and the whisker is a 95% interval. Click a row to read replies.' })),
      controls,
      h('div', { class: 'bl-scroll', style: 'max-height:none' }, h('table', { class: 'bl-table rs-table' }, h('thead', null, head), h('tbody', null, body))));
  }

  function renderDetail(g) {
    const list = R.filter(r => inScope(r) && keyOf(r) === g.id && (!S.dm.size || S.dm.has(r.m)) &&
      (S.outcome === 'all' || (S.outcome === 'correct') === r.ok));
    const total = list.length, shown = list.slice(0, S.limit);
    const accRows = MODELS.map((m, i) => {
      const a = g.accM[i];
      return h('tr', null, h('td', { text: m }), h('td', { class: 'num', text: a.n }), h('td', { class: 'num', text: a.n ? pct(a.k / a.n, 0) : '—' }));
    });
    return h('section', { class: 'bl-card', 'aria-label': 'Replies in this topic' },
      h('div', { class: 'bl-head' }, h('h2', { class: 'bl-h2', text: g.name }),
        h('p', { class: 'bl-sub', text: 'Top words: ' + g.words })),
      h('div', { class: 'bl-scroll', style: 'max-height:none;margin:12px 0' }, h('table', { class: 'bl-table', style: 'min-width:0;max-width:460px' },
        h('thead', null, h('tr', null, h('th', { text: 'Model' }), h('th', { text: 'Replies' }), h('th', { text: 'Correct' }))), h('tbody', null, accRows))),
      h('div', { class: 'bl-frow', style: 'margin-bottom:12px' },
        h('div', { class: 'bl-fgroup' }, h('span', { class: 'bl-flabel', text: 'Model' }),
          h('div', { class: 'bl-chips' }, MODELS.map((m, i) => chip(SHORT(m), S.dm.has(i), () => { S.dm.has(i) ? S.dm.delete(i) : S.dm.add(i); S.limit = 12; render(); })))),
        h('div', { class: 'bl-fgroup' }, h('span', { class: 'bl-flabel', text: 'Outcome' }),
          h('div', { class: 'bl-chips' }, [['all', 'All'], ['correct', 'Correct'], ['wrong', 'Wrong']].map(([v, l]) => chip(l, S.outcome === v, () => { S.outcome = v; S.limit = 12; render(); }))))),
      h('div', { class: 'rs-replies' }, shown.map(r => replyCard(r))),
      h('div', { class: 'bl-more' }, `Showing ${shown.length} of ${total} replies. `,
        total > S.limit && h('button', { class: 'bl-link', type: 'button', onclick: () => { S.limit += 24; render(); } }, 'Show 24 more')));
  }

  function replyCard(r) {
    return h('article', { class: 'rs-reply' },
      h('div', { class: 'rs-meta' },
        h('span', { class: 'bl-res ' + (r.ok ? 'ok' : 'bad'), text: r.ok ? '✓ correct' : '✗ wrong' }),
        h('span', { class: 'rs-model', text: SHORT(MODELS[r.m]) }),
        h('span', { text: `${r.c} ${r.w}` }),
        h('span', null, h('span', { class: 'bl-code', text: r.code }), ' ', LABELS[`${r.c}|${r.w}|${r.code}`] || '')),
      h('p', { class: 'rs-text', text: r.t }));
  }

  /* ---------- related questions: one country, one theme, every model side by side ---------- */
  function renderRelated() {
    const scope = R.filter(r => r.c === S.qc);
    const byTheme = THEMES.map((name, ti) => {
      const rs = scope.filter(r => r.th === ti);
      const qs = new Set(rs.map(r => `${r.w}|${r.code}`));
      const perModel = MODELS.map((_, m) => { const x = rs.filter(r => r.m === m); return { n: x.length, k: x.filter(r => r.ok).length }; });
      return { ti, name, nq: qs.size, perModel };
    }).filter(t => t.nq);
    if (S.theme == null || !byTheme.some(t => t.ti === S.theme)) S.theme = byTheme.length ? byTheme[0].ti : null;

    const grid = h('div', { class: 'bl-scroll', style: 'max-height:none' }, h('table', { class: 'bl-table rs-table', style: 'min-width:620px' },
      h('thead', null, h('tr', null, h('th', { text: 'Theme' }), h('th', { text: 'Questions' }), MODELS.map(m => h('th', { class: 'rs-mh', title: m }, SHORT(m))))),
      h('tbody', null, byTheme.map(t => {
        const pick = () => { S.theme = t.ti; S.openQ = new Set(); render(); };
        return h('tr', { class: 'rs-row' + (S.theme === t.ti ? ' sel' : ''), tabindex: '0', role: 'button', 'aria-pressed': S.theme === t.ti ? 'true' : 'false', onclick: pick,
          onkeydown: e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } } },
          h('td', { class: 'rs-name' }, h('span', { text: t.name })), h('td', { class: 'num', text: t.nq }),
          t.perModel.map((p, i) => h('td', { class: 'rs-cell', style: cellStyle(p.n ? p.k / p.n : 0, 1), title: `${MODELS[i]}: ${p.k} of ${p.n} correct` }, `${p.k}/${p.n}`)));
      }))));

    return h('section', { class: 'bl-card', 'aria-label': 'Related questions' },
      h('div', { class: 'bl-head' }, h('h2', { class: 'bl-h2', text: 'Related questions, country by country' }),
        h('p', { class: 'bl-sub', text: 'Pick a country, then a theme (questions grouped by hand, for example everything about religion). The grid shows how many of each model’s answers were correct (darker = more correct; counts are small, so read them as examples, not rates). Click a theme to compare the models question by question.' })),
      h('div', { class: 'bl-controls' }, h('div', { class: 'bl-fgroup' }, h('span', { class: 'bl-flabel', text: 'Country' }),
        h('div', { class: 'bl-chips' }, COUNTRIES.map(c => chip(cname(c), S.qc === c, () => { S.qc = c; S.openQ = new Set(); render(); }))))),
      grid, renderTheme(scope));
  }

  function renderTheme(scope) {
    if (S.theme == null) return h('div', { class: 'bl-empty', text: 'No questions for this country.' });
    const rs = scope.filter(r => r.th === S.theme);
    const qs = new Map();
    for (const r of rs) { const k = `${r.w}|${r.code}`; if (!qs.has(k)) qs.set(k, []); qs.get(k).push(r); }
    const rows = [...qs.entries()].sort((a, b) => a[0].localeCompare(b[0]));
    const body = [];
    for (const [k, list] of rows) {
      const byM = MODELS.map((_, m) => list.find(r => r.m === m));
      const first = list[0], full = `${first.c}|${first.w}|${first.code}`;
      const open = S.openQ.has(k), nright = list.filter(r => r.ok).length;
      const toggle = () => { open ? S.openQ.delete(k) : S.openQ.add(k); render(); };
      body.push(h('tr', { class: 'bl-qrow' + (open ? ' open' : ''), tabindex: '0', 'aria-expanded': open ? 'true' : 'false', onclick: toggle,
        onkeydown: e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } } },
        h('td', null, h('span', { class: 'bl-code', text: first.code }), ' ', LABELS[full] || '', h('span', { class: 'rs-words', text: `Waves ${first.w.replace('->', '→')} · ${nright} of ${list.length} models right` })),
        h('td', { class: 'rs-truth', text: TRUTH[full] || '' }),
        byM.map(r => h('td', { class: 'rs-pred ' + (r ? (r.ok ? 'ok' : 'bad') : '') }, r ? [h('span', { class: 'bl-res ' + (r.ok ? 'ok' : 'bad'), text: r.ok ? '✓ ' : '✗ ' }), r.pt] : '—'))));
      if (open) body.push(h('tr', { class: 'bl-detail' }, h('td', { colspan: String(2 + MODELS.length) },
        h('div', { class: 'rs-replies' }, byM.filter(Boolean).map(r => replyCard(r))))));
    }
    return h('div', { style: 'margin-top:16px' },
      h('h3', { class: 'rs-h3', text: `${THEMES[S.theme]} in ${cname(S.qc)}` }),
      h('p', { class: 'bl-sub', text: 'Click a question to read every model’s reasoning for it. Green ✓ means the model got the direction right.' }),
      h('div', { class: 'bl-scroll', style: 'max-height:none;margin-top:8px' }, h('table', { class: 'bl-table rs-table', style: 'min-width:900px' },
        h('thead', null, h('tr', null, h('th', { text: 'Question' }), h('th', { text: 'What actually happened' }), MODELS.map(m => h('th', { class: 'rs-mh', title: m }, SHORT(m))))),
        h('tbody', null, body))));
  }

  const CSS = `
  .rs-table{min-width:760px}
  .rs-table th.rs-mh{text-align:center;white-space:normal;min-width:78px}
  .rs-row{cursor:pointer}.rs-row:hover td:not(.rs-cell),.rs-row.sel td:not(.rs-cell){background:var(--accent-soft)}
  .rs-row.sel .rs-name span:first-child{color:var(--accent);font-weight:600}
  .rs-row:focus-visible{outline:2px solid var(--focus);outline-offset:-2px}
  .rs-name{max-width:300px}.rs-name span{display:block}.rs-words{display:block;font-size:11.5px;color:var(--muted);margin-top:2px}
  .rs-cell{text-align:center;font-variant-numeric:tabular-nums;font-weight:500;border-bottom:2px solid var(--surface)!important;border-left:2px solid var(--surface)}
  .rs-table td.num{text-align:right;font-variant-numeric:tabular-nums}
  .rs-acc{min-width:170px;white-space:nowrap}
  .rs-track{display:inline-block;position:relative;width:110px;height:14px;vertical-align:middle;border-right:1px solid var(--border);background:var(--bg);border-radius:0 3px 3px 0}
  .rs-fill{position:absolute;left:0;top:0;height:14px;background:var(--accent);border-radius:0 3px 3px 0}
  .rs-track .bl-ci{top:7px}
  .rs-half{position:absolute;left:50%;top:-2px;height:18px;border-left:1px dashed var(--ink);opacity:.6}
  .rs-accv{display:inline-block;margin-left:8px;font-variant-numeric:tabular-nums;font-size:12.5px}
  .rs-replies{display:flex;flex-direction:column;gap:10px}
  .rs-reply{border:1px solid var(--border);border-radius:8px;padding:10px 12px;background:var(--bg)}
  .bl-detail .rs-reply{background:var(--surface)}
  .rs-meta{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--muted);align-items:baseline}
  .rs-model{font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:var(--ink)}
  .rs-text{margin:6px 0 0;font-size:13.5px;line-height:1.55;max-width:90ch}
  .rs-h3{font-size:16px;font-weight:600}
  .rs-truth{font-size:12.5px;min-width:150px}
  .rs-pred{font-size:12px;min-width:110px}
  .rs-pred.ok{background:var(--up-soft)}.rs-pred.bad{background:var(--down-soft)}
  `;

  window.ReasoningView = {
    mount(el) {
      root = el;
      if (window.BaselineView) window.BaselineView.ensureStyle();   // shared chip / segmented-button / table styles
      if (!document.getElementById('rs-style')) { const s = h('style', { id: 'rs-style' }); s.textContent = CSS; document.head.append(s); }
      render();
    },
  };
})();
