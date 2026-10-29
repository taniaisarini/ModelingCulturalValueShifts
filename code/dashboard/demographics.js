/* Demographics view: pick a wave and a country (or "All" for either) and see which questions show a
   significant difference between demographic groups' answers in that wave. Click a column header to group rows
   by it; click more headers to nest (hierarchical grouping, in click order); click an active header again to
   remove it from the grouping.
   Data: window.DEMOGRAPHICS (demographics_data.js). Mounted into #demographics-view by index.html. */
(function () {
  const D = window.DEMOGRAPHICS;
  if (!D) return;
  const ROWS = D.rows;
  const DEMO_LABEL = { sex: 'Sex', age: 'Age', settlement: 'Settlement size' };
  const DEMO_ORDER = { sex: 0, age: 1, settlement: 2 };
  const GROUPABLE = {
    country: { label: 'Country', key: r => r.country, title: r => r.name, sortKey: k => k },
    wave: { label: 'Wave', key: r => String(r.wave), title: r => `Wave ${r.wave}`, sortKey: k => Number(k) },
    code: { label: 'Question', key: r => r.code, title: r => `${r.code} — ${r.label}`, sortKey: k => k },
    demo: { label: 'Split', key: r => r.demo, title: r => DEMO_LABEL[r.demo], sortKey: k => DEMO_ORDER[k] },
  };

  /* same formatting rules as the Browse tab's fmtEffect/fmtThr in index.html, keyed off effect_name (r.metric) */
  function fmtEffect(r) {
    if (r.metric === 'share_change' || r.metric === 'mean_abs_share_change') return (r.effect * 100).toFixed(1) + ' pts';
    if (r.metric === 'tvd') return (r.effect * 100).toFixed(0) + '%';
    return r.effect.toFixed(2);
  }
  function fmtThr(r) {
    if (r.metric === 'share_change' || r.metric === 'mean_abs_share_change') return (r.thr * 100).toFixed(0) + ' pts';
    if (r.metric === 'tvd') return (r.thr * 100).toFixed(0) + '%';
    return r.thr.toFixed(2);
  }
  function fmtP(p) { return p < 0.0001 ? p.toExponential(1) : p.toFixed(4); }
  /* diff is the plain signed difference (group_b minus group_a): a mean-code difference for scales, a
     percentage-point difference in the tracked answer's share otherwise. Distinct from `effect` (the test
     statistic significance is based on) -- see export_dashboard_demographics.py's module docstring. */
  function fmtDiff(r) { return r.stat_kind === 'mean' ? Math.abs(r.diff).toFixed(2) : Math.abs(r.diff).toFixed(1) + ' pts'; }
  function pillHtml(r) {
    return `<span class="pill ${r.dir}" title="${escapeHtml(r.delta_text)}">${r.dir === 'up' ? '&uarr;' : '&darr;'} ${fmtDiff(r)}</span>`;
  }
  function groupValsHtml(r) {
    if (r.stat_kind === 'mean') {
      return `<span class="mono">${r.group_a}: ${r.mean_a.toFixed(2)}</span><span class="sub">${r.group_b}: ${r.mean_b.toFixed(2)} &middot; scale: ${escapeHtml(r.lo_label)} &hellip; ${escapeHtml(r.hi_label)}</span>`;
    }
    return `<span class="mono">${r.group_a}: ${r.share_a.toFixed(1)}%</span><span class="sub">${r.group_b}: ${r.share_b.toFixed(1)}% &middot; "${escapeHtml(String(r.tracked))}"</span>`;
  }
  function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

  /* D.trend: "country|wave|demo|code" -> {v_a, v_b, stat_kind, significant}, for EVERY tested question
     regardless of significance (ordered/share formats only -- see export_dashboard_demographics.py), so a
     wave-to-wave line doesn't have a hole just because one wave wasn't significant. */
  const TREND = D.trend || {};
  const WAVE_LIST = D.meta.waves;

  function trendPoints(country, code, demo) {
    const pts = [];
    for (const w of WAVE_LIST) {
      const e = TREND[`${country}|${w}|${demo}|${code}`];
      if (e) pts.push(Object.assign({ wave: w }, e));
    }
    return pts;
  }

  function sparklineHtml(row) {
    const pts = trendPoints(row.country, row.code, row.demo);
    if (pts.length < 2) return '';
    const W = 220, H = 60, padX = 10, padY = 10;
    const xs = pts.map((p, i) => padX + i * (W - 2 * padX) / (pts.length - 1));
    const vals = pts.flatMap(p => [p.v_a, p.v_b]);
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
    const y = v => H - padY - (v - lo) / (hi - lo) * (H - 2 * padY);
    const line = (key) => pts.map((p, i) => `${xs[i]},${y(p[key]).toFixed(1)}`).join(' ');
    const fmtV = v => pts[0].stat_kind === 'mean' ? v.toFixed(2) : v.toFixed(0) + '%';
    /* <title> must be nested inside the element it labels to show as a hover tooltip -- not a flat list */
    const dots = (key, name) => pts.map((p, i) =>
      `<circle cx="${xs[i]}" cy="${y(p[key]).toFixed(1)}" r="3.2" fill="${p.significant ? 'currentColor' : 'var(--surface)'}" stroke="currentColor" stroke-width="1.5"><title>Wave ${p.wave}, ${escapeHtml(name)}: ${fmtV(p[key])}${p.significant ? '' : ' (not significant that wave)'}</title></circle>`
    ).join('');
    return `
      <div class="sparkline">
        <div class="sparkline-legend">
          <span class="lg a">&mdash; ${escapeHtml(row.group_a)}</span>
          <span class="lg b">&middot;&middot; ${escapeHtml(row.group_b)}</span>
          <span class="sub">hollow dot = not significant that wave</span>
        </div>
        <svg width="${W}" height="${H + 14}" viewBox="0 0 ${W} ${H + 14}">
          <polyline points="${line('v_a')}" fill="none" stroke="var(--accent)" stroke-width="1.8" class="sp-a"/>
          <polyline points="${line('v_b')}" fill="none" stroke="var(--muted)" stroke-width="1.8" stroke-dasharray="3,3" class="sp-b"/>
          <g class="sp-a">${dots('v_a', row.group_a)}</g>
          <g class="sp-b">${dots('v_b', row.group_b)}</g>
          ${pts.map((p, i) => `<text x="${xs[i]}" y="${H + 12}" text-anchor="middle" class="sp-tick">W${p.wave}</text>`).join('')}
        </svg>
      </div>`;
  }

  let state = { wave: 'all', country: 'all', demo: 'all', q: '', sort: 'strength', groupBy: [] };
  let openGroups = new Set();

  function filtered() {
    return ROWS.filter(r =>
      (state.wave === 'all' || r.wave === Number(state.wave)) &&
      (state.country === 'all' || r.country === state.country) &&
      (state.demo === 'all' || r.demo === state.demo) &&
      (!state.q || r.code.toLowerCase().includes(state.q) || r.label.toLowerCase().includes(state.q) || r.name.toLowerCase().includes(state.q))
    );
  }

  function sortRows(rows) {
    /* whenever more than one wave could show up in the same table (wave isn't pinned by the filter or by
       grouping), order by wave first -- 5, then 6, then 7 -- and only then by the chosen sort */
    const waveFirst = state.wave === 'all' && !state.groupBy.includes('wave');
    return rows.slice().sort((a, b) => {
      if (waveFirst && a.wave !== b.wave) return a.wave - b.wave;
      if (state.sort === 'strength') return (b.effect / b.thr) - (a.effect / a.thr);
      if (state.sort === 'p') return a.p - b.p;
      if (state.sort === 'country') return a.country.localeCompare(b.country) || a.wave - b.wave;
      return 0;
    });
  }

  /* which of the 4 label columns still need their own cell: not fixed by a filter, and not an active group key */
  function visibleCols() {
    const fixed = { country: state.country !== 'all', wave: state.wave !== 'all', demo: state.demo !== 'all', code: false };
    const grouped = new Set(state.groupBy);
    const show = {};
    for (const k of Object.keys(GROUPABLE)) show[k] = !fixed[k] && !grouped.has(k);
    return show;
  }

  function theadHtml() {
    const show = visibleCols();
    const col = k => {
      const idx = state.groupBy.indexOf(k);
      const badge = idx >= 0 ? `<span class="group-badge">${idx + 1}</span>` : '';
      const cls = 'groupable' + (idx >= 0 ? ' grouped' : '');
      return `<th class="${cls}" data-group="${k}" title="Click to group by ${GROUPABLE[k].label}">${GROUPABLE[k].label}${badge}</th>`;
    };
    return (show.country ? col('country') : '') + (show.wave ? col('wave') : '') + col('code') +
      '<th>Label</th>' + (show.demo ? col('demo') : '') +
      '<th>Group averages</th><th>Effect</th><th>Difference</th><th>p-value</th>';
  }

  function rowHtml(r) {
    const show = visibleCols();
    return `
      <tr>
        ${show.country ? `<td class="code">${r.country}</td>` : ''}
        ${show.wave ? `<td class="code">${r.wave}</td>` : ''}
        <td class="code">${escapeHtml(r.code)}</td>
        <td class="label">${escapeHtml(r.label)}</td>
        ${show.demo ? `<td class="fmt"><span class="chip">${DEMO_LABEL[r.demo]}</span><span class="sub">${escapeHtml(r.group_a)} vs. ${escapeHtml(r.group_b)}</span></td>` : ''}
        <td class="means">${groupValsHtml(r)}</td>
        <td class="eff tabular">${fmtEffect(r)}<span class="min">min ${fmtThr(r)}</span></td>
        <td class="delta tabular ${r.dir}">${pillHtml(r)}</td>
        <td class="pval tabular">${fmtP(r.p)}</td>
      </tr>`;
  }

  /* nest rows into {key, title, count, rows | children[]} per state.groupBy, one level per entry */
  function buildTree(rows, keys) {
    if (!keys.length) return rows;
    const def = GROUPABLE[keys[0]];
    const map = new Map();
    for (const r of rows) {
      const k = def.key(r);
      if (!map.has(k)) map.set(k, { key: k, title: def.title(r), rows: [] });
      map.get(k).rows.push(r);
    }
    const nodes = Array.from(map.values()).sort((a, b) => {
      const sa = def.sortKey(a.key), sb = def.sortKey(b.key);
      return typeof sa === 'number' ? sa - sb : String(sa).localeCompare(String(sb));
    });
    for (const node of nodes) {
      node.count = node.rows.length;
      node.children = keys.length > 1 ? buildTree(node.rows, keys.slice(1)) : null;
    }
    return nodes;
  }

  function renderGroups(nodes, path) {
    return nodes.map(node => {
      const gpath = path + '::' + node.key;
      const open = openGroups.has(gpath);
      let trendHtml = '';
      if (!node.children) {
        const triples = new Set(node.rows.map(r => r.country + '|' + r.code + '|' + r.demo));
        if (triples.size === 1) trendHtml = sparklineHtml(node.rows[0]);
      }
      const inner = node.children
        ? `<div class="country-body" style="display:${open ? 'block' : 'none'}">${renderGroups(node.children, gpath)}</div>`
        : `<div class="country-body" style="display:${open ? 'block' : 'none'}">${trendHtml}<table><thead><tr>${theadHtml()}</tr></thead><tbody>${sortRows(node.rows).map(rowHtml).join('')}</tbody></table></div>`;
      return `
        <div class="country-card ${open ? 'open' : ''}" data-gpath="${escapeHtml(gpath)}">
          <button class="country-head" data-gpath="${escapeHtml(gpath)}">
            <svg class="chev" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 6 15 12 9 18"/></svg>
            <span class="code">${escapeHtml(node.title)}</span>
            <span class="count"><strong>${node.count}</strong> row${node.count === 1 ? '' : 's'}</span>
          </button>
          ${inner}
        </div>`;
    }).join('');
  }

  function render(root) {
    const rows = filtered();
    root.querySelector('#demo-count').textContent = `${rows.length.toLocaleString()} significant demographic differences`;
    const wrap = root.querySelector('#demo-table-wrap');

    if (state.groupBy.length === 0) {
      const sorted = sortRows(rows);
      wrap.innerHTML = `<table id="demo-table"><thead><tr>${theadHtml()}</tr></thead>
        <tbody>${sorted.length ? sorted.map(rowHtml).join('') : `<tr><td colspan="9" class="empty-state">No matches.</td></tr>`}</tbody></table>`;
    } else {
      const tree = buildTree(rows, state.groupBy);
      wrap.innerHTML = rows.length
        ? `<div class="country-list">${renderGroups(tree, 'root')}</div>`
        : `<div class="empty-state">No matches.</div>`;
      wrap.querySelectorAll('.country-head').forEach(btn => {
        btn.addEventListener('click', () => {
          const p = btn.dataset.gpath;
          if (openGroups.has(p)) openGroups.delete(p); else openGroups.add(p);
          render(root);
        });
      });
    }

    wrap.querySelectorAll('th.groupable').forEach(th => {
      th.addEventListener('click', () => {
        const k = th.dataset.group;
        const i = state.groupBy.indexOf(k);
        if (i >= 0) state.groupBy.splice(i, 1); else state.groupBy.push(k);
        render(root);
      });
    });

    root.querySelector('#demo-clear-group').hidden = state.groupBy.length === 0;
  }

  function mount(root) {
    const countries = Array.from(new Set(ROWS.map(r => r.country))).sort();
    root.innerHTML = `
      <p class="bl-intro">For each country and wave, were there statistically significant differences in how demographic
        subgroups answered a question? Same reliability + effect-size protocol as the Browse tab's wave-to-wave changes,
        applied to one group vs. another <em>within</em> a single wave (e.g. men vs. women at wave 6), not across waves.
        <strong>Group averages</strong> is each group's own raw answer (mean code for scales, % giving the tracked
        answer otherwise). <strong>Effect</strong> is the statistic the significance test is based on (Wasserstein
        distance, total variation distance, or a share change, depending on format) &mdash; for binary-style questions
        this is mathematically the same size as <strong>Difference</strong> (group B minus group A), but for scale
        questions it usually is not, since Wasserstein distance can pick up a distribution-shape change beyond the mean.
        Click a column header (Country / Wave / Question / Split) to group rows by it; click another to nest inside it.</p>
      <div class="controls">
        <div class="sort-wrap"><label>Wave</label>
          <select id="demo-wave"><option value="all">All waves</option>${D.meta.waves.map(w => `<option value="${w}">Wave ${w}</option>`).join('')}</select>
        </div>
        <div class="sort-wrap"><label>Country</label>
          <select id="demo-country"><option value="all">All countries</option>${countries.map(c => `<option value="${c}">${c}</option>`).join('')}</select>
        </div>
        <div class="sort-wrap"><label>Demographic</label>
          <select id="demo-demo"><option value="all">All</option>${D.meta.demos.map(d => `<option value="${d}">${DEMO_LABEL[d]}</option>`).join('')}</select>
        </div>
        <div class="search-wrap">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          <input id="demo-search" type="text" placeholder="Filter by country or question&hellip;" autocomplete="off">
        </div>
        <div class="sort-wrap"><label>Sort by</label>
          <select id="demo-sort"><option value="strength">Strongest effect</option><option value="p">Smallest p-value</option><option value="country">Country, wave</option></select>
        </div>
        <button id="demo-clear-group" class="chip" style="cursor:pointer" hidden>Clear grouping</button>
        <div class="hint" id="demo-count"></div>
      </div>
      <div class="country-card" id="demo-table-card">
        <div class="country-body" style="display:block" id="demo-table-wrap"></div>
      </div>
    `;

    root.querySelector('#demo-wave').addEventListener('change', e => { state.wave = e.target.value; render(root); });
    root.querySelector('#demo-country').addEventListener('change', e => { state.country = e.target.value; render(root); });
    root.querySelector('#demo-demo').addEventListener('change', e => { state.demo = e.target.value; render(root); });
    root.querySelector('#demo-search').addEventListener('input', e => { state.q = e.target.value.trim().toLowerCase(); render(root); });
    root.querySelector('#demo-sort').addEventListener('change', e => { state.sort = e.target.value; render(root); });
    root.querySelector('#demo-clear-group').addEventListener('click', () => { state.groupBy = []; openGroups.clear(); render(root); });

    render(root);
  }

  window.DemographicsView = { meta: D.meta, mount };
})();
