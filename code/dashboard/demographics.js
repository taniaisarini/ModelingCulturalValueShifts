/* Demographics view: pick a wave and a country (or "All" for either) and see which questions show a
   significant difference between demographic groups' answers in that wave.
   Data: window.DEMOGRAPHICS (demographics_data.js). Mounted into #demographics-view by index.html. */
(function () {
  const D = window.DEMOGRAPHICS;
  if (!D) return;
  const ROWS = D.rows;
  const DEMO_LABEL = { sex: 'Sex', age: 'Age', settlement: 'Settlement size' };

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
  function pillHtml(r) {
    if (r.dir === 'shift') return `<span class="pill shift" title="${escapeHtml(r.delta_text)}">&harr; shift</span>`;
    return `<span class="pill ${r.dir}" title="${escapeHtml(r.delta_text)}">${r.dir === 'up' ? '&uarr;' : '&darr;'} ${fmtEffect(r)}</span>`;
  }
  function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

  let state = { wave: 'all', country: 'all', demo: 'all', q: '', sort: 'strength' };

  function filtered() {
    return ROWS.filter(r =>
      (state.wave === 'all' || r.wave === Number(state.wave)) &&
      (state.country === 'all' || r.country === state.country) &&
      (state.demo === 'all' || r.demo === state.demo) &&
      (!state.q || r.code.toLowerCase().includes(state.q) || r.label.toLowerCase().includes(state.q) || r.name.toLowerCase().includes(state.q))
    );
  }

  function render(root) {
    const rows = filtered();
    const sorted = rows.slice().sort((a, b) => {
      if (state.sort === 'strength') return (b.effect / b.thr) - (a.effect / a.thr);
      if (state.sort === 'p') return a.p - b.p;
      if (state.sort === 'country') return a.country.localeCompare(b.country) || a.wave - b.wave;
      return 0;
    });
    const showCountryCol = state.country === 'all';
    const showWaveCol = state.wave === 'all';

    root.querySelector('#demo-count').textContent = `${rows.length.toLocaleString()} significant demographic differences`;

    const body = root.querySelector('#demo-tbody');
    if (sorted.length === 0) {
      body.innerHTML = `<tr><td colspan="8" class="empty-state">No matches.</td></tr>`;
      return;
    }
    body.innerHTML = sorted.map(r => `
      <tr>
        ${showCountryCol ? `<td class="code">${r.country}</td>` : ''}
        ${showWaveCol ? `<td class="code">${r.wave}</td>` : ''}
        <td class="code">${escapeHtml(r.code)}</td>
        <td class="label">${escapeHtml(r.label)}</td>
        <td class="fmt"><span class="chip">${DEMO_LABEL[r.demo]}</span><span class="sub">${escapeHtml(r.group_a)} vs. ${escapeHtml(r.group_b)}</span></td>
        <td class="eff tabular">${fmtEffect(r)}<span class="min">min ${fmtThr(r)}</span></td>
        <td class="delta tabular ${r.dir}">${pillHtml(r)}</td>
        <td class="pval tabular">${fmtP(r.p)}</td>
      </tr>
    `).join('');
  }

  function mount(root) {
    const countries = Array.from(new Set(ROWS.map(r => r.country))).sort();
    root.innerHTML = `
      <p class="bl-intro">For each country and wave, were there statistically significant differences in how demographic
        subgroups answered a question? Same reliability + effect-size protocol as the Browse tab's wave-to-wave changes,
        applied to one group vs. another <em>within</em> a single wave (e.g. men vs. women at wave 6), not across waves.</p>
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
        <div class="hint" id="demo-count"></div>
      </div>
      <div class="country-card">
        <div class="country-body" style="display:block">
          <table id="demo-table">
            <thead><tr id="demo-thead"></tr></thead>
            <tbody id="demo-tbody"></tbody>
          </table>
        </div>
      </div>
    `;

    function syncHead() {
      const showCountryCol = state.country === 'all';
      const showWaveCol = state.wave === 'all';
      root.querySelector('#demo-thead').innerHTML =
        (showCountryCol ? '<th>Country</th>' : '') + (showWaveCol ? '<th>Wave</th>' : '') +
        '<th>Question</th><th>Label</th><th>Split</th><th>Effect</th><th>Difference</th><th>p-value</th>';
    }

    root.querySelector('#demo-wave').addEventListener('change', e => { state.wave = e.target.value; syncHead(); render(root); });
    root.querySelector('#demo-country').addEventListener('change', e => { state.country = e.target.value; syncHead(); render(root); });
    root.querySelector('#demo-demo').addEventListener('change', e => { state.demo = e.target.value; render(root); });
    root.querySelector('#demo-search').addEventListener('input', e => { state.q = e.target.value.trim().toLowerCase(); render(root); });
    root.querySelector('#demo-sort').addEventListener('change', e => { state.sort = e.target.value; render(root); });

    syncHead();
    render(root);
  }

  window.DemographicsView = { meta: D.meta, mount };
})();
