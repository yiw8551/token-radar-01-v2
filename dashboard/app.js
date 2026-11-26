// Token Radar dashboard: renders window.RADAR (written by main.ts from the snapshot).
// Tabs: pre-token launches (default), all protocols, live tokens, model & fit.
// Every row opens a detail view with its fitted breakdown and its farm plan.
(function () {
  var D = window.RADAR;
  var rows = D.rows;
  var state = { tab: 'launch', q: '', cat: '', sort: 'value', sel: null };

  function $(id) { return document.getElementById(id); }
  function money(n) {
    if (n >= 1e9) return '$' + (n / 1e9).toFixed(2) + 'B';
    if (n >= 1e6) return '$' + (n / 1e6).toFixed(1) + 'M';
    if (n >= 1e3) return '$' + (n / 1e3).toFixed(0) + 'K';
    return '$' + n;
  }
  function pct(x) { return x === null || x === undefined ? '—' : (x > 0 ? '+' : '') + x.toFixed(1) + '%'; }
  function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
  function ageText(a) { return a === null ? 'unknown' : a >= 365 ? (a / 365).toFixed(1) + 'y' : a + 'd'; }

  // ---- header ----
  $('meta').innerHTML = 'Snapshot <b>' + D.as_of + '</b> · DefiLlama public API<br>' +
    'Floor <b>' + money(D.floor_usd) + '</b> · Model <b>' + D.model_version + '</b> · Board <b>' + D.board_size + ' rows</b>';
  var top = rows.filter(function (r) { return r.token_status === 'launch'; })[0];
  $('kpis').innerHTML = [
    [D.board_size, 'protocols ranked'], [D.launch_size, 'launch candidates'],
    [D.watchlist_size, 'watchlist covered'], [D.fit.train_n, 'protocols fitted on'],
    ['R² ' + D.fit.loo_r2, 'leave-one-out, log mcap'], [top ? money(top.implied_value_usd) : '—', 'top launch, implied']
  ].map(function (k, i) {
    return '<div class="kpi' + (i === 4 ? ' warn' : '') + '"><b>' + k[0] + '</b><span>' + k[1] + '</span></div>';
  }).join('');

  // ---- category filter ----
  var cats = {};
  rows.forEach(function (r) { if (r.category) cats[r.category] = 1; });
  Object.keys(cats).sort().forEach(function (c) {
    var o = document.createElement('option'); o.value = c; o.textContent = c; $('cat').appendChild(o);
  });

  // ---- filtering & sorting ----
  function visible() {
    var list = rows.filter(function (r) {
      if (state.tab === 'launch' && r.token_status !== 'launch') return false;
      if (state.tab === 'live' && r.token_status !== 'live') return false;
      if (state.cat && r.category !== state.cat) return false;
      if (state.q) {
        var q = state.q.toLowerCase();
        if ((r.name + ' ' + r.slug + ' ' + (r.category || '')).toLowerCase().indexOf(q) < 0) return false;
      }
      return true;
    });
    var key = state.sort;
    list.sort(function (a, b) {
      if (key === 'tvl') return b.tvl_usd - a.tvl_usd;
      if (key === 'growth') return (b.change_7d || 0) - (a.change_7d || 0);
      if (key === 'age') return (a.age_days === null ? 1e9 : a.age_days) - (b.age_days === null ? 1e9 : b.age_days);
      if (key === 'odds') return b.launch_odds - a.launch_odds;
      return b.implied_value_usd - a.implied_value_usd;
    });
    return list;
  }

  function statusPill(r) {
    if (r.token_status === 'launch') return '<span class="pill launch">PRE-TOKEN</span>';
    if (r.token_status === 'live') return '<span class="pill live">TOKEN TRADES</span>';
    return '<span class="pill no-launch">NO LAUNCH</span>';
  }

  function render() {
    var list = visible();
    var launchNo = {};
    var n = 0;
    rows.forEach(function (r) { if (r.token_status === 'launch') launchNo[r.slug] = ++n; });
    $('count').textContent = list.length + ' rows';
    $('rows').innerHTML = list.slice(0, 120).map(function (r) {
      var g = r.change_7d === null ? '<span class="flat">—</span>'
        : '<span class="' + (r.change_7d > 0 ? 'up' : r.change_7d < 0 ? 'down' : 'flat') + '">' + pct(r.change_7d) + '</span>';
      var no = state.tab === 'launch' ? '#' + launchNo[r.slug] : '#' + r.rank;
      return '<tr data-slug="' + r.slug + '" class="' + (state.sel === r.slug ? 'sel' : '') + '">' +
        '<td>' + no + '<span class="sub">value #' + r.rank + '</span></td>' +
        '<td><b>' + esc(r.name) + '</b><span class="sub">' + esc(r.slug) + '</span></td>' +
        '<td>' + esc(r.category || '—') + '</td>' +
        '<td>' + esc((r.chains || []).slice(0, 2).join(', ') || '—') + '</td>' +
        '<td class="num">' + money(r.tvl_usd) + '</td>' +
        '<td class="num">' + g + '</td>' +
        '<td class="num">' + ageText(r.age_days) + '</td>' +
        '<td class="num"><b>' + money(r.implied_value_usd) + '</b>' +
        (r.value_basis === 'market-cap' ? '<span class="sub">market cap</span>' : '<span class="sub">fitted</span>') + '</td>' +
        '<td class="num">' + money(r.value_low_usd) + ' – ' + money(r.value_high_usd) + '</td>' +
        '<td class="num">' + (r.token_status === 'launch' ? Math.round(r.launch_odds * 100) + '%' : '—') + '</td>' +
        '<td>' + statusPill(r) + '</td></tr>';
    }).join('');
    $('hint').textContent = list.length > 120
      ? 'Showing the first 120 of ' + list.length + ' rows — narrow the search or category to see more.'
      : 'Click any row for its detail view, model breakdown and farm plan.';
    if (!state.sel && list.length) select(list[0].slug);
  }

  // ---- detail view ----
  function select(slug) {
    state.sel = slug;
    var r = null;
    rows.forEach(function (x) { if (x.slug === slug) r = x; });
    if (!r) { $('detail').innerHTML = ''; return; }
    var basis = r.value_basis === 'market-cap'
      ? 'Market value: the token already trades, so this is its recorded market cap, not a model claim.'
      : r.value_basis === 'fitted-no-mcap'
        ? 'Token trades but no market cap is recorded, so the fitted value stands in, flagged as modelled.'
        : 'Fitted value: TVL × fitted multiple (size, growth, age and category terms, all fitted on the snapshot’s own tokenized protocols).';
    var mult = '<div class="dcell"><b>×' + r.model.multiple + '</b><span>fitted multiple</span></div>' +
      '<div class="dcell"><b>×' + r.model.size_multiplier + '</b><span>size term</span></div>' +
      '<div class="dcell"><b>×' + r.model.growth_multiplier + '</b><span>growth term</span></div>' +
      '<div class="dcell"><b>×' + r.model.age_multiplier + '</b><span>age term</span></div>' +
      '<div class="dcell"><b>×' + r.model.category_multiplier + '</b><span>category term</span></div>';
    $('detail').innerHTML = '<h2>' + esc(r.name) + ' <span style="color:var(--mut);font-size:13px">' + esc(r.slug) + '</span></h2>' +
      '<p class="sub2">' + esc(r.category || 'Uncategorised') + ' · ' + esc((r.chains || []).join(', ') || 'no chain listed') +
      ' · listed ' + (r.listed_at || 'unknown') + ' · ' + statusPill(r) + '</p>' +
      '<div class="dgrid">' +
      '<div class="dcell"><b>' + money(r.implied_value_usd) + '</b><span>implied value</span></div>' +
      '<div class="dcell"><b>' + money(r.value_low_usd) + ' – ' + money(r.value_high_usd) + '</b><span>50% interval</span></div>' +
      '<div class="dcell"><b>' + money(r.tvl_usd) + '</b><span>TVL · 7d ' + pct(r.change_7d) + '</span></div>' +
      '<div class="dcell"><b>' + (r.token_status === 'launch' ? Math.round(r.launch_odds * 100) + '%' : '—') + '</b><span>launch odds</span></div>' +
      mult + '</div>' +
      '<p style="color:var(--mut);font-size:13px">' + basis + ' The 50% range comes from the leave-one-out residuals: half the held-out protocols land within ±' + D.fit.loo_q50 + ' log points.</p>' +
      '<h3>Farm plan — what it costs, how long it takes</h3><ol>' +
      r.farm_steps.map(function (s) { return '<li>' + esc(s) + '</li>'; }).join('') + '</ol>';
    document.querySelectorAll('#rows tr').forEach(function (tr) {
      tr.classList.toggle('sel', tr.getAttribute('data-slug') === slug);
    });
  }
  // ---- model tab ----
  $('modeltab').innerHTML = '<h2>How the value is fitted — and how well it holds up</h2>' +
    '<p>Log market cap is regressed on log TVL, seven-day TVL growth, log age and category, ' +
    'ridge λ = ' + D.fit.lambda + ', fitted on the snapshot’s own ' + D.fit.train_n + ' tokenized protocols. ' +
    'Nothing is hardcoded by name: the same function scores all ' + D.board_size + ' protocols. ' +
    'A pre-token protocol’s implied value is its TVL times its fitted multiple; a live token carries its recorded market cap.</p>' +
    '<table><thead><tr><th>Check</th><th class="num">Result</th><th>What it means</th></tr></thead><tbody>' +
    '<tr><td>Leave-one-out R² (log mcap)</td><td class="num">' + D.fit.loo_r2 + '</td><td>Each protocol predicted from a model that never saw it.</td></tr>' +
    '<tr><td>Leave-one-out MAE</td><td class="num">' + D.fit.loo_mae + ' log pts</td><td>Typical held-out miss; the naive median scores ' + D.fit.naive_mae + '.</td></tr>' +
    '<tr><td>Walk-forward R² / MAE</td><td class="num">' + D.fit.walk_r2 + ' / ' + D.fit.walk_mae + '</td><td>Trained on ' + D.fit.walk_train_n + ' protocols listed before ' + D.fit.walk_cut + ', tested on the ' + D.fit.walk_test_n + ' listed after.</td></tr>' +
    '<tr><td>Conformal 50% / 90% band</td><td class="num">±' + D.fit.loo_q50 + ' / ±' + D.fit.loo_q90 + ' log pts</td><td>From the leave-one-out residuals; the 50% band is the range shown per row.</td></tr>' +
    '<tr><td>Base multiple</td><td class="num">×' + D.fit.base_multiple + '</td><td>At median TVL, zero growth, median age, no category effect.</td></tr>' +
    '<tr><td>Size / growth / age coefs</td><td class="num">' + D.fit.size_coef + ' / ' + D.fit.growth_coef + ' / ' + D.fit.age_coef + '</td><td>TVL is sublinear; fast 7-day growth slightly lowers the multiple; age barely matters.</td></tr>' +
    '</tbody></table>' +
    '<h2 style="margin-top:18px">Fitted category multiples</h2>' +
    '<table><thead><tr><th>Category</th><th class="num">Tokenized n</th><th class="num">Multiple</th></tr></thead><tbody>' +
    D.category_multiples.map(function (c) {
      return '<tr><td>' + esc(c.category) + '</td><td class="num">' + c.n + '</td><td class="num">×' + c.multiple + '</td></tr>';
    }).join('') + '</tbody></table>' +
    '<p>Launch odds fall with age: 70% for a protocol under a year with no token, decaying after that — a stated model curve, not a fitted one, because the snapshot records no launches to fit it on. ' +
    'Integrity: the program reads only the sealed snapshot, runs twice byte-identical, uses only the Node standard library, and names no protocol in its code.</p>';

  $('foot').innerHTML = '<b>Token Radar v2.</b> Implied values are the entry’s own claim from the sealed snapshot of ' + D.as_of +
    ' — never a price, never a promise a token will launch. Say a protocol “records no token”, never that it has none. ' +
    'Farm plans are actions with costs and times, not advice; verify every step on the protocol’s own page before moving funds.';

  // ---- events ----
  document.querySelectorAll('#tabs button').forEach(function (btn) {
    btn.addEventListener('click', function () {
      document.querySelectorAll('#tabs button').forEach(function (b) { b.classList.remove('on'); });
      btn.classList.add('on');
      state.tab = btn.getAttribute('data-tab');
      var isModel = state.tab === 'model';
      $('modeltab').hidden = !isModel;
      $('controls').style.display = isModel ? 'none' : 'flex';
      $('board').style.display = isModel ? 'none' : '';
      $('hint').style.display = isModel ? 'none' : '';
      if (!isModel) render();
    });
  });
  $('q').addEventListener('input', function (e) { state.q = e.target.value; render(); });
  $('cat').addEventListener('change', function (e) { state.cat = e.target.value; render(); });
  $('sort').addEventListener('change', function (e) { state.sort = e.target.value; render(); });
  document.querySelectorAll('#board th[data-k]').forEach(function (th) {
    th.addEventListener('click', function () {
      var k = th.getAttribute('data-k');
      state.sort = k === 'rank' || k === 'value' ? 'value' : k === 'name' || k === 'category' || k === 'chains' ? 'value' : k;
      $('sort').value = state.sort;
      render();
    });
  });
  $('rows').addEventListener('click', function (e) {
    var tr = e.target.closest('tr');
    if (tr) select(tr.getAttribute('data-slug'));
  });

  render();
})();
