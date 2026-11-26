// token-radar-01 v2 model: a fitted value model for pre-token protocols.
//
// The model is fitted on the snapshot's own tokenized protocols (those that record a
// market cap): it regresses log market cap on log TVL, seven-day TVL growth, protocol
// age and category, with ridge shrinkage. Every term is fitted, never hardcoded, and
// the fit is validated held-out (leave-one-out and walk-forward) on the same snapshot.
// A protocol that records no token gets its implied value from the fitted model; a
// protocol whose token already trades carries the market's own value (its market cap
// where recorded, the fitted value otherwise, flagged as modelled).
//
// No protocol name is ever special-cased: the same function scores every row.

export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

export class SnapshotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotError';
  }
}

const DAY_MS = 86_400_000;
/** Ridge strengths tried; the one with the best leave-one-out R² wins. */
const LAMBDAS = [3, 10, 30, 100, 300];
const MIN_CATEGORY_MEMBERS = 5;

/** A finite number, else `fallback`. */
function finite(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

/** Whole days from `listedAt` to `asOf`, clamped at zero; null when absent or bad. */
export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

/** Whether the record carries no token at all. */
export function isPreToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

/** A name ending in a version marker is a pool of another protocol, not a launch. */
const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;
/** Exchange names a no-token record may start with (the snapshot's own watchlist rule). */
const EXCHANGE_START = /^(Binance|Coinbase|Bybit|Kraken|Bitget|OKX|Bitfinex|Gemini)\b/i;
/** Whole words that mark a wrapped, staked or bridged product. */
const WRAPPED_WORD = /\b(wrapped|staked|staking)\b/i;

/** Why a pre-token protocol is not a launch candidate, or null when it is one. */
export function noLaunchReason(p: SnapshotProtocol): string | null {
  if (p.category === 'CEX') return 'centralised exchange';
  if (p.category === 'Bridge' || p.category === 'Canonical Bridge') return 'bridge product';
  if (p.category === 'Liquid Staking' || p.category === 'Liquid Restaking' || p.category === 'Restaked BTC') return 'staked product of a token that trades';
  if (VERSIONED.test(p.name)) return 'versioned pool of another protocol';
  if (EXCHANGE_START.test(p.name)) return 'product of a named exchange';
  if (WRAPPED_WORD.test(p.name)) return 'wrapped or staked product';
  return null;
}

export type TokenStatus = 'launch' | 'live' | 'no-launch';

/** A protocol's standing: launch candidate, token already trading, or no launch to farm. */
export function tokenStatus(p: SnapshotProtocol): TokenStatus {
  if (!isPreToken(p)) return 'live';
  return noLaunchReason(p) === null ? 'launch' : 'no-launch';
}

// ------------------------------------------------------------------ ridge fit

/** Solve A x = b by Gauss-Jordan with partial pivoting (A small and dense). */
function solveLinear(A: number[][], b: number[]): number[] {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]!]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r]![c]!) > Math.abs(M[piv]![c]!)!) piv = r;
    const tmp = M[c]!;
    M[c] = M[piv]!;
    M[piv] = tmp;
    const pv = M[c]![c]!;
    if (pv === 0) throw new Error('ridge fit: singular matrix');
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r]![c]! / pv;
      if (f === 0) continue;
      for (let k = c; k <= n; k++) M[r]![k] = M[r]![k]! - f * M[c]![k]!;
    }
  }
  return M.map((row, i) => row[n]! / row[i]!);
}

/** Ridge regression: min ||Xw - y||^2 + lambda * ||w[1:]||^2. Intercept unpenalised. */
function ridgeFit(X: number[][], y: number[], lambda: number): number[] {
  const d = X[0]!.length;
  const A: number[][] = [];
  for (let i = 0; i < d; i++) {
    A.push([]);
    for (let j = 0; j < d; j++) {
      let s = 0;
      for (let k = 0; k < X.length; k++) s += X[k]![i]! * X[k]![j]!;
      if (i === j && i > 0) s += lambda;
      A[i]!.push(s);
    }
  }
  const b: number[] = [];
  for (let i = 0; i < d; i++) {
    let s = 0;
    for (let k = 0; k < X.length; k++) s += X[k]![i]! * y[k]!;
    b.push(s);
  }
  return solveLinear(A, b);
}

const dot = (w: number[], x: number[]): number => {
  let s = 0;
  for (let i = 0; i < w.length; i++) s += w[i]! * x[i]!;
  return s;
};

/** The fitted model: weights, category list, centres, and held-out validation. */
export interface FittedModel {
  weights: number[];
  categories: string[];
  meanLogTvl: number;
  medianAge: number;
  baseMultiple: number;
  /** log-mcap intercept restated as a multiple at median TVL, zero growth, median age. */
  logBase: number;
  trainN: number;
  naiveMae: number;
  lambda: number;
  loo: { r2: number; mae: number; q50: number; q90: number };
  walkForward: { r2: number; mae: number; trainN: number; testN: number; cut: string };
  categoryMultiples: { category: string; n: number; multiple: number }[];
}

/** Feature vector: [1, centred log TVL, clamped growth, log1p age, category dummies]. */
function features(p: SnapshotProtocol, m: { meanLogTvl: number; medianAge: number; categories: string[] }): number[] {
  const growth = Math.min(120, Math.max(-60, finite(p.change_7d, 0))) / 100;
  const age = ageDays(m.asOf, p.listed_at) ?? m.medianAge;
  const x = [1, Math.log(Math.max(1, p.tvl_usd)) - m.meanLogTvl, growth, Math.log1p(Math.max(0, age))];
  for (const c of m.categories) x.push(p.category === c ? 1 : 0);
  return x;
}

function median(values: number[]): number {
  const v = [...values].sort((a, b) => a - b);
  if (v.length === 0) return 0;
  const mid = Math.floor(v.length / 2);
  return v.length % 2 === 1 ? v[mid]! : (v[mid - 1]! + v[mid]!) / 2;
}

/** Fit the value model on the snapshot's own tokenized protocols, with held-out checks. */
export function fitModel(snapshot: Snapshot): FittedModel {
  const asOf = snapshot.as_of;
  const train = snapshot.protocols.filter((p) => p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0);
  const counts = new Map<string, number>();
  for (const p of train) if (p.category !== null) counts.set(p.category, (counts.get(p.category) ?? 0) + 1);
  const categories = [...counts.entries()]
    .filter(([, n]) => n >= MIN_CATEGORY_MEMBERS)
    .map(([c]) => c)
    .sort();
  const meanLogTvl = train.reduce((s, p) => s + Math.log(p.tvl_usd), 0) / train.length;
  const medianAge = median(train.map((p) => ageDays(asOf, p.listed_at)).filter((a): a is number => a !== null));
  const ctx = { meanLogTvl, medianAge, categories, asOf };
  const X = train.map((p) => features(p, ctx));
  const y = train.map((p) => Math.log(p.mcap_usd!));
  const yMean = y.reduce((s, v) => s + v, 0) / y.length;
  const ssTot = y.reduce((s, v) => s + (v - yMean) ** 2, 0);

  // Leave-one-out for each candidate strength; the best held-out R² wins.
  let best: { lambda: number; r2: number; weights: number[]; looErr: number[] } | null = null;
  for (const lambda of LAMBDAS) {
    const looErr: number[] = [];
    for (let i = 0; i < train.length; i++) {
      const w = ridgeFit([...X.slice(0, i), ...X.slice(i + 1)], [...y.slice(0, i), ...y.slice(i + 1)], lambda);
      looErr.push(y[i]! - dot(w, X[i]!));
    }
    const r2 = 1 - looErr.reduce((s, e) => s + e * e, 0) / ssTot;
    if (best === null || r2 > best.r2) {
      best = { lambda, r2, weights: ridgeFit(X, y, lambda), looErr };
    }
  }
  const lambda = best!.lambda;
  const weights = best!.weights;
  const looErr = best!.looErr;
  const absErr = looErr.map((e) => Math.abs(e)).sort((a, b) => a - b);
  const naiveMae = y.reduce((s, v) => s + Math.abs(v - median(y)), 0) / y.length;
  const loo = {
    r2: best!.r2,
    mae: looErr.reduce((s, e) => s + Math.abs(e), 0) / looErr.length,
    q50: absErr[Math.floor(0.5 * absErr.length)]!,
    q90: absErr[Math.min(absErr.length - 1, Math.floor(0.9 * absErr.length))]!,
  };

  // Walk-forward: train on the older half by listing date, test on the newer half.
  const dated = train.map((p, i) => ({ p, i })).filter(({ p }) => p.listed_at !== null).sort((a, b) => (a.p.listed_at! < b.p.listed_at! ? -1 : 1));
  const cut = dated[Math.floor(dated.length / 2)]!.p.listed_at!;
  const trIdx = dated.filter(({ p }) => p.listed_at! < cut).map(({ i }) => i);
  const teIdx = dated.filter(({ p }) => p.listed_at! >= cut).map(({ i }) => i);
  const wWalk = ridgeFit(trIdx.map((i) => X[i]!), trIdx.map((i) => y[i]!), lambda);
  const walkErr = teIdx.map((i) => y[i]! - dot(wWalk, X[i]!));
  const teMean = teIdx.reduce((s, i) => s + y[i]!, 0) / teIdx.length;
  const walkForward = {
    r2: 1 - walkErr.reduce((s, e) => s + e * e, 0) / teIdx.reduce((s, i) => s + (y[i]! - teMean) ** 2, 0),
    mae: walkErr.reduce((s, e) => s + Math.abs(e), 0) / walkErr.length,
    trainN: trIdx.length,
    testN: teIdx.length,
    cut,
  };

  // Base multiple at median TVL, zero growth, median age, no category effect.
  const logBase = weights[0]! - meanLogTvl + weights[3]! * Math.log1p(medianAge);
  const categoryMultiples = categories.map((category, j) => ({
    category,
    n: counts.get(category) ?? 0,
    multiple: Math.exp(weights[4 + j]!),
  }));

  return {
    weights,
    categories,
    meanLogTvl,
    medianAge,
    baseMultiple: Math.exp(logBase),
    logBase,
    trainN: train.length,
    naiveMae,
    lambda,
    loo,
    walkForward,
    categoryMultiples,
  };
}

/** Predicted log market cap for one protocol under the fitted model. */
export function predictLogMcap(p: SnapshotProtocol, model: FittedModel, asOf: string): number {
  return dot(model.weights, features(p, { meanLogTvl: model.meanLogTvl, medianAge: model.medianAge, categories: model.categories, asOf }));
}

/** Launch odds: the model's stated chance a pre-token protocol ever issues a token. */
export function launchOdds(age: number | null): number {
  const a = age ?? 365;
  const odds = 0.7 * Math.exp(-Math.max(0, a - 365) / 1500);
  return Math.max(0.02, Math.min(0.7, odds));
}

// ------------------------------------------------------------------ farm plans

/** Capital one farm position needs, from the protocol's own TVL. */
function capitalFor(p: SnapshotProtocol): number {
  if (p.tvl_usd >= 5_000_000_000) return 250;
  if (p.tvl_usd >= 1_000_000_000) return 150;
  if (p.tvl_usd >= 100_000_000) return 100;
  return 50;
}

/** Hold days from the protocol's age: younger protocols need a longer window. */
function holdDaysFor(age: number | null): number {
  if (age === null) return 30;
  if (age < 180) return 60;
  if (age < 365) return 45;
  return 30;
}

function chainOf(p: SnapshotProtocol): string {
  return p.chains.length > 0 ? p.chains[0]! : 'its main chain';
}

/** The category's main farming action, in words a farmer can follow. */
function actionFor(p: SnapshotProtocol, capital: number, chain: string): { main: string; weekly: string } {
  const c = (p.category ?? '').toLowerCase();
  const name = p.name;
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) {
    return {
      main: `Supply $${capital} of USDC to ${name} on ${chain} and keep the loan position open.`,
      weekly: `Adjust the supplied amount or borrow a small sum against it once a week.`,
    };
  }
  if (c.includes('dex') || c.includes('leveraged farm')) {
    return {
      main: `Provide $${capital} of liquidity to ${name}'s main pool on ${chain} and keep the position open.`,
      weekly: `Swap a small amount through the pool once a week to show repeated use.`,
    };
  }
  if (c.includes('bridge') || c.includes('canonical')) {
    return {
      main: `Bridge $${capital} of USDC to ${chain} through ${name} and use the bridged funds in its app.`,
      weekly: `Bridge a small amount back or onward once a week.`,
    };
  }
  if (c.includes('stak') || c.includes('restak')) {
    return {
      main: `Stake $${capital} of the chain's staking asset through ${name} and keep the receipt token.`,
      weekly: `Restake the rewards or add a small amount once a week.`,
    };
  }
  if (c.includes('perpetual') || c.includes('derivative') || c.includes('options')) {
    return {
      main: `Deposit $${capital} of USDC as margin on ${name} and open one small hedged trade.`,
      weekly: `Place one small trade a week and close it the same day.`,
    };
  }
  if (c.includes('prediction')) {
    return {
      main: `Fund $${capital} and place one market position on ${name} on ${chain}.`,
      weekly: `Place one small position a week on a market you follow.`,
    };
  }
  if (c.includes('yield') || c.includes('aggregator') || c.includes('vault')) {
    return {
      main: `Deposit $${capital} into a ${name} vault on ${chain} and keep the vault shares.`,
      weekly: `Move a small amount between the vault's strategies once a week.`,
    };
  }
  if (c.includes('rwa') || c.includes('real world')) {
    return {
      main: `Mint $${capital} of the protocol's yield-bearing asset on ${chain} and hold it.`,
      weekly: `Check the asset's yield accrual and add a small amount once a week.`,
    };
  }
  if (c.includes('stable')) {
    return {
      main: `Mint or deposit $${capital} of the stable asset through ${name} on ${chain}.`,
      weekly: `Mint or redeem a small amount once a week.`,
    };
  }
  if (c.includes('insurance')) {
    return {
      main: `Buy $${capital} of cover from ${name} on ${chain} and keep the policy active.`,
      weekly: `Check the policy and the covered pools once a week.`,
    };
  }
  if (c.includes('curator') || c.includes('allocator') || c.includes('onchain capital')) {
    return {
      main: `Deposit $${capital} into a ${name} vault on ${chain} and keep the shares.`,
      weekly: `Review the vault's allocation and add a small amount once a week.`,
    };
  }
  if (c.includes('basis')) {
    return {
      main: `Deposit $${capital} into ${name}'s basis strategy on ${chain} and keep the position.`,
      weekly: `Check the funding-rate return and rebalance a small amount once a week.`,
    };
  }
  if (c.includes('launchpad')) {
    return {
      main: `Stake $${capital} through ${name} on ${chain} to qualify for its launches and keep the stake.`,
      weekly: `Join one launch or vote once a week while the stake is held.`,
    };
  }
  return {
    main: `Deposit $${capital} of USDC into ${name} on ${chain} and keep the position open.`,
    weekly: `Make one small deposit, swap or vote in the app once a week.`,
  };
}

export interface FarmStep {
  text: string;
  cost: string;
  time: string;
}

/** One step as a single string: action, cost, time — within the gate's 20..600 chars. */
export function stepText(step: FarmStep): string {
  let s = `${step.text} Cost: ${step.cost}. Time: ${step.time}.`;
  if (s.length > 600) s = `${s.slice(0, 596)}...`;
  if (s.length < 20) s = `${s} Hold the position and check the protocol's own page.`;
  return s;
}

/** A protocol's farm plan: six steps, each with what it costs and how long it takes. */
export function farmSteps(p: SnapshotProtocol, age: number | null, status: TokenStatus): FarmStep[] {
  const chain = chainOf(p);
  const capital = capitalFor(p);
  const hold = holdDaysFor(age);
  const gas = capital >= 150 ? 8 : 4;
  const { main, weekly } = actionFor(p, capital, chain);
  const reason = status === 'no-launch' ? noLaunchReason(p) ?? 'not a launch candidate' : '';
  if (status === 'no-launch') {
    return [
      { text: `Do not farm ${p.name} for a token: it records no token because it is a ${reason}, not a protocol about to launch one.`, cost: 'nothing', time: '5 minutes to check its own page' },
      { text: `Use ${p.name} only for what it is: on ${chain}, use it the way its own docs describe, with money you can afford to move.`, cost: `$${capital} at most, only if you would use it anyway`, time: '15 minutes to set up' },
      { text: `Never send funds to anyone promising a ${p.name} token or airdrop: a ${reason} has no launch to claim.`, cost: 'nothing, and it saves the funds a scam would take', time: '2 minutes' },
      { text: `If you still want exposure to this kind of product, pick a launch candidate from this board instead and farm that one.`, cost: 'nothing to compare', time: '10 minutes' },
      { text: `Check ${p.name}'s own announcements once a month in case its status changes.`, cost: 'nothing', time: '5 minutes a month' },
      { text: `Write the wallet, the dates and every transaction in a farm log so you can repeat the position or unwind it.`, cost: 'nothing but the log', time: '5 minutes' },
    ];
  }
  const tradeNote = status === 'live' ? ` Its token already trades, so farm for points, fees or future programs, not for a launch.` : '';
  return [
    { text: `Use a fresh wallet that holds only the funds you plan to farm, and keep its seed phrase offline.`, cost: 'under $3 of gas for the first transaction', time: '5 minutes' },
    { text: `${main}${tradeNote}`, cost: `$${capital} of capital and about $${gas} of gas`, time: '15 minutes to set up' },
    { text: `${weekly}`, cost: 'about $2 of gas a week', time: '10 minutes a week' },
    { text: `Hold the position for at least ${hold} days so the activity is counted in more than one snapshot.`, cost: `the $${capital} stays in the position`, time: `${hold} days` },
    { text: `Before you withdraw, open the protocol's own points or rewards page and claim only from the link it publishes.`, cost: 'no fee to check, and never pay to claim', time: '10 minutes' },
    { text: `Write the wallet, the dates and every transaction in a farm log so you can repeat the position or unwind it.`, cost: 'nothing but the log', time: '5 minutes' },
  ];
}

// ------------------------------------------------------------------ the board

/** One row of board.json. */
export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
  token_status: TokenStatus;
  age_days: number | null;
  implied_value_usd: number;
  value_low_usd: number;
  value_high_usd: number;
  launch_odds: number;
  value_basis: 'fitted' | 'market-cap' | 'fitted-no-mcap';
  model: {
    multiple: number;
    size_multiplier: number;
    growth_multiplier: number;
    age_multiplier: number;
    category_multiplier: number;
  };
  farm_steps: string[];
}

/** The dashboard's compact view of one row. */
export interface RadarRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  age_days: number | null;
  implied_value_usd: number;
  value_low_usd: number;
  value_high_usd: number;
  launch_odds: number;
  token_status: TokenStatus;
  value_basis: 'fitted' | 'market-cap' | 'fitted-no-mcap';
  model: {
    multiple: number;
    size_multiplier: number;
    growth_multiplier: number;
    age_multiplier: number;
    category_multiplier: number;
  };
  farm_steps: string[];
}

/** A live-token protocol for the dashboard's Live view. */
export interface LiveRow {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  mcap_usd: number | null;
  change_7d: number | null;
}

/** Everything the dashboard renders. */
export interface RadarData {
  as_of: string;
  source: string;
  floor_usd: number;
  model_version: string;
  watchlist_size: number;
  board_size: number;
  launch_size: number;
  live_size: number;
  no_launch_size: number;
  fit: {
    train_n: number;
    lambda: number;
    naive_mae: number;
    loo_r2: number;
    loo_mae: number;
    loo_q50: number;
    loo_q90: number;
    walk_r2: number;
    walk_mae: number;
    walk_train_n: number;
    walk_test_n: number;
    walk_cut: string;
    base_multiple: number;
    growth_coef: number;
    size_coef: number;
    age_coef: number;
  };
  category_multiples: { category: string; n: number; multiple: number }[];
  rows: RadarRow[];
  live: LiveRow[];
}

/** The snapshot out of a file's text, refused unless it is the shape the model scores. */
export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new SnapshotError('snapshot.json is not JSON');
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new SnapshotError('snapshot.json is not a JSON object');
  const o = raw as Record<string, unknown>;
  if (typeof o.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(o.as_of)) throw new SnapshotError('snapshot.json has no as_of date');
  if (typeof o.source !== 'string') throw new SnapshotError('snapshot.json has no source');
  if (typeof o.floor_usd !== 'number' || !Number.isSafeInteger(o.floor_usd) || o.floor_usd <= 0) throw new SnapshotError('snapshot.json has no floor_usd');
  if (!Array.isArray(o.protocols)) throw new SnapshotError('snapshot.json has no protocols array');
  if (!Array.isArray(o.watchlist) || o.watchlist.some((s) => typeof s !== 'string')) throw new SnapshotError('snapshot.json has no watchlist of slugs');
  const protocols = o.protocols.map((v, i) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new SnapshotError(`snapshot.json's protocol ${i} is not an object`);
    const p = v as Record<string, unknown>;
    if (typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number' || !Number.isFinite(p.tvl_usd)) {
      throw new SnapshotError(`snapshot.json's protocol ${i} lacks a slug, a name or a TVL`);
    }
    return {
      slug: p.slug,
      name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
    };
  });
  return { as_of: o.as_of, source: o.source, floor_usd: o.floor_usd, protocols, watchlist: o.watchlist as string[] };
}

const round4 = (n: number): number => Math.round(n * 1e4) / 1e4;

/** The whole board: every protocol in the snapshot, ranked by value, strongest first. */
export function buildBoard(snapshot: Snapshot, model: FittedModel): BoardRow[] {
  const w = model.weights;
  const scored = snapshot.protocols.map((p) => {
    const age = ageDays(snapshot.as_of, p.listed_at);
    const status = tokenStatus(p);
    const growth = Math.min(120, Math.max(-60, finite(p.change_7d, 0))) / 100;
    const ageForModel = age ?? model.medianAge;
    // The multiple is exp(predicted log mcap) / TVL, so the size term carries the
    // elasticity minus one: a bigger protocol's multiple falls with its TVL.
    const sizeMult = Math.exp((w[1]! - 1) * (Math.log(Math.max(1, p.tvl_usd)) - model.meanLogTvl));
    const growthMult = Math.exp(w[2]! * growth);
    const ageMult = Math.exp(w[3]! * (Math.log1p(Math.max(0, ageForModel)) - Math.log1p(model.medianAge)));
    const catIdx = model.categories.indexOf(p.category ?? '');
    const catMult = catIdx >= 0 ? Math.exp(w[4 + catIdx]!) : 1;
    const multiple = model.baseMultiple * sizeMult * growthMult * ageMult * catMult;
    const fitted = Math.max(1, Math.round(p.tvl_usd * multiple));
    let value: number;
    let basis: BoardRow['value_basis'];
    if (status === 'live' && p.mcap_usd !== null && p.mcap_usd > 0) {
      value = Math.round(p.mcap_usd);
      basis = 'market-cap';
    } else {
      value = fitted;
      basis = status === 'live' ? 'fitted-no-mcap' : 'fitted';
    }
    const half = model.loo.q50;
    return {
      p,
      age,
      status,
      value,
      basis,
      multiple: round4(multiple),
      sizeMult: round4(sizeMult),
      growthMult: round4(growthMult),
      ageMult: round4(ageMult),
      catMult: round4(catMult),
      low: Math.max(1, Math.round(value * Math.exp(-half))),
      high: Math.max(1, Math.round(value * Math.exp(half))),
    };
  });
  scored.sort((a, b) => b.value - a.value || (a.p.slug < b.p.slug ? -1 : a.p.slug > b.p.slug ? 1 : 0));
  return scored.map((s, i) => ({
    rank: i + 1,
    slug: s.p.slug,
    name: s.p.name,
    category: s.p.category,
    chains: s.p.chains,
    tvl_usd: s.p.tvl_usd,
    change_7d: s.p.change_7d,
    mcap_usd: s.p.mcap_usd,
    listed_at: s.p.listed_at,
    gecko_id: s.p.gecko_id,
    symbol: s.p.symbol,
    token_status: s.status,
    age_days: s.age,
    implied_value_usd: s.value,
    value_low_usd: s.low,
    value_high_usd: s.high,
    launch_odds: s.status === 'launch' ? round4(launchOdds(s.age)) : 0,
    value_basis: s.basis,
    model: {
      multiple: s.multiple,
      size_multiplier: s.sizeMult,
      growth_multiplier: s.growthMult,
      age_multiplier: s.ageMult,
      category_multiplier: s.catMult,
    },
    farm_steps: farmSteps(s.p, s.age, s.status).map(stepText),
  }));
}

/** Everything the dashboard renders, from the snapshot and the fitted model. */
export function buildRadar(snapshot: Snapshot): { board: BoardRow[]; radar: RadarData; model: FittedModel } {
  const model = fitModel(snapshot);
  const board = buildBoard(snapshot, model);
  const live = snapshot.protocols
    .filter((p) => tokenStatus(p) === 'live')
    .sort((a, b) => (b.mcap_usd ?? 0) - (a.mcap_usd ?? 0) || b.tvl_usd - a.tvl_usd)
    .map((p) => ({ slug: p.slug, name: p.name, category: p.category, chains: p.chains, tvl_usd: p.tvl_usd, mcap_usd: p.mcap_usd, change_7d: p.change_7d }));
  const radar: RadarData = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    model_version: 'radar-2-fitted',
    watchlist_size: snapshot.watchlist.length,
    board_size: board.length,
    launch_size: board.filter((r) => r.token_status === 'launch').length,
    live_size: board.filter((r) => r.token_status === 'live').length,
    no_launch_size: board.filter((r) => r.token_status === 'no-launch').length,
    fit: {
      train_n: model.trainN,
      lambda: model.lambda,
      naive_mae: round4(model.naiveMae),
      loo_r2: round4(model.loo.r2),
      loo_mae: round4(model.loo.mae),
      loo_q50: round4(model.loo.q50),
      loo_q90: round4(model.loo.q90),
      walk_r2: round4(model.walkForward.r2),
      walk_mae: round4(model.walkForward.mae),
      walk_train_n: model.walkForward.trainN,
      walk_test_n: model.walkForward.testN,
      walk_cut: model.walkForward.cut,
      base_multiple: round4(model.baseMultiple),
      growth_coef: round4(model.weights[2]!),
      size_coef: round4(model.weights[1]!),
      age_coef: round4(model.weights[3]!),
    },
    category_multiples: model.categoryMultiples.map((c) => ({ category: c.category, n: c.n, multiple: round4(c.multiple) })),
    rows: board.map((r) => ({
      rank: r.rank,
      slug: r.slug,
      name: r.name,
      category: r.category,
      chains: r.chains,
      tvl_usd: r.tvl_usd,
      change_7d: r.change_7d,
      mcap_usd: r.mcap_usd,
      listed_at: r.listed_at,
      age_days: r.age_days,
      implied_value_usd: r.implied_value_usd,
      value_low_usd: r.value_low_usd,
      value_high_usd: r.value_high_usd,
      launch_odds: r.launch_odds,
      token_status: r.token_status,
      value_basis: r.value_basis,
      model: r.model,
      farm_steps: r.farm_steps,
    })),
    live,
  };
  return { board, radar, model };
}

/** Why a board is not one the gate will take, or null. */
export function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  const slugs = new Set(board.map((r) => r.slug));
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slug${missing.length === 1 ? '' : 's'}: ${missing.slice(0, 5).join(', ')}`;
  const ranks = board.map((r) => r.rank).sort((a, b) => a - b);
  for (let i = 0; i < ranks.length; i++) if (ranks[i] !== i + 1) return `rank ${i + 1} is ${ranks[i]}`;
  for (let i = 1; i < board.length; i++) if (board[i]!.implied_value_usd > board[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i} on implied value`;
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) {
    return `all ${board.length} rows share the same implied value $${board[0]!.implied_value_usd}: a board that ranks nothing by value is not a ranking`;
  }
  for (const r of board) {
    if (r.farm_steps.length < 3 || r.farm_steps.length > 12) return `${r.slug} has ${r.farm_steps.length} farm steps`;
    for (const s of r.farm_steps) if (s.trim().length < 20 || s.trim().length > 600) return `${r.slug} has a farm step of ${s.trim().length} characters`;
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0) return `${r.slug} has implied value ${r.implied_value_usd}`;
  }
  return null;
}

/** A compact data.js: the dashboard's whole dataset as one script, no fetch, no network. */
export function dataScript(radar: RadarData): string {
  return `window.RADAR=${JSON.stringify(radar)};\n`;
}
