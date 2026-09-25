export const OFFLINE_MESSAGE =
  'Dashboard server is offline (the tunnel or `dg dash` was stopped). ' +
  'Showing the last data received; this page reconnects automatically.';

export class OfflineError extends Error {
  constructor() {
    super(OFFLINE_MESSAGE);
    this.offline = true;
  }
}

// 502-504 / Cloudflare 52x-530 mean "origin unreachable", not an API error.
const OFFLINE_STATUSES = new Set([502, 503, 504, 520, 521, 522, 523, 524, 525, 526, 527, 530]);

async function getJson(path) {
  let res;
  try {
    res = await fetch(path, { cache: 'no-store' });
  } catch {
    throw new OfflineError();
  }
  const type = res.headers.get('content-type') || '';
  if (OFFLINE_STATUSES.has(res.status) || (!type.includes('json') && type.includes('html'))) {
    throw new OfflineError();
  }
  if (!res.ok) {
    const text = (await res.text()).replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
    throw new Error(`${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json();
}

/** setTimeout loop: ``fn`` returns the next delay in ms. Backs off while offline. */
export function poll(fn, { interval, maxInterval = 30000 }) {
  let stopped = false;
  let timer = null;
  let delay = interval;
  async function tick() {
    let next = interval;
    try {
      next = (await fn()) ?? interval;
      delay = interval;
    } catch (e) {
      delay = e?.offline ? Math.min(maxInterval, delay * 2) : interval;
      next = delay;
    }
    if (!stopped) timer = setTimeout(tick, next);
  }
  tick();
  return () => {
    stopped = true;
    clearTimeout(timer);
  };
}

export function fetchOverview() {
  return getJson('/api/overview');
}

export function fetchRuns() {
  return getJson('/api/runs');
}

export function fetchRun(stem) {
  return getJson(`/api/runs/${encodeURIComponent(stem)}`);
}

export function fetchCharts(stem) {
  return getJson(`/api/runs/${encodeURIComponent(stem)}/charts`);
}

export function fetchAnalysis(stem) {
  return getJson(`/api/runs/${encodeURIComponent(stem)}/analysis`);
}

export function fetchLive(stem) {
  return getJson(`/api/runs/${encodeURIComponent(stem)}/live`);
}

export function fetchRollout(stem, rolloutId) {
  return getJson(
    `/api/runs/${encodeURIComponent(stem)}/rollouts/${encodeURIComponent(rolloutId)}`,
  );
}

export function parseRoute() {
  const path = window.location.pathname.replace(/\/+$/, '') || '/';
  const parts = path.split('/').filter(Boolean);
  if (parts.length === 0) return { page: 'list' };
  if (parts[0] === 'run' && parts.length === 2) {
    return { page: 'run', stem: decodeURIComponent(parts[1]) };
  }
  if (parts[0] === 'run' && parts.length === 4 && parts[2] === 'rollout') {
    return {
      page: 'rollout',
      stem: decodeURIComponent(parts[1]),
      rolloutId: decodeURIComponent(parts[3]),
    };
  }
  return { page: 'list' };
}

export function navigate(to) {
  if (window.location.pathname !== to) {
    window.history.pushState({}, '', to);
    window.dispatchEvent(new PopStateEvent('popstate'));
  }
}
