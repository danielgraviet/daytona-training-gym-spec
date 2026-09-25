async function getJson(path) {
  const res = await fetch(path, { cache: 'no-store' });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json();
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
