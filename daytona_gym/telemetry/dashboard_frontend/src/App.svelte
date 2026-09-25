<script>
  import { onMount } from 'svelte';
  import { parseRoute } from './lib/api.js';
  import RunsList from './pages/RunsList.svelte';
  import RunDetail from './pages/RunDetail.svelte';
  import RolloutDetail from './pages/RolloutDetail.svelte';

  let route = $state(parseRoute());

  onMount(() => {
    const onNav = () => {
      route = parseRoute();
    };
    window.addEventListener('popstate', onNav);
    return () => window.removeEventListener('popstate', onNav);
  });
</script>

<main class="mx-auto max-w-5xl px-4 py-8">
  {#if route.page === 'run'}
    <RunDetail stem={route.stem} />
  {:else if route.page === 'rollout'}
    <RolloutDetail stem={route.stem} rolloutId={route.rolloutId} />
  {:else}
    <RunsList />
  {/if}
</main>
