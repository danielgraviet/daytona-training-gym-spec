"""Multi-step analytics check on a BYO GPU (run ON the pod, not the laptop).

On a fresh ``slimerl/slime:latest`` pod::

    cd /root && git clone https://github.com/danielgraviet/daytona-training-gym-spec.git
    cd daytona-training-gym-spec && pip install -e .
    export DAYTONA_API_KEY=...          # or put it in .env (auto-loaded)
    python main.py                      # use the image's python, not `uv run`

Then::

    dg stats runs/<run_id>.jsonl        # "where time went" + per-step table
    # dashboard URL is printed at launch (Cloudflare tunnel)

What "working" looks like:
  * the per-step table has 4 rows (steps 0-3), not 1
  * every rollout shows training_step_source=derived
  * GPU util samples land inside the timeline (5s sampler)
  * idle-GPU $ is populated (gpu_cost_per_hour below)
"""

from daytona_gym import PromptJsonlDataset, Qwen25_3B, Qwen25_3B_Recipe, TrainConfig
from daytona_gym.envfile import load_dotenv

load_dotenv()

config = TrainConfig(
    model=Qwen25_3B(),
    dataset=PromptJsonlDataset("examples/coding_dogfood/prompts/coding_pack.jsonl"),
    recipe=Qwen25_3B_Recipe(
        batch_size=4,  # prompts per step
        n_samples=2,  # rollouts per prompt → 8 concurrent sandboxes per step
        num_rollout=4,  # 4 training steps → 4 rows in the step table
        max_turns=6,
        timeout_seconds=180,
        tool_timeout_seconds=60,
    ),
    # Rough RunPod A100 80GB on-demand rate; adjust to what you actually pay.
    gpu_cost_per_hour=1.64,
)

if __name__ == "__main__":
    run = config.launch(open=True)
    print(run.training_run_id)
    print(run.dashboard_url or run.inspect_hint)
    if run.dashboard_url:
        run.wait_dashboard()  # keep the tunnel up; Ctrl+C when done
