"""Fast GSM8K smoke — empty Daytona sandbox, no coding seed (no add(a,b)).

Wall-clock target ~5 min when the 3B checkpoint is already on the pod
(model download/convert dominates first boot).
"""

from daytona_gym import (
    HuggingFaceDataset,
    Qwen25_3B,
    Qwen25_3B_Recipe,
    TrainConfig,
    runpod_worker,
)
from daytona_gym.envfile import load_dotenv

load_dotenv()

config = TrainConfig(
    model=Qwen25_3B(),
    dataset=HuggingFaceDataset(
        hf_repo="openai/gsm8k",
        hf_config="main",
        hf_split="train[:4]",
        input_column="question",
        output_column="answer",
        prompt_template="Solve step by step:\n{input}",
    ),
    recipe=Qwen25_3B_Recipe(
        # No broken.py / add(a,b) — empty sandbox only.
        seed_coding=False,
        bootstrap_run_tests=False,
        require_passing_tests=False,
        generate_path="daytona_gym.adapters.slime.generate.generate",
        # Tiny train so this can finish in ~5 min once Megatron is warm.
        batch_size=1,
        n_samples=1,
        num_rollout=1,
        max_turns=2,
        max_response_len=256,
        timeout_seconds=90,
        tool_timeout_seconds=30,
    ),
)

if __name__ == "__main__":
    run = config.launch(worker=runpod_worker(), detach=True)
    print(run.training_run_id)
    print(run.dashboard_url)
    run.result()
