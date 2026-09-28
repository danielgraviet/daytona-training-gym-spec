"""Vast.ai on-box smoke — identical recipe to ``box_worker.py``.

On the rented 4090 (Cuda 13 template → edit image to daytona-gym-worker)::

    docker pull docker.io/dtgraviet/daytona-gym-worker:latest
    docker run --gpus all --rm -i --ipc=host \\
      -v daytona-gym-models:/models --env-file ~/.daytona-gym.env \\
      docker.io/dtgraviet/daytona-gym-worker:latest \\
      python - < examples/gym_sdk/vast_worker.py

Laptop SSH (optional)::

    from daytona_gym import TrainConfig, vast_worker
    # set VAST_INSTANCE_ID + VAST_API_KEY; then:
    # config.launch(worker=vast_worker(), detach=True)
"""

import runpy
from pathlib import Path

# Keep a single source of truth for the 24 GB card recipe.
_BOX = Path(__file__).with_name("box_worker.py")

if __name__ == "__main__":
    runpy.run_path(str(_BOX), run_name="__main__")
