# Daytona Gym worker image: Slime + Megatron + SGLang (slimerl/slime) + daytona_gym.
#
#   docker run --gpus all --ipc=host --shm-size=16g \
#     -v ~/gym-models:/models -v ~/gym-work:/workspace --env-file ~/.daytona-gym.env \
#     ghcr.io/danielgraviet/daytona-gym-worker:latest python - < examples/gym_sdk/box_worker.py
#
# The base is pinned by digest; bump it deliberately. It requires a host NVIDIA
# driver that supports CUDA >= 12.9 (NVIDIA_REQUIRE_CUDA in the base image).
ARG SLIME_IMAGE=slimerl/slime@sha256:0589e3b6415e79bc43f14ab5bb5fb8a0b89f50b1d31efb2ea76856498ffcbbeb

# Build the wheel in a small image so the 20+ GB base only runs one pip install.
FROM python:3.12-slim AS wheel
WORKDIR /src
COPY pyproject.toml README.md ./
COPY daytona_gym ./daytona_gym
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

FROM ${SLIME_IMAGE}
# Slime's adapter code runs inside Slime's own process, so daytona_gym must be
# installed into the same Python environment (not a separate venv).
COPY --from=wheel /wheels /tmp/wheels
RUN python3 -m pip install --no-cache-dir /tmp/wheels/*.whl \
 && rm -rf /tmp/wheels \
 && python3 -c "import daytona_gym, daytona; print('daytona_gym', daytona_gym.__version__)"

# Paths the gym resolves on the worker (see daytona_gym/gym/models.py, compute.py).
ENV DAYTONA_GYM_WORKDIR=/workspace \
    DAYTONA_GYM_MODELS_DIR=/models \
    SLIME_ROOT=/root/slime \
    MEGATRON_ROOT=/root/Megatron-LM
WORKDIR /workspace
LABEL org.opencontainers.image.source="https://github.com/danielgraviet/daytona-training-gym-spec" \
      org.opencontainers.image.description="Daytona Gym BYO-GPU worker (Slime + Daytona sandboxes)"
CMD ["bash"]
