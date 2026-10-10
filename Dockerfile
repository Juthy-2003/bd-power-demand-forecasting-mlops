# Application image: forecast API and the pipeline CLI (`powerfc`, used by the Kubernetes Jobs).
# Built for linux/amd64 and linux/arm64 (the Oracle A1 node is ARM).
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/venv
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 1000 --create-home app
COPY --from=build /venv /venv
COPY --from=build /app/src /app/src
ENV PATH=/venv/bin:$PATH PYTHONUNBUFFERED=1 POWERFC_HOME=/app POWERFC_DATA=/data MLFLOW_DISABLE_AGENT_HINT=1
WORKDIR /app
USER app
EXPOSE 8000
CMD ["uvicorn", "powerfc.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
