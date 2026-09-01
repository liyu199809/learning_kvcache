"""FastAPI entry point for the DeepCoder/TACO CodeJudge service."""

import uvicorn
from fastapi.responses import JSONResponse

from openenv.core.env_server.http_server import create_app

from ..models import CodeJudgeAction, CodeJudgeObservation
from .code_judge_environment import CodeJudgeEnvironment
from .config import DATA_DIR, MAX_CONCURRENT_ENVS
from .data_loader import TacoDataLoader


_shared_data_loader = TacoDataLoader(DATA_DIR)


def _env_factory() -> CodeJudgeEnvironment:
    return CodeJudgeEnvironment(data_loader=_shared_data_loader)


app = create_app(
    _env_factory,
    CodeJudgeAction,
    CodeJudgeObservation,
    env_name="code_judge_env",
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
)


# A trajectory must keep the same task binding until final verification.
app.routes[:] = [
    route for route in app.routes if getattr(route, "path", None) not in ("/reset", "/step")
]


@app.post("/reset", tags=["disabled"])
async def reset_not_supported():
    return JSONResponse(
        status_code=400,
        content={"error": "HTTP mode is not supported; use the OpenEnv WebSocket client"},
    )


@app.post("/step", tags=["disabled"])
async def step_not_supported():
    return JSONResponse(
        status_code=400,
        content={"error": "HTTP mode is not supported; use the OpenEnv WebSocket client"},
    )


@app.get("/dataset", tags=["monitoring"])
async def dataset_stats():
    try:
        return JSONResponse(content={"status": "ok", **_shared_data_loader.stats()})
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": f"{type(exc).__name__}: {exc}"},
        )


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8901)


if __name__ == "__main__":
    main()
