from fastapi import FastAPI
from contextlib import asynccontextmanager

from .neuro import Neuro


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.neuro = Neuro()
    yield


APP = FastAPI(lifespan=lifespan)
