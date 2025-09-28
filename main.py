import logging
import os
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from models import*
from neuro import*


app = FastAPI()
LOG = logging.getLogger("uvicorn.info")
LOG.info("ЗАПУСК МОДУЛЯ NEURO...")
neuro = Neuro(LOG)
LOG.info("МОДУЛЬ NEURO ЗАПУЩЕН")


@app.post("/analyze", response_model=OutputData)
async def analyze_trees(data: InputData):
    # ОБРАЩЕНИЕ К YOLO и LLM
    trees: list[TreeInfo] = []

    img = from_base64(data.img_base64)
    img_user, img_ollama = neuro.depth_marked(img)
    LOG.info("МАРКИРОВКА ЗАВЕРШЕНА УСПЕШНО")

    json_description = [] # neuro.ollama_json(to_base64(img_ollama))
    LOG.info("ОПИСАНИЕ ПОЛУЧЕНО УСПЕШНО")
    if json_description:
        for tree in json_description:
            trees.append(TreeInfo(**tree)) # type: ignore
    return OutputData(img_marked_base64=to_base64(img_user), trees=trees)

    raise HTTPException(
        status_code=500,
        detail="Ollama error"
    )


@app.get("/tutorial/{step}")
def get_tutorial_image(step: int):
    file_path = os.path.join(prms.TUTORIAL_DIR, f"tutorial{step}.png")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(file_path, media_type="image/png")
