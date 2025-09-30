import os
from fastapi import HTTPException
from fastapi.responses import FileResponse

from . import APP, prms
from .neuro import Neuro, to_base64, from_base64
from .models import*


@APP.post("/analyze", response_model=OutputData)
async def analyze_trees(data: InputData):
    neuro: Neuro = APP.state.neuro

    trees: list[TreeInfo] = []

    img = from_base64(data.img_base64)
    img_user, img_ollama = neuro.depth_marked(img)

    json_description = neuro.ollama_json(to_base64(img_ollama))
    if json_description:
        for tree in json_description:
            trees.append(TreeInfo(**tree)) # type: ignore
        return OutputData(img_marked_base64=to_base64(img_user), trees=trees)

    raise HTTPException(
        status_code=500,
        detail="Ollama error"
    )


@APP.get("/tutorial/{step}")
def get_tutorial_image(step: int):
    file_path = os.path.join(prms.TUTORIAL_DIR, f"tutorial{step}.png")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(file_path, media_type="image/png")
