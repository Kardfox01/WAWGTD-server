import base64
import io
import json
import os
import time
from typing import Optional
from PIL import Image
import cv2
import numpy as np
from ollama import chat, ChatResponse
import requests
import torch

from . import prms
from .logger import LOG


Img = Image.Image

def from_base64(base64_code: str):
    img_data = base64.b64decode(base64_code)
    return Image.open(io.BytesIO(img_data))

def to_base64(img: Img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class Neuro:
    def __init__(self):
        self.load_dpt()
        self.ollama_warmup()

    @LOG("ПРОГРЕВ OLLAMA")
    def ollama_warmup(self):
        try:
            response = requests.post(
                prms.OLLAMA_API,
                json={
                    "model": prms.OLLAMA_MODEL,
                    "prompt": "",
                    "keep_alive": -1
                },
                timeout=10
            )
            if response.status_code != 200:
                raise ConnectionError("ОШИБКА ПОДКЛЮЧЕНИЯ К OLLAMA")
        except Exception as _:
            raise ConnectionError("ОШИБКА ПОДКЛЮЧЕНИЯ К OLLAMA")

    @LOG("ЗАГРУЗКА DPT")
    def load_dpt(self):
        self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", verbose=False)
        self.midas.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas.to(self.device)
        self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms", verbose=False)

    @LOG("МАРКИРОВКА ИЗОБРАЖЕНИЯ ПО ГЛУБИНЕ")
    def depth_marked(self, pil_img: Img) -> tuple[Img, Img]:
        transform = self.transforms.dpt_transform
        img = np.array(pil_img)
        input_tensor = transform(img).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_tensor)
            # Интерполируем карту глубины к размеру изображения
            h, w = img.shape[:2]
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bilinear",
                align_corners=False
            ).squeeze().cpu().numpy()

        depth_min = prediction.min()
        depth_max = prediction.max()
        depth_vis = ((prediction - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

        _, bright_mask = cv2.threshold(depth_vis, prms.THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_with_contours = img.copy()
        cv2.drawContours(img_with_contours, contours, -1, (255, 0, 0), prms.CONTOURS_THICKNESS)

        binary_mask = (depth_vis > prms.THRESHOLD_VALUE).astype(np.uint8)
        dark_image = (img * 0.2).astype(np.uint8)
        binary_mask_3ch = cv2.merge([binary_mask]*3)
        image_dark_bg = np.where(binary_mask_3ch == 1, img, dark_image)
        cv2.drawContours(image_dark_bg, contours, -1, prms.CONTOURS_COLOR, prms.CONTOURS_THICKNESS - 5)

        return Image.fromarray(depth_vis), Image.fromarray(img_with_contours)

    @LOG("ОПИСАНИЕ OLLAMA")
    def ollama_json(self, base64_code: str) -> Optional[dict[str, str]]:
        # response: ChatResponse = chat(model=prms.OLLAMA_MODEL, messages=[
        #     {
        #         "role": "system",
        #         "content": prms.OLLAMA_SYSTEM_PROMPT
        #     },
        #     {
        #         "role": "user",
        #         "content": prms.OLLAMA_USER_PROMPT,
        #         "images": [base64_code]
        #     }
        # ])

        # data = response.message.content

        data = """
        [
            {
                "species": "Хвойное",
                "trunk_rot": false,
                "hollows": false,
                "cracks": false,
                "trunk_damage": false,
                "crown_damage": false,
                "fruiting_bodies": false,
                "diseases": [],
                "dry_branches_percent": 5.0,
                "other": null,
                "description": "Здоровое хвойное дерево.",
                "tree_bounding_boxes": [[40, 100, 200, 600]]
            },
            {
                "species": "Лиственное",
                "trunk_rot": true,
                "hollows": false,
                "cracks": true,
                "trunk_damage": true,
                "crown_damage": true,
                "fruiting_bodies": false,
                "diseases": ["canker"],
                "dry_branches_percent": 60.0,
                "other": "Сильно повреждённая крона.",
                "description": "Лиственное дерево с гнилью ствола, трещинами и усохшей кроной.",
                "tree_bounding_boxes": [[250, 120, 420, 650]]
            }
        ]"""

        time.sleep(2)

        if data != None:
            return json.loads(data.replace("```json", "").replace("```", ""))

        return None
