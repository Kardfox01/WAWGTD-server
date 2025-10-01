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
import open_clip

import prms
from logger import LOG


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
        # self.ollama_warmup()
        self.load_clip()

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

    @LOG("ЗАГРУЗКА CLIP")
    def load_clip(self):
        self.clip, self._, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

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

        # Нормализация в 0–255
        depth_norm = cv2.normalize(prediction, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Сглаживание с сохранением границ
        depth_smooth = cv2.bilateralFilter(depth_norm, d=9, sigmaColor=75, sigmaSpace=75)

        # Otsu threshold вместо фиксированного
        _, bright_mask = cv2.threshold(depth_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Морфология для очистки и сглаживания границ
        kernel = np.ones((5, 5), np.uint8)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)

        # Контуры
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_with_contours = img.copy()
        cv2.drawContours(img_with_contours, contours, -1, (255, 0, 0), prms.CONTOURS_THICKNESS)

        # Тёмный фон с подсветкой объектов
        binary_mask = (bright_mask > 0).astype(np.uint8)
        dark_image = (img * 0.2).astype(np.uint8)
        binary_mask_3ch = cv2.merge([binary_mask] * 3)
        image_dark_bg = np.where(binary_mask_3ch == 1, img, dark_image)
        cv2.drawContours(image_dark_bg, contours, -1, prms.CONTOURS_COLOR, max(1, prms.CONTOURS_THICKNESS - 5))

        return Image.fromarray(depth_smooth), Image.fromarray(img_with_contours)

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

    @LOG("ОПРЕДЕЛЕНИЕ ДЕРЕВА НА ФОТО")
    def tont(self, img: Img): # Tree Or Not Tree
        image = self.preprocess(img).unsqueeze(0)
        text = self.tokenizer(prms.TOKENS)

        with torch.no_grad():
            image_features = self.clip.encode_image(image)
            text_features = self.clip.encode_text(text)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            prediction = prms.LABELS[similarity[0].argmax().item()]
            # for label, score in zip(prms.LABELS, similarity[0].tolist()):
                # print(f"{label}: {score:.4f}")
                # print("Prediction:", prms.LABELS[similarity[0].argmax().item()])
                # prediction *= prms.LABELS[similarity[0].argmax().item()]
            # print("Prediction:", prediction)

        return prediction