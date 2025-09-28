import base64
import io
import json
from logging import Logger
from typing import Optional
from PIL import Image
import cv2
import numpy as np
from ollama import chat, ChatResponse
import torch
import prms


Img = Image.Image

def from_base64(base64_code: str):
    img_data = base64.b64decode(base64_code)
    return Image.open(io.BytesIO(img_data))

def to_base64(img: Img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class Neuro:
    def __init__(self, LOG: Logger):
        self.LOG = LOG

        # self.LOG.info("ЗАГРУЗКА МОДЕЛИ YOLO...")
        # self.yolo_model = YOLO(prms.YOLO_WEIGHTS_PATH)
        # self.LOG.info("МОДЕЛЬ YOLO ЗАГРУЖЕНА УСПЕШНО")
        self.LOG.info("ЗАГРУЗКА DPT...")
        self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")  # можно заменить на "DPT_Large" или "DPT_Hybrid"
        self.midas.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas.to(self.device)
        self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.LOG.info("DPT ЗАГРУЖЕНА")
        self.LOG.info("ИНИЦИАЛИЗАЦИЯ МОДЕЛИ OLLAMA...")
        chat(model=prms.OLLAMA_MODEL, messages=[{
            "role": "user",
            "content": ""
        }])
        self.LOG.info("МОДЕЛЬ OLLAMA ИНИЦИАЛИЗИРОВАНА УСПЕШНО")

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
        cv2.drawContours(img_with_contours, contours, -1, prms.CONTOURS_COLOR, prms.CONTOURS_THICKNESS)

        binary_mask = (depth_vis > prms.THRESHOLD_VALUE).astype(np.uint8)
        dark_image = (img * 0.2).astype(np.uint8)
        binary_mask_3ch = cv2.merge([binary_mask]*3)
        image_dark_bg = np.where(binary_mask_3ch == 1, img, dark_image)
        cv2.drawContours(image_dark_bg, contours, -1, prms.CONTOURS_COLOR, prms.CONTOURS_THICKNESS - 5)

        return Image.fromarray(image_dark_bg), Image.fromarray(img_with_contours)

    def ollama_json(self, base64_code: str) -> Optional[dict[str, str]]:
        self.LOG.info("ОЖИДАНИЕ ОПИСАНИЯ OLLAMA...")
        response: ChatResponse = chat(model=prms.OLLAMA_MODEL, messages=[
            {
                "role": "system",
                "content": prms.OLLAMA_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prms.OLLAMA_USER_PROMPT,
                "images": [base64_code]
            }
        ])

        self.LOG.info("ПОЛУЧЕНО СЫРОЕ ОПИСАНИЕ...")
        print(response.message.content)
        if response.message.content != None:
            self.LOG.info("КОНВЕРТАЦИЯ В JSON...")
            return json.loads(response.message.content.replace("```json", "").replace("```", ""))

        return None
