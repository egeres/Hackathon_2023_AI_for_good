import base64
import datetime
import io
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests
from loguru import logger
from PIL import Image

logger.add("model_runs.log", rotation="1 MB")


class Model(ABC):
    url: str

    def generate_cached(
        self,
    ):
        # TODO: Implement this in the future
        pass

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)",
        number_of_imgs: int = 1,
        steps: int = 35,
        cfg_scale: float = 7.0,
        size: tuple[int, int] = (512, 512),
    ) -> list[Path]:
        """Generates various images from a prompt."""

        assert number_of_imgs > 0, "number_of_imgs must be greater than 0"
        assert isinstance(number_of_imgs, int), "number_of_imgs must be an integer"
        assert isinstance(prompt, str), "prompt must be a string"
        assert isinstance(steps, int), "steps must be an integer"
        assert isinstance(cfg_scale, float), "cfg_scale must be a float"
        assert isinstance(size, tuple), "size must be a tuple"

        # TODO: Add arg "batch_size": 1,

        return [
            self._generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                cfg_scale=cfg_scale,
                size=size,
            )
            for _ in range(number_of_imgs)
        ]

    @abstractmethod
    def _generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 35,
        cfg_scale: float = 7.0,
        size: tuple[int, int] = (512, 512),
    ) -> Path:
        """Generates an image from a prompt."""

        ...  # pragma: no cover

    def get_models(self) -> list[dict[str, Any]]:
        """Gets a list of models."""

        response = requests.get(url=f"{self.url}/sdapi/v1/sd-models")
        assert response.status_code == 200, "Server error"
        return response.json()

    def get_loras(self) -> list[dict[str, Any]]:
        """Gets a list of loras."""

        response = requests.get(url=f"{self.url}/sdapi/v1/loras")
        assert response.status_code == 200, "Server error"
        return response.json()


class Model_SD_0(Model):
    url = "http://127.0.0.1:7860"  # http://127.0.0.1:7860/docs

    def _generate(
        self,
        prompt: str,
        negative_prompt: str = "mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)",
        steps: int = 35,
        cfg_scale: float = 7.0,
        size: tuple[int, int] = (512, 512),
    ) -> Path:
        logger.info(f"Request: {prompt}, {steps}, {cfg_scale}, {size}")

        # Request itself
        response = requests.post(
            url=f"{self.url}/sdapi/v1/txt2img",
            json={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "width": size[0],
                "height": size[1],
                "sampler_name": "DPM++ 2M Karras",
            },
        )
        assert response.status_code == 200, f"Server error: {response.text}"
        r = response.json()
        image = Image.open(io.BytesIO(base64.b64decode(r["images"][0])))

        # File management
        date_format = "%Y-%m-%d_%H-%M-%S"
        date_formatted = datetime.datetime.now().strftime(date_format)
        path = Path(f"outputs/{date_formatted}_{prompt}.png")
        image.save(path)
        return path


if __name__ == "__main__":
    m = Model_SD_0()
    o = m.generate(prompt="a nurse", number_of_imgs=3)
    # o = m.get_models()
    p = 0
