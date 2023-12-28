from __future__ import annotations

import base64
import datetime
import hashlib
import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from utils.config import read_config
from utils.logger import logger

config = read_config("config/config.ini")


def hex_hash(s: str) -> str:
    """Returns the first 10 characters of the hex hash of a string."""
    return hashlib.sha256(s.encode()).hexdigest()[:10]


class Model(ABC):
    url: str

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D ,3D Game, 3D Game Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)",
        number_of_imgs: int = 1,
        steps: int = 35,
        cfg_scale: float = 7.0,
        size: tuple[int, int] = (512, 512),
        cache: bool = True,
    ) -> list[Path]:
        """Generates various images from a prompt.

        # DOCS: Add docs
        cache: If True, the images will be cached based on a hash of the prompt.
        """

        # TODO: Add arg "batch_size": 1,

        assert number_of_imgs > 0, "number_of_imgs must be greater than 0"
        assert isinstance(number_of_imgs, int), "number_of_imgs must be an integer"
        assert isinstance(prompt, str), "prompt must be a string"
        assert isinstance(steps, int), "steps must be an integer"
        assert isinstance(cfg_scale, float), "cfg_scale must be a float"
        assert isinstance(size, tuple), "size must be a tuple"

        if cache:  # Actually, we assume this is the usual case!
            hash_prompt = hex_hash(prompt)
            path = Path(__file__).parent / "outputs" / hash_prompt
            if not path.exists():
                path.mkdir(parents=True)

            # First we check that there are enough images
            images = list(path.glob("*.png"))
            if len(images) < number_of_imgs:
                logger.info(f"Generating {number_of_imgs} images for {prompt}")
                for _ in range(number_of_imgs - len(images)):
                    self._generate(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        steps=steps,
                        cfg_scale=cfg_scale,
                        size=size,
                        path_dir_output=path,
                    )

            # Then we return the images
            return images[:number_of_imgs]

        list_of_list_of_paths = [
            self._generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                cfg_scale=cfg_scale,
                size=size,
            )
            for _ in range(number_of_imgs)
        ]
        return [
            path for list_of_paths in list_of_list_of_paths for path in list_of_paths
        ]

    @abstractmethod
    def _generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 35,
        cfg_scale: float = 7.0,
        size: tuple[int, int] = (512, 512),
        path_dir_output: Path | None = None,
        batch_size: int = 1,
    ) -> Path:
        """Generates an image from a prompt."""

        ...  # pragma: no cover

    def generate_batch(
        self,
        prompt_list: list[str] = None,
        n_images: int = None,
        iterations: int = None,
    ):
        if prompt_list is None:
            prompt_list = config.get("MODEL", "prompts").split(",")
        if n_images is None:
            n_images = int(config.get("MODEL", "n_images"))
        if iterations is None:
            iterations = int(config.get("MODEL", "iterations"))

        for _ in range(iterations):
            for i in prompt_list:
                self.generate(prompt=i, number_of_imgs=n_images)

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
        negative_prompt: str = "(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D ,3D Game, 3D Game Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)",
        steps: int = 35,
        cfg_scale: float = 7.0,
        size: tuple[int, int] = (768, 1168),
        path_dir_output: Path | None = None,
        batch_size: int = 1,
    ) -> list[Path]:
        logger.info(f"Request: {prompt}, {steps}, {cfg_scale}, {size}")

        if path_dir_output is None:
            path_dir_output = Path("outputs")

        # Request itself
        response = requests.post(
            url=f"{self.url}/sdapi/v1/txt2img",
            json={
                # "prompt": prompt + ", photorealistic, soft light, high quality",
                "prompt": prompt + ", photorealistic, soft light",
                "negative_prompt": negative_prompt,
                "sampler_name": "DPM++ 2M Karras",
                "steps": steps,
                "cfg_scale": cfg_scale,
                "width": size[0],
                "height": size[1],
                "batch_size": batch_size,
                "seed": 1,
            },
        )
        assert response.status_code == 200, f"Server error: {response.text}"
        r = response.json()
        # image = Image.open(io.BytesIO(base64.b64decode(r["images"][0])))

        # File management
        paths = []
        for n, img in enumerate(r["images"]):
            image = Image.open(io.BytesIO(base64.b64decode(img)))
            date_formatted = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = path_dir_output / f"{date_formatted}_{prompt}_{n}.png"
            image.save(path)
            paths.append(path)
        return paths


if __name__ == "__main__":
    m = Model_SD_0()
    # Model_SD_0().generate_batch()
    m.generate(
        "a nurse",
    )
