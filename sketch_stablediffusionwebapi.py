import base64
import datetime
import io
import json

import requests
from PIL import Image

url = "http://127.0.0.1:7860"  # http://127.0.0.1:7860/docs

payload = {"prompt": "a nurse", "steps": 25}
print("Running...")
response = requests.post(url=f"{url}/sdapi/v1/txt2img", json=payload)
print(response.status_code)
r = response.json()
image = Image.open(io.BytesIO(base64.b64decode(r["images"][0])))
image.save(f"outputs/{datetime.datetime.now()}_{payload['prompt']}.png")
print("Finished...")
