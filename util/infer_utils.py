import io
import requests
import numpy as np
from PIL import Image

def download_from_cdn(url: str) -> bytes:
    res = requests.get(url, stream=True)
    data = b""
    for chunk in res.iter_content(chunk_size=1024):
        if chunk:
            data += chunk
    return data

def size_checks(img: Image) -> np.ndarray:
    if img.size != (1920, 1080) and img.size == (1080, 1920): #the image is vertical
        img = np.array(img.rotate(90, expand=True))
    elif img.size == (3840, 2160) or (img.size[0] == img.size[1] and img.size[0] >= 1920): #the img is large
        img = img.resize((1920, 1080))
        img = np.array(img)
    elif (img.size[0] < 1920 and img.size[0] != 1080) and (img.size[1] < 1080): #the image is small
        img = img.resize((1920, 1080))
        img = np.array(img)
    else:
        img = np.array(img)
    assert img.shape == (1080, 1920, 3), print(f"Unexpected shape: {img.shape}", flush=True) 
    return img

def get_img(bucket: str, key: str, pc: int) -> np.ndarray:
    
    if bucket != "cdn":
        byte_data = download_from_s3(bucket=bucket, key=key)
    else:
        byte_data = download_from_cdn(key)
    
    img = Image.open(io.BytesIO(byte_data))
    img = size_checks(img) 

    return img 
