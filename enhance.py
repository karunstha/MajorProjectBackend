import os
from datetime import date
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from utils import preprocess_image, save_image, downscale_image
from pathlib import Path

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

def enhance(image_path):
    today = date.today()
    hr_image = preprocess_image(image_path)
    
    # hr_image = downscale_image(tf.squeeze(hr_image))

    # save_image(tf.squeeze(fake_image), filename="Downsampled Resolution")

    model = hub.load(SAVED_MODEL_PATH)

    fake_image = model(hr_image)
    fake_image = tf.squeeze(fake_image)

    save_image(tf.squeeze(fake_image), filename="super_res/"+Path(image_path).stem)

def enhance_with_downsample(image_path):
    today = date.today()
    hr_image = preprocess_image(image_path)
    
    hr_image = downscale_image(tf.squeeze(hr_image))

    model = hub.load(SAVED_MODEL_PATH)

    fake_image = model(hr_image)
    fake_image = tf.squeeze(fake_image)

    save_image(tf.squeeze(fake_image), filename="super_res/"+Path(image_path).stem)
