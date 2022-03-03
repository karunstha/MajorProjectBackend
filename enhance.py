import os
from datetime import date
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

IMAGE_PATH = "Images.jpg"
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

def enhance(image_path):
    today = date.today()
    hr_image = preprocess_image(image_path)

    model = hub.load(SAVED_MODEL_PATH)

    start = time.time()
    fake_image = model(hr_image)
    fake_image = tf.squeeze(fake_image)

    save_image(tf.squeeze(fake_image), filename="Super Resolution")
