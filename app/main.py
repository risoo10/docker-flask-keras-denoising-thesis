import keras
import keras_contrib
import tensorflow
import numpy as np
from flask import Flask, request, abort, jsonify
import cv2
from PIL import Image

from keras.models import load_model
from keras.losses import mean_squared_error
from keras_contrib.losses import dssim
import io
import time

import extract_join_patches

app = Flask(__name__)

FILENAME = "conv-deconv-renoir-64x64-CON-16F-TRANSP-8D"

global model
global loss

# My own loss function
def weighted_loss(y_true, y_pred):
    dssim_loss = loss
    return 0.5 * mean_squared_error(y_true, y_pred) + 0.5 * dssim_loss(y_true, y_pred)


PATCH_SIZE = 64
CHANNELS = 3
STEP = 32

@app.route("/")
def hello():
    return "Hello, this is a perfect site for Artificial intelligence."


# Preprocess image
def preprocess_image(image):
    # Convert to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize image to max 1024 x 1024 and
    image.thumbnail((1024, 1024), Image.ANTIALIAS)
    print(image.width, image.height)
    curWidth, curHeight = image.width, image.height

    # Resize to be divisible by patch size
    width = int(curWidth / PATCH_SIZE) * PATCH_SIZE
    height = int(curHeight / PATCH_SIZE) * PATCH_SIZE
    print('Resize', width, height)
    image = image.resize((width, height), Image.ANTIALIAS)
    image = np.asarray(image)
    print(image.shape)

    # Normalize
    image = image.astype('float32') / 255
    print(image.shape)
    return image


@app.route("/predict-inteligent", methods=["POST"])
def denoise_image_inteligent():
    try:
        if request.method == "POST":
            if request.files.get("image"):

                # Read Image from bytes
                image = request.files["image"].read()
                image = Image.open(io.BytesIO(image))

                # Preprocess image
                image = preprocess_image(image)
                width = image.shape[1]
                height = image.shape[0]

                # Extract noisy patches
                patches = extract_join_patches.split_to_patches(image, shape=(PATCH_SIZE, PATCH_SIZE, CHANNELS), step=STEP)
                patch_count_ver, patch_count_hor = patches.shape[:2]

                # Flatten to 1D array of patches
                patches = patches.reshape((patch_count_ver * patch_count_hor, patches.shape[3], patches.shape[4], patches.shape[5]))

                print('Patches', patches.shape)

                # Predict results
                denoised_patches = model.predict(patches)

                # Reshape back to 2D array of patches
                denoised_patches = denoised_patches.reshape((patch_count_ver, patch_count_hor, patches.shape[1], patches.shape[2], patches.shape[3]))

                # Reconstruct final image from patches
                reconstructed = extract_join_patches.patch_together(denoised_patches, image_size=(width, height))

                # Denormalize and convert color
                result_image = cv2.cvtColor(np.uint8(reconstructed * 255), cv2.COLOR_BGR2RGB)

                # Save to results folder
                time_string = time.strftime("%Y%m%d-%H%M%S")
                image_path = 'results/result_{}.jpg'.format(time_string)
                cv2.imwrite(image_path, result_image)

                data = {
                    "succes": True,
                    "imageUrl": image_path,
                }

                return jsonify(data)
    except IOError:
        abort(404)


if __name__ == "__main__":
    loss = dssim.DSSIMObjective()
    model = load_model("models/{}.h5".format(FILENAME), custom_objects={'weighted_loss': weighted_loss})
    app.run(host="0.0.0.0", port=85)

