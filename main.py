from fastapi import FastAPI, UploadFile, File
import numpy as np
from io import BytesIO
from PIL import Image
import uvicorn
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
origins = [
    "http://localhost",
    "http://localhost:3000",  # Update with your frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../saved_models/1")
CLASS_NAMES = ["Early blight", "Late blight", "Healthy"]

MODELT = tf.keras.models.load_model("../../tomato-Disease-classification/saved_models/version-2")
CLASS_NAMEST = ['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']



@app.get("/ping")
async def ping():
    return "hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predictPotato")
async def predict(file: UploadFile):

    image = read_file_as_image(await file.read())

    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        "class": predicted_class,
        "confidence": float(confidence),
    }

@app.post("/predictTomato")
async def predictT(file: UploadFile):
   
   image = read_file_as_image(await file.read())

   img_batch = np.expand_dims(image,0)

   predictions = MODELT.predict(img_batch)

   predicted_class = CLASS_NAMEST[np.argmax(predictions[0])]
   confidence = np.max(predictions[0])

   return {
       "class": predicted_class,
       "confidence": float(confidence),
   }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
