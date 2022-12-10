from fastapi import FastAPI, status
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from static.utils import models, decode_image, encode_image_to_base64, preprocess_replace_bg_image


VERSION: str = "2.0.0"
STATIC_PATH: str = "static"


class Image(BaseModel):
    imageData: str


class Replace(BaseModel):
    imageData_1: str
    imageData_2: str


origins = [
    "http://localhost:4041",
]


app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return JSONResponse({
        "statusText" : "Root Endpoint of Computer Vision API V2",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.get("/version")
async def get_version():
    return JSONResponse({
        "statusText" : "Computer Vision API Version Fetch Successful",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.get("/classify")
async def get_classifiy_infer():
    return JSONResponse({
        "statusText" : "Classification Inference Endpoint",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.get("/detect")
async def get_detect_infer():
    return JSONResponse({
        "statusText" : "Detection Inference Endpoint",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.get("/segment")
async def get_segment_infer():
    return JSONResponse({
        "statusText" : "Segmentation Inference Endpoint",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.get("/remove")
async def get_remove_bg():
    return JSONResponse({
        "statusText" : "Background Removal Endpoint",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.get("/replace")
async def get_replace_bg():
    return JSONResponse({
        "statusText" : "Background Replacement Endpoint",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.get("/depth")
async def get_depth_infer():
    return JSONResponse({
        "statusText" : "Depth Inference Endpoint",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.get("/face")
async def get_face_detect_infer():
    return JSONResponse({
        "statusText" : "Face Detection Inference Endpoint",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.post("/classify")
async def post_classify_infer(image: Image):
    _, image = decode_image(image.imageData)

    label = models[0].infer(image=image)

    return JSONResponse({
        "statusText" : "Classification Inference Complete",
        "statusCode" : status.HTTP_200_OK,
        "label" : label,
    })


@app.post("/detect")
async def post_detect_infer(image: Image):
    _, image = decode_image(image.imageData)

    label, score, box = models[1].infer(image)

    if label is not None:
        return JSONResponse({
            "statusText" : "Detection Inference Complete",
            "statusCode" : status.HTTP_200_OK,
            "label" : label,
            "score" : str(score),
            "box" : box,
        })
    else:
        return JSONResponse({
            "statusText" : "No Detections",
            "statusCode" : 500,
        })


@app.post("/segment")
async def post_segment_infer(image: Image):
    _, image = decode_image(image.imageData)

    segmented_image, labels = models[2].infer(image)
    
    return JSONResponse({
        "statusText" : "Segmentation Inference Complete",
        "statusCode" : status.HTTP_200_OK,
        "labels" : labels,
        "imageData" : encode_image_to_base64(image=segmented_image)
    })


@app.post("/remove")
async def post_remove_bg(image: Image):
    _, image = decode_image(image.imageData)

    mask = models[3].infer(image=image)
    for i in range(3): image[:, :, i] = image[:, :, i] & mask

    return JSONResponse({
        "statusText" : "Background Removal Complete",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
        "maskImageData" : encode_image_to_base64(image=mask),
        "bglessImageData" : encode_image_to_base64(image=image),
    })


@app.post("/replace")
async def post_replace_bg(images: Replace):
    _, image_1 = decode_image(images.imageData_1)
    _, image_2 = decode_image(images.imageData_2)

    mask = models[3].infer(image=image_1)
    mh, mw = mask.shape
    image_2 = preprocess_replace_bg_image(image_2, mw, mh)
    for i in range(3): 
        image_1[:, :, i] = image_1[:, :, i] & mask
        image_2[:, :, i] = image_2[:, :, i] & (255 - mask) 

    image_2 += image_1   

    return JSONResponse({
        "statusText" : "Background Removal Complete",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
        "bgreplaceImageData" : encode_image_to_base64(image=image_2),
    })


@app.post("/depth")
async def post_depth_infer(image: Image):
    _, image = decode_image(image.imageData)

    image = models[4].infer(image=image)

    return JSONResponse({
        "statusText" : "Depth Inference Complete",
        "statusCode" : status.HTTP_200_OK,
        "imageData" : encode_image_to_base64(image=image),
    })


@app.post("/face")
async def post_face_detect_infer(image: Image):
    _, image = decode_image(image.imageData)

    face_detections_np = models[5].infer(image)

    if len(face_detections_np) > 0:
        face_detections: list = []
        for (x, y, w, h) in face_detections_np:
            face_detections.append([int(x), int(y), int(w), int(h)])
        
        return JSONResponse({
            "statusText" : "Face Detection Complete",
            "statusCode" : status.HTTP_200_OK,
            "face_detections" : face_detections,
        })
    else:
        return JSONResponse({
            "statusText" : "No Detections",
            "statusCode" : status.HTTP_500_INTERNAL_SERVER_ERROR,
        })
