import re

from typing import Union
from fastapi import FastAPI, status
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from static.utils import models, decode_image, encode_image_to_base64, preprocess_replace_bg_image


VERSION: str = "2.0.0"
STATIC_PATH: str = "static"


# class Image(BaseModel):
#     imageData: str


class APIData(BaseModel):
    imageData_1: str
    imageData_2: Union[str, None] = None


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


@app.get("/{infer_type}")
async def get_infer(infer_type: str):
    if re.match(r"^classify$", infer_type, re.IGNORECASE):
        return JSONResponse({
            "statusText" : "Classification Inference Endpoint",
            "statusCode" : status.HTTP_200_OK,
            "version" : VERSION,
        })
    
    elif re.match(r"^detect$", infer_type, re.IGNORECASE):
        return JSONResponse({
            "statusText" : "Detection Inference Endpoint",
            "statusCode" : status.HTTP_200_OK,
            "version" : VERSION,
        })
    
    elif re.match(r"^segment$", infer_type, re.IGNORECASE):
        return JSONResponse({
            "statusText" : "Segmentation Inference Endpoint",
            "statusCode" : status.HTTP_200_OK,
            "version" : VERSION,
        })
    
    elif re.match(r"^remove$", infer_type, re.IGNORECASE):
        return JSONResponse({
            "statusText" : "Background Removal Endpoint",
            "statusCode" : status.HTTP_200_OK,
            "version" : VERSION,
        })
    
    elif re.match(r"^replace$", infer_type, re.IGNORECASE):
        return JSONResponse({
            "statusText" : "Background Replacement Endpoint",
            "statusCode" : status.HTTP_200_OK,
            "version" : VERSION,
        })
    
    elif re.match(r"^depth$", infer_type, re.IGNORECASE):
        return JSONResponse({
            "statusText" : "Depth Inference Endpoint",
            "statusCode" : status.HTTP_200_OK,
            "version" : VERSION,
        })
    
    elif re.match(r"^face-detect$", infer_type, re.IGNORECASE):
        return JSONResponse({
            "statusText" : "Face Detection Inference Endpoint",
            "statusCode" : status.HTTP_200_OK,
            "version" : VERSION,
        })
    
    elif re.match(r"^face-recognize$", infer_type, re.IGNORECASE):
        return JSONResponse({
            "statusText" : "Face Recognition Inference Endpoint",
            "statusCode" : status.HTTP_200_OK,
            "version" : VERSION,
        })
    
    else:
        return JSONResponse({
            "statusText" : f"{infer_type.title()} is Invalid",
            "statusCode" : status.HTTP_400_BAD_REQUEST,
            "version" : VERSION,
        })


@app.post("/{infer_type}")
async def post_infer(infer_type: str, images: APIData):

    if re.match(r"^classify$", infer_type, re.IGNORECASE):
        _, image = decode_image(images.imageData_1)

        label = models[0].infer(image=image)

        return JSONResponse({
            "statusText" : "Classification Inference Complete",
            "statusCode" : status.HTTP_200_OK,
            "label" : label,
        })
    
    elif re.match(r"^detect$", infer_type, re.IGNORECASE):
        _, image = decode_image(images.imageData_1)

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
                "statusCode" : status.HTTP_500_INTERNAL_SERVER_ERROR,
            })
        
    elif re.match(r"^segment$", infer_type, re.IGNORECASE):
        _, image = decode_image(images.imageData_1)

        segmented_image, labels = models[2].infer(image)
        
        return JSONResponse({
            "statusText" : "Segmentation Inference Complete",
            "statusCode" : status.HTTP_200_OK,
            "labels" : labels,
            "imageData" : encode_image_to_base64(image=segmented_image)
        })
    
    elif re.match(r"^remove$", infer_type, re.IGNORECASE):
        _, image = decode_image(images.imageData_1)

        mask = models[3].infer(image=image)
        for i in range(3): image[:, :, i] = image[:, :, i] & mask

        return JSONResponse({
            "statusText" : "Background Removal Complete",
            "statusCode" : status.HTTP_200_OK,
            "version" : VERSION,
            "maskImageData" : encode_image_to_base64(image=mask),
            "bglessImageData" : encode_image_to_base64(image=image),
        })
    
    elif re.match(r"^replace$", infer_type, re.IGNORECASE):
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
    
    elif re.match(r"^depth$", infer_type, re.IGNORECASE):
        _, image = decode_image(images.imageData_1)

        image = models[4].infer(image=image)

        return JSONResponse({
            "statusText" : "Depth Inference Complete",
            "statusCode" : status.HTTP_200_OK,
            "imageData" : encode_image_to_base64(image=image),
        })
    
    elif re.match(r"^face-detect$", infer_type, re.IGNORECASE):
        _, image = decode_image(images.imageData_1)

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
    
    elif re.match(r"^face-recognize$", infer_type, re.IGNORECASE):
        _, image_1 = decode_image(images.imageData_1)
        _, image_2 = decode_image(images.imageData_2)

        cs = models[6].get_cosine_similarity(image_1, image_2)

    if cs is not None:
        return JSONResponse({
            "statusText" : "Face Recognition Inference Complete",
            "statusCode" : status.HTTP_200_OK,
            "cosine_similarity" : str(cs),
        })
    else:
        return JSONResponse({
            "statusText" : "Possible error in APIData; cannot calculate similarity",
            "statusCode" : status.HTTP_500_INTERNAL_SERVER_ERROR,
        })