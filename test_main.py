from fastapi.testclient import TestClient
from main import app, VERSION

client = TestClient(app=app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Root Endpoint of Computer Vision API V2",
        "statusCode" : 200,
        "version" : VERSION,
    }
 

def test_get_version():
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Computer Vision API Version Fetch Successful",
        "statusCode" : 200,
        "version" : VERSION,
    }
    
      
def test_get_classify_infer():
    response = client.get("/classify")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Classification Inference Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    }


def test_get_detect_infer():
    response = client.get("/detect")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Detection Inference Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    }


def test_get_segment_infer():
    response = client.get("/segment")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Segmentation Inference Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    }


def test_get_remove_infer():
    response = client.get("/remove")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Background Removal Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    }


def test_get_replace_infer():
    response = client.get("/replace")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Background Replacement Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    }


def test_get_replace_infer():
    response = client.get("/depth")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Depth Inference Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    }


def test_get_face_detect_infer():
    response = client.get("/face-detect")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Face Detection Inference Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    }


def test_get_face_recognize_infer():
    response = client.get("/face-recognize")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Face Recognition Inference Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    }
