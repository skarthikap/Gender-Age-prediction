# Import the necessary packages
import cv2
import numpy as np
from flask import Flask, render_template, Response, request
from PIL import Image
import io

UPLOAD_FOLDER = './UPLOAD_FOLDER'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frameOpencvDnn, faceBoxes

def gen_frames():
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    video = cv2.VideoCapture(0)
    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            break
        resultImg, faceBoxes = highlightFace(faceNet, frame)
        if faceBoxes:
            for faceBox in faceBoxes:
                face = frame[max(0, faceBox[1]-20):min(faceBox[3]+20, frame.shape[0]-1), max(0, faceBox[0]-20):min(faceBox[2]+20, frame.shape[1]-1)]
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]
                
                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                
                cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        if resultImg is None:
            continue

        ret, encodedImg = cv2.imencode('.jpg', resultImg)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'fileToUpload' not in request.files:
        return 'No file uploaded', 400
    file = request.files['fileToUpload']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        img = Image.open(file)
        img_np = np.array(img.convert('RGB'))
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return Response(gen_frames_photo(frame), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames_photo(frame):
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if faceBoxes:
        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1]-20):min(faceBox[3]+20, frame.shape[0]-1), max(0, faceBox[0]-20):min(faceBox[2]+20, frame.shape[1]-1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            
            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    if resultImg is None:
        return b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b'\r\n'

    ret, encodedImg = cv2.imencode('.jpg', resultImg)
    if not ret:
        return b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b'\r\n'
    
    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')
if __name__ == '__main__':
    app.run(debug=True, port=8080)


