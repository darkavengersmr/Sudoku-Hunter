#!/usr/bin/env python3

import numpy as np
from imutils.video import WebcamVideoStream
import cv2, time, threading, math
from flask import Flask, render_template, Response

cap = WebcamVideoStream(src=0).start()
frame = cap.read()
app = Flask(__name__)

def image2jpeg(image):
    ret, jpeg = cv2.imencode('.jpg', image)
    return jpeg.tobytes()

@app.route('/')
def index():
    return render_template('index_control.html')
def gen_cl():
        while True:
            frame_inet = cap.read()
            frameinet = image2jpeg(frame_inet[40:450, :, :])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frameinet + b'\r\n\r\n')
@app.route('/video_cl')
def video_cl():
    return Response(gen_cl(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
app.run(host='0.0.0.0', debug=False,threaded=True)
