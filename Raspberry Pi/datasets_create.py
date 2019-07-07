#! /usr/bin/python3

# -*- coding: utf-8 -*-

from flask import Flask, render_template, send_file, request, Response
from imutils.video import WebcamVideoStream
import random, time, threading, cv2, math, socket
import numpy as np
import copy, sys, subprocess

game_xmax = 640
game_ymax = 480
game_xmin = 0
game_ymin = 0

draw_points = []

xmin, xmax = 0, 0
ymin, ymax = 0 ,0
rectangle_ok = False

img_size = 128

font = cv2.FONT_HERSHEY_SIMPLEX

cap = WebcamVideoStream(src=0).start()

time_apdate = time.time()

param = sys.argv
number_save_name = param[1]
number_save = param[2]

draw_no_rect = []
draw = []
draw_box = []

cadr = 1
scaning = True

frame = cap.read()
hsv = cv2.cvtColor(frame[game_ymin:game_ymax, game_xmin:game_xmax, :], cv2.COLOR_BGR2HSV)
frame_gray = cv2.inRange(hsv, (0, 0, 0), (150, 255, 50))
frame_gray_ev3 = frame_gray.copy()
frame_ev3 = frame.copy()

def return_cell(x, y):
    global frame_gray_ev3, frame_ev3
    frame_gray_cell = cv2.cvtColor(frame_ev3[game_ymin:game_ymax, game_xmin:game_xmax, :], cv2.COLOR_BGR2GRAY)
    frame_gray_cell = frame_gray_cell[ymin:ymax, xmin:xmax]
    step_x = frame_gray_cell.shape[1]/9
    step_y = frame_gray_cell.shape[0]/9

    cv2.imwrite("/home/pi/neyro_gun_sudoku_solver/frame_gray_cell.jpg", frame_gray_cell)

    return frame_gray_cell[round(step_y*(y-1))+5:round(step_y*y)-5, round(step_x*(x-1))+5:round(step_x*x)-5]

def image2jpeg(image):
    ret, jpeg = cv2.imencode('.jpg', image)
    return jpeg.tobytes()

def camera2inet():
    global sudoku
    app = Flask(__name__, template_folder="templates")
    print("Start inet thread")

    @app.route('/')
    def home():
        return render_template('home.html')

    def gen():
        global draw_points, xmin, xmax, ymin, ymax
        while True:
            inet_gray = cv2.inRange(hsv, (0, 0, 0), (150, 255, 50))
            frame_inet = cv2.cvtColor(inet_gray, cv2.COLOR_GRAY2RGB)

            if(len(draw_points) > 0):
                draw_points2 = draw_points.copy()
                for i in range(3):
                    cv2.line(frame_inet, draw_points2[i], draw_points2[i+1], (0, 0, 255), 3)
                cv2.line(frame_inet, draw_points2[3], draw_points2[0], (0, 0, 255), 3)

                cv2.putText(frame_inet, 'left', draw_points2[0], font, 1, (255,0,0), 1, cv2.LINE_AA)
                cv2.putText(frame_inet, 'up', draw_points2[1], font, 1, (255,0,0), 1, cv2.LINE_AA)
                cv2.putText(frame_inet, 'right', draw_points2[2], font, 1, (255,0,0), 1, cv2.LINE_AA)
                cv2.putText(frame_inet, 'down', draw_points2[3], font, 1, (255,0,0), 1, cv2.LINE_AA)
            elif(rectangle_ok):
                cv2.line(frame_inet, (xmin, ymin), (xmin, ymax), (0, 255, 0), 3)
                cv2.line(frame_inet, (xmin, ymax), (xmax, ymax), (0, 255, 0), 3)
                cv2.line(frame_inet, (xmax, ymax), (xmax, ymin), (0, 255, 0), 3)
                cv2.line(frame_inet, (xmax, ymin), (xmin, ymin), (0, 255, 0), 3)

            frameinet = image2jpeg(frame_inet)
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frameinet + b'\r\n\r\n')

    @app.route('/video')
    def video():
        return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    def gen_color():
        while True: yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + image2jpeg(frame[game_ymin:game_ymax, game_xmin:game_xmax, :]) + b'\r\n\r\n')

    @app.route('/video_color')
    def video_color():
        return Response(return_cell(1, 1),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    def gen_all_color():
        while True:
            frame_inet = frame.copy()
            cv2.line(frame_inet, (game_xmin, game_ymin), (game_xmin, game_ymax), (0, 255, 0), 2)
            cv2.line(frame_inet, (game_xmin, game_ymax), (game_xmax, game_ymax), (0, 255, 0), 2)
            cv2.line(frame_inet, (game_xmax, game_ymax), (game_xmax, game_ymin), (0, 255, 0), 2)
            cv2.line(frame_inet, (game_xmax, game_ymin), (game_xmin, game_ymin), (0, 255, 0), 2)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + image2jpeg(frame_inet) + b'\r\n\r\n')

    @app.route('/all_color')
    def all_color():
        return Response(gen_all_color(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    app.run(host='0.0.0.0', debug=False, threaded=True)

def send0or00(a):
    if(int(a) < 0): a = '000'
    elif(int(a) < 10): a = '00' + a
    elif(int(a) < 100): a = '0' + a
    return a

def ev3():
    global draw_points, frame, frame_gray, xmin, xmax, ymin, ymax, cadr, frame_ev3
    global rectangle_ok, frame_gray_ev3, gamexmin, game_xmax, game_ymin, game_ymax
    client = socket.socket()
    client.connect(("192.168.32.209", 9090))

    draw_points_tmp = []
    while True:
        frame_ev3 = frame.copy()
        frame_gray_ev3 = frame_gray.copy()
        frame_gray_sum_x = np.sum(frame_gray_ev3, axis=0)
        frame_gray_sum_y = np.sum(frame_gray_ev3, axis=1)

        rectangle_ok = False
        for i in range(game_xmax):
            xmin = i
            if(frame_gray_sum_x[i] > 0): break
        for i in range(game_xmax-1, 0, -1):
            xmax = i
            if(frame_gray_sum_x[i] > 0): break
        for i in range(game_ymax):
            ymin = i
            if(frame_gray_sum_y[i] > 0): break
        for i in range(game_ymax-1, 0, -1):
            ymax = i
            if(frame_gray_sum_y[i] > 0): break

        if(abs(abs(xmin-xmax) - abs(ymin-ymax)) < 3 and abs(xmin-xmax) > 240):
            draw_points_tmp.clear()
            draw_points.clear()
            if(np.sum(frame_gray_ev3[xmin:xmin+5, :]) < 8000):
                client.send(b'start')
                if(np.sum(frame_gray_ev3[ymin:ymin+5, :]) < 8000):

                    moments = cv2.moments(frame_gray_ev3[:, xmin], 1) #left
                    ym = int(moments['m01'] / moments['m00'])
                    send = send0or00(str(xmin)) + send0or00(str(ym))
                    client.send(send.encode())
                    draw_points_tmp.append( (xmin, ym) )

                    moments = cv2.moments(frame_gray_ev3[ymin, :], 1) #up
                    xm = int(moments['m01'] / moments['m00'])
                    send = send0or00(str(xm)) + send0or00(str(ymin))
                    client.send(send.encode())
                    draw_points_tmp.append( (xm, ymin) )

                    moments = cv2.moments(frame_gray_ev3[:, xmax], 1) #right
                    ym = int(moments['m01'] / moments['m00'])
                    send = send0or00(str(xmax)) + send0or00(str(ym))
                    client.send(send.encode())
                    draw_points_tmp.append( (xmax, ym) )

                    moments = cv2.moments(frame_gray_ev3[ymax, :], 1) #down
                    xm = int(moments['m01'] / moments['m00'])
                    send = send0or00(str(xm)) + send0or00(str(ymax))
                    client.send(send.encode())
                    draw_points_tmp.append( (xm, ymax) )

                    draw_points = draw_points_tmp.copy()
                else:
                    rectangle_ok = True
                    send = send0or00(str(xmin)) + send0or00(str(ymin))
                    client.send(send.encode())

                    send = send0or00(str(xmax)) + send0or00(str(ymin))
                    client.send(send.encode())

                    send = send0or00(str(xmax)) + send0or00(str(ymax))
                    client.send(send.encode())

                    send = send0or00(str(xmin)) + send0or00(str(ymax))
                    client.send(send.encode())

                data = client.recv(1)
                data = data.decode("utf-8")
                if(data == 's' and rectangle_ok):
                    position_sudoku = True
                    subprocess.run(["mkdir", "-p", "number2/" + str(number_save_name)])

                    hsv_50 = cv2.cvtColor(frame_ev3[game_ymin:game_ymax, game_xmin:game_xmax, :], cv2.COLOR_BGR2HSV)
                    frame_gray_50 = cv2.inRange(hsv_50[ymin:ymax, xmin:xmax], (0, 0, 0), (150, 255, 100))
                    cv2.imwrite("/home/pi/neyro_gun_sudoku_solver/frame_gray.jpg", frame_gray_50[game_ymin:game_ymax, game_xmin:game_xmax])

                    i=0
                    for x in range(9):
                        for y in range(9):
                            i+=1
                            cv2.imwrite("/home/pi/neyro_gun_sudoku_solver/number2/" + str(number_save_name) + "/"+str(i+(int(cadr)-1)*81)+".jpg", return_cell(x, y))
                    print("load number on dataset", cadr)
                    cadr+=1
                    if(cadr > int(number_save)): break
                else:
                    game_xmax = 640
                    game_ymax = 480
                    game_xmin = 0
                    game_ymin = 0
                client.send(b'end')
    print(""" ######### STOP ########## """)

pr1 = threading.Thread(target=camera2inet)
pr1.start()
pr2 = threading.Thread(target=ev3)
pr2.start()

while 1:
    frame = cap.read()
    hsv = cv2.cvtColor(frame[game_ymin:game_ymax, game_xmin:game_xmax, :], cv2.COLOR_BGR2HSV)
    frame_gray = cv2.inRange(hsv, (0, 0, 0), (150, 255, 10))
