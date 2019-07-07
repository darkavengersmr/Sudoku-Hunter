#! /usr/bin/python3

# -*- coding: utf-8 -*-

from flask import Flask, render_template, send_file, request, Response
from imutils.video import WebcamVideoStream
import random, time, threading, cv2, math, socket
from keras.models import model_from_json
import numpy as np
import copy
import tensorflow as tf

size_sudoku = 380
position_sudoku = False

game_xmax = 640
game_ymax = 480
game_xmin = 0
game_ymin = 0

draw_points = []

xmin, xmax = 0, 0
ymin, ymax = 0 ,0
rectangle_ok = False

draw = []

img_size = 64

sudoku = np.zeros((9, 9))
write = np.zeros((9, 9))
neyro_date = np.zeros((9, 9))

writes = True

scaning = True

font = cv2.FONT_HERSHEY_SIMPLEX

cap = WebcamVideoStream(src=0).start()

time_apdate = time.time()

send_package = [0, 0, 0, 0, 0]

frame = cap.read()
hsv = cv2.cvtColor(frame[game_ymin:game_ymax, game_xmin:game_xmax, :], cv2.COLOR_BGR2HSV)
frame_gray = cv2.inRange(hsv, (0, 0, 0), (150, 255, 60))

print("Start loding model")
with open('/home/pi/neyro_gun_sudoku_solver/model.json', 'r') as json_file: model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("/home/pi/neyro_gun_sudoku_solver/model.h5")
print("Loaded model from disk")

#global graph
graph = tf.get_default_graph()

class SudokuSolver:
    def solve( puzzle ):
        solution = copy.deepcopy( puzzle )
        if SudokuSolver.solveHelper( solution ): return solution
        return None
    def solveHelper( solution ):
        minPossibleValueCountCell = None
        while True:
            minPossibleValueCountCell = None
            for rowIndex in range( 9 ):
                for columnIndex in range( 9 ):
                    if solution[ rowIndex ][ columnIndex ] != 0: continue
                    possibleValues = SudokuSolver.findPossibleValues( rowIndex, columnIndex, solution )
                    possibleValueCount = len( possibleValues )
                    if possibleValueCount == 0: return False
                    if possibleValueCount == 1: solution[ rowIndex ][ columnIndex ] = possibleValues.pop()
                    if not minPossibleValueCountCell or possibleValueCount < len( minPossibleValueCountCell[ 1 ] ):
                        minPossibleValueCountCell = ( ( rowIndex, columnIndex ), possibleValues )
            if not minPossibleValueCountCell: return True
            elif 1 < len( minPossibleValueCountCell[ 1 ] ): break
        r, c = minPossibleValueCountCell[ 0 ]
        for v in minPossibleValueCountCell[ 1 ]:
            solutionCopy = copy.deepcopy( solution )
            solutionCopy[ r ][ c ] = v
            if SudokuSolver.solveHelper( solutionCopy ):
                for r in range( 9 ):
                    for c in range( 9 ): solution[ r ][ c ] = solutionCopy[ r ][ c ]
                return True
        return False
    def findPossibleValues( rowIndex, columnIndex, puzzle ):
        values = { v for v in range( 1, 10 ) }
        values -= SudokuSolver.getRowValues( rowIndex, puzzle )
        values -= SudokuSolver.getColumnValues( columnIndex, puzzle )
        values -= SudokuSolver.getBlockValues( rowIndex, columnIndex, puzzle )
        return values
    def getRowValues( rowIndex, puzzle ):
        return set( puzzle[ rowIndex ][ : ] )

    def getColumnValues( columnIndex, puzzle ):
        return { puzzle[ r ][ columnIndex ] for r in range( 9 ) }
    def getBlockValues( rowIndex, columnIndex, puzzle ):
        blockRowStart = 3 * ( rowIndex // 3 )
        blockColumnStart = 3 * ( columnIndex // 3 )
        return {
            puzzle[ blockRowStart + r ][ blockColumnStart + c ]
                for r in range( 3 )
                for c in range( 3 )
        }

def return_cell(x, y, inRange=True):
    global frame_gray_ev3, frame_ev3
    if(inRange):
        frame_gray_cell = cv2.cvtColor(frame_ev3[game_ymin:game_ymax, game_xmin:game_xmax, :], cv2.COLOR_BGR2GRAY)
        frame_gray_cell = frame_gray_cell[ymin:ymax, xmin:xmax]
        step_x = frame_gray_cell.shape[1]/9
        step_y = frame_gray_cell.shape[0]/9
        #cv2.imwrite("/home/pi/neyro_gun_sudoku_solver/img2"+str(x)+str(y)+".jpg", frame_gray_cell[round(step_y*(y-1))+5:round(step_y*y)-5, round(step_x*(x-1))+5:round(step_x*x)-5])
        return cv2.resize(frame_gray_cell[round(step_y*(y-1))+5:round(step_y*y)-5, round(step_x*(x-1))+5:round(step_x*x)-5], (64, 64))
    else:
        hsv_inet = cv2.cvtColor(frame_ev3[game_ymin:game_ymax, game_xmin:game_xmax, :], cv2.COLOR_BGR2HSV)
        frame_gray_cell = cv2.inRange(hsv_inet[ymin:ymax, xmin:xmax], (0, 0, 0), (150, 255, 50))
        step_x = frame_gray_cell.shape[1]/9
        step_y = frame_gray_cell.shape[0]/9
        #return cv2.resize(frame_gray_cell[round(step_y*(y-1))+5:round(step_y*y)-5, round(step_x*(x-1))+5:round(step_x*x)-5], (64, 64))
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

    def gen(mask=False):
        while True:
            frame = cap.read()
            if(not mask): frame_gray_inet = cv2.cvtColor(frame[game_ymin:game_ymax, game_xmin:game_xmax, :], cv2.COLOR_BGR2GRAY)
            else:
                hsv_inet = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                frame_gray_inet = cv2.inRange(hsv_inet, (0, 0, 0), (150, 255, 50))
            frame_inet = cv2.cvtColor(frame_gray_inet, cv2.COLOR_GRAY2RGB)

            step_x = frame_gray.shape[1]/9
            step_y = frame_gray.shape[0]/9

            if(not mask):
                step_x = frame_gray.shape[1]/9
                for i in range(11):
                    cv2.line(frame_inet, (round(step_x*i), 0), (round(step_x*i), game_ymax), (0, 255, 0), 1)
                step_y = frame_gray.shape[0]/9
                for i in range(11):
                    cv2.line(frame_inet, (0, round(step_y*i)), (game_xmax, round(step_y*i)), (0, 255, 0), 1)

            for x in range(9):
                for y in range(9):
                    if(int(neyro_date[y, x]) == 0):
                        if(not mask):
                            if(write is not None): cv2.putText(frame_inet,  str(int(write[y, x])), (round(step_x*x)+10, round(step_y*y)+20), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                            else: cv2.putText(frame_inet,  "ER", (round(step_x*x)+10, round(step_y*y)+20), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                        else:
                            if(write is not None): cv2.putText(frame_inet,  str(int(write[y, x])), (round(step_x*x)+10+game_xmin, round(step_y*y)+20+game_ymin), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                            else: cv2.putText(frame_inet,  "ER", (round(step_x*x)+10+game_xmin, round(step_y*y)+20+game_ymin), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    else: cv2.putText(frame_inet, str(int(neyro_date[y, x])), (round(step_x*x)+10, round(step_y*y)+20), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

            if(True):
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
        return Response(gen(True),
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
    global sudoku, neyro_date, write
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
                    i = 0
                    for x in range(9):
                        for y in range(9):
                            cell = return_cell(x+1, y+1, False)
                            i+=1
                            if(np.sum(cell) > 4000): sudoku[y, x] = int(neyro(return_cell(x+1, y+1), x+1, y+1))
                            else: sudoku[y, x] = -1
                    print("neyro net stop")
                    if(0 not in sudoku):
                        sudoku[sudoku == -1] = 0
                        neyro_date = sudoku.copy()
                        write = SudokuSolver.solve(sudoku)
                        print(neyro_date, "Рапознано нейросетью")
                        print("вызов решателя")
                        if(write is not None):
                            print(write, "Решённый судоку")

                            client.send(b'y')

                            send_string = ""
                            for x in range(9):
                                for y in range(9):
                                    send_string += str(int(write[y, x]))
                            client.send(send_string.encode())

                        else:
                            print("ERROR")
                            client.send(b'n')
                    else:
                        print("ERROR")
                        client.send(b'n')
                else:
                    game_xmax = 640
                    game_ymax = 480
                    game_xmin = 0
                    game_ymin = 0
                client.send(b'end')

def neyro(img, x, y):
    #img = 255 - img
    min_ = np.min(img)
    max_ = np.max(img)
    img = (img - min_) * (255/(max_ - min_))
    img = img.astype("int32")

#    cv2.imwrite("/home/pi/neyro_gun_sudoku_solver/img"+str(x)+str(y)+".jpg", img)
#    print(img.shape)

    img = img.reshape(1, 1, img_size, img_size)
    img = img.astype("float32")
    img /= 255

    with graph.as_default(): out = model.predict([img])

    num = np.argmax(out[0])+1
    print(num, x, y)
    return num

pr1 = threading.Thread(target=camera2inet)
pr1.start()
pr2 = threading.Thread(target=ev3)
pr2.start()

while 1:
    frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_gray = cv2.inRange(hsv, (0, 0, 0), (150, 255, 10))
