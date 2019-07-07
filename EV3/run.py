#!/usr/bin/env python3

from ev3dev.ev3 import *
import time, socket
from PIL import Image, ImageDraw, ImageFont

btn = Button()

lcd = Screen()
lcd.clear()
lcd.update()

line = []

ts = TouchSensor('in3')
server = socket.socket()
server.bind(("192.168.32.209", 9090))
server.listen(10)
Sound.beep()
conn, add = server.accept()
Sound.beep()

sudoku = [[ 0 for j in range(9)] for i in range(9)]

def write(num, x, y):
    f = ImageFont.truetype('FreeMonoBold.ttf', 15)
    lcd.draw.text((int(25+x*14.22), int(14.22*y)), str(num), font=f)

def draw_sudoku():
    for i in range(10): lcd.draw.line((int(14.22*i-3)+25, 0, int(14.22*i-3)+25, 128))
    for i in range(10): lcd.draw.line((22, int(14.22*i), 150, int(14.22*i)))

while True:
    lcd.clear()

    f = ImageFont.truetype('FreeMonoBold.ttf', 10)
    lcd.draw.text((10, 10), "no signal", font=f)
    lcd.update()

    conn.recv(5)

    lcd.clear()
    line.clear()
    for i in range(4):
        data = conn.recv(6)
        data = data.decode("utf-8")
        line.append((int(int(data[:3])/3.59), int(int(data[3:])/3.75)))
    for i in range(3):
        lcd.draw.line((line[i][0], line[i][1], line[i+1][0], line[i+1][1]))
    lcd.draw.line((line[0][0], line[0][1], line[3][0], line[3][1]))
    if(btn.enter or ts.value()): 
        Sound.beep()
        conn.send(b's')
        Sound.speak("Image is being processed, please wait")
        data = conn.recv(1)
        data = data.decode("utf-8")
        if(data == "y"):
            Sound.speak("Solved your sudoku solution")
            data = conn.recv(81)
            data = data.decode("utf-8")
            lcd.clear()
            i=0
            for x in range(9):
                for y in range(9):
                    sudoku[x][y] = int(data[i])
                    write(int(data[i]), x, y)
                    i+=1
            draw_sudoku()
            lcd.update()
            print(sudoku)
            while not btn.backspace: pass            
        else: 
            print("NO ARRAY")
            Sound.speak("Array not passed. Error recognition. Repeat fire")

    else: conn.send(b'n')
    lcd.update()
    conn.recv(3)
    time.sleep(1)
