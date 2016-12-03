#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import socket
import subprocess
import tkinter as tk

from prepare_questions import *

# Record flags
RECORD_FLAG_PAUSE = 0
RECORD_FLAG_QUESTION_TRUE = 1
RECORD_FLAG_QUESTION_FALSE = 2
RECORD_FLAG_ANSWER_TRUE = 3
RECORD_FLAG_ANSWER_FALSE = 4
RECORD_FLAG_CONTROL_SHAPE_TRUE = 5
RECORD_FLAG_CONTROL_SHAPE_FALSE = 6
RECORD_FLAG_END_SESSION = 7

COLOR_TRUE = 'YELLOW'
COLOR_FALSE = 'BLUE'

TIME_BLANK = 500
TIME_QUESTION = 4000
TIME_ANSWER = 5000
TIME_CONTROL_SHAPE = 2000

REPEAT_TIMES = 1


def connect_to_fs_receiver_udp(ip="127.0.0.1", port=33444):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect((ip, port))
    return sock


def send_record_flag_udp(sock, flag=RECORD_FLAG_PAUSE):
    sock.send(bytes(str(flag), 'utf-8'))


def disconnect_from_fs_receiver_udp(sock):
    sock.close()


def set_window_mid_screen():
    root = tk.Tk()  # create a Tk root window

    w = 0  # width for the Tk root
    h = 0  # height for the Tk root

    # get screen width and height
    ws = root.winfo_screenwidth()  # width of the screen
    hs = root.winfo_screenheight()  # height of the screen
    root.overrideredirect(1)
    root.geometry('%dx%d+%d+%d' % (ws, hs, w, h))
    return root, ws, hs


def change_label(label, t):
    label.config(text='%s' % t, fg="black", font=("Helvetica", 72), justify=tk.CENTER, anchor=tk.CENTER)


def show_control_shape(root, flag=COLOR_FALSE, time=1000):
    canvas = tk.Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
    canvas.grid(row=1, column=1)
    cw = canvas.winfo_screenwidth()
    ch = canvas.winfo_screenheight()

    radius = min(cw / 4, ch / 4)

    x0 = int((cw / 2) - radius)
    x1 = int((cw / 2) + radius)

    y0 = int((ch / 2) - radius)
    y1 = int((ch / 2) + radius)

    if flag == COLOR_FALSE:
        canvas.create_oval(x0, y0, x1, y1, fill=flag)
    else:
        canvas.create_rectangle(x0, y0, x1, y1, fill=flag)

    root.after(time, canvas.grid_forget)


def nextQ(colors, sock, root, label, b, q_timeout, a_timeout, q, a):
    b.place_forget()

    shape_color = colors[0]
    colors.pop(0)

    # Show control shape
    root.after(TIME_BLANK, show_control_shape, root, shape_color, TIME_CONTROL_SHAPE)
    root.after(TIME_BLANK, send_record_flag_udp, sock,
               RECORD_FLAG_CONTROL_SHAPE_TRUE if shape_color == COLOR_TRUE else RECORD_FLAG_CONTROL_SHAPE_FALSE)

    # Show blank
    root.after(TIME_BLANK + TIME_CONTROL_SHAPE, change_label, label, '')
    root.after(TIME_BLANK + TIME_CONTROL_SHAPE, send_record_flag_udp, sock, RECORD_FLAG_PAUSE)

    # Show question
    root.after(TIME_BLANK + TIME_CONTROL_SHAPE + TIME_BLANK, change_label, label, a)
    root.after(TIME_BLANK + TIME_CONTROL_SHAPE + TIME_BLANK, send_record_flag_udp, sock,
               RECORD_FLAG_QUESTION_TRUE if shape_color == COLOR_TRUE else RECORD_FLAG_QUESTION_FALSE)

    # # Show blank
    # root.after(TIME_BLANK + TIME_CONTROL_SHAPE + TIME_BLANK + q_timeout, change_label, label, '')
    # root.after(TIME_BLANK + TIME_CONTROL_SHAPE + TIME_BLANK + q_timeout, send_record_flag_udp, sock, RECORD_FLAG_PAUSE)
    #
    # # Show answer format
    # root.after(TIME_BLANK + TIME_CONTROL_SHAPE + TIME_BLANK + q_timeout + TIME_BLANK, change_label, label, answer)
    # root.after(TIME_BLANK + TIME_CONTROL_SHAPE + TIME_BLANK + q_timeout + TIME_BLANK, send_record_flag_udp, sock,
    #            RECORD_FLAG_ANSWER_TRUE if shape_color == COLOR_TRUE else RECORD_FLAG_ANSWER_FALSE)
    #
    root.after(TIME_BLANK + TIME_CONTROL_SHAPE + TIME_BLANK + q_timeout, place_button, b)


def place_button(b):
    b.place(relx=0.5, rely=0.3, anchor=tk.CENTER)


def simplePrint():
    print('Pressed')


def next_question(receiver, sock, questions):
    if not len(questions):
        # End of questions

        send_record_flag_udp(sock, RECORD_FLAG_END_SESSION)

        # Wait for receiver to save data
        receiver.wait()

        # Finish
        exit()

    q = questions[0]
    questions.pop(0)

    a = questions[0]
    questions.pop(0)

    return q, a


def main():
    # Creates the full screen and puts empty first label at top
    root, ws, hs = set_window_mid_screen()
    label = tk.Label(root, text="", wraplength=1200)
    # label.grid(row=1,column=2)
    label.place(relx=0.5, rely=0, anchor=tk.N)

    # Get twice as much questions (half for true half for lies)
    # and sets colors to be tag for lie\truth
    q_list = assoc_array_to_list(prepare_slt())
    q_amount = int(len(q_list) / 2)
    truth_lies_multiplyer = 2
    q_list *= truth_lies_multiplyer  # get questions twice
    colors = [COLOR_FALSE] * q_amount + [COLOR_TRUE] * q_amount  # get equal amount of true\lie amounts

    # Times for repetation of questions:
    colors *= REPEAT_TIMES
    q_list *= REPEAT_TIMES

    tq = list(zip(q_list[::2], q_list[1::2], colors))
    random.shuffle(tq)
    colors = [q[-1] for q in tq]  # get colors out
    tq = [q[:2] for q in tq]  # return tq to tupples of 2
    tq = [q for t in [p for p in tq] for q in t]

    receiver = subprocess.Popen(['python', 'fs_receive.py'])
    sock = connect_to_fs_receiver_udp()

    q_timeout = TIME_QUESTION
    a_timeout = TIME_ANSWER

    b = tk.Button(root, text="לחץ לשאלה הבאה", height=1, width=30, font=("Helvetica", 72), foreground='grey',
                  command=lambda: nextQ(colors, sock, root, label, b, q_timeout, a_timeout,
                                        *next_question(receiver, sock, tq)))
    b.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

    # frame = tk.Frame(root)
    # frame.bind('<Button-1>',*next_question(q_list))
    # frame.place(relx=0.5, rely=0.6, anchor=tk.CENTER, height=50, width=100)

    send_record_flag_udp(sock, 10)

    root.mainloop()


if __name__ == "__main__":
    main()
