#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import socket
import subprocess
import tkinter as tk
import winsound

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
    """
    Connect to FaceShift receiver via UDP
    :param ip:
    :param port:
    :return: connected socket
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect((ip, port))
    return sock


def send_record_flag_udp(sock, flag=RECORD_FLAG_PAUSE):
    """
    Send control flag to fs_receiver via given socket
    :param sock:
    :param flag:
    :return: None
    """
    sock.send(bytes(str(flag), 'utf-8'))


def disconnect_from_fs_receiver_udp(sock):
    """
    Disconnect from fs_receiver
    :param sock:
    :return:
    """
    sock.close()


def set_window_mid_screen():
    """
    Create fullscreen tkinter window
    :return: tk root, width and height of new window
    """
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
    """
    Update question label text
    :param label:
    :param t:
    :return:
    """
    label.config(text='%s' % t, fg="black", font=("Helvetica", 72), justify=tk.CENTER, anchor=tk.CENTER)


def show_control_shape(root, flag=COLOR_FALSE, time=1000):
    """
    Display control shape
    """
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


def show_next_question(colors, sock, root, label, b, q_timeout, q_type, q):
    """
    Display next question
    :param colors: array of shape colors
    :param sock: fs_receiver socket
    :param root: tk window root
    :param label: question label
    :param b: button_next_question
    :param q_timeout: time to display the question
    :param q: the question to display
    :return: None
    """
    b.place_forget()

    shape_color = colors[0]
    colors.pop(0)

    # Show control shape
    root.after(TIME_BLANK, show_control_shape, root, shape_color, TIME_CONTROL_SHAPE)
    # Send shape control flag
    root.after(TIME_BLANK, send_record_flag_udp, sock,
               RECORD_FLAG_CONTROL_SHAPE_TRUE if shape_color == COLOR_TRUE else RECORD_FLAG_CONTROL_SHAPE_FALSE)

    # Show blank
    root.after(TIME_BLANK + TIME_CONTROL_SHAPE, change_label, label, '')
    # Send blank control flag
    root.after(TIME_BLANK + TIME_CONTROL_SHAPE, send_record_flag_udp, sock, RECORD_FLAG_PAUSE)

    # Show question
    root.after(TIME_BLANK + TIME_CONTROL_SHAPE + TIME_BLANK, change_label, label,
               q['true'] if shape_color == COLOR_TRUE else q['false'])
    # Send question control flag
    root.after(TIME_BLANK + TIME_CONTROL_SHAPE + TIME_BLANK, send_record_flag_udp, sock,
               RECORD_FLAG_QUESTION_TRUE if shape_color == COLOR_TRUE else RECORD_FLAG_QUESTION_FALSE)
    # Read question out loud
    root.after(TIME_BLANK + TIME_CONTROL_SHAPE + TIME_BLANK,
               winsound.PlaySound, 'voice/{}_{}.wav'.format(q_type, shape_color == COLOR_TRUE),
               winsound.SND_FILENAME | winsound.SND_ASYNC)

    # Show button_next_question
    root.after(TIME_BLANK + TIME_CONTROL_SHAPE + TIME_BLANK + q_timeout, place_button, b)


def place_button(b):
    """
    Display button_next_question
    :param b: button handle
    """
    b.place(relx=0.5, rely=0.4, anchor=tk.CENTER)


def get_next_question(receiver, sock, questions):
    """
    Get next question from question list
    If no more questions left, send stop flag and exit
    :param receiver:
    :param sock:
    :param questions:
    :return:
    """
    if not len(questions):
        # End of questions

        send_record_flag_udp(sock, RECORD_FLAG_END_SESSION)

        # Wait for receiver to save data
        receiver.wait()

        # Finish
        exit()

    q_type = questions[0]
    questions.pop(0)

    q = questions[0]
    questions.pop(0)

    return q_type, q


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

    b = tk.Button(root, text="לחץ לשאלה הבאה", height=1, width=30, font=("Helvetica", 72), foreground='grey',
                  command=lambda: show_next_question(colors, sock, root, label, b, q_timeout,
                                                     *get_next_question(receiver, sock, tq)))
    b.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

    # frame = tk.Frame(root)
    # frame.bind('<Button-1>',*next_question(q_list))
    # frame.place(relx=0.5, rely=0.6, anchor=tk.CENTER, height=50, width=100)

    send_record_flag_udp(sock, 10)

    root.mainloop()


if __name__ == "__main__":
    main()
