#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import socket
import subprocess
import tkinter as tk
import argparse
import pygame

from fs_receive import RecordFlags
from prepare_questions import *

COLOR_TRUE = 'GREEN'
COLOR_FALSE = 'RED'

TIME_BLANK = 500
TIME_QUESTION = 4000
TIME_ANSWER = 2000
TIME_CONTROL_SHAPE = 2000

REPEAT_TIMES = 4

AUDIO_SEX = 'male'
SUBJECT_ID = None

IDX_AUDIO = IDX_QUESTION_TYPE = 0
IDX_TEXT = IDX_QUESTION_DATA = 1


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


def send_record_flag_udp(sock, flag=RecordFlags.RECORD_FLAG_PAUSE):
    """
    Send control flag to fs_receiver via given socket
    :param sock:
    :param flag:
    :return: None
    """
    sock.send(bytes(str(int(flag)), 'utf-8'))


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


def show_next_question(sock, root, label, b, q):
    """
    Display next question
    :param sock: fs_receiver socket
    :param root: tk window root
    :param label: question label
    :param b: button_next_question
    :param q: the question and its answers
    :return: None
    """
    b.place_forget()

    # Show blank
    root.after(TIME_BLANK, change_label, label, '')
    # Send blank control flag
    root.after(TIME_BLANK, send_record_flag_udp, sock, RecordFlags.RECORD_FLAG_PAUSE)

    # Show question
    root.after(TIME_BLANK + TIME_BLANK, change_label, label, q[IDX_QUESTION_DATA]['question'][IDX_TEXT])
    # Send question control flag
    root.after(TIME_BLANK + TIME_BLANK, send_record_flag_udp, sock, RecordFlags.RECORD_FLAG_QUESTION)
    # Read question out loud
    root.after(TIME_BLANK + TIME_BLANK, read_question, q[IDX_QUESTION_DATA]['question'][IDX_AUDIO])

    # Show answers in random order
    answers = q[IDX_QUESTION_DATA]['false'] + [q[IDX_QUESTION_DATA]['true']]
    random.shuffle(answers)

    for i, a in enumerate(answers):
        # Show answer
        root.after(TIME_BLANK + TIME_BLANK + TIME_QUESTION + i * (TIME_BLANK + TIME_ANSWER),
                   change_label, label, a[IDX_TEXT])
        # Send answer control flag
        root.after(TIME_BLANK + TIME_BLANK + TIME_QUESTION + i * (TIME_BLANK + TIME_ANSWER),
                   send_record_flag_udp, sock, RecordFlags.RECORD_FLAG_ANSWER_FALSE)
        # Read answer out loud
        root.after(TIME_BLANK + TIME_BLANK + TIME_QUESTION + i * (TIME_BLANK + TIME_ANSWER),
                   read_question, a[IDX_AUDIO])

    # Show button_next_question
    root.after(TIME_BLANK + TIME_BLANK + TIME_QUESTION + len(answers) * (TIME_BLANK + TIME_ANSWER) + TIME_BLANK,
               place_button, b)


def read_question(audio_flag):
    """
    Read question out loud from wave file
    :param audio_flag: predefined audio path
    :return: None
    """
    if audio_flag != 'no_audio':
        pygame.mixer.music.load(audio_flag)
        pygame.mixer.music.play()


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

        send_record_flag_udp(sock, RecordFlags.RECORD_FLAG_END_SESSION)

        # Wait for receiver to save data
        receiver.wait()

        # Finish
        exit()

    question_holder = questions[0]
    questions.pop(0)

    return question_holder


def main():
    # Creates the full screen and puts empty first label at top
    root, ws, hs = set_window_mid_screen()
    label = tk.Label(root, text="", wraplength=1200)
    # label.grid(row=1,column=2)
    label.place(relx=0.5, rely=0, anchor=tk.N)

    pygame.mixer.init()

    # Get twice as much questions (half for true half for lies)
    # and sets colors to be tag for lie\truth
    qlist = [item for item in prepare_cit(path='data/{}/questions_cit.csv'.format(SUBJECT_ID), male=AUDIO_SEX == 'male').items()]

    qlist *= REPEAT_TIMES

    random.shuffle(qlist)

    receiver = subprocess.Popen(['python', 'fs_receive.py'])
    sock = connect_to_fs_receiver_udp()

    b = tk.Button(root, text="לחץ לשאלה הבאה", height=1, width=30, font=("Helvetica", 72), foreground='grey',
                  command=lambda: show_next_question(sock, root, label, b, get_next_question(receiver, sock, qlist)))
    b.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

    send_record_flag_udp(sock, RecordFlags.RECORD_FLAG_START_SESSION)

    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--voice-sex', dest='sex', choices=['male', 'female'])

    parser.add_argument('-R', '--repeat', dest='repeat', type=int)

    parser.add_argument('-I', '--id', dest='subject_id', required=True)

    args = parser.parse_args()

    SUBJECT_ID = args.subject_id

    if args.repeat is not None:
        REPEAT_TIMES = args.repeat
    if args.sex is not None:
        AUDIO_SEX = args.sex

    main()
