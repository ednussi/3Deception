#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import socket
import subprocess
import tkinter as tk
import argparse
import pygame

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
RECORD_FLAG_START_SESSION = 8

COLOR_TRUE = 'GREEN'
COLOR_FALSE = 'RED'

TIME_BLANK = 500
TIME_QUESTION = 4000
TIME_ANSWER = 5000
TIME_CONTROL_SHAPE = 2000

REPEAT_TIMES = 4

AUDIO_SEX = 'male'
AUDIO_FALSE_OPTIONS = {
    'first_name': 'daniel',
    'surname': 'fridman',
    'mother_name': 'abigail',
    'birth_country': 'ita',
    'birth_month': 'sep'
}
SUBJECT_ID = None


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


def show_next_question(audio_flags, sock, root, label, b, q_type, q):
    """
    Display next question
    :param audio_flags:
    :param sock: fs_receiver socket
    :param root: tk window root
    :param label: question label
    :param b: button_next_question
    :param q_type: type of question
    :param q: the question to display
    :return: None
    """
    b.place_forget()

    audio_flag = audio_flags[0]
    audio_flags.pop(0)

    # Show blank
    root.after(TIME_BLANK + TIME_CONTROL_SHAPE, change_label, label, '')
    # Send blank control flag
    root.after(TIME_BLANK + TIME_CONTROL_SHAPE, send_record_flag_udp, sock, RECORD_FLAG_PAUSE)

    # Show question
    root.after(TIME_BLANK + TIME_CONTROL_SHAPE + TIME_BLANK, change_label, label, q)
    # Send question control flag
    root.after(TIME_BLANK + TIME_CONTROL_SHAPE + TIME_BLANK, send_record_flag_udp, sock,
               RECORD_FLAG_QUESTION_TRUE if q_type.endswith('true') else RECORD_FLAG_QUESTION_FALSE)
    # Read question out loud
    root.after(TIME_BLANK + TIME_CONTROL_SHAPE + TIME_BLANK,
               read_question, audio_flag, q_type.endswith('true'))

    # Show button_next_question
    root.after(TIME_BLANK + TIME_CONTROL_SHAPE + TIME_BLANK + TIME_QUESTION, place_button, b)


def read_question(audio_flag, truth):
    """
    Read question out loud from wave file
    :param audio_flag: predefined question type
    :param truth:
    :return: None
    """
    if not truth:
        pygame.mixer.music.load('voice/{}/{}.mp3'.format(AUDIO_SEX, audio_flag))
        pygame.mixer.music.play()
    else:
        pygame.mixer.music.load('data/{}/voice/{}.mp3'.format(SUBJECT_ID, audio_flag))
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

    pygame.mixer.init()

    # Get twice as much questions (half for true half for lies)
    # and sets colors to be tag for lie\truth
    q_list = assoc_array_to_list(prepare_cit(path='data/{}/questions_cit.csv'.format(SUBJECT_ID)))

    audio_flags = []

    for qtype, ans in zip(q_list[::2], q_list[1::2]):
        audio_flag = '_'.join(qtype.split('_')[:2]) + '/' + ans[0]
        audio_flags.append(audio_flag)

    q_list *= REPEAT_TIMES
    audio_flags *= REPEAT_TIMES

    tq = list(zip(q_list[::2], q_list[1::2], audio_flags))
    random.shuffle(tq)
    audio_flags = [q[-1] for q in tq]  # get audio flags out
    tq = [(q[0], q[1][1]) for q in tq]  # return tq to tuples of 2
    tq = [q for t in [p for p in tq] for q in t]

    receiver = None  # subprocess.Popen(['python', 'fs_receive.py'])
    sock = connect_to_fs_receiver_udp()

    b = tk.Button(root, text="לחץ לשאלה הבאה", height=1, width=30, font=("Helvetica", 72), foreground='grey',
                  command=lambda: show_next_question(audio_flags, sock, root, label, b,
                                                     *get_next_question(receiver, sock, tq)))
    b.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

    send_record_flag_udp(sock, RECORD_FLAG_START_SESSION)

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
