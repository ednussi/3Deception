#!/usr/bin/env python
# -*- coding: utf-8 -*-

import socket
import subprocess
import tkinter as tk
import argparse
import pygame
from PIL import ImageTk, Image
from fs_receive import RecordFlags
from prepare_questions import *

COLOR_TRUE = 'GREEN'
COLOR_FALSE = 'RED'

TIME_BLANK = 500
TIME_QUESTION = 4000
TIME_ANSWER = 2000
TIME_CONTROL_SHAPE = 2000
TIME_CATCH_ITEM = 4000

REPEAT_TIMES = 4

AUDIO_SEX = 'male'
SUBJECT_ID = None

IDX_AUDIO = IDX_QUESTION_TYPE = 0
IDX_TEXT = IDX_QUESTION_DATA = 1

RUN_EXAMPLE = True

QUESTION_NUMBER = 0
IMAGE_COUNTER = 0

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
    root.configure(background='black')

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
    label.config(text='%s' % t, fg="grey", bg='black', font=("Helvetica", 72), justify=tk.CENTER, anchor=tk.CENTER)


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


def show_fixation_cross(root, time=5000):
    """
    Displays fixation cross in the mid screen for focus
    """

    canvas = tk.Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
    canvas.grid(row=1, column=1)
    cw = canvas.winfo_screenwidth()
    ch = canvas.winfo_screenheight()

    line_length = 100
    x = cw / 2
    y = ch / 10

    canvas.create_line(x-line_length, y, x+line_length, y, width=5.0)
    canvas.create_line(x, y-line_length, x, y+line_length, width=5.0)

    root.after(time, canvas.grid_forget)


def get_random_image():
    global IMAGE_COUNTER
    if IMAGE_COUNTER % 5 == 0:
        img = ImageTk.PhotoImage(Image.open('img/tortoise.png'))
    elif IMAGE_COUNTER % 5 == 1:
        img = ImageTk.PhotoImage(Image.open('img/elephant.png'))
    elif IMAGE_COUNTER % 5 == 2:
        img = ImageTk.PhotoImage(Image.open('img/dog.png'))
    elif IMAGE_COUNTER % 5 == 3:
        img = ImageTk.PhotoImage(Image.open('img/cat.png'))
    else:
        img = ImageTk.PhotoImage(Image.open('img/bird.png'))

    IMAGE_COUNTER += 1
    return img


def show_catch_item(sock, root, label, receiver, qlist, b, time=TIME_CATCH_ITEM):
    global img
    global cb
    img = get_random_image()

    cb = tk.Button(root, bd=0, command=lambda: handle_catch_button_click(sock, root, label, b, receiver, qlist))
    cb.config(image=img)

    cw = root.winfo_screenwidth()
    ch = root.winfo_screenheight()

    x = random.randint(img.width(), cw - img.width())
    y = random.randint(img.height(), ch - img.height())

    cb.place(x=x, y=y, anchor=tk.CENTER)


def handle_catch_button_click(sock, root, label, b, receiver, qlist):
    cb.place_forget()
    show_next_question(sock, root, label, b, get_next_question(root, receiver, sock, qlist), receiver, qlist)


def show_example_q(sock, root, label, b, q, b_q):

    b.place_forget()

    tb = TIME_BLANK

    # Show blank
    root.after(tb, change_label, label, '')
    # Send blank control flag
    root.after(tb, send_record_flag_udp, sock, RecordFlags.RECORD_FLAG_CHANGE)
    root.after(tb, send_record_flag_udp, sock, RecordFlags.RECORD_FLAG_PAUSE)

    # Show question
    root.after(tb + TIME_BLANK, change_label, label, q[IDX_QUESTION_DATA]['question'][IDX_TEXT])
    # Send question control flag
    root.after(tb + TIME_BLANK, send_record_flag_udp, sock, RecordFlags.RECORD_FLAG_CHANGE)
    root.after(tb + TIME_BLANK, send_record_flag_udp, sock, RecordFlags.RECORD_FLAG_EXAMPLE_QUESTION)
    # Read question out loud
    root.after(tb + TIME_BLANK, read_question, q[IDX_QUESTION_DATA]['question'][IDX_AUDIO])

    # randomize answers order
    answers = q[IDX_QUESTION_DATA]['false'][:]
    random.shuffle(answers)

    # get two false answers
    answers = answers[:2]

    # true answer is never first
    true_idx = random.randint(1, len(answers))
    answers.insert(true_idx, q[IDX_QUESTION_DATA]['true'])

    for i, a in enumerate(answers):
        flag = RecordFlags.RECORD_FLAG_EXAMPLE_ANSWER_TRUE \
            if i == true_idx else RecordFlags.RECORD_FLAG_EXAMPLE_ANSWER_FALSE
        # Show answer
        root.after(tb + TIME_BLANK + TIME_QUESTION + i * (TIME_BLANK + TIME_ANSWER),
                   change_label, label, a[IDX_TEXT])
        # Send answer control flag
        root.after(tb + TIME_BLANK + TIME_QUESTION + i * (TIME_BLANK + TIME_ANSWER),
                   send_record_flag_udp, sock, RecordFlags.RECORD_FLAG_CHANGE)
        root.after(tb + TIME_BLANK + TIME_QUESTION + i * (TIME_BLANK + TIME_ANSWER),
                   send_record_flag_udp, sock, flag)
        # Read answer out loud
        root.after(tb + TIME_BLANK + TIME_QUESTION + i * (TIME_BLANK + TIME_ANSWER),
                   read_question, a[IDX_AUDIO])

    # Show button_next_question
    root.after(tb + TIME_BLANK + TIME_QUESTION + len(answers) * (TIME_BLANK + TIME_ANSWER) + TIME_BLANK,
               lambda: b_q.place(relx=0.5, rely=0.4, anchor=tk.CENTER))


def show_next_question(sock, root, label, b, q, receiver, qlist):
    """
    Display next question
    :param sock: fs_receiver socket
    :param root: tk window root
    :param label: question label
    :param b: button_next_question
    :param q: the question and its answers
    :return: None
    """
    global QUESTION_NUMBER
    QUESTION_NUMBER += 1

    b.place_forget()

    tb = TIME_BLANK

    # Show blank
    root.after(tb, change_label, label, '')
    # Send blank control flag
    root.after(tb, send_record_flag_udp, sock, RecordFlags.RECORD_FLAG_CHANGE)
    root.after(tb, send_record_flag_udp, sock, RecordFlags.RECORD_FLAG_PAUSE)

    if (QUESTION_NUMBER) % 4 == 0:
        qlist.insert(0, q)  # put question back to queue
        root.after(tb + TIME_BLANK, show_catch_item, sock, root, label, receiver, qlist, b, TIME_CATCH_ITEM)
        return

    # Show question
    root.after(tb + TIME_BLANK, change_label, label, q[IDX_QUESTION_DATA]['question'][IDX_TEXT])
    # Send question control flag
    root.after(tb + TIME_BLANK, send_record_flag_udp, sock, RecordFlags.RECORD_FLAG_CHANGE)
    root.after(tb + TIME_BLANK, send_record_flag_udp, sock, RecordFlags.RECORD_FLAG_QUESTION)
    # Read question out loud
    root.after(tb + TIME_BLANK, read_question, q[IDX_QUESTION_DATA]['question'][IDX_AUDIO])

    # randomize answers order
    answers = q[IDX_QUESTION_DATA]['false'][:]
    random.shuffle(answers)

    # get two false answers
    answers = answers[:2]

    # true answer is never first
    true_idx = random.randint(1, len(answers))
    answers.insert(true_idx, q[IDX_QUESTION_DATA]['true'])

    for i, a in enumerate(answers):
        flag = RecordFlags.RECORD_FLAG_ANSWER_TRUE if i == true_idx else RecordFlags.RECORD_FLAG_ANSWER_FALSE

        # Show answer
        root.after(tb + TIME_BLANK + TIME_QUESTION + i * (TIME_BLANK + TIME_ANSWER),
                   change_label, label, a[IDX_TEXT])
        # Send answer control flag
        root.after(tb + TIME_BLANK + TIME_QUESTION + i * (TIME_BLANK + TIME_ANSWER),
                   send_record_flag_udp, sock, RecordFlags.RECORD_FLAG_CHANGE)
        root.after(tb + TIME_BLANK + TIME_QUESTION + i * (TIME_BLANK + TIME_ANSWER),
                   send_record_flag_udp, sock, flag)
        # Read answer out loud
        root.after(tb + TIME_BLANK + TIME_QUESTION + i * (TIME_BLANK + TIME_ANSWER),
                   read_question, a[IDX_AUDIO])

    # Show button_next_question
    root.after(tb + TIME_BLANK + TIME_QUESTION + len(answers) * (TIME_BLANK + TIME_ANSWER) + TIME_BLANK,
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
    b['text'] = 'לחץ לשאלה הבאה'


def get_next_question(root, receiver, sock, questions):
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
    else:
        question_holder = questions[0]
        questions.pop(0)

        return question_holder


def run_qs(root, sock, receiver, b):

    b.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

    send_record_flag_udp(sock, RecordFlags.RECORD_FLAG_START_SESSION)


def run_example_qs(root, sock, receiver, b_q, label):

    qlist = [item for item in prepare_cit(path='data/example_questions_cit.csv', male=AUDIO_SEX == 'male').items()]

    b = tk.Button(root,bd=0, text="לחץ להתחיל שאלת דוגמא", height=1, width=30, font=("Helvetica", 72),
                  fg='gray', bg='black', activebackground='black', activeforeground='gray',
                  command=lambda: show_example_q(sock, root, label, b, get_next_question(root, receiver, sock, qlist), b_q))
    b.place(relx=0.5, rely=0.1, anchor=tk.CENTER)


def main():
    # Creates the full screen and puts empty first label at top
    root, ws, hs = set_window_mid_screen()

    # Get twice as much questions (half for true half for lies)
    # and sets colors to be tag for lie\truth
    qlist = [item for item in prepare_cit(path='data/{}/questions_cit.csv'.format(SUBJECT_ID), male=AUDIO_SEX == 'male').items()]
    qlist *= REPEAT_TIMES # Multiple each question
    random.shuffle(qlist) # randomize order of questions

    pygame.mixer.init()

    receiver = subprocess.Popen(['python', 'fs_receive.py'])
    sock = connect_to_fs_receiver_udp()

    label = tk.Label(root, text="", wraplength=1200, bg='black')
    # label.grid(row=1,column=2)
    label.place(relx=0.5, rely=0, anchor=tk.N)

    b = tk.Button(root, bd=0, text="לחץ להתחלת השאלון", height=1, width=30, font=("Helvetica", 72),
                  fg='grey', bg='black', activebackground='black', activeforeground='gray',
                  command=lambda: show_next_question(sock, root, label, b,
                                                     get_next_question(root, receiver, sock, qlist), receiver, qlist))

    global main_button
    main_button = b

    if (RUN_EXAMPLE):
        run_example_qs(root, sock, receiver, b, label)
    else:
        run_qs(root, sock, receiver, b)

    #show_fixation_cross()
    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--voice-sex', dest='sex', choices=['male', 'female'])

    parser.add_argument('-r', '--repeat', dest='repeat', type=int)

    parser.add_argument('-i', '--id', dest='subject_id', required=True)

    parser.add_argument('-d', '--devmode', dest='devmode', action='store_true')

    parser.add_argument('-a', '--numanswers', dest='numanswers', type=int, choices=[3, 4, 5, 6])

    parser.set_defaults(devmode=False, sex='male', repeat=4, numanswers=3)

    args = parser.parse_args()

    SUBJECT_ID = args.subject_id

    if args.repeat is not None:
        REPEAT_TIMES = args.repeat
    if args.sex is not None:
        AUDIO_SEX = args.sex

    if args.devmode:
        TIME_BLANK = 50
        TIME_QUESTION = 400
        TIME_ANSWER = 200
        TIME_CONTROL_SHAPE = 200
        TIME_CATCH_ITEM = 400

    main_button = None

    main()
