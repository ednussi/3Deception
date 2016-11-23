#!/usr/bin/env python
# -*- coding: utf-8 -*-

import socket
import tkinter as tk
import random
import subprocess
import signal
from prepare_questions import *
from prepare_questions_multiple import *

def connect_to_fs_receiver_udp(ip="127.0.0.1", port=33444):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect((ip, port))
    return sock


def send_record_flag_udp(sock, flag=2):
    """
    flag = 0 - blank
    flag = 1 - question
    flag = 2 - answer true
    flag = 3 - answer false
    """
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
    label.config(text=('%s') % t, fg="black", font=("Helvetica", 72), justify=tk.CENTER, anchor=tk.CENTER)


def changeRect(root, rect, color, ws):
    rect = tk.Canvas(root, width=ws, height=300)
    # rect.coords()
    # rect.pack(side=tk.RIGHT,anchor=tk.NE)
    rect.grid(row=2,column=3)

    # w.create_line(0, 0, 200, 100)
    # w.create_line(0, 100, 200, 0, fill="red", dash=(4, 4))

    rect.create_rectangle(0, 0, ws, 100, fill="%s" % color)


def nextQ(rect, colors, sock, root, label, b, q_timeout, a_timeout, q, a):
    b.place_forget()
    
    # Change rect color accordingly
    rect_color = colors[0]
    colors.pop(0)
    # print('\n\n'+rect_color)
    # changeRect(root, rect, rect_color, root.winfo_screenwidth())

    blank_time = 500
    after_answer_time = 1000
    # Show blank between questions
    root.after(0, change_label, label, '')
    root.after(0, send_record_flag_udp, sock, 0)

    # Show question
    root.after(blank_time, change_label, label, q)
    root.after(blank_time, send_record_flag_udp, sock, 1)
    # curTime = curTime + QintervalTime

    # Show answer format
    root.after(blank_time + q_timeout, change_label, label, '')
    root.after(blank_time + q_timeout + blank_time, change_label, label, a)
    # if rect_color == 'RED':
    #     flag = 3
    # else:
    #    flag = 2
    flag = 2
    root.after(blank_time + q_timeout + blank_time, send_record_flag_udp, sock, flag)
    # curTime = curTime + AintervalTime

    root.after(blank_time + a_timeout + blank_time + after_answer_time, place_button, b)
    # curTime = curTime + blankWindowTime

def nextQMultiple(rect, colors, sock, root, label, b, q_timeout, a_timeout, q, a):
    b.place_forget()
    
    # Change rect color accordingly
    rect_color = colors[0]
    colors.pop(0)
    # print('\n\n'+rect_color)
    # changeRect(root, rect, rect_color, root.winfo_screenwidth())

    blank_time = 500
    after_answer_time = 1000
    # Show blank between questions
    root.after(0, change_label, label, '')
    root.after(0, send_record_flag_udp, sock, 0)

    # Show question
    root.after(blank_time, change_label, label, q)
    root.after(blank_time, send_record_flag_udp, sock, 1)
    # curTime = curTime + QintervalTime

    # Show answer format
    ans_interval = 0
    ans_time = 3000

    #runs and shows all answers
    for ans in a:
        root.after(blank_time + q_timeout + ans_interval, change_label, label, '')
        root.after(blank_time + q_timeout + blank_time + ans_interval, change_label, label, ans)
        ans_interval += ans_interval + ans_time
    # if rect_color == 'RED':
    #     flag = 3
    # else:
    #    flag = 2
    flag = 2
    root.after(blank_time + q_timeout + blank_time+ans_interval, send_record_flag_udp, sock, flag)
    # curTime = curTime + AintervalTime

    root.after(blank_time + a_timeout + blank_time + after_answer_time+ans_interval, place_button, b)
    # curTime = curTime + blankWindowTime

def place_button(b):
    b.place(relx=0.5, rely=0.3, anchor=tk.CENTER)


def simplePrint():
    print('Pressed')


def next_question(receiver, root, rect, sock, questions, colors):
    if not len(questions):
        send_record_flag_udp(sock, 11)

        # wait for receiver to save data
        receiver.wait()

        # finish
        exit()

    q = questions[0]
    questions.pop(0)

    a = questions[0]
    questions.pop(0)

    return (q, a)

def main():
    #TEST_TYPE = 'SINGLE_OPTION' or TEST_TYPE = 'MULTIPLE_OPTION'
    TEST_TYPE = 'MULTIPLE_OPTION'

    # Creates the full screen and puts empty first label at top
    root, ws, hs = set_window_mid_screen()
    label = tk.Label(root, text="",wraplength=1200)
    # label.grid(row=1,column=2)
    label.place(relx=0.5, rely=0, anchor=tk.N)

    # Chose q_list path according
    if TEST_TYPE == 'SINGLE_OPTION':
        csv_path = "data/questions_vocal_single_option.csv"
        parse_csv_func = prepare_vocal_single_option
        nextQ_func = nextQ
    elif TEST_TYPE == 'MULTIPLE_OPTION':
        csv_path = "data/questions_vocal_multiple_option.csv"
        parse_csv_func = prepare_vocal_multiple_option
        nextQ_func = nextQMultiple
    
    q_list = assoc_array_to_list(parse_csv_func(csv_path))

    tq = list(zip(q_list[::2], q_list[1::2]))
    random.shuffle(tq)
    tq = [q for t in [p for p in tq] for q in t]

    colors = [random.choice(('RED', 'GREEN')) for x in tq]

    receiver = subprocess.Popen(['python', 'fs_receive.py'])
    sock = connect_to_fs_receiver_udp()
    
    rect = tk.Canvas(root, width=200, height=100)
    # changeRect(root, rect, 'green', ws)

    q_timeout = 5000
    a_timeout = 7000

    b = tk.Button(root, text="לחץ לשאלה הבאה", height=1, width=30, font=("Helvetica", 72), foreground='grey', \
                  command=lambda: nextQ_func(rect, colors, sock, root, label, b, q_timeout, a_timeout, *next_question(receiver, root,rect, sock, tq, colors)))
    b.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

    # frame = tk.Frame(root)
    # frame.bind('<Button-1>',*next_question(q_list))
    # frame.place(relx=0.5, rely=0.6, anchor=tk.CENTER, height=50, width=100)

    send_record_flag_udp(sock, 10)

    root.mainloop()

# curTime = 1000
# QintervalTime = 1000
# AintervalTime = 1000
# blankWindowTime = 1000

# for q in range(int(len(q_list) / 2)):

#     # Show question
#     root.after(curTime,change_label, label, q_list[q*2])
#     curTime = curTime + QintervalTime

#     # Show answer format
#     root.after(curTime,change_label, label, q_list[q*2+1])
#     curTime = curTime + AintervalTime

#     # Show blank between questions
#     root.after(curTime,change_label, label, '')
#     curTime = curTime + blankWindowTime

# # for q in range(int(len(q_list) / 2)):
# #     root.after(curTime,change_label, label, q_list[q*6])
# #     curTime = curTime + QintervalTime

# #     for a in range(4):
# #         print("size of q_list is: %d"%len(q_list))
# #         print("q: %d, a: %d, q*6+a: %d"%(q ,a,q*6+a))
# #         root.after(curTime,change_label, label, q_list[q*6+a])
# #         curTime = curTime + AintervalTime

# #         # Show blank window
# #         root.after(curTime,change_label, label, '')
# #         curTime = curTime + blankWindowTime

# #     #Show Question Again
# #     questString = q_list[q*6]
# #     root.after(curTime, change_label, label, questString)
# #     curTime = curTime + QintervalTime



# root.mainloop()

if __name__ == "__main__":
    main()