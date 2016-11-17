import socket
import tkinter as tk
from prepare_questions import *


def connect_to_fs_receiver_udp(ip="127.0.0.1", port=33444):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect((ip, port))
    return sock


def send_record_flag_udp(sock, flag=True):
    sock.send(str(int(flag)))


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
    label.config(text=('%s') % t, fg="green", font=("Helvetica", 72), justify=tk.CENTER, anchor=tk.CENTER)


def changeRect(root, rect, color, ws):
    rect = tk.Canvas(root, width=ws, height=100)
    # rect.coords()
    # rect.pack(side=tk.RIGHT,anchor=tk.NE)
    rect.pack()

    # w.create_line(0, 0, 200, 100)
    # w.create_line(0, 100, 200, 0, fill="red", dash=(4, 4))

    rect.create_rectangle(0, 0, ws, 100, fill="%s" % color)


def nextQ(word, root, label, q_list):
    for q in range(len(q_list) / 2):
        # Show question
        root.after(0, change_label, label, q_list[q * 2])
        # curTime = curTime + QintervalTime

        # Show answer format
        root.after(1000, change_label, label, q_list[q * 2 + 1])
        # curTime = curTime + AintervalTime

        # Show blank between questions
        root.after(2000, change_label, label, '')
        # curTime = curTime + blankWindowTime

        yield 0


def simplePrint():
    print('Pressed')


def main():
    root, ws, hs = set_window_mid_screen()
    label = tk.Label(root, text="")
    label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    q_list = assoc_array_to_list(prepare_vocal_single_option("data/questions_single_options.csv"))

    rect = tk.Canvas(root, width=200, height=100)
    changeRect(root, rect, 'green', ws)

    b = tk.Button(root, text="Next Question", command=lambda: nextQ('hey', root, label, q_list))
    b.pack()

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