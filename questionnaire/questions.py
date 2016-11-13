import socket
import csv
from collections import OrderedDict
import tkinter as tk
import time


def connect_to_fs_receiver_udp(ip="127.0.0.1", port=33444):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect((ip, port))
    return sock


def send_record_flag_udp(sock, flag=True):
    sock.send(str(int(flag)))


def disconnect_from_fs_receiver_udp(sock):
    sock.close()


def load_questions(path='data/questions.csv'):
    """
    Read questions from csv file at path:
    Each row is in format "question","answer_1","answer_2","answer_3","answer_4"

    @returns ordered dictionary with questions as keys and dict{options, answer} as value; and a concatenated list
        of questions and their answers
    """
    display_list = []
    questions = OrderedDict()

    error = "{}: every question must have at least one answer and an index of right answer".format(path)

    with open(path) as out:
        reader = csv.reader(out)

        header = next(reader, [])

        if len(header) <= 2:
            print(error)
            return []

        for row in reader:
            if len(header) != len(row):
                print(error)
                return []

            questions[row[0]] = {"options": row[1:-1], "answer": int(row[-1])}
            display_list.extend(row[:-1])

    return questions, display_list


def set_window_mid_screen():
    root = tk.Tk()  # create a Tk root window

    w = 0  # width for the Tk root
    h = 0  # height for the Tk root

    # get screen width and height
    ws = root.winfo_screenwidth()  # width of the screen
    hs = root.winfo_screenheight()  # height of the screen
    root.overrideredirect(1)
    root.geometry('%dx%d+%d+%d' % (ws, hs, w, h))
    return root,ws,hs


def change_label(label, t):
    label.config(text=('%s') % t, fg="green", font=("Helvetica", 72), justify=tk.CENTER, anchor=tk.CENTER)

def changeRect(root,rect,color,ws):
    rect = tk.Canvas(root, width=ws, height=100)
    # rect.coords()
    # rect.pack(side=tk.RIGHT,anchor=tk.NE)
    rect.pack()

    #w.create_line(0, 0, 200, 100)
    #w.create_line(0, 100, 200, 0, fill="red", dash=(4, 4))

    rect.create_rectangle(0, 0, ws, 100, fill="%s"%color)

def main():

    root,ws,hs = set_window_mid_screen()
    label = tk.Label(root, text="")
    label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    
    rect = tk.Canvas(root, width=200, height=100)
    changeRect(root,rect,'green',ws)
    
    question, q_list = load_questions()

    curTime = 1000
    QintervalTime = 1000
    AintervalTime = 1000
    blankWindowTime = 1000
    for q in range(int(len(q_list) / 5)):
        root.after(curTime,change_label, label, q_list[q*6])
        curTime = curTime + QintervalTime
        
        for a in range(4):
            print("size of q_list is: %d"%len(q_list))
            print("q: %d, a: %d, q*6+a: %d"%(q ,a,q*6+a))
            root.after(curTime,change_label, label, q_list[q*6+a])
            curTime = curTime + AintervalTime
            
            # Show blank window
            root.after(curTime,change_label, label, '')
            curTime = curTime + blankWindowTime
        
        #Show Question Again
        questString = q_list[q*6]
        root.after(curTime, change_label, label, questString)
        curTime = curTime + QintervalTime
    
    root.mainloop()

if __name__ == "__main__":
    main()