#!/usr/bin/env python

# Importing the required libraries
import os
import pandas as pd
import numpy as np
import time
import datetime
import csv
import cv2
import tkinter as tk
import tkinter.font as TkFont
from PIL import Image
from tkinter import messagebox


# Graphical User Interface
window = tk.Tk()

window.title("Enrollment & Attendance System")
window.iconbitmap('picture.ico')

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

window.configure(bg='#edeef7')
x_cord = 75
y_cord = 20
checker = 0

font_config = TkFont.Font(family="Candara", size=42,
                          weight="bold", slant="roman")

message = tk.Label(window, text="Face Recognition Based Attendance System",
                   bg="#edeef7", height=3, font=font_config, width=50, fg="#7868e6")
message.place(x=-110, y=5)

label = tk. Label(window, text="Enter ID: ", width=20, height=2,
                  fg="#1b1a17", bg="#b8b5ff", font=('Candara', 15, 'bold'))
label.place(x=350, y=200)

txt = tk.Entry(window, width=20, bg="#b8b5ff",
               fg="#1b1a17", font=('Candara', 15, 'bold'))
txt.place(x=650, y=215)

label2 = tk.Label(window, text="Enter Name: ", width=20, fg="#1b1a17",
                  bg="#b8b5ff", height=2, font=('Candara', 15, 'bold'))
label2.place(x=350, y=300)

txt2 = tk.Entry(window, width=20, bg="#b8b5ff",
                fg="#1b1a17", font=('Candara', 15, 'bold'))
txt2.place(x=650, y=315)

label3 = tk.Label(window, text="Notification: ", width=20, fg="#1b1a17",
                  bg="#b8b5ff", height=2, font=('Candara', 15, 'bold'))
label3.place(x=350, y=400)

message = tk.Label(window, text="", bg="#b8b5ff", fg="#1b1a17", width=30,
                   height=2, activebackground="#b8b5ff", font=('Candara', 15, 'bold'))
message.place(x=650, y=400)

label4 = tk.Label(window, text="Log: ", width=20, fg="#1b1a17",
                  bg="#b8b5ff", height=2, font=('Candara', 15, 'bold'))
label4.place(x=350, y=500)

message2 = tk.Label(window, text="", fg="#1b1a17", bg="#b8b5ff",
                    activeforeground="#b8b5ff", width=30, height=2, font=('Candara', 15, 'bold'))
message2.place(x=650, y=500)


# It will clear the text
def clear1():
    txt.delete(0, 'end')
    res = ""
    message.configure(text=res)

# Function for clear text


def clear2():
    txt2.delete(0, 'end')
    res = ""
    message.configure(text=res)


# Clear button style
clearButton = tk.Button(window, text="Clear", command=clear1, fg="#e4fbff", bg="#7868e6",
                        width=12, height=1, activebackground="#B6D0E2", font=('Candara', 15, 'bold'))
clearButton.place(x=900, y=205)

clearButton2 = tk.Button(window, text="Clear", command=clear2, fg="#e4fbff", bg="#7868e6",
                         width=12, height=1, activebackground="#B6D0E2", font=('Candara', 15, 'bold'))
clearButton2.place(x=900, y=305)

# Function to validate the input as numeric


def is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

# Function to assist the input data utilizing camera and store in local space.


def get_data():
    Id = (txt.get())
    name = (txt2.get())
    if not Id:
        res = "Please enter the ID"
        message.configure(text=res)
        MsgBox = tk.messagebox.askquestion(
            "Warning", "Please enter roll number properly, press yes if you understood", icon='warning')
        if MsgBox == 'no':
            tk.messagebox.showinfo(
                'Your need', 'Please go through the documentation thoroughly.')
    elif not name:
        res = "Please enter the name"
        message.configure(text=res)
        MsgBox = tk.messagebox.askquestion(
            "Warning", "Please enter your name properly, press yes if you understood", icon='warning')
        if MsgBox == 'no':
            tk.messagebox.showinfo(
                'Your need', 'Please go through the documentation thoroughly.')

    elif (is_num(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        # Classifier to detect faces from frame
        harcascadePath = "Classifier\haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while (True):
            ret, img = cam.read()
            # Converting the img into gray scale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                # Bound Box for the detected faces
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum = sampleNum+1
                cv2.imwrite("TrainingImage\ " + name + "." + Id +
                            '.' + str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
                cv2.imshow('frame', img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum > 69:  # It will only take 69 input images
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Saved for ID: " + Id + " Name: " + name
        row = [Id, name]
        with open('EntityDetails\entity_details.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text=res)
    else:
        if (is_num(Id)):
            res = "Kindly enter alphabetical name"
            message.configure(text=res)
        if (name.isalpha()):
            res = "Kindly enter numeric ID"
            message.configure(text=res)


# Function for labelling the data with ID
def fetch_data(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


# Function to extract the facial features then store it implementing data serialization
def training_function():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    faces, Id = fetch_data("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("FeaturePool\Feature.yml")
    res = "Image Trained"
    clear1()
    clear2()
    message.configure(text=res)
    tk.messagebox.showinfo(
        'Completed', 'The model has been trained successfully with new inputs!')


# Function helps to recognize the face, extract features and predict the confidence of match(facial features)
def predict_function():
    # LocalBinaryPatternHistogram Algorithm for classification
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("FeaturePool\Feature.yml")
    harcascadePath = "Classifier\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("EntityDetails\entity_details.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if (conf < 50):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(
                    ts).strftime('%H:%M:%S')
                aa = df.loc[df['Id'] == Id]['Name'].values
                tt = str(Id) + "-" + aa
                attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]
            else:  # Intruder will be traced and stored
                Id = 'Intruder'
                tt = str(Id)
            if (conf > 75):
                noOfFile = len(os.listdir("Intruders")) + 1
                cv2.imwrite("Intruders\Image" + str(noOfFile) +
                            ".jpg", im[y:y+h, x:x+w])
            cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('im', im)
        if (cv2.waitKey(1) == ord('q')):  # Close the camera frame
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = "AttendanceLogs\Log"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName, index=False)
    cam.release()
    cv2.destroyAllWindows()
    res = attendance
    message2.configure(text=res)
    res = "Attendance Log Updated"
    message.configure(text=res)
    tk.messagebox.showinfo(
        'Completed', 'Congratulations! Your attendance has been marked successfully for the day.')

# Quit the window


def quit_window():
    MsgBox = tk.messagebox.askquestion(
        'Exit Application', 'Are you sure you want to exit the application?', icon='warning')
    if MsgBox == 'yes':
        tk.messagebox.showinfo(
            "Greetings", "Thank You! Your contribution to save environment is valuable.")
        window.destroy()


# Software GUI
takeImg = tk.Button(window, text="Register", command=get_data, fg="#e4fbff", bg="#7868e6",
                    width=15, height=2, borderwidth=5, activebackground="Red", font=('Candara', 15, 'bold'))
takeImg.place(x=250, y=580)

trainImg = tk.Button(window, text="Train Inputs", command=training_function, fg="#e4fbff", bg="#7868e6",
                     width=15, height=2, borderwidth=5, activebackground="Red", font=('Candara', 15, 'bold'))
trainImg.place(x=450, y=580)

trackImg = tk.Button(window, text="Mark Attendance", command=predict_function, fg="#e4fbff", bg="#7868e6",
                     width=15, height=2, borderwidth=5, activebackground="Red", font=('Candara', 15, 'bold'))
trackImg.place(x=650, y=580)

quitWindow = tk.Button(window, text="Exit", command=quit_window, fg="#e4fbff", bg="#7868e6",
                       width=15, height=2, borderwidth=5, activebackground="#B6D0E2", font=('Candara', 15, 'bold'))
quitWindow.place(x=850, y=580)

window.mainloop()
