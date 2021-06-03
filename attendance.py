#importing all the necessary libraries

import tkinter as tk
import cv2,os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
from tkinter import messagebox
import tkinter.font as TkFont


#Graphical User Interface
window = tk.Tk()

window.title("Attendance System")
window.iconbitmap('Picture2.ico')

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

window.configure(bg='#edeef7')
x_cord = 75
y_cord = 20
checker=0

myf= TkFont.Font(family="Candara", size=42, weight="bold", slant="roman")

message = tk.Label(window, text="Face Recognition Based Attendance System" ,bg="#edeef7" ,height=3, font=myf, width=50, fg="#7868e6")
message.place(x=-110, y=5) 

lbl = tk. Label(window, text="Enter ID",width=20 ,height=2 ,fg="#1b1a17", bg="#b8b5ff" ,font=('Candara', 15, 'bold') ) 
lbl.place(x=350, y=200) 

txt = tk.Entry (window, width=20  , bg="#b8b5ff" ,fg="#1b1a17", font=('Candara', 15, ' bold '))
txt.place (x=650, y=215)

lbl2 = tk.Label(window, text="Enter Name", width=20 ,fg="#1b1a17", bg="#b8b5ff" ,height=2 ,font=('Candara', 15, 'bold')) 
lbl2.place(x=350, y=300)

txt2 = tk.Entry(window, width=20 , bg="#b8b5ff" , fg="#1b1a17", font=('Candara', 15, ' bold ') ) 
txt2.place(x=650, y=315)

lbl3 = tk.Label(window, text="Notification" , width=20 ,fg="#1b1a17", bg="#b8b5ff" ,height=2 ,font=('Candara', 15, ' bold  ')) 
lbl3.place(x=350, y=400)

message = tk.Label(window, text="" ,bg="#b8b5ff" ,fg="#1b1a17"  ,width=30  ,height=2, activebackground = "#b8b5ff" ,font=('Candara', 15, ' bold '))
message.place(x=650 , y=400)
 
lbl3 = tk.Label(window, text="Attendance : ", width=20 ,fg="#1b1a17", bg="#b8b5ff",height = 2 , font=('Candara' , 15, 'bold' )) 
lbl3.place(x=350, y=500)

message2 = tk.Label(window, text="" ,fg="#1b1a17" ,bg="#b8b5ff" ,activeforeground = "#b8b5ff",width=30  ,height=2  ,font=('Candara', 15, ' bold ')) 
message2.place(x=650, y=500)




#function for the clear button1 
def clear1():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)
    
#function for the clear button2
def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)    

#clear button 1 and 2 GUI
clearButton = tk.Button(window, text="Clear", command=clear1 ,fg="#e4fbff"  ,bg="#7868e6"  ,width=12  ,height=1 ,activebackground = "#B6D0E2" ,font=('Candara', 15, ' bold '))
clearButton.place(x=900, y=205)
clearButton2 = tk.Button(window, text="Clear", command=clear2  ,fg="#e4fbff"  ,bg="#7868e6"  ,width=12  ,height=1, activebackground = "#B6D0E2" ,font=('Candara', 15, ' bold '))
clearButton2.place(x=900, y=305)
   
#function for checking number     
def is_number(s):
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

#function for accessing device camera and capturing images 
def TakeImages():        
    Id=(txt.get())
    name=(txt2.get())
    if not Id:
        res="Please enter Id"
        message.configure(text = res)
        MsgBox = tk.messagebox.askquestion ("Warning","Please enter roll number properly , press yes if you understood",icon = 'warning')
        if MsgBox == 'no':
            tk.messagebox.showinfo('Your need','Please go through the documentation file properly')
    elif not name:
        res="Please enter Name"
        message.configure(text = res)
        MsgBox = tk.messagebox.askquestion ("Warning","Please enter your name properly , press yes if you understood",icon = 'warning')
        if MsgBox == 'no':
            tk.messagebox.showinfo('Your need','Please go through the documentation file properly')
        
    elif(is_number(Id) and name.isalpha()):
            cam = cv2.VideoCapture(0)
            #classifier used for detecting faces
            harcascadePath = "Classifier\haarcascade_frontalface_default.xml"
            detector=cv2.CascadeClassifier(harcascadePath)
            sampleNum=0
            while(True):
                ret, img = cam.read()
                #converting img to gray color using cv2
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                    #incrementing sample number 
                    sampleNum=sampleNum+1
                    #saving the captured face in the dataset folder TrainingImage
                    cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                    #display the frame
                    cv2.imshow('frame',img)
                #wait for 100 miliseconds 
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                # break if the sample number is more than 60
                elif sampleNum>60:
                    break
            cam.release()
            cv2.destroyAllWindows() 
            res = "Images Saved for ID : " + Id +" Name : "+ name
            row = [Id , name]
            with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()
            message.configure(text= res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text= res)
            
#function for training the train model after capturing images is done    
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained"
    clear1();
    clear2();
    message.configure(text= res)
    tk.messagebox.showinfo('Completed','Your model has been trained successfully!!')
    

def getImagesAndLabels(path):

    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    
    faces=[]

    Ids=[]

    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids

#after capturing images tracking this captured images with images precaptured and stored in images folder
def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create() #LocalBinaryPatternHistogram Algorithm for face recognition
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "Classifier\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    
    #if the images captured got track with prestored images then attendence will be updated
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
      #if the images doesn't got tracked then the images will be stored in the Images unknown folder          
            else:
                Id='Unknown'                
                tt=str(Id)  
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    res=attendance
    message2.configure(text= res)
    res = "Attendance Taken"
    message.configure(text= res)
    tk.messagebox.showinfo('Completed','Congratulations ! Your attendance has been marked successfully for the day!!')

#fuction for quitting the attendence window
def quit_window():
   MsgBox = tk.messagebox.askquestion ('Exit Application','Are you sure you want to exit the application?',icon = 'warning')
   if MsgBox == 'yes':
       tk.messagebox.showinfo("Greetings", "Thank You very much for using our software. Have a nice day ahead!!")
       window.destroy()
    
# panel GUI
takeImg = tk.Button(window, text="Register", command=TakeImages  ,fg="#e4fbff"  ,bg="#7868e6" ,width=15  ,height=2, borderwidth=5, activebackground = "Red" ,font=('Candara', 15, ' bold '))
takeImg.place(x=250, y=580)

trainImg = tk.Button(window, text="Train Model", command=TrainImages  ,fg="#e4fbff"  ,bg="#7868e6" ,width=15  ,height=2, borderwidth=5, activebackground = "Red" ,font=('Candara', 15, ' bold '))
trainImg.place(x=450, y=580)

trackImg = tk.Button(window, text="Mark Attendance", command=TrackImages  ,fg="#e4fbff"  ,bg="#7868e6"  ,width=15  ,height=2, borderwidth=5, activebackground = "Red" ,font=('Candara', 15, ' bold '))
trackImg.place(x=650, y=580)

quitWindow = tk.Button(window, text="Exit", command=quit_window  ,fg="#e4fbff"  ,bg="#7868e6"  ,width=15  ,height=2, borderwidth=5, activebackground = "#B6D0E2" ,font=('Candara', 15, ' bold '))
quitWindow.place(x=850, y=580)

window.mainloop()
