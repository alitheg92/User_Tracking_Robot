import common as cm
import cv2
import numpy as np
from PIL import Image
import time
from threading import Thread

import util as ut
ut.init_gpio()

from Motor import *            
PWM=Motor()
def Forward_s1():
    PWM.setMotorModel(1000,1000,1000,1000)

def Forward_s2():
    PWM.setMotorModel(2000,2000,2000,2000)

def Back():
    PWM.setMotorModel(-1200,-1200,-1200,-1200)   #Back
    
def Right():
    PWM.setMotorModel(-1450,-1450,1450,1450)
    
def Left():    
    PWM.setMotorModel(1450,1450,-1450,-1450)       #Right    


def Forward_carpet():
    PWM.setMotorModel(1300,1300,1300,1300)

def Back_carpet():
    PWM.setMotorModel(-1300,-1300,-1300,-1300)   #Back
    
def Right_carpet():
    PWM.setMotorModel(-1700,-1300,1300,1300)
    
def Left_carpet():    
    PWM.setMotorModel(1700,1700,-1700,-1700)       #Right    

def Stop():
    PWM.setMotorModel(0,0,0,0)  

cap = cv2.VideoCapture(0)
threshold=0.2 
top_k=5 #first five objects with prediction probability above threshhold (0.2) to be considered
edgetpu=1

model_dir = '/home/pi/Project/odbot_tracking/models'
model = 'mobilenet_ssd_v2_coco_quant_postprocess.tflite'
model_edgetpu = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
lbl = 'coco_labels.txt'


tolerance=0.1
x_deviation=0
y_max=0
arr_track_data=[0,0,0,0,0,0]
#distance=0

user=['person']

def user_tracking(objs,labels):
    
    global x_deviation, y_max, tolerance, arr_track_data
    
    if(len(objs)==0):
        print("NO User")
        Back()
        time.sleep(0.2)
        Stop()
        arr_track_data=[0,0,0,0,0,0]
        return

    k=0
    flag=0
    for obj in objs:
        lbl=labels.get(obj.id, obj.id)
        k = user.count(lbl)
        if (k>0):
            x_min, y_min, x_max, y_max = list(obj.bbox)
            flag=1
            break
        
    #print(x_min, y_min, x_max, y_max)
    if(flag==0):
        print("User Out of Vision")
        return
        
    x_diff=x_max-x_min
    y_diff=y_max-y_min
   
    obj_x_center=x_min+(x_diff/2)
    obj_x_center=round(obj_x_center,3)
    
    obj_y_center=y_min+(y_diff/2)
    obj_y_center=round(obj_y_center,3) - 0.05
    
        
    #print("[",obj_x_center, obj_y_center,"]")
        
    x_deviation=round(0.5-obj_x_center,3)
    y_max=round(y_max,3)
        
   
    thread = Thread(target = move_robot)
    thread.start()
    
    arr_track_data[0]=obj_x_center
    arr_track_data[1]=obj_y_center
    arr_track_data[2]=x_deviation
    arr_track_data[3]=y_max
    

def move_robot():
    global x_deviation, y_max, tolerance, arr_track_data
    
    
    if(abs(x_deviation)<tolerance):
        delay1=0
        
        if(y_max>0.93):
            cmd="STOP"
            Stop()
            
        else:
            cmd="FORWARD"
            Forward_s1()
          
    else:
        
        if(x_deviation>=tolerance):
            cmd="LEFT"
            delay1=get_delay(x_deviation)
            Left()
            time.sleep(delay1)
            Stop()
                
        if(x_deviation<=-1*tolerance):
            cmd="RIGHT"
            delay1=get_delay(x_deviation)
            Right()
            time.sleep(delay1)
            Stop()
            
    arr_track_data[4]=cmd
    arr_track_data[5]=delay1

def get_delay(deviation):
    
    deviation=abs(deviation)
    
    if(deviation>=0.4):
        d=0.080
    elif(deviation>=0.35 and deviation<0.40):
        d=0.060
    elif(deviation>=0.20 and deviation<0.35):
        d=0.050
    else:
        d=0.040
    
    return d
    
    

def main():
    
    if (edgetpu==1):
        mdl = model_edgetpu
    else:
         mdl = model
        
    interpreter, labels =cm.load_model(model_dir,mdl,lbl,edgetpu)
    
    fps=1
    arr_dur=[0,0,0]

    while True:
        start_time=time.time()
        
        #-Capture Camera Frame
        start_t0=time.time()
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2_im = frame
        #cv2_im = cv2.flip(cv2_im, 0)
        #cv2_im = cv2.flip(cv2_im, 1)

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)
       
        arr_dur[0]=time.time() - start_t0
       
        #Inference
        start_t1=time.time()
        cm.set_input(interpreter, pil_im)
        interpreter.invoke()
        objs = cm.get_output(interpreter, score_threshold=threshold, top_k=top_k)
        
        arr_dur[1]=time.time() - start_t1

       
       #other
        start_t2=time.time()
        user_tracking(objs,labels)#tracking  <<<<<<<
       

        arr_dur[2]=time.time() - start_t2
        fps = round(1.0 / (time.time() - start_time),1)
        print("****** FPS: ",fps," ******")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


