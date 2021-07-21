import cv2
import numpy as np
from random import randint

def findDist(a,b):
    x1=a[0]+a[2]/2
    x2 = b[0] + b[2] / 2
    y1 = a[1] + a[3] / 2
    y2 = b[1] + b[3] / 2
    d_1=(x1-x2)*(x1-x2)
    d_2=(y1-y2)*(y1-y2)
    d12=d_1+d_2
    d=int(np.sqrt(d12))
    return d

def sfindDist(a,b):
    x1=a[0]+a[2]/2
    x2 = b[0] + b[2] / 2
    y1 = a[1] + a[3] / 2
    y2 = b[1] + b[3] / 2
    h1=a[3]
    h2=b[3]
    d_1=(x1-x2)*(x1-x2)
    d_2=(y1-y2)*(y1-y2)
    d_3=(h1-h2)*(h1-h2)
    d12=d_1+d_2+d_3
    d=int(np.sqrt(d12))
    return d

def createTrackerByName(trackerType):
    tracker = cv2.TrackerCSRT_create()
    return tracker
def drawBox(img,p1,p2,color,):
    cv2.rectangle(img, p1,p2,color,3,1)
    cv2.putText(img,f"Tracking:",(75,75),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)
def update_boxes(pbox,cbox,d_th=1000):
    d=[]
    c=0
    cc=0
    idx_arr=[]
    p_temp=[]
    c_temp=[]
    new_entry=0
    for idx,i in enumerate(pbox):
        for jidx,j in enumerate(cbox):
            dd=findDist(i,j)
            d.append(dd)
            if dd < d_th:
                if j not in c_temp:
                    c_temp.append(j)
    for j in cbox:
        if j not in c_temp:
            p_temp.append(j)
            new_entry+=1
    return p_temp,idx_arr,new_entry,d

def check_distance(box,d_thresh=100):
    red_box=[]
    for i_indx, i in enumerate(box):
        new_list = box[i_indx + 1:]
        for j_idx, j in enumerate(new_list):
            dd = findDist(i, j)
            if dd<d_thresh:
                red_box.append(i)
                red_box.append(j)
    return red_box










