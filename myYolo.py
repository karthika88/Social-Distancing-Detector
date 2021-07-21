import cv2
import numpy as np
from random import randint

class myYolo:
	def __init__(self,modelConfig = 'yolov3.cfg',weights = 'yolov3.weights',classFileName = 'coco.names'):
		self.modelConfig=modelConfig
		self.weights= weights
		classFileName=classFileName
		with open(classFileName,'rt') as f:
    			classNames=f.read()
    			classNames=classNames.rstrip('\n')
    			self.classNames=classNames.split('\n')
	def make_model(self,img):
		net= cv2.dnn.readNetFromDarknet(self.modelConfig,self.weights)
		net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
		net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
		w,h= 320,320
		self.img=img
		blob=cv2.dnn.blobFromImage(self.img, 1/255,(w,h), [0,0,0],1,crop=False )
		net.setInput(blob)
		outputNames=net.getUnconnectedOutLayersNames()
		self.outputs= net.forward(outputNames)
	def findObjects(self,conf_thresh=0.3, nms_thresh=0.3):
		ht,wt,ch=self.img.shape
		self.xx=0
		self.yy=0
		self.ww=0
		self.hh=0
		self.bbox=[]
		self.confs=[]
		self.conf_thresh=conf_thresh
		self.nms_thresh=nms_thresh
		for output in self.outputs:
			for det in output:
					scores=det[5:]
					classID=np.argmax(scores)
					conf=scores[classID]
					if conf>conf_thresh:
						if classID==0:
								w,h= int(det[2]*wt),int(det[3]*ht)
								x,y= int((det[0]*wt)-w/2),int((det[1]*ht)-h/2)
								self.bbox.append((x,y,w,h))
								self.confs.append(float(conf))
		self.indices = cv2.dnn.NMSBoxes(self.bbox, self.confs, self.conf_thresh, self.nms_thresh)
		return self.bbox,self.indices

	def find_boxes(self):
		self.boxes = []
		self.colors = []
		for i in self.indices:
			i = i[0]
			boxx = self.bbox[i]
			self.boxes.append([boxx[0], boxx[1], boxx[2], boxx[3]])
			self.colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
		return self.boxes, self.colors
