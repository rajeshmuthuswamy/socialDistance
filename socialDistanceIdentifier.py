from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
from os.path import join as pjoin
from os.path import expanduser
from collections import Counter
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
# import facenet
import detect_face
import shutil
import sys
import time
import copy
import math
import pickle
import dist_matrix_script
import json
import redis
import time
import xml.dom.minidom
import MySQLdb
import base64
import glob
from datetime import datetime
from io import BytesIO
from math import pow, sqrt


#!--------------Color Code-----------------
RED="\033[1;91m"
GREEN="\033[1;32m"
CYAN="\033[1;34m"
blue="\033[1;44m" #Blue Bagkground
magenta="\033[1;45m"
NOCOLOR="\033[0m"
#!------------------------------------------

#!--------------MySQL Config----------------
# Parse an xml file by name
path = os.path.expanduser(os.getcwd())
mydoc = xml.dom.minidom.parse(path+'/socialDistance_recog_param.xml')

# Parameters for mysql connection
mysql_xml = mydoc.getElementsByTagName('mysql')

global host, username, password, database
host = str(mysql_xml[0].firstChild.data)
username = str(mysql_xml[1].firstChild.data)
password = "Glueck@321"
database = str(mysql_xml[2].firstChild.data)
table = str(mysql_xml[3].firstChild.data)

# Parameters for face_recognition
recog_xml = mydoc.getElementsByTagName('recog')

input_size = int(recog_xml[0].firstChild.data) 				# Minimum size of input face
image_directory = str(recog_xml[1].firstChild.data)			# Image directory
model = str(recog_xml[2].firstChild.data)					# Model directory
threshold_loss = float(recog_xml[3].firstChild.data) 		# Threshold limit for loss of distance matrix,higher threshold values will accept less similarity of faces.
threshold_ID = float(recog_xml[4].firstChild.data) 			# Threshold limit for ratio of correct faces/enrolled faces for each person
QA_test = bool(int(recog_xml[5].firstChild.data)) 			# Output folders and files for QA testing
gpu_memory_fraction=float(recog_xml[6].firstChild.data)     # Upper bound on the amount of GPU memory that will be used by the process
#!------------------------------------------

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
redis_db = redis.StrictRedis(host="localhost", port=6379, db=0)

#Load Social Distance Model,Prototext
'''
Reading SSD_MobileNet caffe model and its prototxt
'''
#Read model,prototxt,label for socialDistance_model

confidences = 0.2

label = parent_dir + '/glueck-ce-services/model/social_distance_model/class_labels.txt'
labels = [line.strip() for line in open(label)]

# Generate random bounding box bounding_box_color for each label
bounding_box_color = np.random.uniform(0, 255, size=(len(labels), 3))

# Load model
print("\nLoading model...\n")
model = parent_dir + '/glueck-ce-services/model/social_distance_model/SSD_MobileNet.caffemodel'
prototxt = parent_dir + '/glueck-ce-services/model/social_distance_model/SSD_MobileNet_prototxt.txt'
network = cv2.dnn.readNetFromCaffe(prototxt,model)

print("\nSocialDistance prediction in progress ...\n")


#Mysql Connectivity
def mysqldb_connect():
	"""
	Activate connection to MySQLdb
	"""
	mysql_db = MySQLdb.connect(host,username,password,database)
	cursor = mysql_db.cursor()
	return mysql_db, cursor

#Insert Values to socialDistance Table
def insert_mysql(timestamp, camID, trackID, imagePath):
	ts = int(timestamp)/1000
	dateTime = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
	mysql_db, cursor = mysqldb_connect()

	insert2db = 'INSERT INTO socialDistance' + '(dateTime, camID, trackID, imagePath) VALUES (%s, %s, %s, %s)'
	cursor.execute(insert2db, (dateTime, camID, trackID, imagePath))
	mysql_db.commit()
	mysql_db.close()


def retrieve_Dir_FP(Path):
	"""
	Retrieve all image folder by accessing all cam folders within the path.
	"""
	lstName = []
	for camName in os.listdir(Path):
		# camName : Name of the camera folder
		# camName_FP : Full path of the camera folder
		# camName_FP = os.path.join(Path, camName)
		# for imgDirName in os.listdir(camName_FP):
		# 	# print("DirImgNamr:",imgDirName)
		imgDir_FP = os.path.join(Path, camName)
		if os.path.isdir(imgDir_FP):
			lstName.append(imgDir_FP)
	lstName.sort(key=lambda s: os.path.getmtime(os.path.join(Path, s)), reverse=False)
	return lstName


def main(input_size, model, gpu_memory_fraction, threshold_loss, threshold_ID, QA_test, table):
	global finalValue
	recogDirPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+ image_directory
	calcPath = parent_dir + '/glueck-ce-services/PostClassifier/post_distancing'

	while True:
		npy_name_list, emb_npy = dist_matrix_script.load_npy(table, host, username, password, database)
		camDirList = retrieve_Dir_FP(recogDirPath)
		for camDirPath in camDirList:
			FolderID = camDirPath.split("/")[-1]
			cameraID = FolderID.split("_")[-1]
			for _,_,imgList in os.walk(camDirPath):
				if imgList == []:
					print("socialDistance_Frame {} have no Images for Prediction".format(GREEN+FolderID+NOCOLOR))
				else:
					for imgName in imgList:
						timestamp, camID, _, _ = imgName.split("_")
						imageName=imgName.split(".")[0]
						if imgName.endswith(".jpg"):
							imagePath = camDirPath + "/"+ imgName
							image = cv2.imread(imagePath)
							ImageCount = len(imgList)
							if(image is None):
								print("No Images Found")
							else:
								(h, w) = image.shape[:2]
							# Resize the frame to suite the model requirements. Resize the frame to 300X300 pixels
								blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
								network.setInput(blob)
								detections = network.forward()
								pos_dict = dict()
								coordinates = dict()
								F = 615
								for i in range(detections.shape[2]):
									confidence = detections[0, 0, i, 2]
									if confidence > confidences:
										class_id = int(detections[0, 0, i, 1])
										box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
										(startObjX, startObjY, endObjX, endObjY) = box.astype('int')
									# Filtering only persons detected in the frame. Class Id of 'person' is 15
										if class_id == 15.00:
											label = "{}: {:.2f}%".format(labels[class_id], confidence * 100)
										# print("{}".format(label))
											coordinates[i] = (startObjX, startObjY, endObjX, endObjY)
										# Mid point of bounding box
											ObjX_midpt = round((startObjX+endObjX)/2,4)
											ObjY_midpt = round((startObjY+endObjY)/2,4)
											height = round(endObjY-startObjY,4)
										# Distance from `camera` based on triangle similarity
											distance = (165 * F)/height
										# print("Distance(cm):{dist}\n".format(dist=distance))
										# Mid-point of bounding boxes (in cm) based on triangle similarity technique
											ObjX_midpt_cm = (ObjX_midpt * distance) / F
											ObjY_midpt_cm = (ObjY_midpt * distance) / F
											pos_dict[i] = (ObjX_midpt_cm,ObjY_midpt_cm,distance)
											postValue = pos_dict[i]
											close_objects = set()
											for i in pos_dict.keys():
												for j in pos_dict.keys():
													if i < j:
														dist = sqrt(pow(pos_dict[i][0]-pos_dict[j][0],2) + pow(pos_dict[i][1]-pos_dict[j][1],2) + pow(pos_dict[i][2]-pos_dict[j][2],2))
														finalValue = dist
													# Check if distance less than 2 metres or 200 centimetres
														if dist < 200:
															close_objects.add(i)
															close_objects.add(j)
											for i in pos_dict.keys():
												if i in close_objects:
													COLOR = np.array([0,0,255])
												else:
													COLOR = np.array([0,255,0])
												(startObjX, startObjY, endObjX, endObjY) = coordinates[i]
												cv2.rectangle(image, (startObjX, startObjY), (endObjX, endObjY), COLOR, 2)
												cv2.imwrite(calcPath + '/' + imgName + '.jpg', image)
						insert_mysql(timestamp, camID=camID, trackID=imageName, imagePath=imagePath)
						os.remove(imagePath)


if __name__ == '__main__':
	main(input_size, model, gpu_memory_fraction, threshold_loss, threshold_ID, QA_test, table)
