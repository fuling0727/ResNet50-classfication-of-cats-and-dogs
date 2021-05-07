import sys
import cv2
import numpy as np
import glob
import random
from numpy import asarray
import PyQt5
from PyQt5 import *
from PyQt5.QtGui import QPixmap, QIcon, QFont, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QPushButton, QWidget, QLineEdit, QComboBox
from tqdm import tqdm
import PIL
from PIL import ExifTags, Image
import matplotlib
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL
#global combIndex

#print(cv2.__version__)

class MainWindow(QMainWindow):
    def __init__(self):
        
        super(MainWindow, self).__init__()
        self.resize(1300,600) 
        self.title = "2020 Opencvdl HW2"
        self.setWindowTitle(self.title)

        label1 = QLabel(self)
        label1.setText("1. Find Contour")
        label1.setFixedWidth(200)
        label1.setFixedHeight(35)
        label1.setStyleSheet("color:#3C3C3C	")
        myFont=QFont("Arial Font",14,QFont.Bold)
        label1.setFont(myFont)
        label1.move(50,50)

        button1_1 = QPushButton(self)
        button1_1.setText("1.1 Draw Contour")  # 建立名字
        button1_1.setStyleSheet("background-color:#FCFCFC; ")
        button1_1.setFixedHeight(35)
        button1_1.setFixedWidth(150)
        button1_1.move(60,120)  # 移動位置
        button1_1.clicked.connect(self.buttonClicked1_1) # 設置button啟動function

        button1_2 = QPushButton(self)
        button1_2.setText("1.2 Count Coins")  # 建立名字
        button1_2.setStyleSheet("background-color:#FCFCFC; ")
        button1_2.setFixedHeight(35)
        button1_2.setFixedWidth(150)
        button1_2.move(60,200)  # 移動位置
        button1_2.clicked.connect(self.buttonClicked1_2) # 設置button啟動function
    
        self.label1_1 = QLabel(self)
        self.label1_1.setText("There are ___ coins in coin01.jpg")
        self.label1_1.setFixedWidth(230)
        self.label1_1.setFixedHeight(35)
        self.label1_1.setStyleSheet("color:#3C3C3C	")
        myFont=QFont("Arial Font",13,QFont.Bold)
        self.label1_1.setFont(myFont)
        self.label1_1.move(40,300)

        self.label1_2 = QLabel(self)
        self.label1_2.setText("There are ___ coins in coin02.jpg")
        self.label1_2.setFixedWidth(230)
        self.label1_2.setFixedHeight(35)
        self.label1_2.setStyleSheet("color:#3C3C3C	")
        myFont=QFont("Arial Font",13,QFont.Bold)
        self.label1_2.setFont(myFont)
        self.label1_2.move(40,350)

        button2_1 = QPushButton(self)
        button2_1.setText("2.1 Find Corners")  # 建立名字
        button2_1.setStyleSheet("background-color:#FCFCFC; ")
        button2_1.setFixedHeight(35)
        button2_1.setFixedWidth(150)
        button2_1.move(300,120)  # 移動位置
        button2_1.clicked.connect(self.buttonClicked2_1) # 設置button啟動function

        button2_2 = QPushButton(self)
        button2_2.setText("2.2 Find Intrinsic")  # 建立名字
        button2_2.setStyleSheet("background-color:#FCFCFC; ")
        button2_2.setFixedHeight(35)
        button2_2.setFixedWidth(150)
        button2_2.move(300,200)  # 移動位置
        button2_2.clicked.connect(self.buttonClicked2_2) # 設置button啟動function
    
        choices = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
        self.comboBox = QComboBox(self)
        self.comboBox.addItems(choices)
        self.comboBox.move(300, 280)
        self.comboBox.currentIndexChanged[str].connect(self.comboPrint)

        button2_3 = QPushButton(self)
        button2_3.setText("2.3 Find Extrinsic")  # 建立名字
        button2_3.setStyleSheet("background-color:#FCFCFC; ")
        button2_3.setFixedHeight(35)
        button2_3.setFixedWidth(150)
        button2_3.move(300,360)  # 移動位置
        button2_3.clicked.connect(self.buttonClicked2_3) # 設置button啟動function
    
        button2_4 = QPushButton(self)
        button2_4.setText("2.4 Find Distortion")  # 建立名字
        button2_4.setStyleSheet("background-color:#FCFCFC; ")
        button2_4.setFixedHeight(35)
        button2_4.setFixedWidth(150)
        button2_4.move(300,440)  # 移動位置
        button2_4.clicked.connect(self.buttonClicked2_4) # 設置button啟動function

        button3 = QPushButton(self)
        button3.setText("3 Augmented Reality")  # 建立名字
        button3.setStyleSheet("background-color:#FCFCFC; ")
        button3.setFixedHeight(35)
        button3.setFixedWidth(150)
        button3.move(540,120)  # 移動位置
        button3.clicked.connect(self.buttonClicked3) # 設置button啟動function
        
        button4 = QPushButton(self)
        button4.setText("2.4 Find Distortion")  # 建立名字
        button4.setStyleSheet("background-color:#FCFCFC; ")
        button4.setFixedHeight(35)
        button4.setFixedWidth(150)
        button4.move(780,120)  # 移動位置
        button4.clicked.connect(self.buttonClicked4) # 設置button啟動function

        button5_1 = QPushButton(self)
        button5_1.setText("5.1 Training")  # 建立名字
        button5_1.setStyleSheet("background-color:#FCFCFC; ")
        button5_1.setFixedHeight(35)
        button5_1.setFixedWidth(150)
        button5_1.move(1020,120)  # 移動位置
        button5_1.clicked.connect(self.buttonClicked5_1) # 設置button啟動function
    
        button5_3 = QPushButton(self)
        button5_3.setText("5.3 Classification")  # 建立名字
        button5_3.setStyleSheet("background-color:#FCFCFC; ")
        button5_3.setFixedHeight(35)
        button5_3.setFixedWidth(150)
        button5_3.move(1020,360)  # 移動位置
        button5_3.clicked.connect(self.buttonClicked5_3) # 設置button啟動function
    
    def buttonClicked1_1(self):
        img = cv2.imread("Datasets/Q1_Image/coin01.jpg")
        DrawContours(img,'image01')
        img2 = cv2.imread("Datasets/Q1_Image/coin02.jpg")
        DrawContours(img2,'image02')
        
    def buttonClicked1_2(self):
        print("1.2")
        img = cv2.imread("Datasets/Q1_Image/coin01.jpg")
        img2 = cv2.imread("Datasets/Q1_Image/coin02.jpg")
        count1 = CountContours(img,0)
        count2 = CountContours(img2,1)
        s1 = "There are " + str(count1) + " coins in coin01.jpg"
        s2 = "There are " + str(count2) + " coins in coin02.jpg"
        self.label1_1.setText(s1)
        self.label1_2.setText(s2)

    def buttonClicked2_1(self):
        print("2.1")
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((11*8,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        images = glob.glob('Datasets/Q2_Image/*.bmp')
        count = 0
        for fname in images:
            count += 1
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (11, 8),None)
        #print(type(corners))
        # If found, add object points, image points (after refining them)
            if ret == True:
                #print(count)
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray,corners,(15,15),(-1,-1),criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (11,8), corners2,ret)
                cv2.imshow(str(count),img)
                cv2.waitKey(1500)
                cv2.destroyAllWindows()
        
    def buttonClicked2_2(self): #intrinsic
        print("2.2")
        global K, dist
        K, dist = FindInDis()
        print(K)

    def buttonClicked2_3(self): #ex
        print("2.3")
        print(combIndex)
        #file = "Datasets/Q2_Image/" + str(combIndex) + ".bmp"
        #img = cv2.imread(file)
        K, dist = FindInDis()
        num = int(combIndex)
        print(result[num])

    def comboPrint(self,i): 
        global combIndex
        combIndex = i
        

    def buttonClicked2_4(self): #dis
        print("2.4")
        K, dist = FindInDis()
        print(dist)

    def buttonClicked3(self):
        #print("3")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((11*8,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        #images = glob.glob('Datasets/Q3_Image/*.bmp')
        count = 0
        #for fname in images:
        for i in range(5):
            count += 1
            fname = 'Datasets/Q3_Image/' + str(i+1) + '.bmp'
            img = cv2.imread(fname)
            print(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (11, 8),None)
            if ret == True:
                print(count)
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray,corners,(15,15),(-1,-1),criteria)
                imgpoints.append(corners)

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,gray.shape[::-1], None, None)
        point3d = np.float32([[3,3,-3],[3,5,0],[5,1,0],[1,1,0]])
        for i in range(5):
            rmatrix, _ = cv2.Rodrigues(rvecs[i])
            point2d, _ = cv2.projectPoints(point3d, rmatrix, tvecs[i], K, dist)
            #print(imgpoints)
            #print(dist)
            #print(rmatrix)
            #print(tvecs[3])
            point2d = point2d.ravel()
            print(point2d)
            pathname = 'Datasets/Q3_Image/' + str(i+1) + '.bmp'
            img1 = cv2.imread(pathname)
            x1 = point2d[0].astype(np.int32)
            y1 = point2d[1].astype(np.int32)
            x2 = point2d[2].astype(np.int32)
            y2 = point2d[3].astype(np.int32)
            x3 = point2d[4].astype(np.int32)
            y3 = point2d[5].astype(np.int32)
            x4 = point2d[6].astype(np.int32)
            y4 = point2d[7].astype(np.int32)
            cv2.line(img1,(x1,y1),(x2,y2),(0, 0, 255),5)
            cv2.line(img1,(x1,y1),(x3,y3),(0, 0, 255),5)
            cv2.line(img1,(x1,y1),(x4,y4),(0, 0, 255),5)
            cv2.line(img1,(x2,y2),(x3,y3),(0, 0, 255),5)
            cv2.line(img1,(x2,y2),(x4,y4),(0, 0, 255),5)
            cv2.line(img1,(x3,y3),(x4,y4),(0, 0, 255),5)
            cv2.imshow('My Image', img1)

            # 按下任意鍵則關閉所有視窗
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
        #print(point2d)
        #cv2.line()

    def buttonClicked4(self):
        print("4")
        imgL = cv2.imread('Datasets/Q4_image/imgL.png',0)
        imgR = cv2.imread('Datasets/Q4_image/imgR.png',0)
        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=11)
        disparity = stereo.compute(imgL,imgR)
        disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dshow = cv2.cvtColor(disparity, cv2.COLOR_GRAY2RGB)

        def mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(disparity[y,x])
                dshow = cv2.cvtColor(disparity,cv2.COLOR_GRAY2RGB)
                cv2.rectangle(dshow, (2400, 1800), (2820, 1920), (255, 255, 255), -1)
                
                cv2.putText(dshow, "Disparity: {} pixels".format(disparity[y, x]), (2415, 1840), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 2)
                cv2.putText(dshow, "Depth: {} mm".format(int(178*2826/(123+disparity[y, x]))), (2415, 1900), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 2)
                cv2.imshow("disparity", dshow)
        print(disparity)
        #plt.imshow(disparity,'gray')
        #plt.show()
        cv2.namedWindow("disparity", cv2.WINDOW_NORMAL)
        cv2.imshow("disparity", dshow)
        cv2.setMouseCallback("disparity", mouse)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def buttonClicked5_1(self):
        img = cv2.imread('5-1image.png')
        cv2.imshow("img",img)
        cv2.waitKey(2500)
        cv2.destroyAllWindows()

    def buttonClicked5_3(self):
        print("5-3")
        CatorDog = random.randint(0,1)
        num = random.randint(0,12499)
        if(CatorDog == 0): #cats
            path = 'Image/PetImages/Cat/' + str(num) + '.jpg'
        else:
            path = 'Image/PetImages/Dog/' + str(num) + '.jpg'
        ##print(CatorDog)
        #print(num)
        img_width, img_height = 414, 500

        # load the model we saved
        model = load_model('model.h5')
        model.compile(loss='binary_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])
        print(model.summary())
        #path = 'reach3.jpg'
        # predicting images
        img = image.load_img(path, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        classes = model.predict_classes(images, batch_size=10)
        print(classes)
        image_show = cv2.imread(path)
        if(classes == 0):
            print("cat")
            cv2.imshow("cat",image_show)
        else:
            cv2.imshow("dog",image_show)
            print("dog")

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def FindInDis():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((11*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('Datasets/Q2_Image/*.bmp')
    count = 0
    for fname in images:
        count += 1
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (11, 8),None)
        #print(type(corners))
        # If found, add object points, image points (after refining them)
        if ret == True:
            #print(count)
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(15,15),(-1,-1),criteria)
            imgpoints.append(corners2)

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,gray.shape[::-1], None, None)
    a, b = cv2.Rodrigues(rvecs[0])
    #print(a)
    global result 
    result = np.zeros((15,3,4))
    for m in range (15):
        result[m] = np.array([[0.0,0.0,0.0,0.0],
                        [0.0,0.0,0.0,0.0],
                        [0.0,0.0,0.0,0.0]])
        a, b = cv2.Rodrigues(rvecs[m])
        emtrx = a
    
        b = tvecs[m]
    
        #print(b)
        #print(emtrx)
        l = 0
        for i in range(3):
            for j in range(4):
                if(j != 3):
                    result[m][i][j] = emtrx[i][j]
                else:
                    result[m][i][j] = b[l]
                    l += 1
    #tvecs[0] = tvecs[0].ravel()
    
    #print(emtrx)
    #print(tvecs[0])
    #np.insert(emtrx,3,[1,2,3],1)
        
    #print(emtrx)
    float_K = K.astype(np.float32) 
    float_dist = dist.astype(np.float32)
    
    #print(float_dist)
    return float_K, float_dist

def CountContours(img1,x):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if x==1:
        threshold = 145
    else:
        threshold = 130
    ret,img1 = cv2.threshold(img1,threshold,255,1)
    edges1 = cv2.Canny(img1, 10, 200)
    contours1, hierarchy = cv2.findContours(edges1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    print(len(contours1))
    return len(contours1)
    
    

def DrawContours(img,windowName):
    #print("1.1")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #ret,img = cv2.threshold(img,145,255,1)
    cv2.GaussianBlur(img, (3, 3), 0)
    
    edges = cv2.Canny(img, 10, 200)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    #print(len(contours))
    cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
    cv2.imshow(windowName,img)
    cv2.waitKey(1500)
    cv2.destroyAllWindows() 


app = QApplication(sys.argv)
w = MainWindow()
w.show()
sys.exit(app.exec_())

