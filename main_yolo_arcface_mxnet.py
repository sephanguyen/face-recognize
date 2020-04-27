import cv2
from imutils.video import WebcamVideoStream
from PyQt5 import QtCore, QtGui, QtWidgets
from Ui_main import Ui_mainwindow
import sys
import time
import numpy as np
import math
import pickle
import mxnet as mx
from numpy.linalg import norm

def load_model_recognize(link_model,input_shape):
    sym,arg_param,aux_param=mx.model.load_checkpoint(link_model,0)
    all_layer=sym.get_internals()

    sym=all_layer['fc1_output']
    model=mx.mod.Module(symbol=sym,context=[mx.cpu()],label_names= None)
    model.bind(data_shapes=[('data',input_shape)])
    model.set_params(arg_params=arg_param,aux_params=aux_param)
    return model

def load_model_detect(link_cfg,load_weight):
    model = cv2.dnn.readNetFromDarknet(link_cfg,load_weight)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return model

def load_database(link_database):
    file = open(link_database,'rb')
    database = pickle.load(file)
    return database

def detect_face(net_detect, image):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (320, 320), [0, 0, 0], swapRB=True, crop=False)
    net_detect.setInput(blob)
    outs = net_detect.forward(['yolo_16', 'yolo_23'])
    
    width = image.shape[1]
    height = image.shape[0]
    confidences = []
    boxes = []
    position_face=[]

    for out in outs:
        for detection in out:
            scores = detection[5:]
            confidence = scores[0]
            if confidence > 0.5:
                box = detection[0:4] * np.array([width, height, width, height])
                x = int(math.ceil(box[0] - (box[2] / 2)))
                y = int(math.ceil(box[1] - (box[3] / 2)))
                boxes.append([x, y, int(round(box[2], 0)), int(round(box[3], 0))])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    for i in indices:
        i = i[0]
        x, y = boxes[i][0], boxes[i][1]
        w, h = boxes[i][2], boxes[i][3]
        position_face.append([x, y,x + w, y + h])
    return position_face

    
def recognize_face(net_recognize, database, image):
    image = np.transpose(image,(2,0,1) )
    input_blob = np.expand_dims(image, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    net_recognize.forward(db, is_train=False)
    emb = net_recognize.get_outputs()[0].asnumpy()
    emb = emb.flatten()
    max_sim=0
    min_distance=0
    final_label="Not In Database"
    for label, emb_base in database.items():
        sim = np.dot(emb, emb_base)/(norm(emb_base)*norm(emb_base))
        if(sim > max_sim):
            final_label=label
            max_sim=sim
            min_distance=norm(emb - emb_base)
            
    return [final_label,max_sim,min_distance]
        

def combine_detect_recognize(net_detect,net_recognize,database, image):
    position_face = detect_face(net_detect, image)
    img_face=image.copy()
    img_face = cv2.cvtColor(img_face, cv2.COLOR_BGR2RGB)
    if(len(position_face)>=1):
        pos = position_face[0]
        img_face = img_face[pos[1]:pos[3],pos[0]:pos[2]]
        img_face = cv2.resize(img_face,(112,112))
        label,percent,distance = recognize_face(net_recognize, database, img_face)
        cv2.rectangle(image,(pos[0], pos[1]),(pos[2],pos[3]),(0,255,0),2)
        text1="Similarity: " + str(percent)
        text2="Distance:" + str(distance)
        cv2.putText(image, label, (pos[0]-10,pos[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 2)
        cv2.putText(image, text1, (pos[0]-10,pos[1]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 2)
        cv2.putText(image, text2, (pos[0]-10,pos[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 2)
    return image


class Video(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(Video, self).__init__(parent)
        self.ui = Ui_mainwindow()
        self.ui.setupUi()

        self.ui.btn_start.clicked.connect(self.start_camera)
        self.ui.btn_browser.clicked.connect(self.browser)

        self.ui.image_label.setScaledContents(True)
        self.cap = None
        
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
    
        self.net_detect = load_model_detect('face_detect/yolo_model/face_detect_v6.cfg',
                                            'face_detect/yolo_model/model_detect_v6.weights')
      
        self.net_recognize= load_model_recognize('face_recognize/mobilenet_arcface_mxnet_model_v1/model',(1,3,112,112))
        self.database = load_database('face_recognize/database/mobilenet_arcface_mxnet_database_v1.pkl')

    @QtCore.pyqtSlot()
    def start_camera(self):
        if self.cap is None:
            self.cap = WebcamVideoStream(0).start()
        self.timer.start()

    @QtCore.pyqtSlot()
    def update_frame(self):
        img = self.cap.read()
        img = cv2.resize(img, (320, 320))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        start = time.process_time()
        img = combine_detect_recognize(self.net_detect,self.net_recognize, 
                                       self.database,img)
        self.ui.status_bar.showMessage(str(time.process_time() - start))
        self.display_image(img, True)

    def display_image(self, img, windows=True):
        image_show = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], QtGui.QImage.Format_RGB888)
        if windows:
            self.ui.image_label.setPixmap(QtGui.QPixmap.fromImage(image_show))

    def show(self):
        self.ui.showUi()

    def browser(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        folder_path = QtWidgets.QFileDialog.getExistingDirectory()
        self.ui.line_path.setText(folder_path)
        self.link_save = folder_path


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Video()
    window.show()
    sys.exit(app.exec_())
