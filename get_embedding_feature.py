#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:08:13 2019

@author: vinhchau
"""
import mxnet as mx

sym,arg_param,aux_param=mx.model.load_checkpoint('model-r50-am-lfw/model',0)
all_layer=sym.get_internals()

sym=all_layer['fc1_output']
model=mx.mod.Module(symbol=sym,context=[mx.cpu()],label_names= None)
model.bind(data_shapes=[('data',(1,3,112,112))])
model.set_params(arg_params=arg_param,aux_params=aux_param)


from imutils.paths import list_images
import cv2 
import numpy as np
import pickle
import time

database=dict()

for i in list_images('data/database'):
    temp_label=i.split('/')
    label=temp_label[2][:-4]
    img = cv2.imread(i)
    img = cv2.resize(img,(112,112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img,(2,0,1) )
    input_blob = np.expand_dims(img, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    start = time.process_time()
    model.forward(db, is_train=False)
    embedding = model.get_outputs()[0].asnumpy()
    database[label]=embedding.flatten()
    print(time.process_time() - start)
    
f = open('mobilenet_arcface_mxnet_database_v1.pkl',"wb")
pickle.dump(database,f)
f.close()