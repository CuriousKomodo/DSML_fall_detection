import numpy as np
import os
import sys
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/demo/')
from read_real_time_skeletons import *
from load_models import *
#sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/filter/')
#from kalman_filter import *
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/tool/')
from functions import *
import argparse
sys.path.insert(0, '/Users/kefei/Documents/openpose/build/examples/tutorial_api_python/')
import pyglet
#import capture_realtime_skeletons
import time
from datetime import datetime
import gc
import librosa
import soundfile as sf
from play_fall_skeletons import *
import matplotlib.pyplot as plt
#load model!
#kf = KfPose()
#x,_ = librosa.load('/Users/kefei/Downloads/woop.mp3', sr=16000)
#sf.write('/Users/kefei/Downloads/tmp.wav', x, 16000)
#wave.open('tmp.wav','r')
from PIL import Image


#past_ae_weight_path = '/Users/kefei/Documents/mm_fall_detection/demo/models/2D/past/dropout/past_VAE_enc_model5_weights.h5'
past_ae_weight_path ='/Users/kefei/Documents/mm_fall_detection/demo/models/2D/past/past_VAE_enc_model4_weights.h5'
pca_dir = '/Users/kefei/Documents/mm_fall_detection/demo/models/2D/classification/pca.pkl'
svm_dir = '/Users/kefei/Documents/mm_fall_detection/demo/models/2D/classification/svm.pkl'
enc_model = load_encoder(weight_path=past_ae_weight_path)
pca,svm = load_classifier(pca_dir,svm_dir)
#classifier_model = load_encoder_new(weight_path=None)
#poses = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/fall_test.npy')


#make prediction every 2 seconds
interval = 2
#alarm = pyglet.media.load('/Users/kefei/Downloads/woop.mp3')
fall_interval = []

def fall_detection(real_time=False):
    file_path = '/Users/kefei/Documents/real_time_skeletons/joints24.npy'
    window=10
    pose,confidence = obtain_realtime_integrated_joints(keep_3D=False, file_path = file_path)
    if not real_time:
        timelength = int(pose.shape[0]*1.0/window)
    else:
        timelength = 20 #set by default, equivalent to seeing a sequence of just over 200 frames

    start_time = time.time()
    for i in range(timelength):
        startTime = datetime.now()
        time.sleep(start_time + i*interval - time.time())
        if real_time:
            pose,confidence = obtain_realtime_integrated_joints(keep_3D=False, file_path = file_path) #maxmimum 200 frames, (T,num_joints * dim)
        #print('time taken for loading dataset',datetime.now() - startTime)
        startTime = datetime.now()
        pmax = pose.max()
        pmin = pose.min()
        pmean = pose.mean()
        mask = pose>0

        pose[mask] = (pose[mask]-pmean)/(pmax-pmin) #normalize to be within [-1,1].
        print(pmax,pmin)
        if real_time:
            if pose[None,-30:,:].sum() !=0.0:
                z_mean, z_log_var =enc_model.predict(pose[None,-30:,:]) #ONLY PREDICT USING THE LAST 30 FRAMES!
            else:
                print('no skeletons detected')
        else:
            if pose[None,i*window:i*window+30,:].sum() !=0.0:
                z_mean, z_log_var =enc_model.predict(pose[None,i*window:i*window+30,:])
            else:
                print('no skeletons detected')

        features = pca.transform(np.concatenate([z_mean,z_log_var],axis=-1))
        pred = svm.predict(features)
        print('Fall:',pred)
        if pred==1:
            if real_time:
                fall_interval.append(max(pose.shape[0]-30,0))#extract the point where fall starts
            else:
                fall_interval.append(max(i*window,0)) #extract the point where fall starts
            print('fall_interval=',fall_interval)
            try:
                ImageAddress = '/Users/kefei/Documents/Thesis/newalert.png'
                ImageItself = Image.open(ImageAddress)
                ImageNumpyFormat = np.asarray(ImageItself)
                plt.imshow(ImageNumpyFormat)
                plt.draw()
                plt.pause(interval-0.5) # pause how many seconds
                plt.close()
            except Exception:
                pass
            #alarm.play()
            #pyglet.app.run()
        print('time taken for prediction',datetime.now() - startTime)


    print('Now visualise the skeletons')
    time.sleep(3)
    pose, confidence = obtain_realtime_integrated_joints(keep_3D=False, file_path = file_path)
    if len(fall_interval)>0:
        visualize_integrated(pose, keep_3D=False, fall_interval=fall_interval, confidence=confidence)
    else:
        print('no fall detected')
        visualize_integrated(pose, keep_3D=False, fall_interval=[10000], confidence=confidence)
fall_detection(real_time=False)
