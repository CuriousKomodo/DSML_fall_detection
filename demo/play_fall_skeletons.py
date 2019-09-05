import numpy as np
import cv2
import sys
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/demo/')
from openpose_kinect_integrate import *
from read_real_time_skeletons import *

def visualize_integrated(pose, keep_3D=False, fall_interval=None, confidence=None):
    joints_OP, joints_K, connection = joints_to_extract()
    joints = 17
    if keep_3D:
        dims = 3
    else:
        dims  = 2
    dims = 2
    skeletons_to_plot=[]

    pose = np.reshape(pose,(pose.shape[0], dims, joints)) #transpose to t x 3 x joints
    pose = np.transpose(pose,axes=(0,2,1)) #transpose to t x joints x 3
    v_min = pose.min()
    v_max = pose.max()
    pose = (pose - v_min) / (v_max - v_min)  # ensure values between 0 and 1.
    pose_n = 1000 * pose
    skeletons_to_plot.append(pose_n)
    color_map = np.zeros(pose.shape[0])

    if len(fall_interval)>1:
        end_interval =[x+30 for x in fall_interval]
        t = 0
        interval=[]
        interval.append(fall_interval[0])
        global_max = end_interval[0]
        for i in range(len(fall_interval)-1):
            if end_interval[i]< fall_interval[i+1]:
                interval.append(end_interval[i])#end of fall, even
                interval.append(fall_interval[i+1])#beginning of fall, odd
        if len(interval)%2 ==1:
            interval.append(pose.shape[0]-1) #add an ending, if fall is detected till the end
        print('fall_interval=',fall_interval)
        print('interval=',interval)

        for i in np.arange(0,len(interval),2):
            color_map[interval[i]:interval[i+1]] = 1
    t=0
    while True:
        t+=1
        #print('t=',t)
        '''
        if (t > fall_interval[0]) & (t - fall_interval[0] < 30):
            color = (0,0,255)
            if t==fall_interval[0]+29:
                fall_interval = fall_interval[1:]
        else:
            color = (255, 0, 0)
        '''

        img = np.ones((1000,1000,3))
        if color_map[t]==1:
            color = (0,0,255)
        else:
            color = (255, 0, 0)
        pose = skeletons_to_plot[0]

        for joint in range(pose.shape[1]):
            cv2.circle(img, (int(pose[t,joint,0]), int(pose[t,joint,1])),2, color, 3)

        for link in connection:
            joint1 = link[0]
            joint2 = link[1]

            conf1 = confidence[t,joint1]
            conf2 = confidence[t,joint2]
                #only plot line when we are confident enough like > 0.5
            if (conf1 > 0.1) & (conf2 >0.1):
                cv2.line(img, (int(pose[t,joint1,0]),int(pose[t,joint1,1])), (int(pose[t,joint2,0]),int(pose[t,joint2,1])),color, 1)
            #else:
                #if (sum(pose[t,joint1,:])!=0) & (sum(pose[t,joint1,:])!=0):
                #    cv2.line(img, (int(pose[t,joint1,0]),int(pose[t,joint1,1])), (int(pose[t,joint2,0]),int(pose[t,joint2,1])),color, 1)

        cv2.imshow('image',img)

        k = cv2.waitKey(200)
        if k==27:    # Esc key to stop
            break
        elif k==-1:  # normally -1 returned,so don't print it
            continue
        else:
            print(k) # else print its value

    cvMoveWindow("WndowName",250,250)
    cv2.destroyAllWindows()
