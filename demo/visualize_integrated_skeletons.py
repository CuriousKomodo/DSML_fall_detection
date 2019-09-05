import numpy as np
import cv2
import sys
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/demo/')
from openpose_kinect_integrate import *
from read_real_time_skeletons import *

#only visualise fall examples
#load model!
def crop_2D(X):#takes 3D skeleton
    X = X.reshape(X.shape[0],X.shape[1],3,int(X.shape[-1]/3))
    X = X[:,:,:2,:] #only extract x,y cooridnates
    X = X.reshape(X.shape[0],X.shape[1],2*X.shape[-1])
    return X

def visualize_integrated(kinect=True,keep_3D=True, view_fall=False):
    joints_OP, joints_K, connection = joints_to_extract()
    if kinect:
        if view_fall:
            #truth = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/fall_test.npy')
            truth = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_test_60_2_untrimmed.npy')

            fall = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/fall_test.npy')

            if not keep_3D:
                fall = crop_2D(fall)
            truth = fall[700]

        else:
            x = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_test_60_2_trimmed.npy')
            y =  np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/y_test_60_2_trimmed.npy')
            non_fall = x[y!=43]
            if not keep_3D:
                non_fall = crop_2D(non_fall)
            truth = non_fall[500]
    else:
        file_path = '/Users/kefei/Documents/real_time_skeletons/joints15.npy'
        truth = read_realtime_skeleton(file_path)
        print(truth.shape)
        transposed_op_pose, confidence = transpose_openpose(joints_OP, truth, keep_3D=keep_3D)
        reshaped_pose = reshape_openpose(transposed_op_pose, keep_3D=keep_3D)
        truth = normalize_openpose(reshaped_pose)

        #truth= truth[0:30]
        print('truth shape', truth.shape)
        #poses reconstructed by DAE but really bad.
        #
        #truth = np.load('/Users/kefei/Documents/mm_fall_detection/demo/results/recon_pose_2D.npy')
        #print('recon shape',truth.shape)
    #truth = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/2D_DAE/test_realtime.npy')
    #truth = np.load('/Users/kefei/Documents/mm_fall_detection/demo/results/2D/DAE/DAE_recon.npy')
    #truth = np.load('/Users/kefei/Documents/mm_fall_detection/demo/results/recon_pose_2D.npy')
    #truth = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/demo_data/classification/2D/X_test_cropped.npy')
    #target = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/demo_data/classification/2D/target_test.npy')
    #truth = truth[target==1]
    #print('max truth', truth.max())
    #print('min truth', truth.min())
    #truth = truth[10]

    #print(truth.shape)
    joints = len(joints_OP)
    pred = truth.copy()
    if keep_3D:
        dims = 3
    else:
        dims  = 2
    dims = 2
    skeletons_to_plot=[]

    for pose in [truth,pred]:
        pose = np.reshape(pose,(pose.shape[0], dims, joints)) #transpose to t x 3 x joints
        print(pose.shape)
        pose = np.transpose(pose,axes=(0,2,1)) #transpose to t x joints x 3
        v_min = pose.min()
        v_max = pose.max()
        pose = (pose - v_min) / (v_max - v_min)  # ensure values between 0 and 1.
        pose_n = 1000 * pose
        skeletons_to_plot.append(pose_n)

    t = 0

    while True:
        t+=1
        img = np.ones((1000,1000,3))
        colors = [(255, 0, 0),(0,0,255)]

        for i in range(1):
            pose = skeletons_to_plot[i]
            if i==0:
                color = (255, 0, 0)
            else:
                color = (0,0,255)

            for joint in range(pose.shape[1]):
                cv2.circle(img, (int(pose[t,joint,0]), int(pose[t,joint,1])),2, color, 3)

            for link in connection:
                joint1 = link[0]
                joint2 = link[1]
                if not kinect:
                    conf1 = confidence[t,joint1]
                    conf2 = confidence[t,joint2]
                #only plot line when we are confident enough like > 0.5
                    if (conf1 > -0.1) & (conf2 >-0.1):
                        cv2.line(img, (int(pose[t,joint1,0]),int(pose[t,joint1,1])), (int(pose[t,joint2,0]),int(pose[t,joint2,1])),color, 1)
                else:
                    cv2.line(img, (int(pose[t,joint1,0]),int(pose[t,joint1,1])), (int(pose[t,joint2,0]),int(pose[t,joint2,1])),color, 1)

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

visualize_integrated(kinect=True,keep_3D=False,view_fall=False)
