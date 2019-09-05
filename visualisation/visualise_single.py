import cv2
import numpy as np

#Expect poses generated in the form of t x (3 x 25).


#truth = np.load('/Users/kefei/Documents/Dataset/one_pose.npy')
truth = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_test_60_2_untrimmed.npy')
truth = truth[38702]
pred = truth

skeletons_to_plot=[]

for pose in [truth,pred]:
    #print(pose)
    pose = np.reshape(pose,(pose.shape[0], 3, 25))
    pose[:,1,:] = pose[:,1,:]*-1.0
    pose = np.transpose(pose,axes=(0,2,1)) #transpose to t x 25 x 3
    print(pose.shape)

    v_min = pose.min()
    v_max = pose.max()
    pose = (pose - v_min) / (v_max - v_min)  # ensure values between 0 and 1.
    pose_n = 300 * pose
    skeletons_to_plot.append(pose_n)


connection = [[0, 1], [1, 2], [2, 3],[2, 4], [2,8],[8,9],[4,5],[9,10],[5,6],[16,17],[12,13],[17,18],[13,14]]

t = 0

while True:
    t+=1
    img = np.ones((300,300,3))
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
