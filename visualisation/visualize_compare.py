import cv2
import numpy as np

#Expect poses generated in the form of t x (3 x 25).


#truth = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_test_60_2_untrimmed.npy')
truth = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/prediction/future_poses_trimmed_15_train_valid.npy')
#labels = list(np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/y_test_fall_aug.npy'))# TEMP:

pred = np.load('/Users/kefei/Documents/results/win60/future/past_ae_recon1.npy')
pred = np.flip(pred,axis=1)

truth = truth * 0.8 + 0.1

def extract_fall_sequences(test, num_fall_examples=179):
  cycle =int(test.shape[0]/24)
  fall_inds = []
  for i in range(24):
    ind = np.arange(cycle*i,cycle*(i+1),1)
    fall_inds.append(ind[-179*5:]) #number of fall examples in each cycle
  return fall_inds

view_fall = True
example = 6

if view_fall:
    fall_inds = extract_fall_sequences(truth, num_fall_examples=179)
    cycle = 0
    fall_truth = truth[fall_inds[cycle]]
    fall_pred =  pred[fall_inds[cycle]]
    sequences = [fall_truth, fall_pred]
else:
    sequences = [truth, pred]
#np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/fall_test_30_untrimmed.npy',fall_truth)
#np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/fall_recon5_30_untrimmed.npy',fall_pred)
#truth = np.load('/Users/kefei/Documents/Dataset/NTU/data_t.npy')
#truth = [truth]

def center_by_joint_location(joint_num, dataset):
    #if joint num = 0, then for x axis, ind = 0, for y: ind = 25, for z: ind = 50
    dataset[:,:,0:25] = dataset[:,:,0:25] - dataset[:,:,joint_num,None]
    dataset[:,:,25:50] = dataset[:,:,25:50] - dataset[:,:,joint_num+25,None]
    dataset[:,:,50:] = dataset[:,:,50:] - dataset[:,:,joint_num+50,None]
    return dataset
#truth = center_by_joint_location(0, truth)

skeletons_to_plot=[]

for skeleton_sequence in sequences:
    pose = skeleton_sequence[example]
    pose = np.reshape(pose,(pose.shape[0], 3, 25))
    pose[:,1,:] = pose[:,1,:]*-1.0
    pose = np.transpose(pose,axes=(0,2,1)) #transpose to t x 25 x 3

    v_min = pose.min()
    v_max = pose.max()
    pose = (pose - v_min) / (v_max - v_min)  # ensure values between 0 and 1.
    pose_n = 1000 * pose
    skeletons_to_plot.append(pose_n)


connection = [[0, 1], [1, 2], [2, 3],[2, 4], [2,8],[8,9],[4,5],[9,10],[5,6],[16,17],[12,13],[17,18],[13,14]]

t = 0

while True:
    t+=1
    img = np.ones((500,1000,3))
    colors = [(255, 0, 0),(0,0,255)]

    for i in range(2):
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
