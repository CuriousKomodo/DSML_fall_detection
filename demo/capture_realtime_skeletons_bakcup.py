# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
from pathlib import Path
import numpy as np
from datetime import datetime
import sys
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/demo/')
from read_real_time_skeletons import *

def main():
    #reset the joint.npy file at the beginning of time
    reset_realtime_skeleton()

    #saving the models
    p = Path('/Users/kefei/Documents/real_time_skeletons/joints.npy')

    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()


    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    try:
        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()


        #Capture video from webcam
        vid_capture = cv2.VideoCapture(0)
        vid_cod = cv2.VideoWriter_fourcc(*'XVID')
        joints = []
        #output = cv2.VideoWriter("videos/cam_video.mp4", vid_cod, 20.0, (640,480))


        t = 0
        s=0

        while(True):
            # Capture each frame of webcam video
            ret,frame = vid_capture.read()

            # Process Image
            datum = op.Datum()
            #imageToProcess = cv2.imread(args[0].image_path)
            datum.cvInputData = frame
            opWrapper.emplaceAndPop([datum])

            cv2.imshow("My cam video", datum.cvOutputData) # it was: frame

            keypoints = datum.poseKeypoints #(1, 25, 3)

            #only save coordinates of one person. only save when detected
            #Maybe set an upper limit, like 200 frames in storage?
            print(keypoints.shape)
            if t%2 ==0:
                if keypoints.shape == (1,25,3):
                    with p.open('ab') as f:
                        s+=1
                        joints.append(keypoints)
                        np.save(f, keypoints)
                        if t>200:
                            startTime = datetime.now()
                            joints.pop(0) #remove the first observation. then save the entire 200-frame
                            np.save('/Users/kefei/Documents/real_time_skeletons/joints.npy',np.asarray(joints))
                            print(datetime.now() - startTime)

                elif keypoints.shape[0]> 1:
                    if keypoints[0,:,:].shape == (25,3):
                        first_person_points = keypoints[None,:,:]
                        print(first_person_joints.shape)
                        with p.open('ab') as f:
                            s+=1
                            joints.append(first_person_points)
                            np.save(f,first_person_points) #save the first person it sees.
                            if t>200:
                                joints.pop(0)
                                np.save('/Users/kefei/Documents/real_time_skeletons/joints.npy',np.asarray(joints))
            t+=1


            #output.write(frame)
            # Close and break the loop after pressing "x" key
            k = cv2.waitKey(1)

            if t>200:
                print('exceeds 200 frames')
            if  k == ord('x'):
                break
            elif k == ord('q'):
                print('t=',t)
                break
        # close the already opened camera
        print('t=',t)
        print('s=',s)
        #np.save('/Users/kefei/Documents/joints.npy',joints)
        vid_capture.release()

        # close the already opened file
        #output.release()
        # close the window and de-allocate any associated memory usage
        cv2.destroyAllWindows()

        # Display Image
        #print("Body keypoints: \n" + str(datum.poseKeypoints))
        #cv2.imshow("OpenPose 1.5.0 - Tutorial Python API", datum.cvOutputData)
        #cv2.waitKey(0)
    except Exception as e:
        # print(e)
        sys.exit(-1)

if __name__ == "__main__":
    main()
