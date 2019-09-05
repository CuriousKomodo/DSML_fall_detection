import numpy as np
import os


def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence',
'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX',
'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key,
f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence

def get_data_info(skeleton_path):

  #obtain data
  data = read_xyz(skeleton_path)

  #determine if single or doulbe
  if data[:,:,:,1].min()==data[:,:,:,1].max():
    single=True
  else:
    single=False

  #obtain label
  res = re.findall("A(\d+).skeleton",skeleton_path)

  label=int(res[0])

  return label, single, data



def read_xyz(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
    return data

def zero_padding(data_list,single=True):
  max_length = 0
  if single:
    dim = 75
  else:
    dim = 150
  for i in range(len(data_list)):
    length = data_list[i].shape[0]
    if length>max_length:
      max_length =length

  dataset = np.zeros((len(data_list),max_length,dim))
  for i in range(len(data_list)):
    dataset[i,:data_list[i].shape[0],:] = data_list[i]
  return dataset


from os import listdir
from os.path import isfile, join

data_path = '/Users/kefei/Documents/Dataset/NTU/nturgbd_skeletons_s001_to_s017/skeletons/'

import re
from os import listdir
from os.path import isfile, join

onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,
f))]

single_data= []
single_label = []
double_data = []
double_label = []
i=0

'''
skeleton_path =data_path+'/S001C001P001R001A001.skeleton'
label, single, data = get_data_info(skeleton_path)
np.save('/Users/kefei/Documents/Dataset/NTU/one_data.npy',data )
'''

for f in onlyfiles:
  print(i,f)
  skeleton_path = data_path+'/'+f
  label, single, data = get_data_info(skeleton_path)
  if single==True:
    #data_r = np.reshape(data[:,:,:,0],[data.shape[1],data.shape[2]*data.shape[0]])
    data_t = np.transpose(data[:,:,:,0],axes=(1,0,2)) #transpose to shape [t, 3, 25]
    data_r = data_t.reshape(data_t.shape[0],75) #then perform reshape! [t,3x25] lol
    single_data.append(data_r)
    single_label.append(label)
  else:
    data_r = np.reshape(data,[data.shape[1],data.shape[0]*data.shape[2]*data.shape[-1]])
    double_data.append(data_r)
    double_label.append(label)
  i+=1


single_dataset = zero_padding(single_data,single=True)
double_dataset= zero_padding(double_data,single=False)

np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/single_dataset.npy',single_dataset)
np.save('/Users/kefei/Documents/Dataset/NTU/double_poses/double_dataset.npy',double_dataset)
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/single_labels.npy',single_label)
np.save('/Users/kefei/Documents/Dataset/NTU/double_poses/double_labels.npy',double_label)
