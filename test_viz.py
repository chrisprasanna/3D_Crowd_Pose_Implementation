#%%
import poseviz
import pickle
import os
import numpy as np
import imageio
from PIL import Image
import glob
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "notebook_connected"
import pandas as pd

#%%
main_dir = r"C:\Users\David Boe\Documents\C Prasanna\3D-Multi-Person-Pose-main\mupots"
sub_dirs = ['pred', 'pred_dep', 'pred_dep_bu', 'pred_bu', 'pred_inte', 'pred_dep_inte']

d = {}
directory = os.path.join(main_dir, 'pred_inte')
for file in os.listdir(directory):
    if file.endswith('.pkl'):
        filename = os.path.join(directory, file)
        with open(filename, 'rb') as f:
            d[file.split('.')[0]] = pickle.load(f)
names = list(d.keys())

Data = d[names[0]]
print(type(Data), Data.shape)

# [frame number, person, coordinate, keypoint]

#%%

# joint_names = [
#     'nose', # 0
#     # 'neck', # 1
#     'right shoudler', # 1
#     'right elbow', # 2
#     'right wrist', # 3
#     'left shoudler', # 4
#     'left elbow', # 5
#     'left wrist', # 6
#     'right hip', # 7
#     'right knee', # 8
#     'right ankle', # 9
#     'left hip', # 10
#     'left knee', # 11
#     'left ankle', # 12
#     'right eye', # 13
#     'left eye', # 14
#     'right ear', # 15
#     'left ear' # 16
# ]

# joint_edges = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
#     [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
#     [1,0], [0,14], [14,16], [0,15], [15,17],
#     [2,17], [5,16]]

joint_names = [
    'nose',
    'leftEye',
    'rightEye',
    'leftEar',
    'rightEar',
    'leftShoulder',
    'rightShoulder',
    'leftElbow',
    'rightElbow',
    'leftWrist',
    'rightWrist',
    'leftHip',
    'rightHip',
    'leftKnee',
    'rightKnee',
    'leftAnkle',
    'rightAnkle'
 ]

NUM_KEYPOINTS = len(joint_names)

PART_IDS = {pn: pid for pid, pn in enumerate(joint_names)}

CONNECTED_PART_NAMES = [
    ("leftHip", "leftShoulder"), ("leftElbow", "leftShoulder"),
    ("leftElbow", "leftWrist"), ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"), ("rightHip", "rightShoulder"),
    ("rightElbow", "rightShoulder"), ("rightElbow", "rightWrist"),
    ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
    ("leftShoulder", "rightShoulder"), ("leftHip", "rightHip")
]
joint_edges = [(PART_IDS[a], PART_IDS[b]) for a, b in CONNECTED_PART_NAMES]

# joint_edges, joint_edges.shape

# viz = poseviz.PoseViz(joint_names, joint_edges)

#%%
imagedir = r"C:\Users\David Boe\Documents\C Prasanna\3D-Multi-Person-Pose-main\MultiPersonTestSet\TS1"

frames = {}
filenames = []
for filename in glob.glob(imagedir + '/*.JPG'): #assuming jpeg
    im=Image.open(filename)
    # image_list.append(im) 
    filenames.append(filename)
    frames[filename] = im

#%%

# num_frames = Data.shape[0]
# num_people = Data.shape[1]

#  # Iterate over the frames of e.g. a video
# for i in range(num_frames):
#     # Get the current frame
#     frame = frames[filenames[i]]

#     # Make predictions here
#     # ...

#     # Update the visualization
#     viz.update(
#         frame=frame,
#         boxes=np.array([[10, 20, 100, 100]], np.float32),
#         poses=np.array([[[100, 100, 2000], [-100, 100, 2000]]], np.float32),
#         camera=poseviz.Camera.from_fov(55, frame.shape[:2]))

# %%

def plot_landmarks(landmarks, landmark_names, connections=None, frame_num=0, fig=None):
    """_summary_

    Args:
        landmarks (_type_): [landmark, coordainte (x,y,z)]
        connections (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    plotted_landmarks = {}
    for idx, landmark in enumerate(landmarks):
        # plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)
        plotted_landmarks[idx] = (-landmark[0], landmark[1], -landmark[2])
    if connections:
        out_cn = []
        num_landmarks = len(landmarks)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(
                    f"Landmark index is out of range. Invalid connection "
                    f"from landmark #{start_idx} to landmark #{end_idx}."
                )
            if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                landmark_pair = [
                    plotted_landmarks[start_idx],
                    plotted_landmarks[end_idx],
                ]
                out_cn.append(
                    dict(
                        xs=[landmark_pair[0][0], landmark_pair[1][0]],
                        ys=[landmark_pair[0][1], landmark_pair[1][1]],
                        zs=[landmark_pair[0][2], landmark_pair[1][2]],
                    )
                )
        cn2 = {"xs": [], "ys": [], "zs": []}
        for pair in out_cn:
            for k in pair.keys():
                cn2[k].append(pair[k][0])
                cn2[k].append(pair[k][1])
                cn2[k].append(None)

    df = pd.DataFrame(plotted_landmarks).T.rename(columns={0: "z", 1: "x", 2: "y"})
    df["lm"] = df.index.map(lambda s: landmark_names[s]).values
    # if frame_num == 0:
    fig = (
        px.scatter_3d(df, x="z", y="x", z="y", hover_name="lm")
        .update_traces(marker={"color": "red"})
        .update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            scene={"camera": {"eye": {"x": 2.1, "y": 0, "z": 0}}},
        )
    )
    fig.add_traces(
        [
            go.Scatter3d(
                x=cn2["xs"],
                y=cn2["ys"],
                z=cn2["zs"],
                mode="lines",
                line={"color": "black", "width": 5},
                name="connections",
            )
        ]
    )
    

    return fig

num_frames = Data.shape[0]
num_people = Data.shape[1]
fig = None

for i in range(num_frames):
    
    # print(f"frame {i+1}")
    
    results = Data[i,0,:,:]
    results = np.transpose(results)
    
    fig = plot_landmarks(results, joint_names, joint_edges, i, fig)
    
    pio.show(fig)
   
    
# %%
