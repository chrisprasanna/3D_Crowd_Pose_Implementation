#%% Imports
import pickle
import os
import numpy as np

from scipy.io import savemat

import re

# Folder containing your files
main_dir = r"C:\Users\David Boe\Documents\C Prasanna\3D-Multi-Person-Pose-main\mupots"
sub_dirs = ['pred', 'pred_dep', 'pred_dep_bu', 'pred_bu', 'pred_inte', 'pred_dep_inte']

#%%
# Loop over files and read pickles
for folder in sub_dirs:
    data = {}
    directory = os.path.join(main_dir, folder)
    for file in os.listdir(directory):
        if file.endswith('.pkl'):
            filename = os.path.join(directory, file)
            with open(filename, 'rb') as f:
                data[file.split('.')[0]] = pickle.load(f)

    names = list(data.keys())
    # print(folder)
    # print(np.shape(data[names[0]]))
    # print(np.shape(data[names[1]]))
    # print(np.shape(data[names[15]]))

#%%
d = {}
directory = os.path.join(main_dir, 'pred_inte')
for file in os.listdir(directory):
    if file.endswith('.pkl'):
        filename = os.path.join(directory, file)
        with open(filename, 'rb') as f:
            d[file.split('.')[0]] = pickle.load(f)
names = list(d.keys())
    
#%% 

print(names)

Data = d[names[0]]
print(type(Data))

for i, name in enumerate(names):
    Data = d[name]
    mdic = {"data": Data, "label": f"MuPots TS{name}"}
    
    savepath = os.path.abspath(main_dir + "/../")
    savepath = os.path.join(savepath, "MultiPersonTestSet", f"TS{name}")
    print(savepath)
    savefile = os.path.join(savepath, f"MuPots_TS{name}.mat")
    savemat(savefile, mdic)

# %%

dep_data = {}
directory = os.path.join(main_dir, 'pred_dep_inte')
for file in os.listdir(directory):
    if file.endswith('.pkl'):
        filename = os.path.join(directory, file)
        
        f = os.path.splitext(file)[0]
        tmp = re.split('_',f)
        test_num = tmp[0]
        # test_num  = test_num.lstrip('0')
        test_num = int(test_num) + 1
        person_num = int(tmp[1]) + 1
        print([test_num, person_num])
        
        with open(filename, 'rb') as f:
            dep_data[file.split('.')[0]] = pickle.load(f)
            # print(filename)
            # print(dep_data[file.split('.')[0]].shape)
            
            
        savepath = os.path.abspath(main_dir + "/../")
        savepath = os.path.join(savepath, "MultiPersonTestSet", f"TS{test_num}")
        savefile = os.path.join(savepath, f"MuPots_TS{test_num}_P{person_num}_depth.mat")
        
        mdic = {"data": dep_data[file.split('.')[0]], "label": f"MuPots TS{test_num} P{person_num}"}
        savemat(savefile, mdic)

# %%

# with open(r"C:\Users\David Boe\Documents\C Prasanna\3D-Multi-Person-Pose-main\mupots\pred_dep_inte\00_00.pkl", 
#           'rb') as f:
#     dep = pickle.load(f)

# print(dep)

# %%
