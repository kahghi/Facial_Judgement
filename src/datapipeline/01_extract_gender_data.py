#%% import files
import tarfile
import os
import pandas as pd
from PIL import Image
import numpy as np
import ast

#%% loop through tar.gz
targz_files=["part1", "part2", "part3"]
for file in targz_files:
    tar = tarfile.open(file + ".tar.gz")
    tar.extractall()
    tar.close()

#%% save into csv - filename, age, gender, race, datetime
base_dir = "C:/Users/KahGhi/Downloads/" 
filename = []
details = []
img_tensor = []
img_format = []
img_size = []
img_mode = []

for folder in targz_files:
    for root, dirs, files in os.walk(base_dir + folder):
        filename.append(files)
        for name in files:
            details.append(name.rstrip('.jpg').split('_'))
            if name == '.DS_Store':
                img_tensor.append([])
                img_format.append([])
                img_size.append([])
                img_mode.append([])
            else:
                img = Image.open(base_dir + folder + "/" + name)
                img_tensor.append(np.array(img))
                img_format.append(img.format)
                img_size.append(img.size)
                img_mode.append(img.mode)

detail_df = pd.DataFrame.from_records(details)
detail_df.columns = ['age', 'gender', 'race', 'datetime']
file_name= np.hstack(filename).tolist()

data = {'file_name': file_name,'img_tensor':img_tensor,'img_format':img_format,
       'img_size':img_size,'img_mode':img_mode}
data_df = pd.DataFrame.from_dict(data)

final_df = pd.concat([data_df, detail_df], axis=1)
df = final_df[~final_df.file_name.str.contains(".DS_Store")]
df.to_csv(base_dir + 'utkface.csv')

#%% read in csv file
df = pd.read_csv(base_dir +  'utkface.csv')
df.head()

def from_np_array(array_string):
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))

df2 = pd.read_csv(base_dir +  'utkface.csv', converters={'img_tensor': from_np_array})