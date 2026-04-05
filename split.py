import os
from pathlib import Path
import shutil
from tqdm import tqdm

res_name = "img"

BATCH_SIZE = 64

dataset_path = "./dataset_campus_1"

output_path = "./dataset_campus_2"

cloud_path = os.path.join(dataset_path,res_name)

cloud_list = os.listdir(cloud_path)[0::3]

cloud_list = sorted(cloud_list,key=lambda x: int(Path(x).stem.split('_')[1]))

if not os.path.exists(output_path):
    os.mkdir(output_path)
    
if not os.path.exists(f"{output_path}/{res_name}"):
    os.mkdir(f"{output_path}/{res_name}")
    
cloud_batches = [cloud_list[i:i+BATCH_SIZE] for i in range(0,len(cloud_list),BATCH_SIZE)]

print("Batches: ",len(cloud_batches))

target_cloud_dir = os.path.join(output_path,res_name)

for i,batch in tqdm(enumerate(cloud_batches),position=0):
    batch_dir = f"batch_{i}"
    batch_dir_path = os.path.join(target_cloud_dir,batch_dir)
    if not os.path.exists(batch_dir_path):
        os.mkdir(batch_dir_path)
    for name in batch:
        in_file_path = os.path.join(cloud_path,name)
        out_file_path = os.path.join(batch_dir_path,name)
        shutil.copy(in_file_path,out_file_path)
        