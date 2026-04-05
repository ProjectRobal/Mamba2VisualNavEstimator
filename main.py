import torch
from fastkan import FastKANLayer
from LegendreKANLayer import LegendreKANLayer
from mamba2 import Mamba2Simple
from mamba_ssm import Mamba2
from concurrent.futures import ProcessPoolExecutor

from timeit import default_timer as timer
from pathlib import Path

import numpy as np
import os
import cv2

import open3d as o3d
from chamferdist import ChamferDistance
from torchkan.KALnet import KAL_Net as KAN
from kitti_to_3d_pointmap import Position

from tqdm import tqdm

device = torch.device("cuda")

class DatasetMemorizerSignleCloud:
    def __init__(self,image_dir:str,batch_size:int=1):
        super().__init__()

        self.image_dir = os.path.join(image_dir,"img")

        files = os.listdir(self.image_dir)[:20]

        files = sorted(files)

        self.batches_count = int(( len(files)*64 )/batch_size)

        # Split files into batches
        self._batch_size = batch_size
        self.img_batches = files

        self.batch_cache = {}

        pcd = o3d.io.read_point_cloud(f"{image_dir}/point_cloud_4.ply")

        self.cloud = np.array(pcd.points,dtype=np.float32)

        # load map with all points to memorize
        #
        # points format: x,y,z, point_class

    def batch_size(self):
        return self._batch_size

    @staticmethod
    def load_image(args):
        image_dir,dir = args
        images = []
        files = os.listdir(f"{image_dir}/{dir}")
        files = sorted(files,key=lambda x: int(Path(x).stem.split('_')[1]))
        for f in files:
          image = np.load(f"{image_dir}/{dir}/{f}").reshape((-1,8*8))
          images.append(image)
        return images

    def __len__(self):
        return self.batches_count

    def __getitem__(self,idx:int):

        if idx in self.batch_cache.keys():
          return self.batch_cache[idx]

        file_idx = int( ( idx*self._batch_size ) / 64)

        file_batch = self.img_batches[file_idx]

        images = []


        images = DatasetMemorizer.load_image((self.image_dir,file_batch,))

        to_split = int(len(images)/self._batch_size)

        for i in range(to_split):

            start = i*self._batch_size
            end = start + self._batch_size

            _images = images[start:end]

            _images = np.array(_images)/255.0

            clouds = [self.cloud]*len(_images)
            clouds = np.array(clouds)

            np.random.shuffle(_images)


            image_batch = torch.tensor(_images,dtype=torch.float32)
            cloud_batch = torch.tensor(clouds,dtype=torch.float32)

            self.batch_cache[idx+i] = (image_batch,cloud_batch)

        return self.batch_cache[idx]

'''

Idea is to split image into chunks and then
pass them to Mamba Model, and then it will
generate more tokens which will represents
estimated map.

Input image will have size of 224 x 224.

'''


class DatasetMemorizer:
    def __init__(self,image_dir:str,batch_size:int=1):
        super().__init__()

        self.image_dir = os.path.join(image_dir,"img")
        self.cloud_dir = os.path.join(image_dir,"clouds")

        files = os.listdir(self.image_dir)

        files = sorted(files)[:1]

        self.batches_count = int(( len(files)*64 )/batch_size)

        # Split files into batches
        self._batch_size = batch_size
        self.img_batches = files
        # self.img_batches = [
        #     files[i:i+batch_size] for i in range(0,len(files),batch_size)
        # ]

        points_files = os.listdir(self.cloud_dir)
        points_files = sorted(points_files)[:1]

        # points_files = points_files[0::3]
        self.cloud_batches = points_files
        # self.cloud_batches = [
        #     points_files[i:i+batch_size] for i in range(0,len(points_files),batch_size)
        # ]

        self.batch_cache = {}

        # load map with all points to memorize
        #
        # points format: x,y,z, point_class

    @staticmethod
    def padd(x):
        x = x.flatten()
        numbers_count = int(x.shape[0])

        numbers_count_missing = numbers_count % 1024

        # append dummy values to make it divisible by 1024
        if numbers_count_missing != 0:
            padding_count = 1024 - numbers_count_missing

            padding = np.zeros(padding_count,dtype=np.float32)

            x = np.concatenate([x,padding],axis=0)

        return x

    def batch_size(self):
        return self._batch_size

    @staticmethod
    def load_image(args):
        image_dir,dir = args
        images = []
        files = os.listdir(f"{image_dir}/{dir}")
        files = sorted(files,key=lambda x: int(Path(x).stem.split('_')[1]))
        for f in files:
          image = np.load(f"{image_dir}/{dir}/{f}").reshape((-1,8*8))
          images.append(image)
        return images

    @staticmethod
    def load_cloud(args):
        cloud_dir,dir = args
        clouds = []
        files = os.listdir(f"{cloud_dir}/{dir}")
        files = sorted(files,key=lambda x: int(Path(x).stem.split('_')[1]))
        for f in files:
          cloud = np.load(f"{cloud_dir}/{dir}/{f}")
          cloud = cloud.reshape((-1,4))[:,:3]
          clouds.append(cloud)
        return clouds

    def __len__(self):
        return self.batches_count

    def __getitem__(self,idx:int):

        if idx in self.batch_cache.keys():
          return self.batch_cache[idx]

        file_idx = int( ( idx*self._batch_size ) / 64)

        file_batch = self.img_batches[file_idx]
        cloud_batch = self.cloud_batches[file_idx]

        images = []
        clouds = []

        images = DatasetMemorizer.load_image((self.image_dir,file_batch,))
        clouds = DatasetMemorizer.load_cloud((self.cloud_dir,cloud_batch,))
        # for file in file_batch:
        #     image = np.load(f"{self.image_dir}/{file}").reshape((-1,8*8))
        #     images.append(image)

        # for file in cloud_batch:
        #     cloud = np.load(f"{self.cloud_dir}/{file}")
        #     cloud = self.padd(cloud).reshape((-1,1024))
        #     clouds.append(cloud)

        max_cloud_dim=max(clouds,key=lambda x: x.shape[0]).shape[0]

        def pad_cloud(x):
          n_x = np.zeros((int(max_cloud_dim),3),dtype=np.float32)

          n_x[:x.shape[0],:] = x

          return n_x

        cloud_orig_len = [cloud.shape[0] for cloud in clouds]

        clouds = [pad_cloud(cloud) for cloud in clouds]

        to_split = int(len(clouds)/self._batch_size)

        for i in range(to_split):

            start = i*self._batch_size
            end = start + self._batch_size

            _images = images[start:end]
            _clouds = clouds[start:end]

            _images = np.array(_images)
            _clouds = np.array(_clouds)

            image_batch = torch.tensor(_images,dtype=torch.float32)
            cloud_batch = torch.tensor(_clouds,dtype=torch.float32)

            self.batch_cache[idx+i] = (image_batch,cloud_batch,cloud_orig_len[start:end])

        return self.batch_cache[idx]
    
class MambaBlock(torch.nn.Module):
    def __init__(self,input_dim: int,output_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.mamba = Mamba2Simple(
                  d_model=input_dim,
                  d_state=64,
                  d_conv=4,
                  expand=2,
                  headdim=2*input_dim,
                  num_grids=8,
                  grid_min=-2,
                  grid_max=2,
                  use_mem_eff_path=False
              )
        
        self.linear = FastKANLayer(input_dim,output_dim,num_grids=8,grid_min=-2.0,grid_max=2.0)
        
        self._layers = torch.nn.ModuleList([self.mamba,self.linear])
        
    def forward(self,x):
        x = self.mamba.forward(x)
        x = self.linear.forward(x)
        return x


class MapMemorizerDropout(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        layers:list[MambaBlock] = []

        layers.append(MambaBlock(64,32))
        layers.append(MambaBlock(32,16))
        layers.append(MambaBlock(16,8))
        layers.append(MambaBlock(8,3))

        self._layers = torch.nn.ModuleList([*layers])
        
    def forward(self,x):
        for layer in self._layers[:-1]:
            x = layer.forward(x)
        return self._layers[-1].forward(x)
    
    def fit(self,epoches:int,dataset:DatasetMemorizer,checkpoint_path:str):

        self.train(True)

        optimizer = torch.optim.AdamW(self.parameters(),lr=0.001)

        loss_fn = ChamferDistance()

        best_error = 10**9

        mean_error = 0
        last_error = 0
        first = True

        for i in range(epoches):

            mean_error = 0

            for x,y in tqdm(dataset):

                optimizer.zero_grad()

                _x = x.to(device=device)
                _y = y.to(device=device)

                output = self.forward(_x)

                while output.shape[1] < _y.shape[1]:
                # for i in range(2):
                    _x = torch.cat([_x,output],dim=1)

                    # output = self._forward(_x)

                    output = torch.cat([output,self.forward(_x)],dim=1)

                    # output = torch.cat([output,_out],dim=1)
                                
                # print(output)

                # loss = loss_fn(_y,output)
                loss = 0

                loss = loss_fn(_y,output)

                loss.backward()

                mean_error += loss.item()

                optimizer.step()

                _x = _x.to("cpu")
                _y = _y.to("cpu")

                # Free GPU memory
                del _x
                del _y

            mean_error /= dataset.batch_size()

            if mean_error - last_error > 0.0 and not first:
              print("Error increased, stopping training")
              return

            first = False

            last_error = mean_error

            if mean_error < best_error:
              torch.save(self.state_dict(),checkpoint_path)
              best_error = mean_error

            if mean_error < 0.1:
              return

            print(f"Epoch: {i+1} loss: {mean_error}")

        self.train(False)
    

def convert_point_cloud_into_numpy_points_set(cloud:o3d.geometry.PointCloud):
    """
    Docstring for convert_point_cloud_into_numpy_points_set

    """

    points_set = []

    for point in cloud.points:
        points_set.append(
            (point[0],point[1],point[2],1.0)
        )

    points_set.append((0.0,0.0,0.0,255.0))

    points_set = np.array(points_set,dtype=np.float32)

    return points_set

def read_and_parse_images(image_dir:str,target_dir:str):

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    for i,file in enumerate(os.listdir(image_dir)):
        img = cv2.imread(f"{image_dir}/{file}")

        img_resize = cv2.resize(img,(224,224))
        image_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)

        image = image_rgb.reshape((-1,8,8))

        np.save(f"{target_dir}/img_{i}.npy",image)


def convert_point_cloud_into_numpy_points_set(cloud:o3d.geometry.PointCloud):
    """
    Docstring for convert_point_cloud_into_numpy_points_set

    """

    points_set = []

    for point in cloud.points:
        points_set.append(
            (point[0],point[1],point[2],1.0)
        )

    points_set.append((0.0,0.0,0.0,255.0))

    points_set = np.array(points_set,dtype=np.float32)
    print(f"Cloud amount: {points_set.shape[0]}")

    return points_set

def convert_point_cloud_into_numpy_points_set_simple(cloud:o3d.geometry.PointCloud):
    """
    Docstring for convert_point_cloud_into_numpy_points_set

    """

    points_set = []

    for point in cloud.points:
        points_set.append(
            (point[0],point[1],point[2])
        )

    points_set.append((0.0,0.0,0.0))

    points_set = np.array(points_set,dtype=np.float32)
    print(f"Cloud amount: {points_set.shape[0]}")

    return points_set

def read_and_parse_images(image_dir:str,target_dir:str):

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    for i,file in enumerate(os.listdir(image_dir)):
        img = cv2.imread(f"{image_dir}/{file}")

        img_resize = cv2.resize(img,(224,224))
        image_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)

        image = image_rgb.reshape((-1,8,8))

        np.save(f"{target_dir}/img_{i}.npy",image)

def generate_dataset(dataset_path,timestamp_file_path,image_dir):
    
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
        
    if not os.path.exists(f"{dataset_path}/clouds"):
        os.mkdir(f"{dataset_path}/clouds")
        
    if not os.path.exists(f"{dataset_path}/img"):
        os.mkdir(f"{dataset_path}/img")
        
    read_and_parse_images(image_dir,f"{dataset_path}/img")
    
    pcd = o3d.io.read_point_cloud('point_cloud_3.ply')
    
    view_limits = o3d.geometry.AxisAlignedBoundingBox()
    view_limits.max_bound = np.ones(3)*10.0
    view_limits.min_bound = np.ones(3)*-10.0
    
    positions = []
    
    with open(timestamp_file_path) as file:
        lines = file.readlines()
        
        for line in lines:
            if line[0] == '#':
                continue
            pos = Position()
            pos.read_from_line(line)
            positions.append(pos)

    for i,pos in enumerate(positions):
        
        _pos = pos._position
        
        _view_limits = o3d.geometry.AxisAlignedBoundingBox()
        _view_limits.max_bound = view_limits.max_bound + _pos  
        _view_limits.min_bound = view_limits.min_bound + _pos  
        
        _cloud = pcd.crop(_view_limits)
        
        _point_cloud = convert_point_cloud_into_numpy_points_set(_cloud)
        
        np.save(f"{dataset_path}/clouds/cloud_{i}.npy",_point_cloud)
        
def generate_dataset_simple(dataset_path,timestamp_file_path,image_dir):
    
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
        
    if not os.path.exists(f"{dataset_path}/clouds"):
        os.mkdir(f"{dataset_path}/clouds")
        
    if not os.path.exists(f"{dataset_path}/img"):
        os.mkdir(f"{dataset_path}/img")
        
    read_and_parse_images(image_dir,f"{dataset_path}/img")
    
    pcd = o3d.io.read_point_cloud('point_cloud_3.ply')
    
    view_limits = o3d.geometry.AxisAlignedBoundingBox()
    view_limits.max_bound = np.ones(3)*10.0
    view_limits.min_bound = np.ones(3)*-10.0
    
    positions = []
    
    with open(timestamp_file_path) as file:
        lines = file.readlines()
        
        for line in lines:
            if line[0] == '#':
                continue
            pos = Position()
            pos.read_from_line(line)
            positions.append(pos)

    for i,pos in enumerate(positions):
        
        _pos = pos._position
        
        _view_limits = o3d.geometry.AxisAlignedBoundingBox()
        _view_limits.max_bound = view_limits.max_bound + _pos  
        _view_limits.min_bound = view_limits.min_bound + _pos  
        
        _cloud = pcd.crop(_view_limits)
        
        _point_cloud = convert_point_cloud_into_numpy_points_set_simple(_cloud)
        
        np.save(f"{dataset_path}/clouds/cloud_{i}.npy",_point_cloud)
        
        
def main1():
    net = MapMemorizer(1)
    
    x = torch.randn((4,2500,64))
    y = torch.randn((4,2500,3))
    
    loss_fn = ChamferDistance()
    
    optimizer = torch.optim.AdamW(
        net.parameters(),lr=0.001
    )
    
    x = x.to(device=device)
    y = y.to(device=device)
    net = net.to(device=device)
    
    for i in range(1000):
        
        optimizer.zero_grad()
                
        output = net.forward(x)
        
        loss = loss_fn(y,output)
        
        loss.backward()
        
        optimizer.step()
        
        print(f"Error {i+1}: {loss.item()/x.shape[0]}")
        

def main():    
    torch.cuda.empty_cache()

    # print("Processes count: ",os.cpu_count())

    # generate_dataset_simple('./dataset_campus_1',"/home/projectrobal/data/vbr_slam/campus/campus_train0/campus_train0_gt.txt",
    #                  "/home/projectrobal/data/campus_train0/camera_left/data")
    # # # test split points
    # exit()
    dataset = DatasetMemorizerSignleCloud("./dataset7",batch_size=2)

    print("Preloading dataset start")

    # # pre load dataset
    for batch in tqdm(dataset):
      pass
  
    x,y = dataset[0]
        
    # print(f"Max: {np.max(x)}, Min: {np.min(x)}")
    
    x = x.to(device=device)
    
    # exit()
      
    net = MapMemorizerDropout()
    # net = MapMemorizer(1)
    
    net = net.to(device=device)
    
    net.fit(1000,dataset,"checkpoint_1.pth")
    
    exit()
    
    net.load_state_dict(torch.load('checkpoint.pt',weights_only=True))
    
    with torch.no_grad():
        
        pcd = o3d.geometry.PointCloud()
      
        output = net.forward(x)
        
        output = output.cpu().detach().numpy()
        
        output = output[0]
        
        y = y.numpy()
        
        y = y[0]
        
        pcd.points = o3d.utility.Vector3dVector(output)
        
        o3d.visualization.draw_geometries([pcd])
        

if __name__ == "__main__":
    main()