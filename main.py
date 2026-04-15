import torch
from fastkan import FastKANLayer
from mamba2 import Mamba2Simple
from mamba_ssm import Mamba2
from concurrent.futures import ProcessPoolExecutor

from geomloss import SamplesLoss

from timeit import default_timer as timer
from pathlib import Path

import numpy as np
import os
import cv2

import open3d as o3d
from chamferdist import ChamferDistance

from tqdm import tqdm

device = torch.device("cuda")

class DatasetMemorizerSignleCloud:
    def __init__(self,image_dir:str,batch_size:int=1,reduction_factor:int=1):
        super().__init__()

        self.image_dir = os.path.join(image_dir,"img")

        folders = os.listdir(self.image_dir)

        files = []
        for folder in folders:
            subfiles = os.listdir(os.path.join(self.image_dir, folder))
            files.extend([os.path.join(folder, f) for f in subfiles])

        np.random.shuffle(files)
        
        # Split files into batches
        self._batch_size = batch_size
        self.imgs = files
        self.imgs = self.imgs[0::reduction_factor]

        self.batch_cache = {}

        pcd = o3d.io.read_point_cloud(f"{image_dir}/point_cloud_4.ply")

        self.cloud = np.array(pcd.points,dtype=np.float32)
        
        self.batches_count = int(( len(self.imgs) )/batch_size)
        
        batched_clouds = [self.cloud]*batch_size
        batched_clouds = np.array(batched_clouds)

        self.clouds = torch.tensor(batched_clouds,dtype=torch.float32)

    def batch_size(self):
        return self._batch_size
    
    def __len__(self):
        return self.batches_count

    def __getitem__(self,idx:int):
        
        if idx >= self.batches_count:
            raise IndexError("Index out of range")

        if idx in self.batch_cache.keys():
          return self.batch_cache[idx]

        file_batch = self.imgs[idx*self._batch_size:(idx+1)*self._batch_size]

        images = []

        # images = DatasetMemorizer.load_image((self.image_dir,file_batch,))
        for file in file_batch:
            image = np.load(f"{self.image_dir}/{file}").reshape((-1,8*8))
            images.append(image)
            
        _images = np.array(images)/255.0

        np.random.shuffle(_images)

        image_batch = torch.tensor(_images,dtype=torch.float32)

        self.batch_cache[idx] = (image_batch,self.clouds)

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
    def __init__(self,input_dim: int,output_dim: int,N:int=1,loops:int=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mambas = []
        for i in range(N):
          mamba = Mamba2Simple(
                    d_model=input_dim,
                    d_state=64,
                    d_conv=4,
                    expand=8,
                    headdim=input_dim,
                    num_grids=16,
                    grid_min=-2,
                    grid_max=2,
                    use_mem_eff_path=False
                ).to(device=device)
          self.mambas.append(mamba)

        self.linear = FastKANLayer(input_dim,output_dim,num_grids=16,grid_min=-2.0,grid_max=2.0).to(device=device)
        # self.linear = torch.nn.Linear(input_dim,output_dim).to(device=device)

        self._layers = torch.nn.ModuleList([*self.mambas,self.linear])
        self.loops = loops

    def forward(self,x):
        orig_shape = x.shape[1]
        for i in range(self.loops):
          y = x
          for mamba in self.mambas:
            y = mamba.forward(y)
          x = torch.cat([x,y],dim=1)

        x = x[:,-orig_shape:,:]
        x = self.linear.forward(x)
        return x




class MapMemorizerEncoder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._layers = torch.nn.ModuleList()

        layers:list[MambaBlock] = []
        # Encoder
        layers.append(MambaBlock(64,16,1,1))
        layers.append(MambaBlock(16,8,1,1))
        layers.append(MambaBlock(8,3,1,1))
        # Decoder
        layers.append(MambaBlock(3,8,1,1))
        layers.append(MambaBlock(8,16,1,1))
        layers.append(MambaBlock(16,64,1,1))
        
        self._layers.extend(layers)

    def forward(self,x):
        for layer in self._layers:
            x = layer.forward(x)
        return x
    
    def fit(self,epoches:int,dataset:DatasetMemorizer,checkpoint_path:str):

        self.train(True)

        optimizer = torch.optim.AdamW(self.parameters(),lr=0.001)

        # loss_fn = ChamferDistance()
        # loss_fn = SamplesLoss(loss="sinkhorn", p=1)
        loss_fn = torch.nn.MSELoss()

        best_error = 10**9

        mean_error = 0
        last_error = 0
        first = True
        
        embedding = np.random.random(256*3).astype(np.float32)
        embedding = embedding.reshape((256,3))
        
        # y = torch.tensor([embedding]*dataset.batch_size(),dtype=torch.float32).to(device=device)
        
        # y = y.reshape((dataset.batch_size(),256,3))

        for i in range(epoches):

            mean_error = 0

            for x,_ in tqdm(dataset):

                optimizer.zero_grad()

                _x = x.to(device=device)
                # _y = y.to(device=device)

                output = self.forward(_x)

                loss = loss_fn(_x,output)

                loss.backward()

                mean_error += loss.item()

                optimizer.step()

                _x = _x.to("cpu")
                # _y = _y.to("cpu")

                # Free GPU memory
                del _x

            mean_error /= dataset.batch_size()

            if mean_error - last_error > 0.5 and not first:
              print("Error increased, stopping training")
              return

            first = False

            last_error = mean_error

            if mean_error < best_error:
              torch.save(self.state_dict(),checkpoint_path)
              best_error = mean_error

            # if mean_error < 0.1:
            #   return

            print(f"Epoch: {i+1} loss: {mean_error}")

        self.train(False)



class MapMemorizerDropout(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._layers = torch.nn.ModuleList()

        layers:list[MambaBlock] = []

        layers.append(MambaBlock(64,32,1,1))
        layers.append(MambaBlock(32,16,1,1))
        layers.append(MambaBlock(16,8,1,1))
        layers.append(MambaBlock(8,3,1,1))
        
        self._layers.extend(layers)

    def forward(self,x):
        for layer in self._layers:
            x = layer.forward(x)
        return x
    
    def emd_loss(self,p1, p2):
        # EMD for 1D distributions is the L1 distance between their CDFs
        cdf1 = torch.cumsum(p1, dim=-1)
        cdf2 = torch.cumsum(p2, dim=-1)
        return torch.mean(torch.abs(cdf1 - cdf2))
    
    def sinkhorn_loss(self,x, y, epsilon=0.05, n_iter=100):
        """
        Computes the Sinkhorn distance between two sets of points.
        x: (batch_size, n, d) - Source point cloud
        y: (batch_size, m, d) - Target point cloud
        epsilon: Regularization strength
        n_iter: Number of Sinkhorn iterations
        """
        batch_size, n, d = x.shape
        m = y.shape[1]

        # Compute the cost matrix (squared Euclidean distance)
        # C_ij = ||x_i - y_j||^2
        C = torch.cdist(x, y, p=2)**2 

        # Gibbs kernel
        K = torch.exp(-C / epsilon)
        
        # Initialize scaling factors
        u = torch.ones((batch_size, n), device=x.device) / n
        v = torch.ones((batch_size, m), device=x.device) / m

        for _ in range(n_iter):
            # Update u and v
            u = 1.0 / (torch.matmul(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)
            v = 1.0 / (torch.matmul(K.transpose(-1, -2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)

        # Compute the optimal transport plan and the loss
        # T = diag(u) * K * diag(v)
        # Loss = Trace(T^T * C)
        T = u.unsqueeze(-1) * K * v.unsqueeze(-2)
        loss = torch.sum(T * C, dim=(1, 2))
        
        return loss.mean()
    
    def uniformity_loss(p, k=5):
        """
        p: (batch_size, n, 3) - predicted point cloud
        k: number of neighbors to check
        """
        # Compute distances to k-nearest neighbors within the same cloud
        dist = torch.cdist(p, p)
        dist, _ = torch.topk(dist, k=k+1, dim=-1, largest=False)
        
        # Ignore the first neighbor (which is the point itself, dist=0)
        nn_dist = dist[:, :, 1:] 
        
        # We want these distances to be somewhat uniform/large
        # This acts as a 'repulsion' force so points don't clump
        loss = torch.mean(1.0 / (nn_dist + 1e-6)) 
        return loss
    
    def density_aware_chamfer_distance(self,pred, target, alpha=1000, n_lambda=1.0):
        """
        DCD Loss Implementation
        pred: (B, N, 3) - Predicted point cloud
        target: (B, M, 3) - Ground truth point cloud
        alpha: Density factor (higher = stricter uniformity)
        n_lambda: Balance factor between completeness and accuracy
        """
        batch_size = pred.shape[0]
        
        # 1. Compute Pairwise Squared Distance Matrix (B, N, M)
        # dist[b, i, j] = ||pred[i] - target[j]||^2
        dist_matrix = torch.cdist(pred, target, p=2)**2
        
        # 2. Find nearest neighbors (Standard Chamfer components)
        # d1: for each pred, distance to nearest target
        # d2: for each target, distance to nearest pred
        d1, idx1 = torch.min(dist_matrix, dim=2) # (B, N)
        d2, idx2 = torch.max(torch.min(dist_matrix, dim=1)[0], dim=1) # (B,) - simplified version
        
        # Better d2 calculation for density:
        dist_to_pred, _ = torch.min(dist_matrix, dim=1) # (B, M)

        # 3. Compute Density Weights
        # For each target point, how many pred points 'claim' it as their nearest?
        # We use an exponential kernel to softly count this density.
        # exp(-alpha * dist) will be ~1 if dist is 0, and ~0 if dist is large.
        weight_matrix = torch.exp(-alpha * dist_matrix) # (B, N, M)
        
        # Sum weights along the prediction axis to see how "crowded" each target point is
        # density_per_target[b, j] = sum of weights from all pred points for target j
        density_per_target = torch.sum(weight_matrix, dim=1) # (B, M)
        
        # 4. Final Loss Calculation
        # We want to minimize the distance AND maximize the "uniqueness" of the match
        # Term 1: Completeness (Target -> Pred) with density penalty
        loss_target_to_pred = torch.mean(dist_to_pred / (density_per_target + 1e-6), dim=1)
        
        # Term 2: Accuracy (Pred -> Target)
        loss_pred_to_target = torch.mean(d1, dim=1)
        
        return (loss_pred_to_target + n_lambda * loss_target_to_pred).mean()

    def fit(self,epoches:int,dataset:DatasetMemorizer,checkpoint_path:str):

        self.train(True)

        optimizer = torch.optim.AdamW(self.parameters(),lr=0.001)

        # loss_fn = ChamferDistance()
        # loss_fn = SamplesLoss(loss="sinkhorn", p=1)
        # loss_fn = torch.nn.MSELoss()

        best_error = 10**9

        mean_error = 0
        last_error = 0
        first = True
        
        # embedding = np.random.random(256*3).astype(np.float32)
        # embedding = embedding.reshape((256,3))
        
        # y = torch.tensor([embedding]*dataset.batch_size(),dtype=torch.float32).to(device=device)
        
        # y = y.reshape((dataset.batch_size(),256,3))

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

                    output = torch.cat([output,self.forward(_x)],dim=1)

                # print(output)
                output = output[:,-y.shape[1]:,:]
                # output = output.reshape((-1,2352))

                # loss = loss_fn(_y,output)
                loss = 0
                
                # print(output.shape)
                # print(_y.shape)
                # loss = self.emd_loss(_y,output)
                loss = self.density_aware_chamfer_distance(_y,output)

                loss.backward()

                mean_error += loss.item()

                optimizer.step()

                _x = _x.to("cpu")
                _y = _y.to("cpu")

                # Free GPU memory
                del _x
                del _y

            mean_error /= dataset.batch_size()

            if mean_error - last_error > 0.5 and not first:
              print("Error increased, stopping training")
              return

            first = False

            last_error = mean_error

            if mean_error < best_error:
              torch.save(self.state_dict(),checkpoint_path)
              best_error = mean_error

            # if mean_error < 0.1:
            #   return

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

    from kitti_to_3d_pointmap import Position
    
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


        
        

def main():    
    torch.cuda.empty_cache()

    # print("Processes count: ",os.cpu_count())

    # generate_dataset_simple('./dataset_campus_1',"/home/projectrobal/data/vbr_slam/campus/campus_train0/campus_train0_gt.txt",
    #                  "/home/projectrobal/data/campus_train0/camera_left/data")
    # # # test split points
    # exit()
    
    # pcd = o3d.geometry.PointCloud()
    
    # pcd.points = o3d.utility.Vector3dVector(np.random.random((256,3)).astype(np.float32))
    
    # o3d.io.write_point_cloud("map_id_1.ply",pcd)
    
    # exit()
    dataset = DatasetMemorizerSignleCloud("./dataset7",batch_size=2,reduction_factor=16)

    print("Preloading dataset start")

    # # pre load dataset
    for batch in tqdm(dataset):
      pass
  
    print("Preloading dataset end")
    # exit()
  
    x,y = dataset[0]
        
    # print(f"Max: {np.max(x)}, Min: {np.min(x)}")
    
    x = x.to(device=device)
    
    # exit()
      
    net = MapMemorizerDropout()
    # net = MapMemorizerEncoder()
    # net = MapMemorizer(1)
    
    # net = net.to(device=device)
    
    net.fit(1000,dataset,"checkpoint_encoder_1.pth")
    
    exit()
    
    # net.load_state_dict(torch.load('checkpoint_1.pth',weights_only=True))
    
    with torch.no_grad():
        
        pcd = o3d.geometry.PointCloud()
      
        output = net.forward(x)
        
        output = output.cpu().detach().numpy()[0]
        # output = output[-y.shape[1]:,:]
        print("Output shape: ",output.shape)
                
        y = y.numpy()
        
        y = y[0]
        
        pcd.points = o3d.utility.Vector3dVector(y)
        # pcd = pcd.remove_duplicated_points()
        # pcd = pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)[0]
        # print("Chamfer distance: ",ChamferDistance()(torch.tensor(y).unsqueeze(0),torch.tensor(output).unsqueeze(0)).item())
        
        o3d.visualization.draw_geometries([pcd])
        

if __name__ == "__main__":
    main()