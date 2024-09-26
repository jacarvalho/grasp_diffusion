import gc
import glob
import copy
import multiprocessing
import pdb
import time

import numpy as np
import psutil
import trimesh

from scipy.stats import special_ortho_group

import os
import torch

from torch.utils.data import DataLoader, Dataset
import json
import pickle
import h5py
from se3dif.utils import get_data_src

from se3dif.utils import to_numpy, to_torch, get_grasps_src
from mesh_to_sdf.surface_point_cloud import get_scan_view, get_hq_scan_view
from mesh_to_sdf.scan import ScanPointcloud



import os, sys

import logging
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)


# class AcronymGrasps():
#     def __init__(self, filename):

#         scale = None
#         if filename.endswith(".json"):
#             data = json.load(open(filename, "r"))
#             self.mesh_fname = data["object"].decode('utf-8')
#             self.mesh_type = self.mesh_fname.split('/')[1]
#             self.mesh_id = self.mesh_fname.split('/')[-1].split('.')[0]
#             self.mesh_scale = data["object_scale"] if scale is None else scale
#         elif filename.endswith(".h5"):
#             data = h5py.File(filename, "r")
#             self.mesh_fname = data["object/file"][()].decode('utf-8')
#             self.mesh_type = self.mesh_fname.split('/')[1]
#             self.mesh_id = self.mesh_fname.split('/')[-1].split('.')[0]
#             self.mesh_scale = data["object/scale"][()] if scale is None else scale
#         else:
#             raise RuntimeError("Unknown file ending:", filename)

#         self.grasps, self.success = self.load_grasps(filename)
#         good_idxs = np.argwhere(self.success==1)[:,0]
#         bad_idxs  = np.argwhere(self.success==0)[:,0]
#         self.good_grasps = self.grasps[good_idxs,...]
#         self.bad_grasps  = self.grasps[bad_idxs,...]

#     def load_grasps(self, filename):
#         """Load transformations and qualities of grasps from a JSON file from the dataset.

#         Args:
#             filename (str): HDF5 or JSON file name.

#         Returns:
#             np.ndarray: Homogenous matrices describing the grasp poses. 2000 x 4 x 4.
#             np.ndarray: List of binary values indicating grasp success in simulation.
#         """
#         if filename.endswith(".json"):
#             data = json.load(open(filename, "r"))
#             T = np.array(data["transforms"])
#             success = np.array(data["quality_flex_object_in_gripper"])
#         elif filename.endswith(".h5"):
#             data = h5py.File(filename, "r")
#             T = np.array(data["grasps/transforms"])
#             success = np.array(data["grasps/qualities/flex/object_in_gripper"])
#         else:
#             raise RuntimeError("Unknown file ending:", filename)
#         return T, success

#     def load_mesh(self):
#         mesh_path_file = os.path.join(get_data_src(), self.mesh_fname)

#         mesh = trimesh.load(mesh_path_file,  file_type='obj', force='mesh')

#         mesh.apply_scale(self.mesh_scale)
#         if type(mesh) == trimesh.scene.scene.Scene:
#             mesh = trimesh.util.concatenate(mesh.dump())
#         return mesh

class AcronymGrasps():
    def __init__(self, filename):
        self.filename = filename  # Store filename for later use
        self.mesh_fname = None
        self.mesh_type = None
        self.mesh_id = None
        self.mesh_scale = None
        self.grasps = None
        self.success = None
        self.good_grasps = None
        self.bad_grasps = None

    def load_data(self):
        # process = psutil.Process(os.getpid())
        # print(f"Memory used before loading data: {process.memory_info().rss / 1024 ** 2:.2f} MB")
        scale = None
        filename = self.filename

        if filename.endswith(".json"):
            data = json.load(open(filename, "r"))
            self.mesh_fname = data["object"].decode('utf-8')
            self.mesh_type = self.mesh_fname.split('/')[1]
            self.mesh_id = self.mesh_fname.split('/')[-1].split('.')[0]
            self.mesh_scale = data["object_scale"] if scale is None else scale
        elif filename.endswith(".h5"):
            with h5py.File(filename, "r") as data:
                # self.mesh_fname = data["object/file"][()].decode('utf-8')
                self.mesh_fname = copy.deepcopy(data["object/file"][()]).decode('utf-8')
                self.mesh_type = self.mesh_fname.split('/')[1]
                self.mesh_id = self.mesh_fname.split('/')[-1].split('.')[0]
                # self.mesh_scale = data["object/scale"][()] if scale is None else scale
                self.mesh_scale = data["object/scale"][()].copy() if scale is None else scale
            # data = h5py.File(filename, "r")
            # self.mesh_fname = data["object/file"][()].decode('utf-8')
            # self.mesh_type = self.mesh_fname.split('/')[1]
            # self.mesh_id = self.mesh_fname.split('/')[-1].split('.')[0]
            # self.mesh_scale = data["object/scale"][()] if scale is None else scale
        else:
            raise RuntimeError("Unknown file ending:", filename)

        self.grasps, self.success = self.load_grasps(filename)
        good_idxs = np.argwhere(self.success == 1)[:, 0]
        bad_idxs = np.argwhere(self.success == 0)[:, 0]
        self.good_grasps = self.grasps[good_idxs, ...]
        self.bad_grasps = self.grasps[bad_idxs, ...]
        
        # print(f"Memory used after loading data: {process.memory_info().rss / 1024 ** 2:.2f} MB")
        # del data, filename

    def load_grasps(self, filename):
        """Load transformations and qualities of grasps from a JSON or HDF5 file.

        Args:
            filename (str): HDF5 or JSON file name.

        Returns:
            np.ndarray: Homogenous matrices describing the grasp poses. 2000 x 4 x 4.
            np.ndarray: List of binary values indicating grasp success in simulation.
        """
        if filename.endswith(".json"):
            data = json.load(open(filename, "r"))
            T = np.array(data["transforms"])
            success = np.array(data["quality_flex_object_in_gripper"])
        elif filename.endswith(".h5"):
            with h5py.File(filename, "r") as data:
                T = data["grasps/transforms"][:].copy()
                success = data["grasps/qualities/flex/object_in_gripper"][:].copy()
        else:
            raise RuntimeError("Unknown file ending:", filename)
        
        # del data, filename
        return T, success

    # def load_mesh(self):
    #     if self.mesh_fname is None:
    #         self.load_data()

    #     mesh_path_file = os.path.join(get_data_src(), self.mesh_fname)
    #     # mesh = trimesh.load(mesh_path_file, file_type='obj', force='mesh')
    #     # mesh.apply_scale(self.mesh_scale)
    #     # if type(mesh) == trimesh.scene.scene.Scene:
    #     #     mesh = trimesh.util.concatenate(mesh.dump())
        
    #     with trimesh.load(mesh_path_file, file_type='obj', force='mesh') as mesh:
    #         mesh.apply_scale(self.mesh_scale)
    #         if type(mesh) == trimesh.scene.scene.Scene:
    #             mesh = trimesh.util.concatenate(mesh.dump())
                 
    #     # del mesh_path_file
    #     return mesh
        
    def load_mesh(self):
        trimesh.util.attach_to_log(level='ERROR')
        trimesh.caching.CACHE_DISABLE = True
        
        if self.mesh_fname is None:
            self.load_data()

        mesh_path_file = os.path.join(get_data_src(), self.mesh_fname)
        
        try:
            mesh = trimesh.load(mesh_path_file, file_type='obj', force='mesh',process=False)
            mesh.apply_scale(self.mesh_scale)
            if isinstance(mesh, trimesh.scene.scene.Scene):
                mesh = trimesh.util.concatenate(mesh.dump())
            
            return mesh
        finally:
       
            del mesh_path_file


class AcronymGraspsDirectory():
    def __init__(self, filename=get_grasps_src(), data_type='Mug',dataset_dir=None, allowed_categories=None, split='test'):
        if dataset_dir is None and allowed_categories is None:
            grasps_files = sorted(glob.glob(filename + '/' + data_type + '/*.h5'))

            self.avail_obj = []
            for grasp_file in grasps_files:
                self.avail_obj.append(AcronymGrasps(grasp_file))
                
            # Determine the JSON file path based on opt.allowed_categories
            splits_dir = os.path.join(dataset_dir, 'splits')
            json_file_name = f"{allowed_categories}.json"
            json_file_path = os.path.join(splits_dir, json_file_name)
            
        # Using splited dataset
        else:
            # Determine the JSON file path based on opt.allowed_categories
            splits_dir = os.path.join(dataset_dir, 'splits')
            json_file_name = f"{allowed_categories}.json"
            json_file_path = os.path.join(splits_dir, json_file_name)

            # Load the JSON file content
            with open(json_file_path, 'r') as json_file:
                split_data = json.load(json_file)

            # Extract training and testing file names from the JSON content
            self.avail_obj = []
            if split == 'train':
                train_file_names = split_data.get('train', [])
                train_files = [os.path.join(dataset_dir, 'grasps', fname) for fname in train_file_names]
                train_files = [os.path.normpath(fpath) for fpath in train_files]
                self.avail_obj_train = []
                for grasp_file in train_files:
                    self.avail_obj.append(AcronymGrasps(grasp_file))
            elif split == 'test':
                test_file_names = split_data.get('test', [])
                test_files = [os.path.join(dataset_dir, 'grasps', fname) for fname in test_file_names]
                test_files = [os.path.normpath(fpath) for fpath in test_files]
                self.avail_obj_test = []
                for grasp_file in test_files:
                    self.avail_obj.append(AcronymGrasps(grasp_file))



class AcronymAndSDFDataset(Dataset):
    'DataLoader for training DeepSDF Auto-Decoder model'
    def __init__(self, class_type='Mug', se3=False, phase='train', one_object=False,
                 n_pointcloud = 1000, n_density = 200, n_coords = 1500,
                 augmented_rotation=True, visualize=False, split = True):

        self.class_type = class_type
        self.data_dir = get_data_src()
        self.acronym_data_dir = self.data_dir

        self.grasps_dir = os.path.join(self.acronym_data_dir, 'grasps')
        self.sdf_dir = os.path.join(self.acronym_data_dir, 'sdf')

        self.generated_points_dir = os.path.join(self.acronym_data_dir, 'train_data')

        grasps_files = sorted(glob.glob(self.grasps_dir+'/'+class_type+'/*.h5'))

        points_files = []
        sdf_files = []
        for grasp_file in grasps_files:
            g_obj = AcronymGrasps(grasp_file)
            mesh_file = g_obj.mesh_fname
            txt_split = mesh_file.split('/')

            sdf_file = os.path.join(self.sdf_dir, class_type, txt_split[-1].split('.')[0]+'.json')
            point_file = os.path.join(self.generated_points_dir, class_type, '4_points', txt_split[-1]+'.npz')

            sdf_files.append(sdf_file)
            points_files.append(point_file)

        ## Split Train/Validation
        n = len(grasps_files)
        indexes = np.arange(0, n)
        self.total_len = n
        if split:
            idx = int(0.9 * n)
        else:
            idx = int(n)

        if phase == 'train':
            self.grasp_files = grasps_files[:idx]
            self.points_files = points_files[:idx]
            self.sdf_files = sdf_files[:idx]
            self.indexes = indexes[:idx]
        else:
            self.grasp_files = grasps_files[idx:]
            self.points_files = points_files[idx:]
            self.sdf_files = sdf_files[idx:]
            self.indexes = indexes[idx:]


        self.len = len(self.points_files)

        self.n_pointcloud = n_pointcloud
        self.n_density  = n_density
        self.n_occ = n_coords

        ## Variables on Data
        self.one_object = one_object
        self.augmented_rotation = augmented_rotation
        self.se3 = se3

        ## Visualization
        self.visualize = visualize
        self.scale = 8.

    def __len__(self):
        return self.len

    def _get_item(self, index):
        if self.one_object:
            index = 0

        index_obj = self.indexes[index]
        ## Load Files ##
        grasps_obj = AcronymGrasps(self.grasp_files[index])
        sdf_file = self.sdf_files[index]
        with open(sdf_file, 'rb') as handle:
            sdf_dict = pickle.load(handle)

        ## PointCloud
        p_clouds = sdf_dict['pcl']
        rix = np.random.permutation(p_clouds.shape[0])
        p_clouds = p_clouds[rix[:self.n_pointcloud],:]

        ## Coordinates XYZ
        coords  = sdf_dict['xyz']
        rix = np.random.permutation(coords.shape[0])
        coords = coords[rix[:self.n_occ],:]

        ### SDF value
        sdf = sdf_dict['sdf'][rix[:self.n_occ]]
        grad_sdf = sdf_dict['grad_sdf'][rix[:self.n_occ], ...]

        ### Scale and Loc
        scale = sdf_dict['scale']
        loc = sdf_dict['loc']

        ## Grasps good/bad
        rix = np.random.randint(low=0, high=grasps_obj.good_grasps.shape[0], size=self.n_density)
        H_grasps = grasps_obj.good_grasps[rix, ...]
        rix = np.random.randint(low=0, high=grasps_obj.bad_grasps.shape[0], size=self.n_density)
        H_bad_grasps = grasps_obj.bad_grasps[rix, ...]

        ## Rescale Pointcloud and Occupancy Points ##
        coords = (coords + loc)*scale*grasps_obj.mesh_scale * self.scale
        p_clouds = (p_clouds + loc)*scale*grasps_obj.mesh_scale * self.scale

        sdf = sdf*scale*grasps_obj.mesh_scale * self.scale
        grad_sdf = -grad_sdf*scale*grasps_obj.mesh_scale * self.scale

        H_grasps[:,:-1,-1] = H_grasps[:,:-1,-1] * self.scale
        H_bad_grasps[:,:-1,-1] = H_bad_grasps[:,:-1,-1]*self.scale

        ## Random rotation ##
        if self.augmented_rotation:
            R = special_ortho_group.rvs(3)
            H = np.eye(4)
            H[:3,:3] = R

            coords = np.einsum('mn,bn->bm',R, coords)
            p_clouds = np.einsum('mn,bn->bm',R, p_clouds)

            H_grasps = np.einsum('mn,bnd->bmd', H, H_grasps)
            H_bad_grasps = np.einsum('mn,bnd->bmd', H, H_bad_grasps)

            grad_sdf = np.einsum('mn,bn->bm', R, grad_sdf)


        # Visualize
        if self.visualize:
            ## 3D matplotlib ##
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(p_clouds[:,0], p_clouds[:,1], p_clouds[:,2], c='r')

            n = 10
            x = coords[:n,:]
            ## grad sdf ##
            x_grad = grad_sdf[:n, :]
            ax.quiver(x[:,0], x[:,1], x[:,2], x_grad[:,0], x_grad[:,1], x_grad[:,2], length=0.3)

            ## sdf visualization ##
            x_sdf = sdf[:n]
            x_sdf = 0.9*x_sdf/np.max(x_sdf)
            c = np.zeros((n, 3))
            c[:, 1] = x_sdf
            ax.scatter(x[:,0], x[:,1], x[:,2], c=c)

            plt.show(block=True)

        # del sdf_dict

        res = {'point_cloud': torch.from_numpy(p_clouds).float(),
               'x_sdf': torch.from_numpy(coords).float(),
               'x_ene_pos': torch.from_numpy(H_grasps).float(),
               'x_neg_ene': torch.from_numpy(H_bad_grasps).float(),
               'scale': torch.Tensor([self.scale]).float(),
               'visual_context':  torch.Tensor([index_obj])}

        return res, {'sdf': torch.from_numpy(sdf).float(), 'grad_sdf': torch.from_numpy(grad_sdf).float()}
    

    def __getitem__(self, index):
        'Generates one sample of data'
        return self._get_item(index)


class PointcloudAcronymAndSDFDataset(Dataset):
    'DataLoader for training DeepSDF with a Rotation Invariant Encoder model'
    def __init__(self, class_type=['Cup', 'Mug', 'Fork', 'Hat', 'Bottle', 'Bowl', 'Car', 'Donut', 'Laptop', 'MousePad', 'Pencil',
                                   'Plate', 'ScrewDriver', 'WineBottle','Backpack', 'Bag', 'Banana', 'Battery', 'BeanBag', 'Bear',
                                   'Book', 'Books', 'Camera','CerealBox', 'Cookie','Hammer', 'Hanger', 'Knife', 'MilkCarton', 'Painting',
                                   'PillBottle', 'Plant','PowerSocket', 'PowerStrip', 'PS3', 'PSP', 'Ring', 'Scissors', 'Shampoo', 'Shoes',
                                   'Sheep', 'Shower', 'Sink', 'SoapBottle', 'SodaCan','Spoon', 'Statue', 'Teacup', 'Teapot', 'ToiletPaper',
                                   'ToyFigure', 'Wallet','WineGlass',
                                   'Cow', 'Sheep', 'Cat', 'Dog', 'Pizza', 'Elephant', 'Donkey', 'RubiksCube', 'Tank', 'Truck', 'USBStick'],
                 se3=False, phase='train', one_object=False,
                 n_pointcloud = 1000, n_density = 200, n_coords = 1000,
                 augmented_rotation=True, visualize=False, split = True):

        #class_type = ['Mug']
        self.class_type = class_type
        self.data_dir = get_data_src()

        self.grasps_dir = os.path.join(self.data_dir, 'grasps')

        self.grasp_files = []
        for class_type_i in class_type:
            cls_grasps_files = sorted(glob.glob(self.grasps_dir+'/'+class_type_i+'/*.h5'))

            for grasp_file in cls_grasps_files:
                g_obj = AcronymGrasps(grasp_file)

                ## Grasp File ##
                if g_obj.good_grasps.shape[0] > 0:
                    self.grasp_files.append(grasp_file)

        ## Split Train/Validation
        n = len(self.grasp_files)
        train_size = int(n*0.9)
        test_size  =  n - train_size

        self.train_grasp_files, self.test_grasp_files = torch.utils.data.random_split(self.grasp_files, [train_size, test_size])

        self.type = 'train'
        self.len = len(self.train_grasp_files)

        self.n_pointcloud = n_pointcloud
        self.n_density  = n_density
        self.n_occ = n_coords

        ## Variables on Data
        self.one_object = one_object
        self.augmented_rotation = augmented_rotation
        self.se3 = se3

        ## Visualization
        self.visualize = visualize
        self.scale = 8.
        
        # del n, train_size, test_size

    def __len__(self):
        return self.len

    def set_test_data(self):
        self.len = len(self.test_grasp_files)
        self.type = 'test'

    def _get_grasps(self, grasp_obj):
        try:
            rix = np.random.randint(low=0, high=grasp_obj.good_grasps.shape[0], size=self.n_density)
        except:
            print('lets see')
        H_grasps = grasp_obj.good_grasps[rix, ...]
        return H_grasps

    def _get_sdf(self, grasp_obj, grasp_file):

        mesh_fname = grasp_obj.mesh_fname
        mesh_scale = grasp_obj.mesh_scale

        mesh_type = mesh_fname.split('/')[1]
        mesh_name = mesh_fname.split('/')[-1]
        filename  = mesh_name.split('.obj')[0]
        sdf_file = os.path.join(self.data_dir, 'sdf', mesh_type, filename+'.json')

        with open(sdf_file, 'rb') as handle:
            sdf_dict = pickle.load(handle)

        loc = sdf_dict['loc']
        scale = sdf_dict['scale']
        xyz = (sdf_dict['xyz'] + loc)*scale*mesh_scale
        rix = np.random.permutation(xyz.shape[0])
        xyz = xyz[rix[:self.n_occ], :]
        sdf = sdf_dict['sdf'][rix[:self.n_occ]]*scale*mesh_scale
        return xyz, sdf

    def _get_mesh_pcl(self, grasp_obj):
        
        # mesh = grasp_obj.load_mesh()
        # mesh_sample = mesh.sample(self.n_pointcloud)
        with grasp_obj.load_mesh() as mesh:
            mesh_sample = mesh.sample(self.n_pointcloud)
        del mesh
        return mesh_sample

    def _get_item(self, index):
        if self.one_object:
            index = 0

        ## Load Files ##
        if self.type == 'train':
            grasps_obj = AcronymGrasps(self.train_grasp_files[index])
        else:
            grasps_obj = AcronymGrasps(self.test_grasp_files[index])

        ## SDF
        xyz, sdf = self._get_sdf(grasps_obj, self.train_grasp_files[index])

        ## PointCloud
        pcl = self._get_mesh_pcl(grasps_obj)

        ## Grasps good/bad
        H_grasps = self._get_grasps(grasps_obj)

        ## rescale, rotate and translate ##
        xyz = xyz*self.scale
        sdf = sdf*self.scale
        pcl = pcl*self.scale
        H_grasps[..., :3, -1] = H_grasps[..., :3, -1]*self.scale
        ## Random rotation ##
        R = special_ortho_group.rvs(3)
        H = np.eye(4)
        H[:3, :3] = R
        mean = np.mean(pcl, 0)
        ## translate ##
        xyz = xyz - mean
        pcl = pcl - mean
        H_grasps[..., :3, -1] = H_grasps[..., :3, -1] - mean
        ## rotate ##
        pcl = np.einsum('mn,bn->bm',R, pcl)
        xyz = np.einsum('mn,bn->bm',R, xyz)
        H_grasps = np.einsum('mn,bnk->bmk', H, H_grasps)
        #######################

        # Visualize
        if self.visualize:

            ## 3D matplotlib ##
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(pcl[:,0], pcl[:,1], pcl[:,2], c='r')

            x_grasps = H_grasps[..., :3, -1]
            ax.scatter(x_grasps[:,0], x_grasps[:,1], x_grasps[:,2], c='b')

            ## sdf visualization ##
            n = 100
            x = xyz[:n,:]

            x_sdf = sdf[:n]
            x_sdf = 0.9*x_sdf/np.max(x_sdf)
            c = np.zeros((n, 3))
            c[:, 1] = x_sdf
            ax.scatter(x[:,0], x[:,1], x[:,2], c=c)

            plt.show()
            #plt.show(block=True)

        res = {'visual_context': torch.from_numpy(pcl).float(),
               'x_sdf': torch.from_numpy(xyz).float(),
               'x_ene_pos': torch.from_numpy(H_grasps).float(),
               'scale': torch.Tensor([self.scale]).float()}

        # del pcl, xyz, H_grasps, sdf, grasps_obj, R, H, mean, x_grasps, x_sdf, c, x, n, sdf_dict, loc, scale, mesh, mesh_sample, sdf_file, mesh_fname, mesh_scale, filename, mesh_name, mesh_type, rix
        return res, {'sdf': torch.from_numpy(sdf).float()}

    def __getitem__(self, index):
        'Generates one sample of data'
        return self._get_item(index)


# class PartialPointcloudAcronymAndSDFDataset(Dataset):
#     'DataLoader for training DeepSDF with a Rotation Invariant Encoder model'
#     def __init__(self, class_type=['Cup', 'Mug', 'Fork', 'Hat', 'Bottle'],
#                  se3=False, phase='train', one_object=False,
#                  n_pointcloud=1000, n_density=200, n_coords=1000,
#                  augmented_rotation=True, visualize=False, split=True,
#                  train_files=None, test_files=None):

#         self.class_type = class_type
#         self.data_dir = get_data_src()

#         self.grasps_dir = os.path.join(self.data_dir, 'grasps')

#         self.grasp_files = []
#         for class_type_i in class_type:
#             cls_grasps_files = sorted(glob.glob(self.grasps_dir + '/' + class_type_i + '/*.h5'))

#             for grasp_file in cls_grasps_files:
#                 g_obj = AcronymGrasps(grasp_file)

#                 ## Grasp File ##
#                 if g_obj.good_grasps.shape[0] > 0:
#                     self.grasp_files.append(grasp_file)

#         ## Handle custom train and test files
#         if train_files is not None:
#             self.train_grasp_files = train_files
#         else:
#             self.train_grasp_files = self.grasp_files

#         if test_files is not None:
#             self.test_grasp_files = test_files
#         else:
#             ## Split Train/Test if test_files not provided
#             n = len(self.grasp_files)
#             train_size = int(n * 0.9)
#             test_size = n - train_size
#             self.train_grasp_files, self.test_grasp_files = torch.utils.data.random_split(
#                 self.grasp_files, [train_size, test_size])

#         ## Set dataset phase and length
#         self.phase = phase
#         if self.phase == 'train':
#             self.len = len(self.train_grasp_files)
#         else:
#             self.len = len(self.test_grasp_files)

#         self.n_pointcloud = n_pointcloud
#         self.n_density = n_density
#         self.n_occ = n_coords

#         ## Variables on Data
#         self.one_object = one_object
#         self.augmented_rotation = augmented_rotation
#         self.se3 = se3

#         ## Visualization
#         self.visualize = visualize
#         self.scale = 8.

#         ## Sampler
#         self.scan_pointcloud = ScanPointcloud()

#     def __len__(self):
#         return self.len

#     def _get_grasps(self, grasp_obj):
#         try:
#             rix = np.random.randint(low=0, high=grasp_obj.good_grasps.shape[0], size=self.n_density)
#         except:
#             print('Error in sampling grasps.')
#         H_grasps = grasp_obj.good_grasps[rix, ...]
#         return H_grasps

#     def _get_sdf(self, grasp_obj, grasp_file):
#         mesh_fname = grasp_obj.mesh_fname
#         mesh_scale = grasp_obj.mesh_scale

#         mesh_type = mesh_fname.split('/')[1]
#         mesh_name = mesh_fname.split('/')[-1]
#         filename = mesh_name.split('.obj')[0]
#         sdf_file = os.path.join(self.data_dir, 'sdf', mesh_type, filename + '.json')

#         with open(sdf_file, 'rb') as handle:
#             sdf_dict = pickle.load(handle)

#         loc = sdf_dict['loc']
#         scale = sdf_dict['scale']
#         xyz = (sdf_dict['xyz'] + loc) * scale * mesh_scale
#         rix = np.random.permutation(xyz.shape[0])
#         xyz = xyz[rix[:self.n_occ], :]
#         sdf = sdf_dict['sdf'][rix[:self.n_occ]] * scale * mesh_scale
#         return xyz, sdf

#     def _get_mesh_pcl(self, grasp_obj):
#         mesh = grasp_obj.load_mesh()
#         ## 1. Mesh Centroid ##
#         centroid = mesh.centroid
#         H = np.eye(4)
#         H[:3, -1] = -centroid
#         mesh.apply_transform(H)
#         ######################
#         P = self.scan_pointcloud.get_hq_scan_view(mesh)
#         P += centroid
#         try:
#             rix = np.random.randint(low=0, high=P.shape[0], size=self.n_pointcloud)
#         except:
#             print('Error in sampling point cloud.')
#         return P[rix, :]

#     def _get_item(self, index):
#         if self.one_object:
#             index = 0

#         ## Load Files ##
#         if self.phase == 'train':
#             grasp_file = self.train_grasp_files[index]
#         else:
#             grasp_file = self.test_grasp_files[index]

#         grasps_obj = AcronymGrasps(grasp_file)

#         ## SDF
#         xyz, sdf = self._get_sdf(grasps_obj, grasp_file)

#         ## PointCloud
#         pcl = self._get_mesh_pcl(grasps_obj)

#         ## Grasps good/bad
#         H_grasps = self._get_grasps(grasps_obj)

#         ## rescale, rotate and translate ##
#         xyz = xyz * self.scale
#         sdf = sdf * self.scale
#         pcl = pcl * self.scale
#         H_grasps[..., :3, -1] = H_grasps[..., :3, -1] * self.scale
#         ## Random rotation ##
#         R = special_ortho_group.rvs(3)
#         H = np.eye(4)
#         H[:3, :3] = R
#         mean = np.mean(pcl, 0)
#         ## translate ##
#         xyz = xyz - mean
#         pcl = pcl - mean
#         H_grasps[..., :3, -1] = H_grasps[..., :3, -1] - mean
#         ## rotate ##
#         pcl = np.einsum('mn,bn->bm', R, pcl)
#         xyz = np.einsum('mn,bn->bm', R, xyz)
#         H_grasps = np.einsum('mn,bnk->bmk', H, H_grasps)
#         #######################

#         # Visualize
#         if self.visualize:
#             ## 3D matplotlib ##
#             import matplotlib.pyplot as plt

#             fig = plt.figure()
#             ax = fig.add_subplot(projection='3d')
#             ax.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], c='r')

#             x_grasps = H_grasps[..., :3, -1]
#             ax.scatter(x_grasps[:, 0], x_grasps[:, 1], x_grasps[:, 2], c='b')

#             ## sdf visualization ##
#             n = 100
#             x = xyz[:n, :]

#             x_sdf = sdf[:n]
#             x_sdf = 0.9 * x_sdf / np.max(x_sdf)
#             c = np.zeros((n, 3))
#             c[:, 1] = x_sdf
#             ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=c)

#             plt.show()

#         res = {'visual_context': torch.from_numpy(pcl).float(),
#                'x_sdf': torch.from_numpy(xyz).float(),
#                'x_ene_pos': torch.from_numpy(H_grasps).float(),
#                'scale': torch.Tensor([self.scale]).float()}
        
#         process = psutil.Process(os.getpid())
        
#         return res, {'sdf': torch.from_numpy(sdf).float()}

#     def __getitem__(self, index):
#         'Generates one sample of data'
#         return self._get_item(index)


# class PartialPointcloudAcronymAndSDFDataset(Dataset):
#     'DataLoader for training DeepSDF with a Rotation Invariant Encoder model'
#     def __init__(self, class_type=['Cup', 'Mug', 'Fork', 'Hat', 'Bottle'],
#                  se3=False, phase='train', one_object=False,
#                  n_pointcloud=1000, n_density=200, n_coords=1000,
#                  augmented_rotation=True, visualize=False, split=True,
#                  train_files=None, test_files=None):

#         # self.class_type = class_type
#         self.data_dir = get_data_src()

#         # self.grasps_dir = os.path.join(self.data_dir, 'grasps')

#         # Collect grasp file paths without opening them
#         # self.grasp_files = []
#         # for class_type_i in class_type:
#         #     cls_grasps_files = sorted(glob.glob(os.path.join(self.grasps_dir, class_type_i, '*.h5')))
#         #     self.grasp_files.extend(cls_grasps_files)

#         # Handle custom train and test files
#         if train_files is not None and test_files is None:
#             self.train_grasp_files = train_files
#         # elif train_files is None and test_files is None:
#         #     self.train_grasp_files = self.grasp_files

#         if test_files is not None and train_files is None:
#             self.test_grasp_files = test_files
#         # else:
#         #     # Split Train/Test if test_files not provided
#         #     n = len(self.grasp_files)
#         #     train_size = int(n * 0.9)
#         #     test_size = n - train_size
#         #     self.train_grasp_files, self.test_grasp_files = torch.utils.data.random_split(
#         #         self.grasp_files, [train_size, test_size])

#         # Set dataset phase and length
#         self.phase = phase
#         if self.phase == 'train':
#             self.len = len(self.train_grasp_files)
#         else:
#             self.len = len(self.test_grasp_files)

#         self.n_pointcloud = n_pointcloud
#         self.n_density = n_density
#         self.n_occ = n_coords

#         # Variables on Data
#         self.one_object = one_object
#         self.augmented_rotation = augmented_rotation
#         self.se3 = se3

#         # Visualization
#         self.visualize = visualize
#         self.scale = 8.

#         # Sampler
#         self.scan_pointcloud = ScanPointcloud()
        
#         # del class_type, se3, phase, one_object, n_pointcloud, n_density, n_coords, augmented_rotation, visualize, split, train_files, test_files

#     def __len__(self):
#         return self.len

#     def _get_grasps(self, grasp_obj):
#         try:
#             rix = np.random.randint(low=0, high=grasp_obj.good_grasps.shape[0], size=self.n_density)
#             H_grasps = grasp_obj.good_grasps[rix, ...]
#         except:
#             print('Error in sampling grasps.')
#             raise ValueError("No valid grasps available.")
#         # del rix
#         return H_grasps

#     def _get_sdf(self, grasp_obj):
#         mesh_fname = grasp_obj.mesh_fname
#         mesh_scale = grasp_obj.mesh_scale

#         mesh_type = mesh_fname.split('/')[1]
#         mesh_name = mesh_fname.split('/')[-1]
#         filename = mesh_name.split('.obj')[0]
#         sdf_file = os.path.join(self.data_dir, 'sdf', mesh_type, filename + '.json')

#         with open(sdf_file, 'rb') as handle:
#             sdf_dict = pickle.load(handle)

#         loc = sdf_dict['loc']
#         scale = sdf_dict['scale']
#         xyz = (sdf_dict['xyz'] + loc) * scale * mesh_scale
#         rix = np.random.permutation(xyz.shape[0])
#         xyz = xyz[rix[:self.n_occ], :]
#         sdf = sdf_dict['sdf'][rix[:self.n_occ]] * scale * mesh_scale
        
#         # del rix, sdf_dict, loc, scale, mesh_type, mesh_name, filename, sdf_file
#         return xyz, sdf

#     def _get_mesh_pcl(self, grasp_obj):
#         process = psutil.Process(os.getpid())
#         mesh = grasp_obj.load_mesh()
        
#         # 1. Mesh Centroid
#         centroid = mesh.centroid
        
#         H = np.eye(4)
#         H[:3, -1] = -centroid
#         mesh.apply_transform(H)
#         # Generate point cloud
#         P = self.scan_pointcloud.get_hq_scan_view(mesh)
#         P += centroid

#         try:
#             rix = np.random.randint(low=0, high=P.shape[0], size=self.n_pointcloud)
#         except:
#             print('Error in sampling point cloud.')
#         return P[rix, :]
    
#     def get_mesh_pcl_subprocess(self, queue, filename, n_pointcloud):
#         grasp_obj = AcronymGrasps(filename)
#         mesh = grasp_obj.load_mesh()
#         ret = queue.get()
#         ret['pcl'] = mesh.sample(n_pointcloud)
#         queue.put(ret)

#     def _get_item(self, index):
#         if self.one_object:
#             index = 0

#         # Load Files
#         if self.phase == 'train':
#             grasp_file = self.train_grasp_files[index]
#         else:
#             grasp_file = self.test_grasp_files[index]
        
#         # Open the h5 file here instead of __init__
#         grasps_obj = AcronymGrasps(grasp_file)
#         grasps_obj.load_data()
        
#         # Check if there are good grasps
#         if grasps_obj.good_grasps.shape[0] == 0:
#             # Skip this sample or handle it appropriately
#             # Here, we'll raise an exception to indicate invalid data
#             raise ValueError(f"No good grasps in file: {grasp_file}")
        
        
#         # SDF
#         xyz, sdf = self._get_sdf(grasps_obj)
        
#         # PointCloud
#         # pcl = self._get_mesh_pcl(grasps_obj)
#         # fix for memory leak, by using a subprocess
#         queue = multiprocessing.Queue()
#         ret = {'pcl': None}
#         queue.put(ret)
#         p = multiprocessing.Process(target=self.get_mesh_pcl_subprocess, args=(queue, grasp_file, self.n_pointcloud))
#         p.start()
#         p.join()
#         pcl = queue.get()['pcl']
        
#         # Grasps good/bad
#         H_grasps = self._get_grasps(grasps_obj)
        
#         # Rescale, rotate and translate
#         xyz = xyz * self.scale
#         sdf = sdf * self.scale
#         pcl = pcl * self.scale
#         H_grasps[..., :3, -1] = H_grasps[..., :3, -1] * self.scale

#         # Random rotation
#         R = special_ortho_group.rvs(3)
#         H = np.eye(4)
#         H[:3, :3] = R
#         mean = np.mean(pcl, 0)

#         # Translate
#         xyz = xyz - mean
#         pcl = pcl - mean
#         H_grasps[..., :3, -1] = H_grasps[..., :3, -1] - mean

#         # Rotate
#         pcl = np.einsum('mn,bn->bm', R, pcl)
#         xyz = np.einsum('mn,bn->bm', R, xyz)
#         H_grasps = np.einsum('mn,bnk->bmk', H, H_grasps)

        
#         # Visualization
#         if self.visualize:
#             import matplotlib.pyplot as plt
#             fig = plt.figure()
#             ax = fig.add_subplot(projection='3d')
#             ax.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], c='r')

#             x_grasps = H_grasps[..., :3, -1]
#             ax.scatter(x_grasps[:, 0], x_grasps[:, 1], x_grasps[:, 2], c='b')

#             # SDF visualization
#             n = 100
#             x = xyz[:n, :]
#             x_sdf = sdf[:n]
#             x_sdf = 0.9 * x_sdf / np.max(x_sdf)
#             c = np.zeros((n, 3))
#             c[:, 1] = x_sdf
#             ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=c)
#             plt.show()


        
#         res = {'visual_context': torch.from_numpy(pcl).float(),
#                'x_sdf': torch.from_numpy(xyz).float(),
#                'x_ene_pos': torch.from_numpy(H_grasps).float(),
#                'scale': torch.Tensor([self.scale]).float()}
        
        
#         # del pcl, xyz, H_grasps, grasps_obj, R, H, mean
#         return res, {'sdf': torch.from_numpy(sdf).float()}
    

#     def __getitem__(self, index):
#         'Generates one sample of data'
#         try:
#             return self._get_item(index)
#         except ValueError as e:
#             # Handle the case when there is no valid data
#             print(e)
#             # For simplicity, raise the exception to let DataLoader handle it
#             raise e

class NoPositiveGraspsException(Exception):
    pass

class PartialPointcloudAcronymAndSDFDataset(Dataset):
    'DataLoader for training DeepSDF with a Rotation Invariant Encoder model'

    def __init__(self, class_type=['Cup', 'Mug', 'Fork', 'Hat', 'Bottle'],
                 se3=False, phase='train', one_object=False,
                 n_pointcloud=1000, n_density=200, n_coords=1000,
                 augmented_rotation=True, visualize=False, split=True,
                 train_files=None, test_files=None):

        # self.class_type = class_type
        self.data_dir = get_data_src()

        # Handle custom train and test files
        if train_files is not None and test_files is None:
            self.train_grasp_files = train_files

        if test_files is not None and train_files is None:
            self.test_grasp_files = test_files

        # Set dataset phase and length
        self.phase = phase
        if self.phase == 'train':
            self.len = len(self.train_grasp_files)
        else:
            self.len = len(self.test_grasp_files)

        self.n_pointcloud = n_pointcloud
        self.n_density = n_density
        self.n_occ = n_coords

        # Variables on Data
        self.one_object = one_object
        self.augmented_rotation = augmented_rotation
        self.se3 = se3

        # Visualization
        self.visualize = visualize
        self.scale = 8.

        # Sampler
        self.scan_pointcloud = ScanPointcloud()
        
        # del class_type, se3, phase, one_object, n_pointcloud, n_density, n_coords, augmented_rotation, visualize, split, train_files, test_files

    def __len__(self):
        return self.len

    def _get_grasps(self, grasp_obj):
        try:
            rix = np.random.randint(low=0, high=grasp_obj.good_grasps.shape[0], size=self.n_density)
            H_grasps = grasp_obj.good_grasps[rix, ...]
        except:
            print('Error in sampling grasps.')
            raise ValueError("No valid grasps available.")
        # del rix
        return H_grasps

    def _get_sdf(self, grasp_obj):
        mesh_fname = grasp_obj.mesh_fname
        mesh_scale = grasp_obj.mesh_scale

        mesh_type = mesh_fname.split('/')[1]
        mesh_name = mesh_fname.split('/')[-1]
        filename = mesh_name.split('.obj')[0]
        sdf_file = os.path.join(self.data_dir, 'sdf', mesh_type, filename + '.json')

        with open(sdf_file, 'rb') as handle:
            sdf_dict = pickle.load(handle)

        loc = sdf_dict['loc']
        scale = sdf_dict['scale']
        xyz = (sdf_dict['xyz'] + loc) * scale * mesh_scale
        rix = np.random.permutation(xyz.shape[0])
        xyz = xyz[rix[:self.n_occ], :]
        sdf = sdf_dict['sdf'][rix[:self.n_occ]] * scale * mesh_scale
        
        # del rix, sdf_dict, loc, scale, mesh_type, mesh_name, filename, sdf_file
        return xyz, sdf

    def _get_mesh_pcl(self, grasp_obj):
        process = psutil.Process(os.getpid())
        mesh = grasp_obj.load_mesh()
        
        # 1. Mesh Centroid
        centroid = mesh.centroid
        
        H = np.eye(4)
        H[:3, -1] = -centroid
        mesh.apply_transform(H)
        # Generate point cloud
        P = self.scan_pointcloud.get_hq_scan_view(mesh)
        P += centroid

        try:
            rix = np.random.randint(low=0, high=P.shape[0], size=self.n_pointcloud)
        except:
            print('Error in sampling point cloud.')
        return P[rix, :]
    
    def get_mesh_pcl_subprocess(self, queue, filename, n_pointcloud):
        grasp_obj = AcronymGrasps(filename)
        mesh = grasp_obj.load_mesh()
        ret = queue.get()
        ret['pcl'] = mesh.sample(n_pointcloud)
        queue.put(ret)

    def _get_item(self, index):
        if self.one_object:
            index = 0

        # Load Files
        if self.phase == 'train':
            grasp_file = self.train_grasp_files[index]
        else:
            grasp_file = self.test_grasp_files[index]
        
        # Open the h5 file here instead of __init__
        grasps_obj = AcronymGrasps(grasp_file)
        grasps_obj.load_data()
        
        # Check if there are good grasps
        if grasps_obj.good_grasps.shape[0] == 0:
            # Skip this sample by raising a custom exception
            raise NoPositiveGraspsException()
        
        # SDF
        xyz, sdf = self._get_sdf(grasps_obj)
        
        # PointCloud
        # pcl = self._get_mesh_pcl(grasps_obj)
        # fix for memory leak, by using a subprocess
        queue = multiprocessing.Queue()
        ret = {'pcl': None}
        queue.put(ret)
        p = multiprocessing.Process(target=self.get_mesh_pcl_subprocess, args=(queue, grasp_file, self.n_pointcloud))
        p.start()
        p.join()
        pcl = queue.get()['pcl']
        
        # Grasps good/bad
        H_grasps = self._get_grasps(grasps_obj)
        
        # Rescale, rotate and translate
        xyz = xyz * self.scale
        sdf = sdf * self.scale
        pcl = pcl * self.scale
        H_grasps[..., :3, -1] = H_grasps[..., :3, -1] * self.scale

        # Random rotation
        R = special_ortho_group.rvs(3)
        H = np.eye(4)
        H[:3, :3] = R
        mean = np.mean(pcl, 0)

        # Translate
        xyz = xyz - mean
        pcl = pcl - mean
        H_grasps[..., :3, -1] = H_grasps[..., :3, -1] - mean

        # Rotate
        pcl = np.einsum('mn,bn->bm', R, pcl)
        xyz = np.einsum('mn,bn->bm', R, xyz)
        H_grasps = np.einsum('mn,bnk->bmk', H, H_grasps)

        # Visualization
        if self.visualize:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], c='r')

            x_grasps = H_grasps[..., :3, -1]
            ax.scatter(x_grasps[:, 0], x_grasps[:, 1], x_grasps[:, 2], c='b')

            # SDF visualization
            n = 100
            x = xyz[:n, :]
            x_sdf = sdf[:n]
            x_sdf = 0.9 * x_sdf / np.max(x_sdf)
            c = np.zeros((n, 3))
            c[:, 1] = x_sdf
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=c)
            plt.show()

        res = {'visual_context': torch.from_numpy(pcl).float(),
               'x_sdf': torch.from_numpy(xyz).float(),
               'x_ene_pos': torch.from_numpy(H_grasps).float(),
               'scale': torch.Tensor([self.scale]).float()}
        
        # del pcl, xyz, H_grasps, grasps_obj, R, H, mean
        return res, {'sdf': torch.from_numpy(sdf).float()}

    def __getitem__(self, index):
        'Generates one sample of data'
        starting_index = index
        while True:
            try:
                return self._get_item(index)
            except NoPositiveGraspsException as e:
                # Handle the case when there is no valid data
                print(e)
                # Try the next index
                index = (index + 1) % self.len
                if index == starting_index:
                    # No valid data found in the entire dataset
                    raise StopIteration("No valid data found in dataset.")

if __name__ == '__main__':

    # ## Index conditioned dataset
    # dataset = AcronymAndSDFDataset(visualize=True, augmented_rotation=True, one_object=False)

    # ## Pointcloud conditioned dataset
    # dataset = PointcloudAcronymAndSDFDataset(visualize=True, augmented_rotation=True, one_object=False)

    ## Partial Pointcloud conditioned dataset with custom train and test files
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the base path to the dataset directory
    dataset_dir = os.path.join(script_dir, '../../../../grasp_diffusion_network/dataset_acronym_shapenetsem')

    # Define paths to the grasp files within the dataset
    train_file_relative = os.path.join(
        dataset_dir,
        'grasps/Mug_6a9b31e1298ca1109c515ccf0f61e75f_0.029998777830639544.h5'
    )
    test_file_relative = os.path.join(
        dataset_dir,
        'grasps/Mug_ba10400c108e5c3f54e1b6f41fdd78a_0.01695507616001961.h5'
    )

    # Ensure the paths are normalized
    train_file_path = os.path.normpath(train_file_relative)
    test_file_path = os.path.normpath(test_file_relative)

    # Define the train and test files using the paths
    train_files = [train_file_path]
    test_files = [test_file_path]

    # Instantiate the dataset with custom train files
    train_dataset = PartialPointcloudAcronymAndSDFDataset(
        visualize=False,
        augmented_rotation=True,
        one_object=False,
        phase='train',
        train_files=train_files
    )

    # Instantiate the dataset with custom test files
    test_dataset = PartialPointcloudAcronymAndSDFDataset(
        visualize=False,
        augmented_rotation=True,
        one_object=False,
        phase='test',
        test_files=test_files
    )

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    # # Iterate over the train DataLoader
    # print("Training Data:")
    # for x, y in train_dataloader:
    #     print(x)
    #     break  # Remove or adjust as needed

    # # Iterate over the test DataLoader
    # print("\nTesting Data:")
    # for x, y in test_dataloader:
    #     print(x)
    #     break  # Remove or adjust as needed
    
    # epoch = 0
    # while True:
    #     process = psutil.Process(os.getpid())
    #     print(f"Memory usage before epoch {epoch}: {process.memory_info().rss / 1024 ** 2} MB")
    #     # Iterate over the train DataLoader
    #     print(f"Training Data {epoch}:")
    #     for x in train_dataloader:
    #         pass
    #     epoch += 1
    #     gc.collect()
    #     print(f"Memory usage after epoch {epoch}: {process.memory_info().rss / 1024 ** 2} MB")
