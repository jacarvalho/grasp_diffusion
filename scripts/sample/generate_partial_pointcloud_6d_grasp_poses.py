import time
import scipy.spatial.transform
import torch
import numpy as np
import os
import json
from se3dif.datasets import AcronymGraspsDirectory
from se3dif.models.loader import load_model
from se3dif.samplers import ApproximatedGrasp_AnnealedLD, Grasp_AnnealedLD
from se3dif.utils import to_numpy, to_torch
import configargparse
from se3dif.utils import get_root_src

from mesh_to_sdf.scan import ScanPointcloud

device = 'cpu'
saving_folder = os.path.join(get_root_src(), 'logs', 'multiobject_partial_graspdif')

base_dir = os.path.join(get_root_src(), 'grasp_diffusion')
train_params_dir = os.path.join(base_dir, 'scripts', 'train', 'params')
root_dir = os.path.abspath(os.path.dirname(__file__ + '/../../../../../'))
dataset_dir = os.path.join(root_dir, 'grasp_diffusion_network/dataset_acronym_shapenetsem')


def parse_args():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--obj_id', type=str, default='3')
    p.add_argument('--n_grasps', type=str, default='10')
    p.add_argument('--obj_class', type=str, default='Mug')
    p.add_argument('--allowed_categories', type=str, default='Mug-v00', help='Just for using our splied dataset')
    p.add_argument('--split', type=str, choices=['test', 'train'], default='test', help='Using split dataset test or train')
    p.add_argument('--model_from', type=str, choices=['pretrained_model', 'saving_folder'], default='saving_folder',help='Load model from given pretrained model or saving folder of trained model')

    opt = p.parse_args()
    return opt


def get_approximated_grasp_diffusion_field(p, device='cpu',model_from='saving_folder'):
    model_params = 'partial_grasp_dif'
    batch = 10
    ## Load model
    if model_from == 'pretrained_model':
        model_args = {
            'device': device,
            'pretrained_model': model_params,
        }
    else:
        model_args = {
            'device': device,
            'saving_folder': saving_folder,
        }
    ## read model params    
    if model_args['saving_folder'] is not None and model_params is 'partial_grasp_dif':
       model_args['params_dir'] = os.path.join(train_params_dir, 'multiobject_partialp_graspdif')  
    elif model_args['saving_folder'] is not None and model_params is 'grasp_dif_multi':
       model_args['params_dir'] = os.path.join(train_params_dir, 'multiobject_p_graspdif')
    
    model = load_model(model_args)

    context = to_torch(p[None,...], device)
    model.set_latent(context, batch=batch)

    ########### 2. SET SAMPLING METHOD #############
    generator = Grasp_AnnealedLD(model, batch=batch, T = 70, T_fit=50, k_steps=1, device=device)

    #generator.set_latent_code(obj_id)
    return generator, model


def sample_pointcloud(obj_id=0, obj_class='Mug', dataset_dir=None, allowed_categories=None, split='test'):
    # acronym_grasps = AcronymGraspsDirectory(data_type=obj_class)
    # mesh = acronym_grasps.avail_obj[obj_id].load_mesh()
    
    # for using our splitted dataset
    acronym_grasps = AcronymGraspsDirectory(data_type=obj_class,dataset_dir=dataset_dir, allowed_categories=allowed_categories, split=split)
    if split == 'test':
        mesh = acronym_grasps.avail_obj_test[obj_id].load_mesh()
    elif split == 'train':
        mesh = acronym_grasps.avail_obj_train[obj_id].load_mesh()

    centroid = mesh.centroid
    H = np.eye(4)
    H[:3,-1] = -centroid
    mesh.apply_transform(H)

    scan_pointcloud = ScanPointcloud()
    P = scan_pointcloud.get_hq_scan_view(mesh)


    P *= 8.
    P_mean = np.mean(P, 0)
    P += -P_mean

    rot = scipy.spatial.transform.Rotation.random().as_matrix()
    P = np.einsum('mn,bn->bm', rot, P)

    mesh.apply_scale(8.)
    H = np.eye(4)
    H[:3,-1] = -P_mean
    mesh.apply_transform(H)
    H = np.eye(4)
    H[:3,:3] = rot
    mesh.apply_transform(H)

    if dataset_dir is not None:
        return P, mesh, len(acronym_grasps.avail_obj_test)
    return P, mesh


if __name__ == '__main__':

    args = parse_args()

    print('##########################################################')
    print('Object Class: {}'.format(args.obj_class))
    print('Category: {}'.format(args.allowed_categories))
    print('##########################################################')

    n_grasps = int(args.n_grasps)
    obj_id = int(args.obj_id)
    obj_class = args.obj_class
    n_envs = 30

    ## Set Model and Sample Generator ##
    start_time = time.time()
    P, mesh, num_objs = sample_pointcloud(obj_id, obj_class,dataset_dir, args.allowed_categories, args.split)
    print('Dataset split: {}'.format(args.split))
    print('Number of object from datsset: {}'.format(num_objs))
    print('Selected object: {}'.format(obj_id))
    end_time = time.time()
    print(f"Time taken to sample point cloud: {end_time - start_time:.4f} seconds")
    
    start_time = time.time()
    generator, model = get_approximated_grasp_diffusion_field(P, device, args.model_from)
    end_time = time.time()
    print(f"Time taken to set up grasp diffusion field: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    H = generator.sample()
    end_time = time.time()
    print(f"Time taken to sample grasps: {end_time - start_time:.4f} seconds")
    
    H[..., :3, -1] *=1/8.

    ## Visualize results ##
    from se3dif.visualization import grasp_visualization

    vis_H = H.squeeze()
    P *=1/8
    mesh = mesh.apply_scale(1/8)
    grasp_visualization.visualize_grasps(to_numpy(H), p_cloud=P)#, mesh=mesh)

