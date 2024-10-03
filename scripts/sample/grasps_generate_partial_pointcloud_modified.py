import time
import shutil
from datetime import datetime
import scipy.spatial.transform
import torch
import numpy as np
import os
import json
from se3dif.datasets import AcronymGraspsDirectory, AcronymGraspsDirectory_
from se3dif.models.loader import load_model, load_model_pointcloud_grasp_diffusion
from se3dif.samplers import ApproximatedGrasp_AnnealedLD, Grasp_AnnealedLD
from se3dif.utils import to_numpy, to_torch
import configargparse
from se3dif.utils import get_root_src

from mesh_to_sdf.scan import ScanPointcloud

device = 'cpu'
base_dir = os.path.join(get_root_src(), 'grasp_diffusion')
train_params_dir = os.path.join(base_dir, 'scripts', 'train', 'params')
root_dir = os.path.abspath(os.path.dirname(__file__ + '/../../../../../'))


allowed_categories = 'Cup'  # 'Cup' or 'Mug-v00' or 'CAT10'
n_grasps = 5
# path to the checkpoints directory of different categories
checkpoints_dir = os.path.join(get_root_src(), 'logs/multiobject_partial_graspdif/Cup/bs_4_2024-09-25_20-50/checkpoints')
# checkpoints_dir = os.path.join(get_root_src(), 'logs/multiobject_partial_graspdif/Mug-v00/bs_4_2024-09-25_20-50/checkpoints')
# checkpoints_dir = os.path.join(get_root_src(), 'logs/multiobject_partial_graspdif/Book_Hammer_Cup_Mug_Teapot_Shampoo_Bottle_Bowl_RubiksCube_MilkCarton/bs_4_2024-09-26_10-08/checkpoints')
dataset_root_folder = os.path.join(root_dir, 'grasp_diffusion_network/dataset_acronym_shapenetsem')



def parse_args():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--obj_id', type=int, default=2, help='Object id to sample from the dataset split')
    p.add_argument('--split', type=str, choices=['test', 'train'], default='test', help='Using split dataset test or train')
    p.add_argument('--scale', type=float, default=8., help='Scale of the point cloud, mesh and grasps')
    p.add_argument('--allowed_categories', type=str, default=allowed_categories, choices='Cup' or 'Mug-v00' or 'CAT10', help='Just for using dataset_acronym_shapenetsem')
    p.add_argument('--n_grasps', type=int, default=n_grasps, help='Number of grasps to generate')
    p.add_argument('--checkpoints_dir', type=str, default=checkpoints_dir, help='Path to the checkpoints directory')
    p.add_argument('--dataset_root_folder', type=str, default=dataset_root_folder, help='Path to the dataset folder')
    opt = p.parse_args()
    return opt


def check_and_update_checkpoint_dir(checkpoints_dir):
    if checkpoints_dir.endswith('.pth') and os.path.isfile(checkpoints_dir):
        return checkpoints_dir

    model_current_path = os.path.join(checkpoints_dir, 'model_current.pth')
    if os.path.exists(model_current_path):
        return model_current_path

    for file_name in os.listdir(checkpoints_dir):
        if file_name.endswith('.pth'):
            return os.path.join(checkpoints_dir, file_name)

    raise FileNotFoundError('No .pth file found in the directory')



def get_approximated_grasp_diffusion_field(p, device='cpu', batch=10, checkpoints_dir=None):
    checkpoints_dir = check_and_update_checkpoint_dir(checkpoints_dir)
    model_args = {
        'device': device,
        'checkpoints_dir': checkpoints_dir
    }

    model = load_model_pointcloud_grasp_diffusion(model_args)

    # context = to_torch(p[None, ...], device)
    if isinstance(p, np.ndarray):
        context = to_torch(p[None, ...], device)
    else:
        context = p[None, ...].to(device)
    model.set_latent(context, batch=batch)

    generator = Grasp_AnnealedLD(model, batch=batch, T=70, T_fit=50, k_steps=1, device=device)
    return generator, model


def sample_pointcloud(obj_id=0, dataset_dir=None, allowed_categories=None, split='test', scale=8.):

    acronym_grasps = AcronymGraspsDirectory_(dataset_dir=dataset_dir, allowed_categories=allowed_categories, split=split)
    mesh = acronym_grasps.avail_obj[obj_id].load_mesh(dataset_dir=dataset_dir)

    centroid = mesh.centroid
    H = np.eye(4)
    H[:3, -1] = -centroid
    mesh.apply_transform(H)

    scan_pointcloud = ScanPointcloud()
    P = scan_pointcloud.get_hq_scan_view(mesh)

    P *= scale
    P_mean = np.mean(P, 0)
    P += -P_mean

    rot = scipy.spatial.transform.Rotation.random().as_matrix()
    P = np.einsum('mn,bn->bm', rot, P)

    mesh.apply_scale(scale)
    H = np.eye(4)
    H[:3, -1] = -P_mean
    mesh.apply_transform(H)
    H = np.eye(4)
    H[:3, :3] = rot
    mesh.apply_transform(H)

    return P, mesh


def transform_grasp_to_pointcloud_center(pointcloud, H, return_chain=False):
    if isinstance(pointcloud, np.ndarray):
        pointcloud = torch.tensor(pointcloud, dtype=torch.float32)
    if isinstance(H, np.ndarray):
        H = torch.tensor(H, dtype=torch.float32)

    pointcloud_center = torch.mean(pointcloud, dim=0)
    translation_matrix = torch.eye(4)
    translation_matrix[:3, 3] = -pointcloud_center

    if return_chain:
        m, n, _, _ = H.shape
        translation_matrix = translation_matrix.unsqueeze(0).expand(m, -1, -1)
        H_transformed = torch.matmul(translation_matrix.unsqueeze(1), H)
    else:
        H_transformed = torch.matmul(translation_matrix, H)

    return H_transformed


def generate_grasps(checkpoints_dir, n_grasps, pointcloud, device='cpu', scale=8., return_chain=False):
    generator, model = get_approximated_grasp_diffusion_field(p=pointcloud, device=device, batch=n_grasps, checkpoints_dir=checkpoints_dir)
    H_chain = generator.sample(return_chain=return_chain)
    
    # H_chain = transform_grasp_to_pointcloud_center(pointcloud, H_chain, return_chain=return_chain)
    H_chain[..., :3, -1] *= 1 / scale
    return H_chain


if __name__ == '__main__':

    args = parse_args()

    if args.allowed_categories == 'CAT10':
        # random choose one category from Book, Hammer, Cup, Mug, Teapot, Shampoo_Bottle, Bowl, RubiksCube, MilkCarton 
        allowed_categories = np.random.choice(['Book', 'Hammer', 'Cup', 'Mug', 'Teapot', 'Shampoo_Bottle', 'Bowl', 'RubiksCube', 'MilkCarton'])
        test_categories = allowed_categories
    else:
        allowed_categories = args.allowed_categories
        test_categories = allowed_categories
    pointcloud, mesh = sample_pointcloud(obj_id=args.obj_id, dataset_dir=args.dataset_root_folder, allowed_categories=allowed_categories, split=args.split, scale=args.scale)
    pointcloud = torch.tensor(pointcloud, dtype=torch.float32)
    
    print('##########################################################')
    print('Train category / Test category: {}/{}'.format(args.allowed_categories, test_categories))
    print('##########################################################')
    print('Dataset split: {}'.format(args.split))
    print('Selected object: {}'.format(args.obj_id))

    return_chain = False
    H_chain = generate_grasps(checkpoints_dir=checkpoints_dir, n_grasps=args.n_grasps, pointcloud=pointcloud, device=device, scale=args.scale, return_chain=return_chain)
    if return_chain:
        H = H_chain[0]
    else:
        H = H_chain
    
    # Visualize results
    from se3dif.visualization import grasp_visualization
    grasp_visualization.visualize_grasps(H, p_cloud=pointcloud, scale_pc=args.scale)

    # visualize the grasps with the mesh
    # grasp_visualization.visualize_grasps(H, p_cloud=pointcloud, scale_pc=args.scale, mesh=mesh.apply_scale(1 / args.scale))

