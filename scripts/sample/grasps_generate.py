import time
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


def parse_args():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--obj_id', type=str, default='2')
    p.add_argument('--n_grasps', type=str, default='10')
    p.add_argument('--allowed_categories', type=str, default='CAT10', choices='Cup' or 'Mug-v00' or 'CAT10', help='Just for using our splied dataset')
    p.add_argument('--split', type=str, choices=['test', 'train'], default='test', help='Using split dataset test or train')
    p.add_argument('--model_from', type=str, choices=['pretrained_model', 'saving_folder'], default='saving_folder', help='Load model from given pretrained model or saving folder of trained model')
    p.add_argument('--scale', type=float, default=8., help='Scale of the point cloud, mesh and grasps')
    opt = p.parse_args()
    return opt


def get_approximated_grasp_diffusion_field(p, device='cpu', batch=10, checkpoints_dir=None):
    model_args = {
        'device': device,
        'checkpoints_dir': checkpoints_dir
    }

    model = load_model_pointcloud_grasp_diffusion(model_args)

    context = to_torch(p[None, ...], device)
    model.set_latent(context, batch=batch)

    generator = Grasp_AnnealedLD(model, batch=batch, T=70, T_fit=50, k_steps=1, device=device)
    return generator, model


def sample_pointcloud(obj_id=0, dataset_dir=None, allowed_categories=None, split='test', scale=8.):

    acronym_grasps = AcronymGraspsDirectory_(dataset_dir=dataset_dir, allowed_categories=allowed_categories, split=split)
    mesh = acronym_grasps.avail_obj[obj_id].load_mesh()

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


def transform_grasp_to_pointcloud_center(pointcloud, H):
    pointcloud_center = np.mean(pointcloud, axis=0)
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = -pointcloud_center
    H_transformed = np.zeros_like(H)
    for i in range(H.shape[0]):
        H_transformed[i] = np.dot(translation_matrix, H[i])
    return H_transformed


def generate_grasps(checkpoints_dir, n_grasps, pointcloud, device='cpu'):
    generator, model = get_approximated_grasp_diffusion_field(p=pointcloud, device=device, batch=n_grasps, checkpoints_dir=checkpoints_dir)
    H = to_numpy(generator.sample())
    H = transform_grasp_to_pointcloud_center(pointcloud, H)
    return H


if __name__ == '__main__':

    args = parse_args()
    obj_id = int(args.obj_id)
    scale = args.scale
    dataset_dir = os.path.join(root_dir, 'grasp_diffusion_network/dataset_acronym_shapenetsem')
    if args.allowed_categories == 'CAT10':
        # random choose one category from Book, Hammer, Cup, Mug, Teapot, Shampoo_Bottle, Bowl, RubiksCube, MilkCarton 
        allowed_categories = np.random.choice(['Book', 'Hammer', 'Cup', 'Mug', 'Teapot', 'Shampoo_Bottle', 'Bowl', 'RubiksCube', 'MilkCarton'])
        test_categories = allowed_categories
    else:
        allowed_categories = args.allowed_categories
        test_categories = allowed_categories
    P, mesh = sample_pointcloud(obj_id=obj_id, dataset_dir=dataset_dir, allowed_categories=allowed_categories, split=args.split, scale=scale)

        # Input parameters
    if args.allowed_categories == 'Cup':
        checkpoints_dir = os.path.join(get_root_src(), 'logs', 'multiobject_partial_graspdif', 'Cup/bs_4_2024-09-25_20-50/checkpoints/', 'model_current.pth')
    elif args.allowed_categories == 'Mug-v00':
        checkpoints_dir = os.path.join(get_root_src(), 'logs', 'multiobject_partial_graspdif', 'Mug-v00/bs_4_2024-09-25_20-50/checkpoints/', 'model_current.pth')
    elif args.allowed_categories == 'CAT10':
        checkpoints_dir = os.path.join(get_root_src(), 'logs', 'multiobject_partial_graspdif', 'Book_Hammer_Cup_Mug_Teapot_Shampoo_Bottle_Bowl_RubiksCube_MilkCarton/bs_4_2024-09-26_10-08/checkpoints/', 'model_current.pth')
    
    
    n_grasps = 5
    pointcloud = P

    print('##########################################################')
    print('Train category / Test category: {}/{}'.format(args.allowed_categories, test_categories))
    print('##########################################################')
    print('Dataset split: {}'.format(args.split))
    print('Selected object: {}'.format(obj_id))

    H = generate_grasps(checkpoints_dir=checkpoints_dir, n_grasps=n_grasps, pointcloud=pointcloud, device=device)

    H[..., :3, -1] *= 1 / scale

    # Visualize results
    from se3dif.visualization import grasp_visualization
    P *= 1 / scale
    grasp_visualization.visualize_grasps(H, p_cloud=P)

    # Uncomment the following lines to visualize the grasps with the mesh
    # P *= 1 / scale
    # mesh = mesh.apply_scale(1 / scale)
    # grasp_visualization.visualize_grasps(H, p_cloud=P, mesh=mesh)

