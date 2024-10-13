import yaml
import os

def generate_opt_yaml(save_path, allowed_categories):
    if isinstance(allowed_categories, (list, set)):
        allowed_categories_dict = {category: None for category in allowed_categories}
    elif isinstance(allowed_categories, str):
        allowed_categories_dict = {allowed_categories: None}
    else:
        raise ValueError("allowed_categories should be a list, set, or string")
    
    data = {
        'allowed_categories': allowed_categories_dict,
        'arch': 'GraspGeneratorSe3Diffusion',
        'balanced_data': False,
        'beta1': 0.9,
        'blacklisted_categories': '',
        'bps_encoder': False,
        'cluster_grasps': False,
        'confidence_weight': 1.0,
        'continue_train': False,
        'dataset_split': 'train',
        'debug': False,
        'gpu_ids': [0],
        'grasps_folder_name': 'grasps',
        'grasps_ratio': 1.0,
        'gripper': 'panda',
        'gripper_pc_npoints': -1,
        'init_gain': 0.02,
        'init_type': 'normal',
        'is_train': True,
        'kl_loss_weight': 0.01,
        'latent_size': 2,
        'lr': 0.0003,
        'lr_decay_iters': 50,
        'lr_policy': 'lambda',
        'max_dataset_size': float('inf'),
        'merge_pcs_in_vae_encoder': 0,
        'model_scale': 1,
        'niter_epoch_decay': 0,
        'niter_epoch_no_decay': 100000,
        'npoints': 1024,
        'num_grasp_clusters': 32,
        'num_grasps_per_object': 32,
        'num_objects_per_batch': 32,
        'num_workers': 16,
        'occlusion_dropout_rate': 0.0, 
        'occlusion_nclusters': 0,
        'optimizer': 'adam',
        'pc_features_size': 256,
        'phase': 'train',
        'pointnet_nclusters': 128,
        'pointnet_radius': 0.02,
        'ratio_hardnegative': 0.0,
        'ratio_positive': 0.5,
        'run_name': 'GraspGeneratorSe3Diffusion',
        'run_test_grasp_success': False,
        'run_test_num_grasps_per_object': 32,
        'run_test_num_objects': 8,
        'sample_full_pc': False,
        'seed': 42,
        'skip_error': False,
        'slurm_job_id': None,
        'splits_folder_name': 'splits',
        'use_uniform_quaternions': False,
        'vnn_encoder_dgcnn': False,
        'vnn_encoder_pointnet': False,
        'wandb_entity': 'grasp-diffusion-network',
        'wandb_group': None,
        'wandb_mode': 'online',
        'wandb_project': 'test_gdn',
        'wandb_run_name': 'GraspGeneratorSe3Diffusion',
        'which_epoch': 'latest'
    }

    checkpoint_dir = os.path.join(save_path, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    yaml_file_path = os.path.join(checkpoint_dir, 'opt.yaml')

    with open(yaml_file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
