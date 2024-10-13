import os
import json
import copy
import configargparse
import numpy as np
from datetime import datetime
from se3dif.utils import get_root_src
from se3dif.utils import generate_opt_yaml

import torch
from torch.utils.data import DataLoader

from se3dif import datasets, losses, summaries, trainer
from se3dif.models import loader

from se3dif.utils import load_experiment_specifications

from se3dif.trainer.learning_rate_scheduler import get_learning_rate_schedules
import torch.multiprocessing as mp

base_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.abspath(os.path.dirname(__file__ + '/../../../../../'))
# path to dataset_acronym_shapenetsem
dataset_root_folder = os.path.join(root_dir, 'grasp_diffusion_network/dataset_acronym_shapenetsem')
train_params_dir = os.path.join(get_root_src(), 'grasp_diffusion', 'scripts', 'train', 'params')

# Selected 10 categories from the ACRONYM dataset
CAT10_set = {
    'Book',
    'Bottle',
    'Bowl',
    'Cup',
    'Hammer',
    'MilkCarton',
    'Mug',
    'RubiksCube',
    'Shampoo',
    'SodaCan',
}

def parse_args():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--specs_file_dir', type=str, default=os.path.join(base_dir, 'params')
                   , help='root for saving logging')

    p.add_argument('--spec_file', type=str, default='multiobject_partialp_graspdif'
                   , help='root for saving logging')

    p.add_argument('--summary', type=bool, default=True
                   , help='activate or deactivate summary')

    p.add_argument('--saving_root', type=str, default=os.path.join(get_root_src(), 'logs')
                   , help='root for saving logging')

    p.add_argument('--models_root', type=str, default=root_dir
                   , help='root for saving logging')

    p.add_argument('--device',  type=str, default='cuda',)
    # p.add_argument('--class_type', type=str, default='Mug')
    
    p.add_argument('--dataset_root_folder', type=str, default=dataset_root_folder, help='Path to dataset_acronym_shapenetsem')
    p.add_argument('--allowed_categories', type=str or set, default="Mug-v00", choices="Mug-v00,Mug-v01,Mug-v04,Cup,CAT10",help='for using dataset_acronym_shapenetsem')
    p.add_argument('--batch_size', type=int, default=4, help='Batch size')

    opt = p.parse_args()
    return opt

def load_train_test_files(allowed_categories, dataset_root_folder):
    """Load and accumulate train and test file paths based on allowed_categories."""
    train_files = []
    test_files = []

    splits_dir = os.path.join(dataset_root_folder, 'splits')

    if isinstance(allowed_categories, str) and allowed_categories != 'CAT10':
        json_file_name = f"{allowed_categories}.json"
        json_file_path = os.path.join(splits_dir, json_file_name)

        with open(json_file_path, 'r') as json_file:
            split_data = json.load(json_file)

        train_file_names = split_data.get('train', [])
        test_file_names = split_data.get('test', [])

        # Combine base path with file names
        train_files = [os.path.join(dataset_root_folder, 'grasps', fname) for fname in train_file_names]
        test_files = [os.path.join(dataset_root_folder, 'grasps', fname) for fname in test_file_names]
    elif isinstance(allowed_categories, str) and allowed_categories == 'CAT10':
        allowed_categories_set = CAT10_set
        # Multiple categories, accumulate files from each
        for category in allowed_categories_set:
            json_file_name = f"{category}.json"
            json_file_path = os.path.join(splits_dir, json_file_name)

            # Load the JSON file content
            with open(json_file_path, 'r') as json_file:
                split_data = json.load(json_file)

            # Extract training and testing file names
            train_file_names = split_data.get('train', [])
            test_file_names = split_data.get('test', [])

            # Combine base path with file names and accumulate
            train_files.extend([os.path.join(dataset_root_folder, 'grasps', fname) for fname in train_file_names])
            test_files.extend([os.path.join(dataset_root_folder, 'grasps', fname) for fname in test_file_names])
    else:
        raise ValueError("allowed_categories must be a string or set of strings.")

    # Normalize the file paths and convert to numpy arrays
    train_files = np.array([os.path.normpath(fpath) for fpath in train_files])
    test_files = np.array([os.path.normpath(fpath) for fpath in test_files])

    return train_files, test_files


def main(opt):
    ## Load training args ##
    spec_file = os.path.join(opt.specs_file_dir, opt.spec_file)
    args = load_experiment_specifications(spec_file)
    
    args['TrainSpecs']['batch_size'] = opt.batch_size
    class_type = opt.allowed_categories if isinstance(opt.allowed_categories, str) else '_'.join(opt.allowed_categories)
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M")

    # saving directories
    root_dir = opt.saving_root
    exp_dir = os.path.join(root_dir, args['exp_log_dir'], class_type)
    exp_dir = os.path.join(exp_dir,f"bs_{opt.batch_size}_{current_time}")
    args['saving_folder'] = exp_dir
    
    # generate opt yaml for evaluation in grasp diffusion network
    generate_opt_yaml(save_path=exp_dir, allowed_categories=opt.allowed_categories if opt.allowed_categories != 'CAT10' else CAT10_set)

    if opt.device =='cuda':
        if 'cuda_device' in args:
            cuda_device = args['cuda_device']
        else:
            cuda_device = 0
        device = torch.device('cuda:' + str(cuda_device) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    train_files, test_files = load_train_test_files(opt.allowed_categories, dataset_root_folder)
    
    ## Dataset
    data_loader_options = {}
    data_loader_options['num_workers'] = 0
    data_loader_options['pin_memory'] = True
    data_loader_options['persistent_workers'] = data_loader_options['num_workers'] > 0
    # data_loader_options['multiprocessing_context'] = mp.get_context('spawn') 
    

    train_dataset = datasets.PartialPointcloudAcronymAndSDFDataset(
        augmented_rotation=True,
        one_object=args['single_object'],
        phase='train',
        train_files=train_files
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args['TrainSpecs']['batch_size'],
        shuffle=True,
        drop_last=True,
        **data_loader_options
    )
    test_dataset = datasets.PartialPointcloudAcronymAndSDFDataset(
        augmented_rotation=True,
        one_object=args['single_object'],
        phase='test',
        test_files=test_files
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args['TrainSpecs']['batch_size'],
        shuffle=True,
        drop_last=True,
        **data_loader_options
    )


    ## Model
    args['device'] = device
    ## model params    
    args['params_dir'] = os.path.join(train_params_dir, 'multiobject_partialp_graspdif')  
    model = loader.load_model(args)

    # Losses
    loss = losses.get_losses(args)
    loss_fn = val_loss_fn = loss.loss_fn

    ## Summaries
    summary = summaries.get_summary(args, opt.summary)

    ## Optimizer
    lr_schedules = get_learning_rate_schedules(args)
    optimizer = torch.optim.Adam([
            {
                "params": model.vision_encoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": model.feature_encoder.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
            {
                "params": model.decoder.parameters(),
                "lr": lr_schedules[2].get_learning_rate(0),
            },
        ])

    print('#'*50)
    print(f"Training {opt.allowed_categories} with batch size {opt.batch_size}")
    # Train
    trainer.train(model=model.float(), train_dataloader=train_dataloader, epochs=args['TrainSpecs']['num_epochs'], model_dir= exp_dir,
                summary_fn=summary, device=device, lr=1e-4, optimizers=[optimizer],
                steps_til_summary=args['TrainSpecs']['steps_til_summary'],
                epochs_til_checkpoint=args['TrainSpecs']['epochs_til_checkpoint'],
                loss_fn=loss_fn, iters_til_checkpoint=args['TrainSpecs']['iters_til_checkpoint'],
                clip_grad=False, val_loss_fn=val_loss_fn, overwrite=True,
                val_dataloader=test_dataloader)


if __name__ == '__main__':
    args = parse_args()
    main(args)