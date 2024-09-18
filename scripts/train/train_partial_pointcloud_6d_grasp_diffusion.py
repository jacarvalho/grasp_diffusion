import os
import json
import copy
import configargparse
from se3dif.utils import get_root_src

import torch
from torch.utils.data import DataLoader

from se3dif import datasets, losses, summaries, trainer
from se3dif.models import loader

from se3dif.utils import load_experiment_specifications

from se3dif.trainer.learning_rate_scheduler import get_learning_rate_schedules

base_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.abspath(os.path.dirname(__file__ + '/../../../../../'))
# Define the base path to the splited dataset directory
# dataset_dir = os.path.join(base_dir, '../../../../grasp_diffusion_network/dataset_acronym_shapenetsem')
dataset_dir = os.path.join(root_dir, 'grasp_diffusion_network/dataset_acronym_shapenetsem')

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
    p.add_argument('--class_type', type=str, default='Mug')
    
    p.add_argument('--allowed_categories', type=str, default='Mug-v00', help='Just for using our splied dataset')

    opt = p.parse_args()
    return opt


def main(opt):

    ## Load training args ##
    spec_file = os.path.join(opt.specs_file_dir, opt.spec_file)
    args = load_experiment_specifications(spec_file)

    # saving directories
    root_dir = opt.saving_root
    exp_dir  = os.path.join(root_dir, args['exp_log_dir'])
    args['saving_folder'] = exp_dir


    if opt.device =='cuda':
        if 'cuda_device' in args:
            cuda_device = args['cuda_device']
        else:
            cuda_device = 0
        device = torch.device('cuda:' + str(cuda_device) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
        
    # Determine the JSON file path based on opt.allowed_categories
    splits_dir = os.path.join(dataset_dir, 'splits')
    json_file_name = f"{opt.allowed_categories}.json"
    json_file_path = os.path.join(splits_dir, json_file_name)

    # Load the JSON file content
    with open(json_file_path, 'r') as json_file:
        split_data = json.load(json_file)

    # Extract training and testing file names from the JSON content
    train_file_names = split_data.get('train', [])
    test_file_names = split_data.get('test', [])

    # Combine the base path with file names to get full paths
    train_files = [os.path.join(dataset_dir, 'grasps', fname) for fname in train_file_names]
    test_files = [os.path.join(dataset_dir, 'grasps', fname) for fname in test_file_names]
    train_files = [os.path.normpath(fpath) for fpath in train_files]
    test_files = [os.path.normpath(fpath) for fpath in test_files]

    ## Dataset
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
        drop_last=True
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
        drop_last=True
    )


    ## Model
    args['device'] = device
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