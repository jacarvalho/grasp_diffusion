import os
import torch
import torch.nn as nn
import numpy as np
import glob

from se3dif import models


from se3dif.utils import get_pretrained_models_src, load_experiment_specifications
pretrained_models_dir = get_pretrained_models_src()


def load_model(args):
    if 'pretrained_model' in args:
        model_args = load_experiment_specifications(os.path.join(pretrained_models_dir,
                                                                      args['pretrained_model']))
        args["NetworkArch"] = model_args["NetworkArch"]
        args["NetworkSpecs"] = model_args["NetworkSpecs"]
    elif 'saving_folder' in args:
        model_args = load_experiment_specifications(args['params_dir'])
        args["NetworkArch"] = model_args["NetworkArch"]
        args["NetworkSpecs"] = model_args["NetworkSpecs"]
        

    if args['NetworkArch'] == 'GraspDiffusion':
        model = load_grasp_diffusion(args)
    elif args['NetworkArch'] == 'PointcloudGraspDiffusion':
        model = load_pointcloud_grasp_diffusion(args)


    if 'pretrained_model' in args:
        # model_path = os.path.join(pretrained_models_dir, args['pretrained_model'], 'model.pth')
        # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model_dir = os.path.join(pretrained_models_dir, args['pretrained_model'])
        model_files = glob.glob(os.path.join(model_dir, 'model*.pth'))
        if not model_files:
            raise FileNotFoundError(f"No model files starting with 'model' found in {model_dir}")
        model_path = max(model_files, key=os.path.getmtime)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        if args['device'] != 'cpu':
            model = model.to(args['device'], dtype=torch.float32)

    elif 'saving_folder' in args:
        load_model_dir = os.path.join(args['saving_folder'], 'checkpoints', 'model_current.pth')
        try:
            if args['device'] == 'cpu':
                model.load_state_dict(torch.load(load_model_dir, map_location=torch.device('cpu')))
            else:
                model.load_state_dict(torch.load(load_model_dir))
        except:
            pass

    return model


def load_model_pointcloud_grasp_diffusion(args):
    args["NetworkArch"] = "PointcloudGraspDiffusion"
    args["NetworkSpecs"] = {
        "feature_encoder": {
        "enc_dim": 132,
        "in_dim": 3,
        "out_dim": 7,
        "dims" : [ 512, 512, 512, 512, 512, 512, 512, 512],
        "dropout" : [0, 1, 2, 3, 4, 5, 6, 7],
        "dropout_prob" : 0.2,
        "norm_layers" : [0, 1, 2, 3, 4, 5, 6, 7],
        "latent_in" : [4],
        "xyz_in_all" : False,
        "use_tanh" : False,
        "latent_dropout" : False,
        "weight_norm" : True
        },
        "encoder": {
        "latent_size": 132,
        "hidden_dim": 512
        },
        "points": {
        "n_points": 30,
        "loc": [0.0, 0.0, 0.5],
        "scale": [0.7, 0.5, 0.7]
        },
        "decoder": {
        "hidden_dim": 512
        }
        }
        
    model = load_pointcloud_grasp_diffusion(args)
    load_model_dir = args['checkpoints_dir']

    try:
        if args['device'] == 'cpu':
            model.load_state_dict(torch.load(load_model_dir, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(load_model_dir))
    except:
        pass

    return model


def load_grasp_diffusion(args):
    device = args['device']
    params = args['NetworkSpecs']
    feat_enc_params = params['feature_encoder']
    lat_params = params['latent_codes']
    points_params = params['points']
    # vision encoder
    vision_encoder = models.vision_encoder.LatentCodes(num_scenes=lat_params['num_scenes'], latent_size=lat_params['latent_size'])
    # Geometry encoder
    geometry_encoder = models.geometry_encoder.map_projected_points
    # Feature Encoder
    feature_encoder = models.nets.TimeLatentFeatureEncoder(
            enc_dim=feat_enc_params['enc_dim'],
            latent_size= lat_params['latent_size'],
            dims = feat_enc_params['dims'],
            out_dim=feat_enc_params['out_dim'],
            dropout=feat_enc_params['dropout'],
            dropout_prob=feat_enc_params['dropout_prob'],
            norm_layers = feat_enc_params['norm_layers'],
            latent_in = feat_enc_params["latent_in"],
            xyz_in_all = feat_enc_params["xyz_in_all"],
            use_tanh = feat_enc_params["use_tanh"],
            latent_dropout = feat_enc_params["latent_dropout"],
            weight_norm= feat_enc_params["weight_norm"]
        )
    # 3D Points
    if 'loc' in points_params:
        points = models.points.get_3d_pts(n_points = points_params['n_points'],
                            loc=np.array(points_params['loc']),
                            scale=np.array(points_params['scale']))
    else:
        points = models.points.get_3d_pts(n_points=points_params['n_points'])
    # Energy Based Model
    in_dim = points_params['n_points']*feat_enc_params['out_dim']
    hidden_dim = 512
    energy_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
    )

    model = models.GraspDiffusionFields(vision_encoder=vision_encoder, feature_encoder=feature_encoder, geometry_encoder=geometry_encoder,
                                       decoder=energy_net, points=points).to(device)
    return model


def load_pointcloud_grasp_diffusion(args):
    device = args['device']
    params = args['NetworkSpecs']
    feat_enc_params = params['feature_encoder']
    v_enc_params = params['encoder']
    points_params = params['points']
    # vision encoder
    vision_encoder = models.vision_encoder.VNNPointnet2(out_features=v_enc_params['latent_size'], device=device)
    # Geometry encoder
    geometry_encoder = models.geometry_encoder.map_projected_points
    # Feature Encoder
    feature_encoder = models.nets.TimeLatentFeatureEncoder(
            enc_dim=feat_enc_params['enc_dim'],
            latent_size= v_enc_params['latent_size'],
            dims = feat_enc_params['dims'],
            out_dim=feat_enc_params['out_dim'],
            dropout=feat_enc_params['dropout'],
            dropout_prob=feat_enc_params['dropout_prob'],
            norm_layers = feat_enc_params['norm_layers'],
            latent_in = feat_enc_params["latent_in"],
            xyz_in_all = feat_enc_params["xyz_in_all"],
            use_tanh = feat_enc_params["use_tanh"],
            latent_dropout = feat_enc_params["latent_dropout"],
            weight_norm= feat_enc_params["weight_norm"]
        )
    # 3D Points
    if 'loc' in points_params:
        points = models.points.get_3d_pts(n_points = points_params['n_points'],
                            loc=np.array(points_params['loc']),
                            scale=np.array(points_params['scale']))
    else:
        points = models.points.get_3d_pts(n_points=points_params['n_points'])
    # Energy Based Model
    in_dim = points_params['n_points']*feat_enc_params['out_dim']
    hidden_dim = 512
    energy_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
    )

    model = models.GraspDiffusionFields(vision_encoder=vision_encoder, feature_encoder=feature_encoder, geometry_encoder=geometry_encoder,
                                       decoder=energy_net, points=points).to(device)
    return model