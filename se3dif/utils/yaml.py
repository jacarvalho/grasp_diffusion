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
    }

    checkpoint_dir = os.path.join(save_path, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    yaml_file_path = os.path.join(checkpoint_dir, 'opt.yaml')

    with open(yaml_file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
