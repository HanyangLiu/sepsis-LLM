import os
import yaml
import argparse
import torch
from archive.dataset import sepsisDataModule

os.chdir("../")

NUM_RUNS = 5

# load configs
parser = argparse.ArgumentParser(description='Generic runner')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/mocca.yaml')
parser.add_argument('--task', '-t',
                    default='AMR')
args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# Set random seed and precision
config["data_params"]["multiclass"] = config["model_params"]["multiclass"]
config['data_params']['task'] = args.task
print(config)


dm = sepsisDataModule(**config["data_params"], pin_memory=torch.cuda.is_available())
data = dm.all_data


a = 1

