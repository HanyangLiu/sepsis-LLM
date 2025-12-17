import os
import yaml
import argparse
import time
import torch
import lightning as L
import pandas as pd
from models import all_models
from experiment import experiment
from dataset_2 import sepsisDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar, Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from utils.utils_data import copy_file_to_memory, paste_file_from_memory


NUM_RUNS = 3
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load configs
parser = argparse.ArgumentParser(description="Run experiment")
parser.add_argument("--config", "-c",
                    dest="filename",
                    metavar="FILE",
                    help="path to the config file",
                    default="configs/agg_mm.yaml")
parser.add_argument("--task", "-t", default="AMR")
args = parser.parse_args()

with open(args.filename, "r") as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# Copy model file to memory
file_in_memory = copy_file_to_memory("models/{}.py".format(args.filename.split("/")[-1].split(".")[0]))

# Update config
config["data_params"]["multiclass"] = config["model_params"]["multiclass"]
config["data_params"]["task"] = args.task
config["data_params"]["llm_type"] = config["model_params"]["llm_type"]
config["model_params"]["modalities"] = config["data_params"]["modalities"]
config["model_params"]["use_precomputed"] = config["data_params"]["use_precomputed"]
print(config)


# Custom callback for tracking epoch time
class TimingCallback(Callback):
    def __init__(self):
        self.epoch_start_time = None
        self.epoch_times = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        print(f"Epoch {trainer.current_epoch} took {epoch_time:.2f} seconds.")


result = pd.DataFrame()
for seed in range(NUM_RUNS):
    config["exp_params"]["manual_seed"] = seed
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")

    # Define data, model, experiment
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(config["logging_params"]["save_dir"], args.task),
        name=config["model_params"]["name"],
    )

    # Retrieve version number (Lightning auto-generates it)
    version_no = tb_logger.version  # Automatically assigned by Lightning
    log_dir = os.path.join(config["logging_params"]["save_dir"], args.task, config["model_params"]["name"], f"version_{version_no}")
    os.makedirs(log_dir, exist_ok=True)

    dm = sepsisDataModule(**config["data_params"], pin_memory=torch.cuda.is_available())
    dm.setup()

    pytorch_model = all_models[config["model_params"]["name"]](
        size_static=dm.size_static,
        size_timeseries=dm.size_timeseries,
        embedding_size=dm.train_dataloader().dataset.embedding_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        **config["model_params"]
    )

    # Save model architecture and code to log_dir
    with open(os.path.join(log_dir, "model_arch.txt"), "w") as f: f.write(str(pytorch_model))
    if file_in_memory: paste_file_from_memory(file_in_memory, os.path.join(log_dir, "model.py"))

    # Lightning model
    LitModel = experiment(pytorch_model, config["exp_params"], config)

    # Initialize the custom timing callback
    timing_callback = TimingCallback()

    trainer = L.Trainer(
        logger=tb_logger,
        callbacks=[
            timing_callback,  # Add timing callback here
            RichProgressBar(),
            LearningRateMonitor(),
            ModelCheckpoint(
                save_top_k=1,
                dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                monitor="val_auprc",
                mode="max",
                save_last=True,
            ),
            EarlyStopping(
                monitor="val_auprc",
                mode="max",
                patience=config["exp_params"]["patience"],
            ),
        ],
        **config["trainer_params"]
    )

    # Train/eval
    print(f"======= Training {config['model_params']['name']} =======")
    trainer.fit(LitModel, datamodule=dm)

    # Test and log results
    result_row = trainer.test(ckpt_path="best", dataloaders=dm.test_dataloader())
    # outputs = trainer.predict(LitModel, dataloaders=dm.test_dataloader())
    # outputs = {key: torch.cat([ele[key] for ele in outputs], dim=0) for key in ["z_mm", "z_um", "imputed", "mod_mask", "labels"]}
    # torch.save(outputs, 'outputs_2.pt')
    # # outputs_loaded = torch.load('outputs.pt')

    # Compute avg epoch time
    avg_epoch_time = sum(timing_callback.epoch_times) / len(
        timing_callback.epoch_times) if timing_callback.epoch_times else 0
    print(f"Average epoch time: {avg_epoch_time:.2f} seconds")

    # Convert test results to DataFrame
    result_row = pd.DataFrame(result_row)
    result_row["avg_epoch_time"] = avg_epoch_time  # Add avg epoch time to results

    # Save final results under logs/{target}/{model_name}/{version_no}/results.csv
    result_row.to_csv(os.path.join(log_dir, "results.csv"), index=False)
    result = pd.concat([result, result_row], ignore_index=True)  # Append result


result.to_csv(os.path.join(log_dir, "all_results.csv"), index=False)
print("======= Final Results =======")
print("config:", config)
print(result.describe().loc[["mean", "std"], :])
# Calculate Mean and Standard Deviation of AUROC and AUPRC
print(f"AUROC: {result['test_auroc'].mean():.4f} ({result['test_auroc'].std():.4f})")
print(f"AUPRC: {result['test_auprc'].mean():.4f} ({result['test_auprc'].std():.4f})")

