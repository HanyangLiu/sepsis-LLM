from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer
import matplotlib.pyplot as plt
import imgkit

import torch
torch.set_float32_matmul_precision('high')
import os
import yaml
from models import all_models
from experiment import experiment
import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from dataset import sepsisDataModule
from torch import nn
import numpy as np
import pandas as pd
os.chdir("../")


def init_model(config, task):
    # Define data, model, experiment
    tb_logger = TensorBoardLogger(save_dir=os.path.join(config['logging_params']['save_dir'], task),
                                  name=config['model_params']['name'],
                                  )
    dm = sepsisDataModule(**config["data_params"], pin_memory=torch.cuda.is_available())
    pytorch_model = all_models[config['model_params']['name']](
        size_static=dm.size_static,
        size_timeseries=dm.size_timeseries,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        **config['model_params']
    )
    LitModel = experiment(pytorch_model, config['exp_params'], config)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
        monitor="val_auprc",
        mode='max',
        save_last=True,
    )
    trainer = L.Trainer(logger=tb_logger,
                        callbacks=[
                            RichProgressBar(),
                            LearningRateMonitor(),
                            checkpoint_callback,
                            EarlyStopping(monitor="val_auprc",
                                          mode="max",
                                          patience=2),
                        ],
                        **config['trainer_params'])

    return trainer, LitModel, dm, checkpoint_callback


config_file = "configs/agg_mm_note.yaml"
task = "AMR"


with open(config_file, 'r') as file:
    config = yaml.safe_load(file)
trainer, LitModel, dm, _ = init_model(config, task=task)
dm.setup()
model_dir = os.path.join(config['logging_params']['save_dir'], task, config['model_params']['name'])


cp = "/data/hanyang/sepsis/sepsis_multimodal/logs/AMR/AggMMNote/version_248/checkpoints/epoch=2-step=786.ckpt"
LitModel.load_state_dict(torch.load(cp)["state_dict"])


tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

def model_forward(notes):
    if isinstance(notes, np.ndarray):  # Convert numpy array to list of strings
        notes = notes.tolist()
    if isinstance(notes, list):
        notes = [str(t) for t in notes]  # Ensure all elements are strings
    inputs = tokenizer(notes, return_tensors="pt", padding=True, truncation=True)
    LitModel.to("cpu")
    model = nn.Sequential(
        LitModel.model.encoder_N,
        LitModel.model.fc_N,
        LitModel.model.fc_mm,
        LitModel.model.predictor,
    ).cpu()
    logits = model(notes)
    return logits


test_loader = dm.test_dataloader()
test_data = next(iter(test_loader))
notes = test_data["note"]


# With both the model and tokenizer initialized we are now able to get explanations on an example text.
cls_explainer = SequenceClassificationExplainer(
    LitModel.model,
    tokenizer)
word_attributions = cls_explainer("This patient is diagnosed to have hypertension and leukemia")

fig = cls_explainer.visualize("file.html")
imgkit.from_file('file.html', 'out.jpg')



###############

import shap
import logging
import numpy as np
from SHAP_for_text import SHAPexplainer
logging.getLogger("shap").setLevel(logging.WARNING)
shap.initjs()

words_dict = {0: None}
words_dict_reverse = {None: 0}
for h, hh in enumerate(bag_of_words):
    words_dict[h + 1] = hh
    words_dict_reverse[hh] = h + 1

predictor = SHAPexplainer(LitModel.model, tokenizer, words_dict, words_dict_reverse)
train_dt = np.array([predictor.split_string(x) for x in np.array(train_data)])
idx_train_data, max_seq_len = predictor.dt_to_idx(train_dt)

explainer = shap.KernelExplainer(model=predictor.predict, data=shap.kmeans(idx_train_data, k=50))

texts_ = [predictor.split_string(x) for x in texts]
idx_texts, _ = predictor.dt_to_idx(texts_, max_seq_len=max_seq_len)

to_use = idx_texts[-1:]
shap_values = explainer.shap_values(X=to_use, nsamples=64, l1_reg="aic")

len_ = len(texts_[-1:][0])
d = {i: sum(x > 0 for x in shap_values[i][0, :len_]) for i, x in enumerate(shap_values)}
m = max(d, key=d.get)
print(" ".join(texts_[-1:][0]))
shap.force_plot(explainer.expected_value[m], shap_values[m][0, :len_], texts_[-1:][0])



a = 1