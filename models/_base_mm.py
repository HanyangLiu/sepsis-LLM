import os.path
import torch
from torch import nn
from utils.types_ import *
import numpy as np
import pandas as pd
from ._modules import comorbidityTransformer, FFNEncoder
from ._grud import BackboneGRUD
from ._text_encoder import TextEncoder
from paths import processed_data_path
from ._modules import focalLoss


class BaseMM(nn.Module):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self, **kwargs) -> None:
        super(BaseMM, self).__init__()

        # Extract key parameters safely
        self.modalities = kwargs.pop("modalities", [True, True, True, True])
        self.compute_device = kwargs.pop("device", "cpu")
        self.embed_size = kwargs.pop("embed_size", 128)
        self.embed_weights = kwargs.pop("embed_weights", None)
        self.max_codes = kwargs.pop("max_codes", 200)
        self.static_hidden = kwargs.pop("static_hidden", 32)
        self.use_pretrain = kwargs.pop("use_pretrain", False)
        self.freeze_embed = kwargs.pop("freeze_embed", False)
        self.num_layers = kwargs.pop("num_layers", 1)
        self.num_heads = kwargs.pop("num_heads", 2)
        self.dropout = kwargs.pop("dropout", 0.3)
        self.multiclass = kwargs.pop("multiclass", False)
        self.clf_thresh = kwargs.pop("clf_thresh", 0.5)
        self.llm_type = kwargs.pop("llm_type", None)
        self.focal_loss = kwargs.pop("focal_loss", False)
        self.use_precomputed = kwargs.pop("use_precomputed", False)

        self.out_dim = 3 if self.multiclass else 1
        self.tau = nn.Parameter(torch.tensor(1 / 0.07), requires_grad=False)

        # Initialize encoders based on active modalities
        if self.modalities[0]:  # Static Data
            self.encoder_S = FFNEncoder(
                input_dim=kwargs["size_static"][0],
                hidden_dim=self.static_hidden,
                output_dim=self.embed_size,
                num_layers=2,
                dropout_prob=self.dropout,
                device=self.compute_device
            )

        if self.modalities[1]:  # Time-Series Data
            self.encoder_T = BackboneGRUD(
                n_steps=kwargs["size_timeseries"][0],
                n_features=kwargs["size_timeseries"][1],
                rnn_hidden_size=self.embed_size,
            )

        if self.modalities[2]:  # Comorbidity Data
            df_embedding = pd.read_csv(os.path.join(processed_data_path, self.embed_weights)).drop(columns=['code'])
            weights = np.zeros((len(df_embedding) + 2, self.embed_size))
            weights[1:-1, :] = df_embedding.values  # "0" as padding, largest index to mark all empty
            self.encoder_C = comorbidityTransformer(
                weights, self.embed_size, self.max_codes, self.use_pretrain, self.freeze_embed, self.num_layers,
                num_heads=self.num_heads,
                output_size=self.embed_size,
                dropout_rate=self.dropout,
                device=self.compute_device
            )

        if self.modalities[3]:  # Text Data
            if not self.use_precomputed:
                out_size = 1024 if self.llm_type == "microsoft/biogpt" else 768
                self.encoder_N = nn.Sequential(
                    TextEncoder(llm_type=self.llm_type, device=self.compute_device),
                    nn.Linear(out_size, self.embed_size),
                )
            else:
                self.encoder_N = nn.Linear(kwargs["embedding_size"], self.embed_size)

        # Fusion layer based on active modalities
        self.fc_mm = nn.Linear(self.embed_size * sum(self.modalities), self.embed_size)

        # Define loss functions
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        # Use Focal Loss instead of BCEWithLogitsLoss and CrossEntropyLoss
        if self.focal_loss:
            self.cls_loss_binary = focalLoss(alpha=0.25, gamma=2.0, reduction='mean')  # Binary classification
            self.cls_loss_multi = focalLoss(alpha=0.25, gamma=2.0, reduction='mean')  # Multi-class classification
        else:
            self.cls_loss_binary = nn.BCEWithLogitsLoss(reduction='mean')
            self.cls_loss_multi = nn.CrossEntropyLoss(reduction='mean')

    def logits_to_probs(self, logits):
        return self.softmax(logits) if self.multiclass else self.sigmoid(logits)

    def prediction_loss(self, logits, targets):
        return self.cls_loss_multi(logits, targets.view((-1,)).long()) if self.multiclass \
            else self.cls_loss_binary(logits, targets.float())

    def encoder_forward(self, batch, **kwargs):
        """Forward pass through individual encoders for each modality."""
        output = []

        if self.modalities[0]:  # Static
            output.append(self.encoder_S(batch["static"]))

        if self.modalities[1]:  # Time-Series
            data_T = batch["ts"]
            _, z_T = self.encoder_T(
                data_T["X"],
                data_T["mask"],
                data_T["delta"],
                data_T["mean"],
                data_T["X_LOCF"],
            )
            output.append(z_T)

        if self.modalities[2]:  # Comorbidity
            output.append(self.encoder_C(batch["comorb"]))

        if self.modalities[3]:  # Text
            output.append(self.encoder_N(batch["note"]))

        return output

    def _eval_cosine(self, features):
        z_MM, z_UMs = features[0], features[1:]
        z_MM = z_MM / (z_MM.norm(dim=-1, keepdim=True) + 1e-8)  # Avoid division by zero
        z_UMs = torch.stack([z / (z.norm(dim=-1, keepdim=True) + 1e-8) for z in z_UMs], dim=0)  # Shape: (num_modalities, batch_size, embed_dim)
        sim = torch.sum(z_MM.unsqueeze(0) * z_UMs, dim=-1).mean()
        return sim

