import torch
from utils.types_ import *
from ._modules import MLP, focalLoss
from ._base_mm import BaseMM
from torch import nn


class AggMM(BaseMM):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self, **kwargs) -> None:
        super(AggMM, self).__init__(**kwargs)  # Pass all arguments to BaseMM

        # Extract key parameters safely
        dropout = kwargs.pop("dropout", 0.3)
        embed_size = kwargs.pop("embed_size", 128)
        modalities = kwargs.pop("modalities", [True, True, True, True])  # Define number of modalities

        # Define prediction head
        self.predictor = MLP(
            dropout=dropout,
            in_dim=embed_size,  # Ensure correct input size
            post_dim=embed_size,
            out_dim=self.out_dim
        )

        # Define a zero embedding for missing modalities
        self.zero_embedding = nn.Parameter(torch.zeros(1, embed_size), requires_grad=False)

    def predict_proba(self, batch, **kwargs):
        logits, _, _ = self.forward(batch)
        return self.logits_to_probs(logits)

    def forward(self, batch, **kwargs) -> torch.Tensor:
        encoded = self.encoder_forward(batch)  # List of modality embeddings (B, D)
        mod_mask = batch["modality_mask"]  # Shape: (B, M) (1 = present, 0 = missing)

        # Replace missing modalities with zero embeddings
        for m in range(len(encoded)):
            missing_indices = ~mod_mask[:, m].bool()
            if missing_indices.any():
                encoded[m] = torch.where(
                    missing_indices.unsqueeze(1),
                    self.zero_embedding.expand_as(encoded[m]),  # Replace missing modalities with zero embedding
                    encoded[m]
                )

        # Concatenate embeddings from all modalities
        embed = torch.cat(encoded, dim=-1) if len(encoded) > 1 else encoded[0]

        z_mm = self.fc_mm(embed)
        logits = self.predictor(z_mm)

        return logits, z_mm, encoded

    def loss_function(self, *args, **kwargs) -> dict:
        logits, z_mm, z_um = args[0]
        loss = self.prediction_loss(logits, kwargs["targets"])
        mm_similarity = self._eval_cosine([z_mm] + z_um)
        uni_similarity = self._eval_cosine(z_um) if len(z_um) > 1 else torch.tensor(0.0)

        return {
            "loss": loss,
            "mm_similarity": mm_similarity,
            "um_similarity": uni_similarity
        }

    def encode(self, input, **kwargs):
        _, z_mm, z_um = self.forward(input, **kwargs)
        return [z_mm] + z_um
