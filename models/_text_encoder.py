import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers import BioGptTokenizer, BioGptForCausalLM
from transformers import AutoTokenizer, AutoModelForMaskedLM


class TextEncoder(nn.Module):
    def __init__(self, llm_type="emilyalsentzer/Bio_ClinicalBERT", device="cpu"):
        super().__init__()
        self.llm_type = llm_type
        self.device = device
        self.max_length = 4096 if llm_type == "microsoft/biogpt" else 512

        # Define model loading map
        model_mapping = {
            "microsoft/biogpt": BioGptForCausalLM,
            "yikuan8/Clinical-Longformer": AutoModelForMaskedLM,
            "emilyalsentzer/Bio_ClinicalBERT": AutoModel,
            "medicalai/ClinicalBERT": AutoModel
        }

        if llm_type not in model_mapping:
            raise ValueError(f"Invalid LLM type: {llm_type}")

        # Load model
        self.model = model_mapping[llm_type].from_pretrained(llm_type, output_hidden_states=True).to(device)

        # Freeze base model parameters
        for param in self.model.base_model.parameters():
            param.requires_grad = False

        # Pooling layer for feature extraction
        self.pool_layer = nn.AdaptiveAvgPool1d(1)  # Adaptive pooling for any sequence length

    def forward(self, text_tokenized):
        with torch.no_grad():  # No gradient computation for efficiency
            output = self.model(**text_tokenized)

        # Handle different models
        if self.llm_type in ["emilyalsentzer/Bio_ClinicalBERT", "medicalai/ClinicalBERT"]:
            embeddings = output.pooler_output if hasattr(output, "pooler_output") else output.last_hidden_state[:, 0, :]

        elif self.llm_type in ["microsoft/biogpt", "yikuan8/Clinical-Longformer"]:
            hidden_states = output.hidden_states[1]  # Use second hidden state
            embeddings = self.pool_layer(hidden_states.permute(0, 2, 1)).squeeze(-1)

        return embeddings

