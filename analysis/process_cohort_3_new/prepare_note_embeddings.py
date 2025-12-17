import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers import BioGptTokenizer, BioGptForCausalLM
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch import nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


project_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
project_name = os.path.basename(project_path)
server_name = os.uname()[1]
# remote_root = "/home/hangyue/sepsis/"
remote_root = "/data/hanyang/sepsis/"


prefix = "cohort3"
raw_data_path = os.path.join(remote_root, "cohort_3_new")
manual_data_path = os.path.join(remote_root, "manual_tables")
remote_project_path = os.path.join(remote_root, project_name)
processed_data_path = os.path.join(raw_data_path, "data_processed")
tmp_data_path = os.path.join(raw_data_path, "data_tmp")
combined_data_path = os.path.join(raw_data_path, "data_combined")

# Define device\
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define tokenizer and model mapping
tokenizer_mapping = {
    "microsoft/biogpt": BioGptTokenizer,
    "yikuan8/Clinical-Longformer": AutoTokenizer,
    "emilyalsentzer/Bio_ClinicalBERT": AutoTokenizer,
    "medicalai/ClinicalBERT": AutoTokenizer,
    "ruslanmv/Medical-Llama3-8B": AutoTokenizer,
}
model_mapping = {
    "microsoft/biogpt": BioGptForCausalLM,
    "yikuan8/Clinical-Longformer": AutoModelForMaskedLM,
    "emilyalsentzer/Bio_ClinicalBERT": AutoModel,
    "medicalai/ClinicalBERT": AutoModel,
    "ruslanmv/Medical-Llama3-8B": AutoModelForCausalLM,
}

# Move tokenizer loading outside the function to prevent repeated initialization
tokenizers = {name: tokenizer_mapping[name].from_pretrained(name) for name in tokenizer_mapping}

# Adaptive Pooling Layer (only needed for Longformer/BioGPT)
pool_layer = nn.AdaptiveAvgPool1d(1).to(device)


# Function to extract embeddings
def extract_embeddings(llm_type, note_type, max_length, batch_size=32):
    # check if embeddings already exist
    embedding_filename = f"note_embeddings_{note_type}_{llm_type.replace('/', '_')}_unpaired.npy"
    save_path = os.path.join(combined_data_path, embedding_filename)
    if os.path.exists(save_path):
        print(f"Embeddings already exist at {save_path}")
        return

    if llm_type not in tokenizer_mapping:
        raise ValueError(f"Invalid LLM type: {llm_type}")

    tokenizer = tokenizers[llm_type]  # Use preloaded tokenizer

    # Load model once per function call
    if llm_type == "ruslanmv/Medical-Llama3-8B":
        # Set 4-bit or 8-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,  # Use 4-bit quantization (change to load_in_8bit=True for 8-bit)
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # Further compression
        )
        # Load model with quantization
        model = model_mapping[llm_type].from_pretrained(
            llm_type,
            quantization_config=bnb_config,  # Apply quantization
            device_map="auto",
            output_hidden_states=True,
        )
    else:
        model = model_mapping[llm_type].from_pretrained(llm_type, output_hidden_states=True).eval().to(device)

    # Ensure parameters are frozen
    for param in model.base_model.parameters():
        param.requires_grad = False

    # Load dataset
    df = pd.read_csv(os.path.join(combined_data_path, "deep_notes_all.csv"))
    df[note_type] = df[note_type].fillna("No note")

    # Store embeddings as a CPU tensor list
    all_embeddings = []

    def encode_batch(text_list):
        """ Encode a batch of texts into fixed-size embeddings. """
        with torch.no_grad():
            encoded_input = tokenizer(
                text_list,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(device)

            output = model(**encoded_input)

            # Extract embeddings safely
            if llm_type in ["emilyalsentzer/Bio_ClinicalBERT", "medicalai/ClinicalBERT"]:
                embeddings = output.pooler_output.detach() if hasattr(output,
                                                                      "pooler_output") else output.last_hidden_state[:,
                                                                                            0, :].detach()

            elif llm_type in ["microsoft/biogpt", "yikuan8/Clinical-Longformer"]:
                hidden_states = output.hidden_states[1]  # Use second hidden state
                embeddings = pool_layer(hidden_states.permute(0, 2, 1)).squeeze(-1).detach()
            elif llm_type in ["ruslanmv/Medical-Llama3-8B"]:
                hidden_states = output.hidden_states[-1]  # Last layer hidden states
                embeddings = hidden_states.mean(dim=1).squeeze().detach()

            # Move to CPU to free GPU memory
            return embeddings.cpu()

    # Encode texts in batches
    for i in tqdm(range(0, len(df), batch_size)):
        batch_texts = df[note_type].iloc[i: i + batch_size].tolist()
        batch_embeddings = encode_batch(batch_texts)

        # Convert to numpy and store in CPU memory
        all_embeddings.append(batch_embeddings.numpy())

        # Free GPU memory
        del batch_embeddings
        torch.cuda.empty_cache()

    # Stack all embeddings into a single array
    embeddings = np.vstack(all_embeddings)

    # Save embeddings
    np.save(save_path, embeddings)

    print(f"Precomputed text embeddings saved as {embedding_filename}!")

    # Delete model to release GPU memory
    del model
    torch.cuda.empty_cache()


# Define LLM models and settings
llm_types = [
    # ("medicalai/ClinicalBERT", 512),
    # ("emilyalsentzer/Bio_ClinicalBERT", 512),
    # ("microsoft/biogpt", 512),
    # ("yikuan8/Clinical-Longformer", 4096),
    ("ruslanmv/Medical-Llama3-8B", 4096),
]
batch_size = 64  # Adjust batch size based on GPU memory
note_types = ["full"]

# Run embedding extraction
for llm_type, max_length in llm_types:
    for note_type in note_types:
        if llm_type in ["microsoft/biogpt", "ruslanmv/Medical-Llama3-8B"] and note_type == "full":
            batch_size = 1
        if note_type == "full" and llm_type not in ["yikuan8/Clinical-Longformer", "ruslanmv/Medical-Llama3-8B"]:
            continue
        extract_embeddings(llm_type, note_type, max_length, batch_size)

