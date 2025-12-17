import os
import torch
import shap
import numpy as np
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    BioGptTokenizer,
    BioGptModel,
)
from utils.utils_evaluate_torch import load_model_and_trainer
import pickle


def run_shap_analysis(task="AMR", method="AggMM", version=182, num_samples=20, save_path=None):
    # === Device setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running SHAP analysis on device: {device}")

    # === Load model and config ===
    trainer, LitModel, dm, config = load_model_and_trainer(task, method, version)
    trainer.test(model=LitModel, dataloaders=dm.test_dataloader())

    LitModel = LitModel.to(device)
    base_model = LitModel.model.eval()

    llm_type = config["data_params"]["llm_type"]

    # === Load tokenizer + encoder ===
    if llm_type == "microsoft/biogpt":
        tokenizer = BioGptTokenizer.from_pretrained(llm_type)
        encoder = BioGptModel.from_pretrained(llm_type, output_hidden_states=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(llm_type)
        encoder = AutoModel.from_pretrained(llm_type, output_hidden_states=True)

    encoder = encoder.eval().to(device)
    pool_layer = nn.AdaptiveAvgPool1d(1)

    # === Sample raw notes ===
    test_dataset = dm.test_dataloader().dataset
    raw_texts = []
    for i in range(num_samples):
        idx = test_dataset.indices.iloc[i]
        PID, AID = idx["PID"], idx["AID"]
        try:
            note_text = test_dataset.note.loc[(PID, AID), "note_text"]
        except KeyError:
            note_text = "No note"
            print(f"[Warning] Note not found for ({PID}, {AID}), using placeholder.")
        raw_texts.append(note_text)

    # === SHAP model wrapper ===
    def model_forward(texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        texts = [str(t) for t in texts]

        tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = encoder(**tokens)

            if llm_type in ["emilyalsentzer/Bio_ClinicalBERT", "medicalai/ClinicalBERT"]:
                embeddings = outputs.pooler_output if hasattr(outputs, "pooler_output") \
                    else outputs.last_hidden_state[:, 0, :]

            elif llm_type in ["microsoft/biogpt", "yikuan8/Clinical-Longformer"]:
                hidden_states = outputs.hidden_states[1]
                embeddings = pool_layer(hidden_states.permute(0, 2, 1)).squeeze(-1)

            elif llm_type in ["ruslanmv/Medical-Llama3-8B"]:
                hidden_states = outputs.hidden_states[-1]
                embeddings = hidden_states.mean(dim=1)

            else:
                raise ValueError(f"Unsupported llm_type: {llm_type}")

            batch = {"note": embeddings}
            preds = base_model(batch)

            logits = preds[0] if isinstance(preds, tuple) else preds
            if logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            return logits.cpu().numpy()

    # === Run SHAP ===
    print("[INFO] Running SHAP Explainer...")
    explainer = shap.Explainer(model_forward, tokenizer)
    shap_values = explainer(raw_texts)

    # === Save SHAP values if desired ===
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(shap_values, f)
        print(f"[INFO] SHAP values saved to: {save_path}")

    # === Visualize ===
    print("\n🧠 SHAP Explanation for First Note:")
    shap.plots.text(shap_values[0])

    print("\n📊 Top Features Summary (Mean Abs SHAP):")
    shap.plots.bar(shap_values)


if __name__ == "__main__":
    run_shap_analysis(
        task="AMR",
        method="AggMM",
        version=182,
        num_samples=20,
        save_path="shap_values_note.pkl"
    )
