import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data from the final version of the community infection table
data = [
    # HPI
    ["HPI", "BioClinicalBERT", 0.6685, 0.1579, 0.8516, 0.3984],
    ["HPI", "ClinicalBERT", 0.6314, 0.1406, 0.8515, 0.4001],
    ["HPI", "BioGPT", 0.6671, 0.1667, 0.8520, 0.3984],
    ["HPI", "Clinical-Longformer", 0.6328, 0.1474, 0.8498, 0.3922],
    ["HPI", "Medical-LLaMa3-8B", 0.6268, 0.1475, 0.8488, 0.3872],
    # Assessment
    ["Assessment", "BioClinicalBERT", 0.5881, 0.1254, 0.8499, 0.3955],
    ["Assessment", "ClinicalBERT", 0.6153, 0.1442, 0.8514, 0.3993],
    ["Assessment", "BioGPT", 0.6861, 0.1786, 0.8521, 0.4103],
    ["Assessment", "Clinical-Longformer", 0.6693, 0.1733, 0.8498, 0.3969],
    ["Assessment", "Medical-LLaMa3-8B", 0.6172, 0.1418, 0.8500, 0.3883],
    # Full H&P
    ["Full H&P", "Clinical-Longformer", 0.6803, 0.1759, 0.8494, 0.3974],
    ["Full H&P", "Medical-LLaMa3-8B", 0.7348, 0.2051, 0.8509, 0.3908],
]

df = pd.DataFrame(data, columns=["NoteType", "Model", "AUROC_Note", "AUPRC_Note", "AUROC_StructNote", "AUPRC_StructNote"])

# Desired note type order
note_order = ["HPI", "Assessment", "Full H&P"]

# Pivot and organize AUROC heatmap data
auroc_note = df.pivot(index="Model", columns="NoteType", values="AUROC_Note")
auroc_struct = df.pivot(index="Model", columns="NoteType", values="AUROC_StructNote")
auroc_note.columns = [f"{col} (Note Only)" for col in auroc_note.columns]
auroc_struct.columns = [f"{col} (Combined)" for col in auroc_struct.columns]
auroc_all = pd.concat([auroc_note, auroc_struct], axis=1)
auroc_all = auroc_all[[f"{n} (Note Only)" for n in note_order] + [f"{n} (Combined)" for n in note_order]]

# Pivot and organize AUPRC heatmap data
auprc_note = df.pivot(index="Model", columns="NoteType", values="AUPRC_Note")
auprc_struct = df.pivot(index="Model", columns="NoteType", values="AUPRC_StructNote")
auprc_note.columns = [f"{col} (Note Only)" for col in auprc_note.columns]
auprc_struct.columns = [f"{col} (Combined)" for col in auprc_struct.columns]
auprc_all = pd.concat([auprc_note, auprc_struct], axis=1)
auprc_all = auprc_all[[f"{n} (Note Only)" for n in note_order] + [f"{n} (Combined)" for n in note_order]]

# Formatting x-axis labels
def format_xtick_labels(labels):
    base = []
    method = []
    for label in labels:
        note_type = label.split(" ")[0] + " " + label.split(" ")[1] if "Full" in label else label.split(" ")[0]
        base.append(note_type)
        method.append("Note Only" if "Note Only" in label else "Combined")
    return base, method

# Create AUROC heatmap
auroc_base, auroc_method = format_xtick_labels(auroc_all.columns)
plt.figure(figsize=(12, 6))
ax = sns.heatmap(auroc_all, annot=True, cmap="YlGnBu", fmt=".3f", vmin=0.6, vmax=0.9,
                 cbar_kws={'label': 'AUROC'})
ax.set_xticklabels([f"{b}\n({m})" for b, m in zip(auroc_base, auroc_method)], rotation=0)
plt.title("Combined Heatmap: AUROC - Note Only vs Combined")
plt.tight_layout()
plt.show()

# Create AUPRC heatmap
auprc_base, auprc_method = format_xtick_labels(auprc_all.columns)
plt.figure(figsize=(12, 6))
ax = sns.heatmap(auprc_all, annot=True, cmap="BuPu", fmt=".3f", vmin=0.1, vmax=0.5,
                 cbar_kws={'label': 'AUPRC'})
ax.set_xticklabels([f"{b}\n({m})" for b, m in zip(auprc_base, auprc_method)], rotation=0)
plt.title("Combined Heatmap: AUPRC - Note Only vs Combined")
plt.tight_layout()
plt.show()
