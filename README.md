# Sepsis-LLM: Multimodal Deep Learning for Antimicrobial Resistance Prediction

A comprehensive machine learning framework for predicting antimicrobial resistance (AMR) in sepsis patients using multimodal clinical data and Large Language Models (LLMs).

## 🏥 Overview

This project implements state-of-the-art multimodal deep learning models to predict antimicrobial resistance patterns in sepsis patients by integrating:

- **Static Data**: Demographics, admission details, comorbidities
- **Time-Series Data**: Vital signs and laboratory values over time  
- **Clinical Notes**: Free-text clinical documentation processed with medical LLMs
- **Comorbidity Codes**: ICD-10 diagnosis codes with learned embeddings

## 🎯 Key Features

- **Multimodal Architecture**: Integrates diverse clinical data modalities using attention mechanisms
- **Clinical LLM Integration**: Supports multiple medical language models (BioGPT, Clinical-Longformer, Bio_ClinicalBERT)
- **Missing Data Handling**: Robust handling of missing modalities common in clinical data
- **Comprehensive Evaluation**: SHAP explanations, subgroup analysis, and performance visualization
- **Production Ready**: PyTorch Lightning framework with automated logging and checkpointing

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch lightning transformers scikit-learn pandas numpy
pip install catboost xgboost shap matplotlib seaborn
```

### Basic Usage

```bash
# Train the multimodal model with default configuration
python run.py --config configs/agg_mm.yaml --task AMR

# Run traditional ML baselines
python simple_ml.py

# Generate SHAP explanations
python analysis/evaluate_llm_shap.py
```

## 📊 Model Architecture

### AggMM (Aggregated Multimodal Model)

The core model (`AggMM`) uses a modular architecture:

```
┌─── Static Encoder (FFN) ────┐
├─── Time-Series Encoder ────┤──► Fusion Layer ──► Prediction Head
├─── Comorbidity Encoder ────┤    (Concatenation)     (Binary/Multi-class)
└─── Clinical LLM Encoder ───┘
```

**Key Components:**
- **Static Encoder**: Feed-forward network for demographic/admission data
- **Time-Series Encoder**: GRU-based recurrent network for vital signs/labs
- **Comorbidity Encoder**: Transformer with pretrained ICD-10 embeddings
- **Text Encoder**: Clinical LLM (BioGPT, Clinical-Longformer, etc.) for clinical notes

## 🗂️ Project Structure

```
sepsis-LLM/
├── configs/                    # Model configurations
│   └── agg_mm.yaml            # AggMM model config
├── models/                    # Model implementations
│   ├── _base_mm.py           # Base multimodal class
│   ├── agg_mm.py             # Aggregated multimodal model
│   ├── _text_encoder.py      # LLM integration
│   └── _modules.py           # Neural network modules
├── analysis/                  # Analysis and evaluation tools
│   ├── explain_llm.py        # LLM interpretability
│   ├── evaluate_llm_shap.py  # SHAP explanations
│   ├── missing_modality.py   # Robustness analysis
│   └── process_cohort_3_new/ # Data preprocessing
├── utils/                     # Utility functions
│   ├── utils_data.py         # Data loading utilities
│   ├── utils_evaluation.py   # Evaluation metrics
│   └── ontology/             # Medical ontologies
├── run.py                     # Main training script
├── experiment.py              # PyTorch Lightning experiment
├── dataset_2.py              # Data loading and preprocessing
├── simple_ml.py              # Traditional ML baselines
└── paths.py                  # Data path configurations
```

## ⚙️ Configuration

The model behavior is controlled through YAML configuration files. Key parameters:

```yaml
model_params:
  name: 'AggMM'
  embed_size: 128
  llm_type: "microsoft/biogpt"  # Clinical LLM to use
  modalities: [True, True, True, False]  # [static, timeseries, comorbidity, notes]

data_params:
  task: "AMR"                   # AMR prediction or GNB detection
  note_type: "hpi"             # Clinical note section
  batch_size: 128
  use_precomputed: True        # Use cached LLM embeddings

exp_params:
  LR: 0.0001
  patience: 2
  max_epochs: 100
```

## 📈 Supported Tasks

### 1. Antimicrobial Resistance (AMR) Prediction
- **Binary**: Resistant vs. Susceptible
- **Multiclass**: Susceptible (SS) / Intermediate (RS) / Resistant (RR)

### 2. Gram-Negative Bacteria (GNB) Detection
- Predict presence of gram-negative bacteria in cultures

## 🔬 Clinical LLM Support

The framework supports multiple clinical language models:

| Model | Description | Use Case |
|-------|-------------|----------|
| **BioGPT** | Biomedical generative model | Long clinical narratives |
| **Clinical-Longformer** | Long-context clinical BERT | Extended clinical documents |
| **Bio_ClinicalBERT** | Clinical domain BERT | Standard clinical notes |
| **ClinicalBERT** | Alternative clinical BERT | General clinical text |

## 📊 Evaluation and Analysis

### Model Performance
```bash
# Comprehensive model evaluation
python analysis/evaluate_hospital_subgroups.py

# Performance visualization
python analysis/llm_performance_heatmap.py
```

### Interpretability
```bash
# SHAP-based model explanations
python analysis/evaluate_llm_shap.py --model AggMM --version 182

# LLM attention visualization
python analysis/explain_llm.py
```

### Robustness Analysis
```bash
# Missing modality analysis
python analysis/missing_modality.py

# Data distribution analysis
python analysis/analyze_data_distribution.py
```

## 📋 Data Requirements

The framework expects processed clinical data in the following format:

- **Labels**: `df_label_full.csv` - AMR labels and metadata
- **Static**: `deep_static.csv` - Patient demographics and admission details
- **Time-series**: Pickle files with vital signs and lab values over time
- **Comorbidities**: `deep_comorb_codes_*.pickle` - ICD-10 diagnosis codes
- **Notes**: `deep_notes.csv` - Clinical documentation text
- **Embeddings**: `icd10_embeddings_128.csv` - Pretrained ICD-10 embeddings

## 🎛️ Advanced Usage

### Custom Model Development
```python
from models._base_mm import BaseMM

class CustomModel(BaseMM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Custom architecture here
    
    def forward(self, batch, **kwargs):
        # Custom forward pass
        pass
```

### Custom Data Processing
```python
from dataset_2 import sepsisDataModule

# Custom data module with different preprocessing
dm = sepsisDataModule(
    max_codes=300,
    batch_size=64,
    note_type="assessment",  # Different note section
    infection_type="community",  # Specific infection type
    llm_type="emilyalsentzer/Bio_ClinicalBERT"
)
```

## 📚 Key Publications & References

This work builds upon research in:
- Clinical language models for healthcare applications
- Multimodal fusion techniques for electronic health records
- Antimicrobial resistance prediction in sepsis patients
- Interpretable machine learning in clinical decision support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-model`)
3. Commit your changes (`git commit -am 'Add new model architecture'`)
4. Push to the branch (`git push origin feature/new-model`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions or issues:
1. Check the existing [Issues](../../issues)
2. Review the documentation in the `analysis/` directory
3. Contact the development team

## 🔧 Troubleshooting

### Common Issues

**CUDA out of memory:**
```yaml
# Reduce batch size in config
data_params:
  batch_size: 32  # Reduce from 128
```

**Missing LLM models:**
```bash
# The models will be automatically downloaded from Hugging Face
# Ensure you have sufficient disk space and internet connection
```

**Data path errors:**
```python
# Update paths.py with your data location
remote_root = "/your/data/path/"
```

---

**Built with ❤️ for advancing clinical AI and improving patient outcomes**
