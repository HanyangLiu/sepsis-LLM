from os.path import join

import pandas as pd
import torch
import torch.utils
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupShuffleSplit
import lightning as L
from utils.utils_data import load_data, RandomizedGroupKFold
import numpy as np
import pickle
from paths import processed_data_path, comorb_vocab_size, remote_root, ID
from transformers import AutoModel, AutoTokenizer
from transformers import BioGptTokenizer, BioGptForCausalLM
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os


class myDataset(Dataset):
    def __init__(self,
                 indices,
                 all_data,
                 max_codes,
                 static_size,
                 ts_size,
                 note_type,
                 llm_type,
                 modalities,
                 use_precomputed=False,
                 embedding_dir=None):
        self.static, self.codes, self.timeseries, self.note, self.labels, self.label_mask = all_data
        self.indices = indices
        self.static_size = static_size
        self.ts_size = ts_size
        self.comorb_size = (max_codes,)
        self.note_type = note_type
        self.llm_type = llm_type
        self.modalities = modalities
        self.use_precomputed = use_precomputed

        # Create a mapping from (PID, AID, infection_id) to global sample ID
        self.global_id_mapping = {
            tuple(row): idx for idx, row in
            enumerate(self.indices[["PID", "AID", "infection_id"]].itertuples(index=False))
        }

        # Load precomputed text embeddings if required
        if self.use_precomputed:
            embedding_filename = f"note_embeddings_{note_type}_{llm_type.replace('/', '_')}.npy"
            self.embeddings = np.load(os.path.join(embedding_dir, embedding_filename), allow_pickle=True)
            self.embedding_size = self.embeddings.shape[1]
        else:
            # Initialize tokenizer only if we are processing raw text
            tokenizer_mapping = {
                "microsoft/biogpt": BioGptTokenizer,
                "yikuan8/Clinical-Longformer": AutoTokenizer,
                "emilyalsentzer/Bio_ClinicalBERT": AutoTokenizer,
                "medicalai/ClinicalBERT": AutoTokenizer,
            }
            if llm_type not in tokenizer_mapping:
                raise ValueError(f"Invalid LLM type: {llm_type}")
            self.tokenizer = tokenizer_mapping[llm_type].from_pretrained(llm_type)
            self.embedding_size = 768
            self.max_len = 4096 if llm_type == "microsoft/biogpt" else 512  # Set max token length

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.indices.iloc[idx]
        PID, AID, infection_id = row[ID['PID']], row[ID['AID']], row.infection_id
        modality_mask = np.zeros((4,))
        sample = {}

        # Generate Global Sample ID
        global_sample_id = self.global_id_mapping.get((PID, AID, infection_id), -1)
        sample["sample_id"] = torch.tensor(global_sample_id, dtype=torch.long)

        # Static
        if self.modalities[0]:
            if (PID, AID, infection_id) in self.static.index:
                static = self.static.loc[(PID, AID, infection_id)].astype(float).values
                modality_mask[0] = 1
            else:
                static = np.zeros(self.static_size)

            sample["static"] = torch.tensor(static, dtype=torch.float32)

        # Longitudinal
        if self.modalities[1]:
            X, X_interv, X_filled, X_mask = self.timeseries
            if (PID, AID, infection_id) in X:
                ts = X[(PID, AID, infection_id)]
                ts_interv = X_interv[(PID, AID, infection_id)]
                ts_filled = X_filled[(PID, AID, infection_id)]
                ts_mask = X_mask[(PID, AID, infection_id)]
                modality_mask[1] = 1
            else:
                ts = np.zeros(self.ts_size)
                ts_interv = np.zeros(self.ts_size)
                ts_filled = np.zeros(self.ts_size)
                ts_mask = np.zeros(self.ts_size)

            sample["ts"] = {
                "X": torch.tensor(ts, dtype=torch.float32),
                "mask": torch.tensor(ts_mask, dtype=torch.float32),
                "delta": torch.tensor(ts_interv, dtype=torch.float32),
                "mean": torch.tensor(np.zeros(self.ts_size[1], dtype=np.float32)),
                "X_LOCF": torch.tensor(ts_filled, dtype=torch.float32),
            }

        # Comorbidity
        if self.modalities[2]:
            modality_mask[2] = 1  # Comorbidity is always present unless set to all zeros
            if PID in self.codes:
                comorb = self.codes[PID]
                if np.sum(comorb) == 0:
                    modality_mask[2] = 0
            else:
                comorb = np.full(self.comorb_size, comorb_vocab_size + 1, dtype=int)
            sample["comorb"] = torch.tensor(comorb, dtype=torch.long)

        # Notes Processing (Precomputed or Tokenized)
        if self.modalities[3]:
            if self.use_precomputed:
                self.note_index_mapping = {
                    (PID, AID): i for i, (PID, AID) in enumerate(self.note.index)
                }
                if (PID, AID) in self.note_index_mapping:
                    emb_idx = self.note_index_mapping[(PID, AID)]
                    note_embedding = self.embeddings[emb_idx]  # Fetch from precomputed embeddings
                    modality_mask[3] = 1
                else:
                    note_embedding = np.zeros((self.embedding_size,))  # Assuming 768-d embeddings

                sample["note"] = torch.tensor(note_embedding, dtype=torch.float32)
            else:
                note_text = self.note.loc[(PID, AID), "note_text"] if (PID, AID) in self.note.index else "No note"
                modality_mask[3] = (PID, AID) in self.note.index
                text_encoding = self.tokenizer(
                    note_text,
                    max_length=self.max_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt"
                )
                sample["note"] = {
                    "input_ids": text_encoding["input_ids"].squeeze(),
                    "attention_mask": text_encoding["attention_mask"].squeeze(),
                }

        # Labels
        label = self.labels.loc[(PID, AID, infection_id)].values
        label_mask = self.label_mask.loc[(PID, AID, infection_id)].values
        sample["label"] = torch.tensor(label, dtype=torch.float32)
        sample["label_mask"] = torch.tensor(label_mask, dtype=torch.bool)
        sample["modality_mask"] = torch.tensor(modality_mask[self.modalities], dtype=torch.bool)

        return sample


class sepsisDataModule(L.LightningDataModule):
    def __init__(self,
                 max_codes=200,
                 batch_size=128,
                 pin_memory=False,
                 num_workers=1,
                 multiclass=False,
                 task="AMR",
                 use_unlabeled=False,
                 note_type="full",
                 infection_type="all",
                 all_patients=False,
                 llm_type="emilyalsentzer/Bio_ClinicalBERT",
                 use_precomputed=False,
                 missing_rate=0.0,
                 **kwargs):
        super().__init__()
        self.max_codes = max_codes
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.data_dir = processed_data_path
        self.multiclass = multiclass
        self.task = task
        self.use_unlabeled = use_unlabeled
        self.note_type = note_type
        self.infection_type = infection_type
        self.all_patients = all_patients
        self.llm_type = llm_type
        self.modalities = kwargs.pop("modalities", [True, True, True, True])
        self.use_precomputed = use_precomputed
        self.missing_rate = missing_rate
        self.missing_on_test = True

    def prepare_labels(self):
        """ Prepares labels based on task type. """
        labels = self.labels_raw.set_index([ID['PID'], ID['AID'], 'infection_id'])

        if self.task == "GNB":
            labels = labels[['GNB', 'UN']].rename(columns={'GNB': 'label'})
        elif self.task == "AMR":
            if self.multiclass:
                labels['label'] = np.select(
                    [labels.SS, labels.RS, labels.RR, labels.UN],
                    [0, 1, 2, 0],  # Multiclass labels
                    default=0
                )
            else:
                labels['label'] = (~labels['SS']).astype(int)
                labels.loc[labels['UN'], 'label'] = 0  # Set unknowns to 0
        else:
            raise ValueError("Please specify task: GNB or AMR.")

        return labels[['label']].astype(int)

    def prepare_static(self):
        return pd.read_csv(join(self.data_dir, 'deep_static.csv')).set_index([ID['PID'], ID['AID'], 'infection_id'])

    def prepare_comorb(self):
        """ Loads comorbidity data and applies missingness if required. """
        with open(join(self.data_dir, f'deep_comorb_codes_{self.max_codes}.pickle'), 'rb') as f:
            comorb_data = pickle.load(f)

        if self.missing_rate > 0:
            # Randomly select patients to have missing comorbidities
            patient_ids = list(comorb_data.keys())
            if not self.missing_on_test:
                patient_ids = [pid for pid in patient_ids if pid in self.idx_train[ID['PID']].values]
            num_missing = int(len(patient_ids) * self.missing_rate)
            missing_patients = np.random.choice(patient_ids, size=num_missing, replace=False)

            # Replace comorbidities with zeros
            for pid in missing_patients:
                comorb_data[pid] = np.zeros((self.max_codes,), dtype=int)

            print(f"🔹 Applied missingness: {num_missing}/{len(patient_ids)} ({self.missing_rate * 100:.1f}%) patients now have missing comorbidity data.")

        return comorb_data

    def prepare_notes(self):
        notes = pd.read_csv(join(remote_root, "cohort_3_new", "data_combined", "deep_notes.csv"))

        if self.missing_rate > 0:
            # Randomly select patients to have missing comorbidities
            patient_ids = list(notes[ID['PID']].unique())
            patient_ids = [pid for pid in patient_ids if pid in self.idx_train[ID['PID']].values]
            num_missing = int(len(patient_ids) * self.missing_rate)
            missing_patients = np.random.choice(patient_ids, size=num_missing, replace=False)
            notes = notes[~notes[ID['PID']].isin(missing_patients)]

            print(f"🔹 Applied missingness: {num_missing}/{len(patient_ids)} ({self.missing_rate * 100:.1f}%) patients now have missing note data.")

        return notes.rename(columns={self.note_type: "note_text"}).set_index([ID['PID'], "AID"])

    def prepare_timeseries(self):
        ts_files = ['deep_timeseries_4H_normalized.pickle',
                    'deep_timeseries_4H_interv.pickle',
                    'deep_timeseries_4H_filled.pickle',
                    'deep_timeseries_4H_mask.pickle']
        return [pickle.load(open(join(self.data_dir, f), 'rb')) for f in ts_files]

    def setup(self, stage=None):
        """ Loads and prepares datasets for training, validation, and testing. """
        self.labels_raw = pd.read_csv(join(self.data_dir, 'df_label_full.csv'))

        # Keep only patients with notes if `all_patients=False`
        notes = pd.read_csv(join(remote_root, "cohort_3_new", "data_combined", "deep_notes.csv")).rename(columns={self.note_type: "note_text"}).set_index([ID['PID'], "AID"])
        labels_other = self.labels_raw[~self.labels_raw["AID"].isin(notes.reset_index()["AID"])]
        self.labels_raw = self.labels_raw[self.labels_raw["AID"].isin(notes.reset_index()["AID"])]

        # Filter by infection type
        if self.infection_type == "community":
            self.labels_raw = self.labels_raw[self.labels_raw["infection_id"] == 0]
        elif self.infection_type == "hospital":
            self.labels_raw = self.labels_raw[self.labels_raw["infection_id"] != 0]

        if not self.use_unlabeled:
            self.labels_raw = self.labels_raw[~self.labels_raw.UN]  # Exclude unknown labels

        # Train/Valid/Test Split
        self.indices_labeled = self.labels_raw[~self.labels_raw['UN']][[ID['PID'], ID['AID'], 'infection_id']]
        cv = RandomizedGroupKFold(groups=self.indices_labeled[ID['AID']].to_numpy(),
                                  n_splits=5,
                                  random_state=42)
        train_val_ix, test_ix = cv[0]
        self.idx_train_val, self.idx_test = self.indices_labeled.iloc[train_val_ix], self.indices_labeled.iloc[test_ix]

        gss = GroupShuffleSplit(n_splits=2, test_size=1 / 8, random_state=42)
        splits = gss.split(self.idx_train_val, groups=self.idx_train_val[ID['AID']])
        train_ix, valid_ix = next(splits)
        self.idx_train, self.idx_valid = self.idx_train_val.iloc[train_ix], self.idx_train_val.iloc[valid_ix]

        if self.use_unlabeled:
            self.idx_train = pd.concat([
                self.idx_train,
                self.labels_raw[self.labels_raw['UN']][[ID['PID'], ID['AID'], 'infection_id']]
            ], axis=0)
        if self.all_patients:
            self.idx_train = pd.concat([
                self.idx_train,
                labels_other[[ID['PID'], ID['AID'], 'infection_id']]
            ], axis=0)
            self.labels_raw = pd.concat([self.labels_raw, labels_other], axis=0)

        labels = self.prepare_labels()
        data_static = self.prepare_static()
        data_comorb = self.prepare_comorb()
        data_timeseries = self.prepare_timeseries()
        data_notes = self.prepare_notes()

        # Store indices of patient admissions
        label_mask = ~self.labels_raw.set_index([ID['PID'], ID['AID'], 'infection_id'])[["UN"]]
        self.all_data = [data_static, data_comorb, data_timeseries, data_notes, labels, label_mask]

        # Store data dimensions
        self.size_static = (len(data_static.columns),)
        self.size_timeseries = np.shape(next(iter(data_timeseries[0].values())))

    def get_dataloader(self, indices, shuffle=False):
        return DataLoader(
            myDataset(indices, self.all_data, self.max_codes, self.size_static, self.size_timeseries,
                      note_type=self.note_type, llm_type=self.llm_type, modalities=self.modalities,
                      use_precomputed=self.use_precomputed, embedding_dir=self.data_dir),
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=shuffle
        )

    def train_dataloader(self):
        return self.get_dataloader(self.idx_train, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(self.idx_valid, shuffle=False)

    def test_dataloader(self, idx_test=None, shuffle=False):
        return self.get_dataloader(idx_test if idx_test is not None else self.idx_test, shuffle=shuffle)
