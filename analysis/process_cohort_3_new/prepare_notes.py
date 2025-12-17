import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import datetime
from tqdm import tqdm
from sklearn.impute import SimpleImputer
import os
import re
import unicodedata


project_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
project_name = os.path.basename(project_path)
server_name = os.uname()[1]
remote_root = "/data/hanyang/sepsis/"


prefix = "cohort3"
raw_data_path = os.path.join(remote_root, "cohort_3_new")
manual_data_path = os.path.join(remote_root, "manual_tables")
remote_project_path = os.path.join(remote_root, project_name)
processed_data_path = os.path.join(raw_data_path, "data_processed")
tmp_data_path = os.path.join(raw_data_path, "data_tmp")
combined_data_path = os.path.join(raw_data_path, "data_combined")


# load notes, crosswalk, and labels
raw_notes = pd.read_csv(join(raw_data_path, "{}_notes-2.csv".format(prefix)))
raw_notes["adm_date"] = pd.to_datetime(raw_notes["hosp_admsn_time"], format='%Y-%m-%dT%H:%M:%SZ').dt.date
raw_notes["mrn_date"] = raw_notes["pat_mrn_id"].astype(str) + "--" + raw_notes["adm_date"].astype(str)


# crosswalk of cohort 3 old
cw_old = pd.read_csv(join(raw_data_path, "cohort3old_crosswalk.csv"))
cw_old[["patient_id", "admission_id", "epic_mrn"]] = cw_old[["patient_id", "admission_id", "epic_mrn"]].astype("Int64")
cw_old["adm_date"] = pd.to_datetime(cw_old["admit_date"], format='%Y-%m-%dT%H:%M:%SZ').dt.date
cw_old = cw_old[~cw_old["epic_mrn"].isna()]
cw_old["mrn_date"] = cw_old["epic_mrn"].astype(str) + "--" + cw_old["adm_date"].astype(str)
pid2mrn_old = dict(zip(cw_old["patient_id"], cw_old["epic_mrn"]))
adm2mrnd_old = dict(zip(cw_old["admission_id"], cw_old["mrn_date"]))

# crosswalk of cohort 3 new
cw_new = pd.read_csv(join(raw_data_path, "cohort3new_crosswalk.csv"))
cw_new[["person_id", "visit_occurrence_id", "epic_mrn"]] = cw_new[["person_id", "visit_occurrence_id", "epic_mrn"]].astype("Int64")
cw_new["adm_date"] = pd.to_datetime(cw_new["admit_date"], format='%Y-%m-%dT%H:%M:%SZ').dt.date
cw_new = cw_new[~cw_new["epic_mrn"].isna()]
cw_new["mrn_date"] = cw_new["epic_mrn"].astype(str) + "--" + cw_new["adm_date"].astype(str)
pid2mrn_new = dict(zip(cw_new["person_id"], cw_new["epic_mrn"]))
adm2mrnd_new = dict(zip(cw_new["visit_occurrence_id"], cw_new["mrn_date"]))

# Overall crosswalk
crosswalk = pd.merge(
    left=cw_new[["person_id", "visit_occurrence_id", "epic_mrn", "mrn_date"]].drop_duplicates(),
    right=cw_old[["patient_id", "admission_id", "epic_mrn", "mrn_date"]].drop_duplicates(),
    how="outer",
    on=["epic_mrn", "mrn_date"]
)


# crosswalk = pd.read_csv(join(raw_data_path, "cohort3_updated_crosswalk.csv"))
# crosswalk = crosswalk[["patient_id", "admission_id", "person_id", "visit_occurrence_id", "epic_mrn", "admit_date"]].drop_duplicates()
# crosswalk[["patient_id", "admission_id", "person_id", "visit_occurrence_id", "epic_mrn"]] = crosswalk[["patient_id", "admission_id", "person_id", "visit_occurrence_id", "epic_mrn"]].astype("Int64")
# crosswalk["adm_date"] = pd.to_datetime(crosswalk["admit_date"], format='%Y-%m-%dT%H:%M:%SZ').dt.date
# crosswalk["mrn_date"] = crosswalk["epic_mrn"].astype(str) + "--" + crosswalk["adm_date"].astype(str)
#
# # Use MRN and admission date to relate across multiple files
# perid2mrn = dict(zip(crosswalk["person_id"], crosswalk["epic_mrn"]))
# patid2mrn = dict(zip(crosswalk["patient_id"], crosswalk["epic_mrn"]))
# occ2mrnd = dict(zip(crosswalk["visit_occurrence_id"], crosswalk["mrn_date"]))
# adm2mrnd = dict(zip(crosswalk["admission_id"], crosswalk["mrn_date"]))


# Add IDs to labels
labels = pd.read_csv(join(combined_data_path, "df_label_full.csv"))
labels["new_pull"] = labels["PID"].apply(lambda x: x[0] == "N")
labels["person_id"] = labels["PID"].apply(lambda x: int(x[1:]) if x[0] == "N" else 0)
labels["patient_id"] = labels["PID"].apply(lambda x: int(x[1:]) if x[0] == "O" else 0)
labels["visit_occurrence_id"] = labels["AID"].apply(lambda x: int(x[1:]) if x[0] == "N" else 0)
labels["admission_id"] = labels["AID"].apply(lambda x: int(x[1:]) if x[0] == "O" else 0)

labels["epic_mrn"] = labels["person_id"].apply(lambda x: pid2mrn_new[x] if x in pid2mrn_new else 0)
labels.loc[labels["epic_mrn"] == 0, "epic_mrn"] = labels.loc[labels["epic_mrn"] == 0, "patient_id"].apply(lambda x: pid2mrn_old[x] if x in pid2mrn_old else 0)
labels["mrn_date"] = labels["visit_occurrence_id"].apply(lambda x: adm2mrnd_new[x] if x in adm2mrnd_new else 0)
labels.loc[labels["mrn_date"] == 0, "mrn_date"] = labels.loc[labels["mrn_date"] == 0, "admission_id"].apply(lambda x: adm2mrnd_old[x] if x in adm2mrnd_old else 0)
labels["epic_mrn"] = labels["mrn_date"].apply(lambda x: x.split("--")[0] if x !=0 else 0)



# process notes -- overlaped with crosswalk
notes = pd.merge(
    left=raw_notes,
    right=crosswalk.drop_duplicates(),
    how="left",
    on="mrn_date"
).drop_duplicates()

notes = notes[~notes["note_text"].isna()]
notes_raw = raw_notes[~raw_notes["note_text"].isna()]


# Overlap
print(len(set(notes["admission_id"].unique()) & set(labels["admission_id"])))
print(len(set(notes["visit_occurrence_id"].unique()) & set(labels["visit_occurrence_id"])))
print(len(set(notes["mrn_date"].unique()) & set(crosswalk["mrn_date"])))
print(len(set(notes["mrn_date"].unique()) & set(labels["mrn_date"])))


# Concatenate notes for the same admission
notes = notes.sort_values(by="mrn_date")
notes_cat = notes.groupby("mrn_date").first()
notes_cat["note_text"] = notes.groupby("mrn_date")["note_text"].agg(func=" ".join)
notes_cat = notes_cat.reset_index()



def extract_hpi(note):
    # Find the start and end indices of the Discharge Instructions section
    start = max(
        note.find("HPI"),
        note.find("History Present Illness"),
        note.find("History of the Present Illness"),
        note.find("Subjective"),
        note.find("SUBJECTIVE"),
    )

    end = max(
        note.find("PMH"),
        note.find("Past Medical History"),
        note.find("Objective"),
        note.find("OBJECTIVE")
    )

    if end == -1:
        end = max(
            note.find("Assessment"),
            note.find("ASSESSMENT"),
            note.find("Plan"),
            note.find("PLAN"),
            note.find("Impression"),
            note.find("IMPRESSION")
        )

    # If both indices are found, extract the section
    if start != -1 and end != -1:
        output = note[start: end].strip()
    elif start != -1 and end == -1:
        output = note[start:].strip()
    else:
        return "No HPI note."

    # Clean up the text for NLP tasks
    output = output.replace("_", " ").replace("\n", " ").strip()
    output = output.split()[:1024]
    output = " ".join(output)

    return output


def extract_assessment(note):
    # Find the start and end indices of the Discharge Instructions section
    start = max(
        note.find("Assessment"),
        note.find("ASSESSMENT"),
        note.find("Plan"),
        note.find("PLAN"),
        note.find("Impression"),
        note.find("IMPRESSION")
    )

    # If indices are found, extract the section
    if start != -1:
        output = note[start:].strip()
    else:
        return "No assessment note."

    # Clean up the text for NLP tasks
    output = output.replace("_", " ").replace("\n", " ").strip()
    output = output.split()[:1024]
    output = " ".join(output)

    return output


def clean_text_for_bert(text, lowercase=True, remove_special_chars=False, expand_contractions=False):
    """
    Cleans text to prepare it for tokenization with a BERT-based model.
    """

    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Convert to lowercase if needed
    if lowercase:
        text = text.lower()

    # Expand contractions if enabled
    if expand_contractions:
        contractions = {
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "can't": "cannot", "couldn't": "could not", "won't": "will not",
            "wouldn't": "would not", "shouldn't": "should not",
            "it's": "it is", "that's": "that is", "what's": "what is",
            "i'm": "i am", "he's": "he is", "she's": "she is", "we're": "we are",
            "they're": "they are", "you're": "you are", "isn't": "is not",
            "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        }
        for contraction, expanded in contractions.items():
            text = re.sub(rf"\b{re.escape(contraction)}\b", expanded, text)

    # Remove special characters if enabled
    if remove_special_chars:
        text = re.sub(r"[^a-zA-Z0-9.,!?'\s]", "", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


tqdm.pandas()

# Get HPI notes
notes_cat = notes_cat[['mrn_date', 'note_attested_yn', 'note_text', 'person_id', 'visit_occurrence_id', 'epic_mrn', 'patient_id', 'admission_id']]
notes_cat["hpi"] = notes_cat.note_text.apply(extract_hpi)
notes_cat.loc[notes_cat.hpi == "", "hpi"] = "No HPI note."
notes_cat["hpi"] = notes_cat.hpi.progress_apply(clean_text_for_bert)
notes_cat["hpi_len"] = notes_cat.hpi.apply(lambda x: len(x.split()) if isinstance(x, str) else 0)

# Get assessment notes
notes_cat["assessment"] = notes_cat.note_text.apply(extract_assessment)
notes_cat.loc[notes_cat.assessment == "", "assessment"] = "No assessment note."
notes_cat["assessment"] = notes_cat.assessment.progress_apply(clean_text_for_bert)
notes_cat["assessment_len"] = notes_cat.assessment.apply(lambda x: len(x.replace("_", " ").replace("\n", " ").strip().split()) if isinstance(x, str) else 0)

# Save the first 1024 words for full notes
notes_cat["note_text"] = notes_cat["note_text"].apply(lambda x: " ".join(x.split()[: 4096]))
notes_cat = notes_cat.rename(columns={"note_text": "full"})
notes_cat["full"] = notes_cat.full.progress_apply(clean_text_for_bert)

notes_cat.to_csv(join(combined_data_path, "deep_notes_all.csv"), index=False)


notes_co3 = pd.merge(
    left=labels[["PID", "AID", "new_pull", "mrn_date"]].drop_duplicates(),
    right=notes_cat,
    how="inner",
    on="mrn_date"
).sort_values(["PID", "AID"]).reset_index(drop=True)
notes_co3.to_csv(join(combined_data_path, "deep_notes.csv"), index=False)


