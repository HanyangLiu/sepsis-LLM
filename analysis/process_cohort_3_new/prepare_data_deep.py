import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import datetime
import os
from joblib import Parallel, delayed
import collections
import pickle
from sklearn import preprocessing
from os.path import join

import multiprocessing
import numpy as np
from utils.utils_graph import GraphEmb
import os
import sys
pd.Timestamp('today')


project_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
project_name = os.path.basename(project_path)
server_name = os.uname()[1]
remote_root = "/data/hanyang/sepsis/"
sys.path.append(join(remote_root, "sepsis_multimodal"))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


prefix = "cohort3"
raw_data_path = os.path.join(remote_root, "cohort_3_new")
manual_data_path = os.path.join(remote_root, "manual_tables")
remote_project_path = os.path.join(remote_root, project_name)
processed_data_path = os.path.join(raw_data_path, "data_processed")
tmp_data_path = os.path.join(raw_data_path, "data_tmp")


def minmax_scale(df):
    for col in df.columns:
        df.loc[:, col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df


def standard_scale(df):
    for col in df.columns:
        df.loc[:, col] = (df[col] - df[col].mean()) / df[col].std()
    return df


def process_static():
    print("Processing static variables...")
    # load input
    labels_raw = pd.read_csv(join(processed_data_path, 'df_label_full.csv'))
    demo = pd.read_csv(join(processed_data_path, 'data_demographics.csv'))
    vasop = pd.read_csv(join(processed_data_path, 'data_last_vasop_by_instance.csv'))
    diagnoses = pd.read_csv(join(processed_data_path, 'data_diagnoses.csv'))
    proc = pd.read_csv(join(raw_data_path, '{}_procedures.csv'.format(prefix)))

    # initial table
    data_table = labels_raw[['person_id', 'visit_occurrence_id', 'infection_id']]
    data_table = data_table.assign(
        admission_instance=data_table['visit_occurrence_id'].astype(str) + '-' + data_table['infection_id'].astype(str))
    # add demographics
    data_table = data_table.merge(demo, how='left', on=['person_id', 'visit_occurrence_id'])
    # add vasopressors
    data_table = data_table.merge(vasop, how='left', on=['person_id', 'visit_occurrence_id', 'admission_instance']).fillna(
        value=0)

    # add intubation
    labels_raw['start_date'] = pd.to_datetime(labels_raw['admit_date']).dt.date
    labels_raw['end_date'] = pd.to_datetime(labels_raw['collection_date']).dt.date
    labels_raw['date_diff'] = (pd.to_datetime(labels_raw['end_date']) - pd.to_datetime(
        labels_raw['start_date'])) / np.timedelta64(1, 'D')

    intu_true = proc[proc.description.str.contains('Endotracheal', na=False)].visit_occurrence_id.unique().tolist()
    data_table['mechanical_ventilation'] = 0
    data_table.loc[data_table.visit_occurrence_id.isin(intu_true), 'mechanical_ventilation'] = 1

    # add history
    # sensitivity history
    labels_raw['resistant'] = np.logical_or(labels_raw['RS'], labels_raw['RR'])
    labels_raw['resistance_history'] = labels_raw.groupby('visit_occurrence_id')['resistant'].shift(1).cumsum().fillna(0)
    labels_raw['resistance_history'] = labels_raw['resistance_history'] > 0
    data_table['resistance_history'] = labels_raw['resistance_history']
    # re-admission (within 3 month)
    adm_hist = labels_raw.sort_values(['person_id', 'admit_date']).groupby(
        ['person_id', 'visit_occurrence_id']).first().reset_index()
    adm_hist['last_end_date'] = adm_hist.groupby('person_id')['end_date'].shift(1)
    adm_hist['readmission_interval'] = (pd.to_datetime(adm_hist['start_date']) - pd.to_datetime(
        adm_hist['last_end_date'])) / np.timedelta64(1, 'D')
    adm_hist['readmission'] = adm_hist['readmission_interval'].transform(
        lambda x: x <= 90 if not np.isnan(x) else False)
    readmission = adm_hist[['visit_occurrence_id', 'readmission']].set_index('visit_occurrence_id')
    data_table['readmission'] = data_table['visit_occurrence_id'].transform(lambda x: readmission.loc[x, 'readmission'])

    # add pneumonia diagnosis
    diagnoses = diagnoses.rename(columns={'instance_id': 'admission_instance'})
    data_table = data_table.merge(diagnoses[['admission_instance', 'pneumonia_community', 'pneumonia_acquired']],
                                  how='left', on='admission_instance')
    data_table[['pneumonia_community', 'pneumonia_acquired']] = data_table[
        ['pneumonia_community', 'pneumonia_acquired']].astype(int)

    # add time intervals
    data_table = data_table.merge(diagnoses[['admission_instance', 'time_since_admission']], how='left',
                                  on='admission_instance')

    # add infection type
    data_table["hospital_acquired"] = data_table["infection_id"].apply(lambda x: 1 if x > 0 else 0)
    data_table["infections"] = data_table["infection_id"]

    # data normalization
    normalizer = preprocessing.MinMaxScaler()
    data_table.iloc[:, 4:] = normalizer.fit_transform(data_table.iloc[:, 4:])
    data_table = data_table.drop(columns=['admission_instance'])

    # save output
    data_table.to_csv(os.path.join(processed_data_path, 'deep_static.csv'), index=False)


def organize_w_infection(data_table, time_column, collection_dates, time_window):
    organized_data = []
    for _, row in tqdm(collection_dates.iterrows()):
        date = pd.to_datetime(row.collection_date)
        data = data_table[data_table.visit_occurrence_id == row.visit_occurrence_id]
        data = data[np.logical_and(pd.to_datetime(data[time_column]).dt.tz_localize(None) < date.to_datetime64(),
                                   pd.to_datetime(data[time_column]).dt.tz_localize(None) >= (date - datetime.timedelta(time_window)).to_datetime64())]
        data['infection_id'] = row.infection_id
        organized_data.append(data)

    organized_data = pd.concat(organized_data, axis=0)
    organized_data = organized_data[
        ['person_id', 'visit_occurrence_id', 'infection_id', time_column] + data_table.columns.tolist()[3:]]
    return organized_data


def process_vitals():
    print("Processing vitals...")
    def process_parallel(adm_ids, data_all, resample='1H'):
        data = data_all.loc[data_all.visit_occurrence_id.isin(adm_ids)]
        data_pivot = (data.pivot_table(index=['person_id', 'visit_occurrence_id', 'meas_date'],
                                       columns='concept_name', values='measurement').add_prefix('vital_').reset_index())
        if resample:
            data_pivot = (data_pivot.groupby(['person_id', 'visit_occurrence_id'])
                          .resample(resample, on='meas_date')
                          .mean(numeric_only=False).drop(columns=['person_id', 'visit_occurrence_id']).reset_index())

        return data_pivot

    # # process vitals
    vital_match = pd.read_csv(join(remote_root, "sepsis_multimodal", "process_cohort_3_new", "vitals_match.csv"))
    vital_selected = vital_match["Selected (new)"].unique().tolist()
    print("Loading raw data file...")
    vitals = pd.read_csv(os.path.join(raw_data_path, '{}_vitals.csv'.format(prefix)))
    print("Loaded.")
    vitals["concept_name"] = vitals["concept_name"].apply(lambda x: x.lower())
    vitals = vitals[vitals["concept_name"].isin(vital_selected)]
    vitals = vitals[['person_id', 'visit_occurrence_id', 'meas_date', 'concept_name', 'measurement']]

    vitals[['person_id', 'visit_occurrence_id']] = vitals[['person_id', 'visit_occurrence_id']].astype(str)
    vitals = vitals.drop(index=vitals[vitals['measurement'] == ' '].index).reset_index()
    vitals['measurement'] = vitals['measurement'].astype(float)
    vitals['meas_date'] = pd.to_datetime(vitals['meas_date'])

    # multiprocessing
    num_cores = multiprocessing.cpu_count() - 2
    adm_ids = vitals.visit_occurrence_id.unique()
    block_size = 200
    adm_ids_list = [adm_ids[i: i + block_size] for i in range(0, len(adm_ids), block_size)]
    data_pivot_list = Parallel(n_jobs=num_cores, timeout=9999)(
        delayed(process_parallel)(adm_ids, vitals, resample_window) for adm_ids in tqdm(adm_ids_list))
    data_pivot = pd.concat(data_pivot_list, axis=0).sort_values(by=['person_id', 'visit_occurrence_id', 'meas_date'])
    data_pivot.to_csv(os.path.join(processed_data_path, 'deep_vitals_22var_{}.csv'.format(resample_window)), index=False)

    # organize w.r.t. infection time
    data_pivot = pd.read_csv(os.path.join(processed_data_path, 'deep_vitals_22var_{}.csv'.format(resample_window)))
    collection_dates = pd.read_csv(os.path.join(processed_data_path, 'df_label_full.csv'))[
        ['person_id', 'visit_occurrence_id', 'infection_id', 'collection_date']]
    organized_data = organize_w_infection(data_pivot, 'meas_date', collection_dates, time_window)
    organized_data = organized_data.rename(columns={'meas_date': 'time'})
    organized_data.to_csv(os.path.join(processed_data_path, 'deep_vitals_22var_{}_organized.csv'.format(resample_window)),
                          index=False)


def process_labs():
    print("Processing labs...")
    def process_parallel(adm_ids, data_all, resample='1H'):
        data = data_all.loc[data_all.visit_occurrence_id.isin(adm_ids)]
        data_pivot = (data.pivot_table(index=['person_id', 'visit_occurrence_id', 'collection_tmstp'],
                                       columns='concept_name', values='result').add_prefix('lab_').reset_index())
        if resample:
            data_pivot = (data_pivot.groupby(['person_id', 'visit_occurrence_id'])
                          .resample(resample, on='collection_tmstp')
                          .mean(numeric_only=False).drop(columns=['person_id', 'visit_occurrence_id']).reset_index())
        return data_pivot

    # process labs
    labs_match = pd.read_csv(join(remote_root, "sepsis_multimodal", "process_cohort_3_new", "labs_match.csv"))
    labs_selected = labs_match["New"].unique().tolist()
    
    print("Loading raw data file...")
    labs = pd.read_csv(os.path.join(raw_data_path, '{}_laboratory.csv'.format(prefix)), low_memory=False)
    print("Loaded.")
    labs["concept_name"] = labs["concept_name"].apply(lambda x: x.lower())
    labs = labs[labs["concept_name"].isin(labs_selected)]

    labs = labs[~labs['concept_name'].isna()]
    labs['person_id'] = labs['person_id'].astype(int)

    labs['result'] = labs['result_value'].astype('str').str.extract('([0-9][,.]*[0-9]*)').astype(float)
    labs = labs[~labs['result'].isna()]
    labs = labs[['person_id', 'visit_occurrence_id', 'collection_tmstp', 'concept_name', 'result']]
    labs[['person_id', 'visit_occurrence_id', 'concept_name']] = labs[['person_id', 'visit_occurrence_id', 'concept_name']].astype(str)
    labs['collection_tmstp'] = pd.to_datetime(labs['collection_tmstp'])

    # multiprocessing
    num_cores = multiprocessing.cpu_count() - 2
    adm_ids = labs.visit_occurrence_id.unique()
    block_size = 200
    adm_ids_list = [adm_ids[i: i + block_size] for i in range(0, len(adm_ids), block_size)]
    data_pivot_list = Parallel(n_jobs=num_cores, timeout=9999)(
        delayed(process_parallel)(adm_ids, labs, resample_window) for adm_ids in tqdm(adm_ids_list))
    data_pivot = pd.concat(data_pivot_list, axis=0).sort_values(by=['person_id', 'visit_occurrence_id', 'collection_tmstp'])
    data_pivot.to_csv(os.path.join(processed_data_path, 'deep_labs_44var_{}.csv'.format(resample_window)), index=False)

    # organize w.r.t. infection time
    data_pivot = pd.read_csv(os.path.join(processed_data_path, 'deep_labs_44var_{}.csv'.format(resample_window)))
    collection_dates = pd.read_csv(os.path.join(processed_data_path, 'df_label_full.csv'))[
        ['person_id', 'visit_occurrence_id', 'infection_id', 'collection_date']]
    organized_data = organize_w_infection(data_pivot, 'collection_tmstp', collection_dates, time_window)
    organized_data = organized_data.rename(columns={'collection_tmstp': 'time'})
    organized_data.to_csv(os.path.join(processed_data_path, 'deep_labs_44var_{}_organized.csv'.format(resample_window)),
                          index=False)


def process_procedure():
    data_raw = pd.read_csv(join(raw_data_path, "{}_procedures.csv".format(prefix))).sort_values(
        by=["visit_occurrence_id", "PROCEDURE_DATE"]).reset_index()
    labels = pd.read_csv(join(processed_data_path, 'df_label_full.csv'))

    time_column = 'PROCEDURE_DATE'
    data_out = pd.DataFrame()
    for infection_id in tqdm(labels['infection_id'].unique()):
        col_dates = labels[labels['infection_id'] == infection_id]
        id2date = dict(zip(col_dates['visit_occurrence_id'], col_dates['collection_date']))
        data = data_raw.copy()
        data['collection_date'] = data['visit_occurrence_id'].apply(lambda x: id2date[int(x)] if int(x) in id2date else np.nan)
        data[time_column] = pd.to_datetime(data[time_column])
        data['collection_date'] = pd.to_datetime(data['collection_date'])
        data = data[data[time_column] <= data['collection_date'] - datetime.timedelta(days=1)]
        data['infection_id'] = infection_id
        data_out = pd.concat([data_out, data], axis=0)

    data_out = data_out.sort_values(['person_id', 'visit_occurrence_id', 'infection_id', time_column]).set_index(
        ["person_id", "visit_occurrence_id", "infection_id"])

    code2idx = dict(zip(data_out["procedure_code"].unique(), range(1, data_out["procedure_code"].nunique() + 1)))
    data_out["code_idx"] = data_out["procedure_code"].apply(lambda x: code2idx[x])
    data_out = data_out.drop(columns=["index"])
    procedure_codes = dict()
    for ind in tqdm(data_out.index.unique()):
        procedure_codes[ind] = data_out.loc[ind]["code_idx"].unique()

    with open(os.path.join(processed_data_path, 'deep_procedure_codes.pickle'), 'wb') as f:
        pickle.dump(procedure_codes, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # paths and global variables
    resample_window = '4H'  # None or '1H'
    time_window = 5  # days

    process_static()
    process_vitals()
    process_labs()
    process_procedure()
