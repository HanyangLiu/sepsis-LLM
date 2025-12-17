import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import datetime
from tqdm import tqdm
from sklearn.impute import SimpleImputer
import os


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


def process_comorb_binary():
    print("Processing comorbidity into binary features...")
    # process comorbidities
    raw_labels = pd.read_csv(join(processed_data_path, 'df_label_full.csv'))
    raw_comorb = pd.read_csv(join(raw_data_path, '{}_diagnoses_for_comorbidities.csv'.format(prefix)), low_memory=False)
    raw_comorb['diagnosis_code'] = raw_comorb['ICDX_DIAGNOSIS_CODE'].astype(str)
    raw_comorb['diagnosis_code_fam'] = raw_comorb['ICDX_DIAGNOSIS_CODE'].astype(str).apply(lambda x: x.split('.')[0])
    comorb = raw_comorb[['reference_no', 'diagnosis_code_fam', 'diagnosis_code']]
    # comorb.to_csv(join(save_dir, 'data_comorb_raw.csv'), index=False)

    df_comorb_fam = pd.pivot_table(comorb[['reference_no', 'diagnosis_code_fam']],
                                   index=['reference_no'],
                                   columns='diagnosis_code_fam',
                                   aggfunc=len,
                                   fill_value=0).astype(bool).astype(int).reset_index()
    df_comorb_fam = (
        raw_labels.iloc[:, :2].merge(df_comorb_fam, how='inner', left_on='person_id', right_on='reference_no')
            .drop(columns=['reference_no']))
    df_comorb_fam.to_csv(join(processed_data_path, 'data_comorb_fam.csv'), index=False)
    sdf_comorb_fam = df_comorb_fam.astype(pd.SparseDtype("int", 0))
    sdf_comorb_fam.to_pickle(join(processed_data_path, 'data_comorb_fam.pickle'))

    df_comorb = pd.pivot_table(comorb[['reference_no', 'diagnosis_code']],
                               index=['reference_no'],
                               columns='diagnosis_code',
                               aggfunc=len,
                               fill_value=0).astype(bool).astype(int).reset_index()
    df_comorb = (raw_labels.iloc[:, :2].merge(df_comorb, how='inner', left_on='person_id', right_on='reference_no')
                 .drop(columns=['reference_no']))
    df_comorb.to_csv(join(processed_data_path, 'data_comorb.csv'), index=False)
    sdf_comorb = df_comorb.astype(pd.SparseDtype("int", 0))
    sdf_comorb.to_pickle(join(processed_data_path, 'data_comorb.pickle'))


def process_vitals_simple():
    print("Processing vitals...")
    # process vitals
    data = pd.read_csv(join(raw_data_path, '{}_vitals.csv'.format(prefix)))
    labels = pd.read_csv(join(processed_data_path, 'df_label_full.csv'))
    data['meas_date'] = pd.to_datetime(data['meas_date'])
    data = (data[['person_id', 'visit_occurrence_id', 'meas_date', 'concept_name', 'measurement']]
            .sort_values(by=['person_id', 'visit_occurrence_id', 'concept_name', 'meas_date']))
    data[['person_id', 'visit_occurrence_id', 'concept_name']] = data[['person_id', 'visit_occurrence_id', 'concept_name']].astype(
        str)
    data = data.drop(index=data[data['measurement'] == ' '].index).reset_index()
    data['measurement'] = data['measurement'].astype(float)
    if resample_window:
        data = (data.groupby(['person_id', 'visit_occurrence_id', 'concept_name'])
                .resample(rule=resample_window, on='meas_date')
                .mean(numeric_only=True).drop(columns=['index']).reset_index())
    data = (data.pivot_table(index=['person_id', 'visit_occurrence_id', 'meas_date'],
                             columns='concept_name', values='measurement').add_prefix('vital_').reset_index())
    data.to_csv(join(processed_data_path, 'data_vitals.csv'), index=False)


    # extract last values before each infections
    processed_data = data
    time_column = 'meas_date'
    data_out = pd.DataFrame()
    for infection_id in tqdm(labels['infection_id'].unique()):
        col_dates = labels[labels['infection_id'] == infection_id]
        id2date = dict(zip(col_dates['visit_occurrence_id'], col_dates['collection_date']))
        data_table = processed_data.copy()
        data_table['collection_date'] = data_table['visit_occurrence_id'].apply(
            lambda x: id2date[int(x)] if int(x) in id2date else np.nan)
        data_table[time_column] = pd.to_datetime(data_table[time_column])
        data_table['collection_date'] = pd.to_datetime(data_table['collection_date'])
        # last value
        data_table = data_table[data_table[time_column] <= data_table['collection_date']]
        data_last = data_table.groupby(['person_id', 'visit_occurrence_id']).last().drop(
            columns=['collection_date', time_column]).reset_index()
        # max/min within 24h before
        data_table = data_table[np.logical_and(data_table[time_column] <= data_table['collection_date'],
                                               data_table[time_column] > data_table[
                                                   'collection_date'] - datetime.timedelta(
                                                   days=1))]
        data_min = data_table.groupby(['person_id', 'visit_occurrence_id']).min().drop(
            columns=['collection_date', time_column]).reset_index()
        data_max = data_table.groupby(['person_id', 'visit_occurrence_id']).max().drop(
            columns=['collection_date', time_column]).reset_index()
        data_table = data_min.merge(data_max.iloc[:, 1:], on='visit_occurrence_id', suffixes=['_min', '_max'])
        data_table = data_table.merge(data_last.iloc[:, 1:], on='visit_occurrence_id')
        data_table['admission_instance'] = data_table['visit_occurrence_id'].astype(str) + '-' + str(infection_id)
        data_out = pd.concat([data_out, data_table], axis=0)

    data_out = data_out.sort_values(['person_id', 'visit_occurrence_id', 'admission_instance'])
    data_cols = processed_data.columns[3:].tolist()
    data_cols = data_cols + [ele + '_min' for ele in data_cols] + [ele + '_max' for ele in data_cols]
    data_out = data_out[['person_id', 'visit_occurrence_id', 'admission_instance'] + data_cols]
    data_out.to_csv(join(processed_data_path, 'data_last_min_max_vitals_by_instance.csv'), index=False)


def process_labs_simple():
    print("Processing labs...")
    # process labs
    data = pd.read_csv(join(raw_data_path, '{}_laboratory.csv'.format(prefix)))
    labels = pd.read_csv(join(processed_data_path, 'df_label_full.csv'))
    data = data[~data['concept_name'].isna()]
    data['person_id'] = data['person_id'].astype(int)

    data['collection_tmstp'] = pd.to_datetime(data['collection_tmstp'])
    data['result'] = data['result_value'].astype('str').str.extract('([0-9][,.]*[0-9]*)').astype(float)
    data = data[~data['result'].isna()]
    data = (data[['person_id', 'visit_occurrence_id', 'collection_tmstp', 'concept_name', 'result']]
            .sort_values(by=['person_id', 'visit_occurrence_id', 'concept_name', 'collection_tmstp']))
    data[['person_id', 'visit_occurrence_id', 'concept_name']] = data[['person_id', 'visit_occurrence_id', 'concept_name']].astype(str)
    if resample_window:
        data = (data.groupby(['person_id', 'visit_occurrence_id', 'concept_name'])
                .resample(rule=resample_window, on='collection_tmstp')
                .mean(numeric_only=True).reset_index())
    data = (data.pivot_table(index=['person_id', 'visit_occurrence_id', 'collection_tmstp'],
                             columns='concept_name', values='result').add_prefix('lab_').reset_index())
    data.to_csv(join(processed_data_path, 'data_labs.csv'), index=False)


    # extract last values before each infections
    processed_data = data
    time_column = 'collection_tmstp'
    data_out = pd.DataFrame()
    for infection_id in tqdm(labels['infection_id'].unique()):
        col_dates = labels[labels['infection_id'] == infection_id]
        id2date = dict(zip(col_dates['visit_occurrence_id'], col_dates['collection_date']))
        data_table = processed_data.copy()
        data_table['collection_date'] = data_table['visit_occurrence_id'].apply(
            lambda x: id2date[int(x)] if int(x) in id2date else np.nan)
        data_table[time_column] = pd.to_datetime(data_table[time_column])
        data_table['collection_date'] = pd.to_datetime(data_table['collection_date'])
        # last value
        data_table = data_table[data_table[time_column] <= data_table['collection_date']]
        data_last = data_table.groupby(['person_id', 'visit_occurrence_id']).last().drop(
            columns=['collection_date', time_column]).reset_index()
        # max/min within 24h before
        data_table = data_table[np.logical_and(data_table[time_column] <= data_table['collection_date'],
                                               data_table[time_column] > data_table[
                                                   'collection_date'] - datetime.timedelta(
                                                   days=1))]
        data_min = data_table.groupby(['person_id', 'visit_occurrence_id']).min().drop(
            columns=['collection_date', time_column]).reset_index()
        data_max = data_table.groupby(['person_id', 'visit_occurrence_id']).max().drop(
            columns=['collection_date', time_column]).reset_index()
        data_table = data_min.merge(data_max.iloc[:, 1:], on='visit_occurrence_id', suffixes=['_min', '_max'])
        data_table = data_table.merge(data_last.iloc[:, 1:], on='visit_occurrence_id')
        data_table['admission_instance'] = data_table['visit_occurrence_id'].astype(str) + '-' + str(infection_id)
        data_out = pd.concat([data_out, data_table], axis=0)

    data_out = data_out.sort_values(['person_id', 'visit_occurrence_id', 'admission_instance'])
    data_cols = processed_data.columns[3:].tolist()
    data_cols = data_cols + [ele + '_min' for ele in data_cols] + [ele + '_max' for ele in data_cols]
    data_out = data_out[['person_id', 'visit_occurrence_id', 'admission_instance'] + data_cols]
    data_out.to_csv(join(processed_data_path, 'data_last_min_max_labs_by_instance.csv'), index=False)


if __name__ == "__main__":
    # paths and global variables
    resample_window = '4H'  # None or '1H'
    time_window = 5  # days

    # process_vitals_simple()
    process_labs_simple()
    process_comorb_binary()
