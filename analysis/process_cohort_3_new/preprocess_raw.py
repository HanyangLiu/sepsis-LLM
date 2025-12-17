import numpy as np
import pandas as pd
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


class dataImputation:
    def __init__(self):
        pass

    def mean_impute(self, data_df, column_name):
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        X = data_df[column_name].values
        imp.fit(X)
        imputed_array = imp.transform(X)
        data_df = data_df.copy()
        data_df.loc[:, column_name] = imputed_array
        return data_df

    def impute_static_dataframe(self, data_df, contin_feat, catego_feat):
        # mean imputation for continuous features
        data_df = self.mean_impute(data_df, contin_feat)
        # impute discrete features with a unique number
        for column in catego_feat:
            data_df[[column]] = data_df[[column]].fillna(value=np.max(data_df[column].unique()) + 1)
        # impute the rest with 0 padding
        data_df = data_df.fillna(value=0)
        return data_df


def process_static():
    print("Processing static variables...")
    ## Demographic data: demographic & comobidities
    raw_table = pd.read_csv(join(raw_data_path, '{}_demographics.csv'.format(prefix)),
                            low_memory=False,
                            dtype={
                                'hospital_id': str,
                                'race': str,
                                'gender': str
                            })
    raw_table['age_yrs'] = raw_table['age']
    raw_table = raw_table[['person_id', 'visit_occurrence_id', 'hospital_id', 'race', 'gender', 'age_yrs']]
    raw_table['gender'] = raw_table['gender'].fillna('U')
    raw_table['race'] = raw_table['race'].fillna(' ')

    hospital_mapping = {
        "17": "2572",
        "8": "2574",
        "11": "3049",
        "18": "3148",
        "793": "3269",
        "4": "4674",
        "7": "5107",
        "14": "5572",
        "9": "6729",
        "6": "160559",
        "10": "N10",
        "790": "N790",
        "791": "N791",
        "15": "N15",
    }
    gender_mapping = {
        "1": "M",
        "2": "F",
        "U": "U"
    }
    raw_table['hospital_id'] = raw_table['hospital_id'].apply(lambda x: hospital_mapping[x])
    raw_table['gender'] = raw_table['gender'].apply(lambda x: gender_mapping[x])


    # one-hot processing for categorical feats
    imputer = dataImputation()
    data_demographic = imputer.mean_impute(data_df=raw_table, column_name=['age_yrs'])
    data_demographic['hospital_id'] = data_demographic['hospital_id'].astype(str)
    one_hot = pd.get_dummies(data_demographic[['hospital_id', 'race', 'gender']].astype(str))
    processed_demo = raw_table[['person_id', 'visit_occurrence_id', 'age_yrs']].join(one_hot)
    processed_demo.to_csv(join(processed_data_path, 'data_demographics.csv'), index=False)


def process_comorb_binary():
    print("Processing comorbidity into binary features...")
    # process comorbidities
    raw_labels = pd.read_csv(join(processed_data_path, 'df_label_full.csv'))
    raw_comorb = pd.read_csv(join(raw_data_path, '{}_diagnoses_for_comorbidities.csv'.format(prefix)), low_memory=False)
    raw_comorb['diagnosis_code'] = raw_comorb['icdx_diagnosis_code'].astype(str)
    raw_comorb['diagnosis_code_fam'] = raw_comorb['icdx_diagnosis_code'].astype(str).apply(lambda x: x.split('.')[0])
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


def process_vasop():
    print("Processing vasopressors...")
    # process vasopressors
    time_column = 'order_start_date'
    data = pd.read_csv(join(raw_data_path, '{}_vasopressors.csv'.format(prefix)))
    labels = pd.read_csv(join(processed_data_path, 'df_label_full.csv'))

    # filter
    data = data[data['drug_route'].isin(['intravenous', 'IV'])]
    data['order_start_date'] = pd.to_datetime(data['order_start_date'], errors = 'coerce')
    data['order_stop_date'] = pd.to_datetime(data['order_stop_date'], errors = 'coerce')
    data['time_diff'] = data['order_stop_date'] - data['order_start_date']
    data = data[data['time_diff'] > pd.Timedelta(5, 'min')]

    # extract binary features
    data['mark'] = 1
    data[time_column] = pd.to_datetime(data[time_column])
    data = (data[['person_id', 'visit_occurrence_id', time_column, 'drug_id', 'mark']]
            .sort_values(by=['person_id', 'visit_occurrence_id', 'drug_id', time_column]))
    data[['person_id', 'visit_occurrence_id', 'drug_id']] = data[['person_id', 'visit_occurrence_id', 'drug_id']].astype(str)
    data = (data.pivot_table(index=['person_id', 'visit_occurrence_id', time_column],
                             columns='drug_id', values='mark').add_prefix('vasop_').reset_index())
    data.to_csv(join(processed_data_path, 'data_vasop_start.csv'), index=False)

    # extract last values before each infections
    processed_data = data
    data_out = pd.DataFrame()
    for infection_id in tqdm(labels['infection_id'].unique()):
        col_dates = labels[labels['infection_id'] == infection_id]
        id2date = dict(zip(col_dates['visit_occurrence_id'], col_dates['collection_date']))
        data = processed_data.copy()
        data['collection_date'] = data['visit_occurrence_id'].apply(lambda x: id2date[int(x)] if int(x) in id2date else np.nan)
        data[time_column] = pd.to_datetime(data[time_column])
        data['collection_date'] = pd.to_datetime(data['collection_date'])
        data = data[np.logical_and(data[time_column] <= data['collection_date'],
                                   data[time_column] > data['collection_date'] - datetime.timedelta(days=1))]
        data = data.groupby(['person_id', 'visit_occurrence_id']).last().drop(
            columns=['collection_date', time_column]).reset_index()
        data['admission_instance'] = data['visit_occurrence_id'].astype(str) + '-' + str(infection_id)
        data_out = pd.concat([data_out, data], axis=0)

    data_out = data_out.sort_values(['person_id', 'visit_occurrence_id', 'admission_instance']).fillna(0)
    data_out['vasop_history'] = (data_out[processed_data.columns[3:]].sum(axis=1) > 1).astype(int)
    data_out = data_out[['person_id', 'visit_occurrence_id', 'admission_instance', 'vasop_history']]
    data_out.to_csv(join(processed_data_path, 'data_last_vasop_by_instance.csv'), index=False)


def process_diagnoses():
    print("Processing diagnoses...")
    data = pd.read_csv(join(raw_data_path, '{}_diagnoses.csv'.format(prefix)))
    proc = pd.read_csv(join(raw_data_path, '{}_procedures.csv'.format(prefix)))
    labels_raw = pd.read_csv(join(processed_data_path, 'df_label.csv'))

    # extract code for pneumonia
    codes_t0 = ['J18.9', 'J13', 'J15.9']
    data_t0 = data[data.icdx_diagnosis_code.isin(codes_t0)]
    adm_id_t0 = data_t0.visit_occurrence_id.unique().tolist()

    adm_id_j189 = data[data.icdx_diagnosis_code == 'J18.9'].visit_occurrence_id.unique().tolist()
    data_j189 = data[data.visit_occurrence_id.isin(adm_id_j189)]
    data_hospital_required = data_j189[data_j189.icdx_diagnosis_code == 'Y95']
    adm_id_hospital_required = data_hospital_required.visit_occurrence_id.unique().tolist()

    # adm_id_intu = proc[proc.DESCRIPTION.str.contains('Endotracheal', na=False)].visit_occurrence_id.unique().tolist()
    adm_id_j95851 = data[data.icdx_diagnosis_code == 'J95.851'].visit_occurrence_id.unique().tolist()
    adm_id_vent_pneu = adm_id_j95851

    labels_raw['instance'] = labels_raw.instance_id.apply(lambda x: x.split('-')[1])
    labels_raw.loc[
        np.logical_and(labels_raw.instance == '0',
                       labels_raw.visit_occurrence_id.isin(adm_id_t0)), 'pneumonia_community'] = True
    labels_raw.loc[np.logical_and(labels_raw.instance != '0',
                                  labels_raw.visit_occurrence_id.isin(adm_id_hospital_required)), 'pneumonia_hospital'] = True
    labels_raw.loc[np.logical_and(labels_raw.instance != '0',
                                  labels_raw.visit_occurrence_id.isin(adm_id_vent_pneu)), 'pneumonia_ventilator'] = True
    labels_raw[['pneumonia_community', 'pneumonia_hospital', 'pneumonia_ventilator']] = labels_raw[
        ['pneumonia_community', 'pneumonia_hospital', 'pneumonia_ventilator']].fillna(value=False)
    labels_raw['pneumonia_acquired'] = labels_raw.apply(
        lambda row: row['pneumonia_hospital'] or row['pneumonia_ventilator'], axis=1)
    pneumonia_history = labels_raw.groupby('visit_occurrence_id').sum()['pneumonia_acquired']
    labels_raw['pneumonia_acquired'] = labels_raw['visit_occurrence_id'].apply(lambda x: pneumonia_history[x] > 0)

    labels_raw['time_since_admission'] = labels_raw.apply(
        lambda row: max(
            0,
            pd.Timedelta(
                pd.to_datetime(row.collection_date) - pd.to_datetime(row.admit_date)
            ).total_seconds() / 3600.0 / 24
        ),
        axis=1
    )
    labels_raw['time_since_admission'] = labels_raw['time_since_admission'].astype(int)
    data_diagnosis = labels_raw[['instance_id', 'pneumonia_community', 'pneumonia_acquired', 'time_since_admission']]
    data_diagnosis.to_csv(join(processed_data_path, 'data_diagnoses.csv'), index=False)


if __name__ == "__main__":
    # paths and global variables
    resample_window = '4H'  # None or '1H'
    time_window = 5  # days

    process_static()
    process_vasop()
    process_diagnoses()
