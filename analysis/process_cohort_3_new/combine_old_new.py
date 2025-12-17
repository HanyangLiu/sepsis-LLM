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


project_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
project_name = os.path.basename(project_path)
server_name = os.uname()[1]
remote_root = "/data/hanyang/sepsis/"


prefix = "cohort3"
raw_data_path = os.path.join(remote_root, "cohort_3_new")
manual_data_path = os.path.join(remote_root, "manual_tables")
remote_project_path = os.path.join(remote_root, project_name)
tmp_data_path = os.path.join(raw_data_path, "data_tmp")

old_data_path = os.path.join(remote_root, "cohort_3", "data_processed")
new_data_path = os.path.join(raw_data_path, "data_processed")
combined_data_path = os.path.join(raw_data_path, "data_combined")
resample_window = "4H"
time_window = 5  # days


def combine_static():
    # combine static
    static_old = pd.read_csv(join(old_data_path, "deep_static.csv"))
    static_new = pd.read_csv(join(new_data_path, "deep_static.csv"))

    vasop_vars = [name for name in static_old.columns if "vasop" in name]
    static_old = static_old.assign(vasop_history=static_old[vasop_vars].sum(axis=1).astype(bool).astype(int))

    static = combine_dfs(static_old, static_new, crosswalk, N_idx=3)
    static.to_csv(os.path.join(combined_data_path, 'deep_static.csv'), index=False)


def combine_labels():
    label_old = pd.read_csv(join(old_data_path, "df_label_full.csv"))
    label_new = pd.read_csv(join(new_data_path, "df_label_full.csv"))
    label = combine_dfs(label_old, label_new, crosswalk, N_idx=3)
    label.to_csv(join(combined_data_path, "df_label_full.csv"), index=False)


def combine_dfs(df_old, df_new, crosswalk, N_idx=4):
    # ID mapping
    personID2patientID = dict(zip(crosswalk["person_id"], crosswalk["patient_id"]))
    visitID2admissionID = dict(zip(crosswalk["visit_occurrence_id"], crosswalk["admission_id"]))
    df_new["patient_id"] = df_new["person_id"].apply(
        lambda x: int(personID2patientID[x]) if x in personID2patientID else np.nan)
    df_new["admission_id"] = df_new["visit_occurrence_id"].apply(
        lambda x: int(visitID2admissionID[x]) if x in visitID2admissionID else np.nan)
    df_old = df_old[~df_old["admission_id"].isin(df_new["admission_id"].unique())]
    common_vars = sorted(list(set(df_new.columns[N_idx:]) & set(df_old.columns[N_idx:])))

    # new patient IDs and admission IDs
    df_old = df_old.assign(PID=df_old["patient_id"].transform(lambda x: "O" + str(x)))
    df_new = df_new.assign(PID=df_new["person_id"].transform(lambda x: "N" + str(x)))
    df_old = df_old.assign(AID=df_old["admission_id"].transform(lambda x: "O" + str(x)))
    df_new = df_new.assign(AID=df_new["visit_occurrence_id"].transform(lambda x: "N" + str(x)))

    # combine
    idx = ["PID", "AID", "infection_id", "time"]
    df_old = df_old[idx[:N_idx] + common_vars]
    df_new = df_new[idx[:N_idx] + common_vars]
    df_combined = pd.concat([df_old, df_new], axis=0).sort_values(by=idx[:N_idx]).reset_index(drop=True)

    return df_combined


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


def combine_vitals_labs():
    # combine vitals
    vitals_match = pd.read_csv(join(remote_root, "sepsis_multimodal", "process_cohort_3_new", "vitals_match.csv"))
    vitals_old = pd.read_csv(join(old_data_path, "deep_vitals_22var_4H_organized.csv"))
    vitals_new = pd.read_csv(join(new_data_path, "deep_vitals_22var_4H_organized.csv"))

    vitals_map = dict(zip(vitals_match["Selected (old)"], vitals_match["Selected (new)"]))
    var_names = ["vital_" + vitals_map[name[6:].lower()] for name in vitals_old.columns[4:]]
    vitals_old = vitals_old.rename(
        columns={old_name: new_name for old_name, new_name in zip(vitals_old.columns[4:], var_names)})

    vitals = combine_dfs(vitals_old, vitals_new, crosswalk, N_idx=4)
    vitals.to_csv(os.path.join(combined_data_path, 'deep_vitals_{}_organized.csv'.format(resample_window)), index=False)

    # combine labs
    labs_match = pd.read_csv(join(remote_root, "sepsis_multimodal", "process_cohort_3_new", "labs_match.csv"))
    labs_old = pd.read_csv(join(old_data_path, "deep_labs_44var_4H_organized.csv"))
    labs_new = pd.read_csv(join(new_data_path, "deep_labs_44var_4H_organized.csv"))

    labs_map = dict(zip(labs_match["Old"], labs_match["New"]))
    old_var_names = [name for name in labs_old.columns[4:] if name[4:].lower() in labs_map]
    var_names = ["lab_" + labs_map[name[4:].lower()] for name in old_var_names]
    labs_old = labs_old.rename(columns={old_name: new_name for old_name, new_name in zip(old_var_names, var_names)})

    labs = combine_dfs(labs_old, labs_new, crosswalk, N_idx=4)
    labs.to_csv(os.path.join(combined_data_path, 'deep_labs_{}_organized.csv'.format(resample_window)), index=False)


def concat_timeseries():
    def standard_scale(df):
        for col in df.columns:
            df.loc[:, col] = (df[col] - df[col].mean()) / df[col].std()
        return df


    print("Combining vitals and labs...")
    df_labs = pd.read_csv(join(combined_data_path, "deep_labs_{}_organized.csv".format(resample_window)))
    df_vitals = pd.read_csv(join(combined_data_path, "deep_vitals_{}_organized.csv".format(resample_window)))

    ratio_labs = df_labs.count() / len(df_labs)
    selected_labs = ratio_labs[ratio_labs > 0.01].index.tolist()

    ratio_vitals = df_vitals.count() / len(df_vitals)
    selected_vitals = ratio_vitals[ratio_vitals > 0.01].index.tolist()

    df_ts = pd.merge(df_vitals[selected_vitals], df_labs[selected_labs], how='outer',
                     on=['PID', 'AID', 'infection_id', 'time'])
    df_ts.to_csv(join(combined_data_path, 'deep_timeseries_organized.csv'), index=False)

    # normalize
    df_ts.iloc[:, 4:] = standard_scale(df_ts.iloc[:, 4:])
    df_ts.fillna(value=0).to_csv(
        os.path.join(combined_data_path, 'deep_timeseries_{}_normalized.csv'.format(resample_window)), index=False)

    # generate mask
    mask = df_ts.copy()
    mask.iloc[:, 4:] = mask.iloc[:, 4:].notnull().astype(int)
    mask.to_csv(os.path.join(combined_data_path, 'deep_timeseries_{}_mask.csv'.format(resample_window)), index=False)

    # generate intervals
    interv = df_ts.set_index(["PID", "AID", "infection_id"]).copy()
    for col in df_ts.columns[4:]:
        interv[col] = (
            interv[col].isnull().astype(int)
                .groupby([interv.index, interv[col].notnull().astype(int).cumsum()])
                .cumsum()
                .groupby(interv.index).shift(periods=1, fill_value=0).astype(int)
                .add(1)
        )
    interv = interv.reset_index()
    interv.iloc[:, 4:] = interv.iloc[:, 4:] / interv.iloc[:, 4:].max().max()
    interv.to_csv(os.path.join(combined_data_path, 'deep_timeseries_{}_interv.csv'.format(resample_window)), index=False)

    # impute missing data
    filled = df_ts.groupby(['PID', 'AID', 'infection_id'], group_keys=False).apply(
        lambda x: x.ffill().bfill().fillna(0)).reset_index(drop=True)
    filled.to_csv(join(combined_data_path, 'deep_timeseries_{}_filled.csv'.format(resample_window)), index=False)

    # convert to arrays
    def save_array(df, name=None):
        ts_arr = collections.defaultdict()
        for group_id, df_group in tqdm(df.groupby(['PID', 'AID', 'infection_id'])):
            ts = np.zeros((24 // int(resample_window[0]) * 5, len(df_ts.columns) - 4))
            ts[-len(df_group):, :] = df_group.iloc[:, 4:].values
            ts_arr[group_id] = ts

        with open(os.path.join(combined_data_path, 'deep_timeseries_{}_{}.pickle'.format(resample_window, name)), 'wb') as f:
            pickle.dump(ts_arr, f, protocol=pickle.HIGHEST_PROTOCOL)

    save_array(df_ts.fillna(value=0), "normalized")
    save_array(filled, "filled")
    save_array(interv, "interv")
    save_array(mask, "mask")


def process_comorb():
    print("Processing comorbidities and combine w/ old pull...")

    def standardize(code):
        if len(str(code)) > 3:
            if '.' not in str(code):
                code = str(code)[:3] + '.' + str(code)[3:]
        return code

    def convert_icd9_to_10(row):
        # code = standardize(row.icdx_diagnosis_code)
        code = row.icdx_diagnosis_code
        if str(code)[0].isdigit():
            if code in icd9to10_dict:
                code = icd9to10_dict[code]

        return code

    # load files
    raw_labels = pd.read_csv(os.path.join(combined_data_path, 'df_label_full.csv'))
    raw_comorb = pd.read_csv(os.path.join(raw_data_path, '{}_comorbidities.csv'.format(prefix)), low_memory=False)

    # graph embedding
    graph_emb = GraphEmb(embed_size=128)
    graph_emb.train_embedding()
    comorb_codes = graph_emb.codes

    # convert ICD9 to ICD10 codes
    icd9to10 = pd.read_csv(os.path.join(manual_data_path, 'icd9to10.csv'))
    icd9to10['icd9cm_standard'] = icd9to10['icd9cm'].apply(lambda x: standardize(x))
    icd9to10['icd10cm_standard'] = icd9to10['icd10cm'].apply(lambda x: standardize(x))
    icd9to10_dict = dict(zip(icd9to10['icd9cm_standard'].values, icd9to10['icd10cm_standard'].values))


    raw_comorb['diagnosis_code'] = raw_comorb.progress_apply(lambda row: convert_icd9_to_10(row), axis=1)
    comorb_new = raw_comorb[raw_comorb['diagnosis_code'].isin(comorb_codes)]
    # comorb_new = raw_comorb.drop(index=raw_comorb[raw_comorb.diagnosis_code == 'NOT FOUND'].index)
    comorb_new = comorb_new.sort_values(by=['person_id'])[['person_id', 'diagnosis_code']]


    # get family codes
    comorb_new['diagnosis_code_fam'] = comorb_new['diagnosis_code'].apply(lambda x: str(x).split('.')[0])
    comorb_new = comorb_new[['person_id', 'diagnosis_code', 'diagnosis_code_fam']]
    comorb_new['person_id'] = comorb_new['person_id'].astype(int)
    comorb_new = comorb_new.drop_duplicates().reset_index()


    # combine
    cohort_old_path = os.path.join(remote_root, "cohort_3", "data_processed")
    comorb_old = pd.read_csv(join(cohort_old_path, 'deep_comorb_raw.csv'))

    comorb_old = comorb_old.rename(columns={
        "reference_no": "patient_id",
    })

    df_old, df_new = comorb_old, comorb_new

    # ID mapping
    personID2patientID = dict(zip(crosswalk["person_id"], crosswalk["patient_id"]))
    df_new["patient_id"] = df_new["person_id"].apply(
        lambda x: int(personID2patientID[x]) if x in personID2patientID else np.nan)

    df_old = df_old[~df_old["patient_id"].isin(df_new["patient_id"].unique())]
    common_vars = ["diagnosis_code", "diagnosis_code_fam"]

    # new patient IDs and admission IDs
    df_old = df_old.assign(PID=df_old["patient_id"].transform(lambda x: "O" + str(x)))
    df_new = df_new.assign(PID=df_new["person_id"].transform(lambda x: "N" + str(x)))
    # combine
    df_old = df_old[["PID"] + common_vars]
    df_new = df_new[["PID"] + common_vars]
    df_combined = pd.concat([df_old, df_new], axis=0).sort_values(by="PID").reset_index(drop=True)

    comorb = df_combined
    comorb.to_csv(join(combined_data_path, "deep_comorb_raw.csv"), index=False)


    df_comorb_fam = pd.pivot_table(comorb[['PID', 'diagnosis_code_fam']],
                                   index=['PID'],
                                   columns='diagnosis_code_fam',
                                   aggfunc=len,
                                   fill_value=0).astype(bool).astype(int).reset_index()
    df_comorb_fam = (
        raw_labels.iloc[:, :2].merge(df_comorb_fam, how='inner', left_on='PID', right_on='PID')
            .drop(columns=['PID']))
    df_comorb_fam.to_csv(os.path.join(combined_data_path, 'deep_comorb_fam.csv'), index=False)

    df_comorb = pd.pivot_table(comorb[['PID', 'diagnosis_code']],
                               index=['PID'],
                               columns='diagnosis_code',
                               aggfunc=len,
                               fill_value=0).astype(bool).astype(int).reset_index()
    df_comorb = (raw_labels.iloc[:, :2].merge(df_comorb, how='inner', left_on='PID', right_on='PID')
                 .drop(columns=['PID']))
    df_comorb.to_csv(os.path.join(combined_data_path, 'deep_comorb_ori.csv'), index=False)


def save_comorb_to_array(maxlen=300):
    # covert into arrays
    comorb = pd.read_csv(os.path.join(combined_data_path, 'deep_comorb_raw.csv'))

    col = 'diagnosis_code'
    code_list = comorb.sort_values(col)[col].unique().tolist()
    code2id = dict(zip(code_list, range(1, len(code_list) + 1)))
    comorb['code_id'] = comorb[col].apply(lambda x: code2id[x])

    col = 'diagnosis_code_fam'
    code_list = comorb.sort_values(col)[col].unique().tolist()
    fam2id = dict(zip(code_list, range(1, len(code_list) + 1)))
    comorb['fam_id'] = comorb[col].apply(lambda x: fam2id[x])

    col = 'diagnosis_code_cat'
    comorb[col] = comorb['diagnosis_code'].apply(lambda x: str(x)[0])
    code_list = comorb.sort_values(col)[col].unique().tolist()
    cat2id = dict(zip(code_list, range(1, len(code_list) + 1)))
    comorb['cat_id'] = comorb[col].apply(lambda x: cat2id[x])

    codes_arr = collections.defaultdict()
    fam_arr = collections.defaultdict()
    cat_arr = collections.defaultdict()
    for pid in tqdm(comorb.PID.unique().tolist()):
        df = comorb[comorb.PID == pid]

        codes = np.array(df['code_id'].unique().tolist())
        tmp = np.zeros((maxlen,))
        tmp[: min(len(codes), maxlen)] = codes[: maxlen]
        codes_arr[pid] = tmp

        fams = np.array(df['fam_id'].unique().tolist())
        tmp = np.zeros((maxlen,))
        tmp[: min(len(fams), maxlen)] = fams[: maxlen]
        fam_arr[pid] = tmp

        cats = np.array(df['cat_id'].unique().tolist())
        tmp = np.zeros((maxlen,))
        tmp[: min(len(cats), maxlen)] = cats[: maxlen]
        cat_arr[pid] = tmp

    with open(os.path.join(combined_data_path, 'deep_comorb_codes_{}.pickle'.format(maxlen)), 'wb') as f:
        pickle.dump(codes_arr, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(combined_data_path, 'deep_comorb_fams_{}.pickle'.format(maxlen)), 'wb') as f:
        pickle.dump(fam_arr, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(combined_data_path, 'deep_comorb_cats_{}.pickle'.format(maxlen)), 'wb') as f:
        pickle.dump(cat_arr, f, protocol=pickle.HIGHEST_PROTOCOL)


def graph_embedding(embed_size):
    print("Training graph embedding...")

    # Graph embedding
    comorb = pd.read_csv(os.path.join(combined_data_path, 'deep_comorb_raw.csv'))
    graph_emb = GraphEmb(embed_size=embed_size)
    graph_emb.train_embedding()

    code_list = comorb.sort_values('diagnosis_code').diagnosis_code.unique()
    comorb_codes = graph_emb.codes

    embeddings = []
    zero_vec = np.zeros((embed_size,))
    count = 0
    outliers = []
    for code in tqdm(code_list):
        if code in comorb_codes:
            embeddings.append(graph_emb.to_vec([code])[0])
        else:
            try:
                embeddings.append(graph_emb.to_vec([code[:5]])[0])
            except:
                count += 1
                embeddings.append(zero_vec)
                outliers.append(code)
    print(count)
    print(outliers)
    df_embeddings = pd.DataFrame(data=code_list, columns=['code'])
    df_embeddings = pd.concat([df_embeddings, pd.DataFrame(embeddings)], axis=1)
    df_embeddings.to_csv(os.path.join(combined_data_path, 'icd10_embeddings_{}.csv'.format(embed_size)), index=False)



if __name__ == "__main__":
    crosswalk = pd.read_csv(os.path.join(raw_data_path, "Crosswalk for patient_admissions_community File.csv"))
    # combine_labels()
    # combine_static()
    # combine_vitals_labs()
    # concat_timeseries()

    # combine comorbidity & train graph embedding
    # process_comorb()
    # save_comorb_to_array(maxlen=300)
    graph_embedding(embed_size=128)










