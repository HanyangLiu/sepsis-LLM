import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
from os.path import join
import io
from paths import processed_data_path, comorb_vocab_size, remote_root, ID


def load_data(full_comorb=False):
    # initial table
    labels_raw = pd.read_csv(join(processed_data_path, 'df_label_full.csv'))
    data_table = labels_raw[[ID['PID'], ID['AID'], 'infection_id', 'SS', 'RS', 'RR', 'UN', 'GNB']]

    # load static features
    static = pd.read_csv(join(processed_data_path, 'deep_static.csv'))
    data_table = data_table.merge(static, how='left', on=[ID['PID'], ID['AID'], 'infection_id'])

    # load time series
    ts = pd.read_csv(join(processed_data_path, 'deep_timeseries_organized.csv'))
    ts_table = ts.groupby([ID['PID'], ID['AID'], 'infection_id'])[ts.columns[4:]].mean().reset_index()
    data_table = data_table.merge(ts_table, how='left', on=[ID['PID'], ID['AID'], 'infection_id']).fillna(0)

    # load comorbidities
    if full_comorb:
        comorb = pd.read_csv(join(processed_data_path, 'deep_comorb_ori.csv'))
    else:
        comorb = pd.read_csv(join(processed_data_path, 'deep_comorb_fam.csv'))
    data_table = data_table.merge(comorb.drop_duplicates(), how='left', on=[ID['AID']]).fillna(0)

    # data normalization
    normalizer = preprocessing.MinMaxScaler()
    tmp = normalizer.fit_transform(data_table.iloc[:, 8:])
    data_table.iloc[:, 8:] = tmp

    return data_table.set_index([ID['PID'], ID['AID'], 'infection_id'])


def RandomizedGroupKFold(groups, n_splits, random_state=None):  # noqa: N802
    """
    Random analogous of sklearn.model_selection.GroupKFold.split.
    :return: list of (train, test) indices
    """
    groups = pd.Series(groups)
    ix = np.arange(len(groups))
    unique = np.unique(groups)
    np.random.RandomState(random_state).shuffle(unique)
    result = []
    for split in np.array_split(unique, n_splits):
        mask = groups.isin(split)
        train, test = ix[~mask], ix[mask]
        result.append((train, test))

    return result


def select_subgroup(df_data, group='1'):
    if group == '0':
        return df_data[[ID['PID'], ID['AID'], 'infection_id']]  # all
    elif group == '1':
        return df_data[df_data.age_yrs >= 65 / 121.0][[ID['PID'], ID['AID'], 'infection_id']]
    elif group == '2':
        return df_data[df_data.age_yrs < 65 / 121.0][[ID['PID'], ID['AID'], 'infection_id']]
    elif group == '3':
        J15 = df_data.filter(like='J15', axis=1).sum(axis=1)
        return df_data[J15 > 0][[ID['PID'], ID['AID'], 'infection_id']]
    elif group == '4':
        A41 = df_data.filter(like='A41', axis=1).sum(axis=1)
        return df_data[A41 > 0][[ID['PID'], ID['AID'], 'infection_id']]
    elif group == '5':
        return df_data[df_data.B96 > 0][[ID['PID'], ID['AID'], 'infection_id']]
    elif group == '6':
        return df_data[df_data.Z16 > 0][[ID['PID'], ID['AID'], 'infection_id']]
    elif group == '7':
        cols = [col for col in df_data.columns if 'A41' in col or 'B96' in col or 'J15' in col or 'Z16' in col]
        if_selected = ~df_data[cols].sum(axis=1).astype(bool)
        return df_data[if_selected][[ID['PID'], ID['AID'], 'infection_id']]
    elif group == '8':
        # any comorbidities in C81-C96
        if_selected = (df_data.filter(regex='C8').sum(axis=1) + df_data.filter(regex='C9').sum(axis=1)).astype(bool)
        return df_data[if_selected][[ID['PID'], ID['AID'], 'infection_id']]
    elif group == '9':
        return df_data[df_data.filter(like='Z94', axis=1).sum(axis=1) > 0][[ID['PID'], ID['AID'], 'infection_id']]
    elif group == '10':
        comorb = pd.read_csv('../cohort_3/cohort3_diagnoses_for_comorbidities.csv', low_memory=False)
        pids = comorb[comorb.ICDX_DIAGNOSIS_CODE.str.contains('K70.3')].reference_no.unique().tolist()
        instances = pd.read_csv('../data_analysis/instance_to_patient_id.csv')
        inst = instances[instances.patient_id.isin(pids)].admission_id.astype(str).unique().tolist()
        return df_data[df_data[ID['AID']].astype(str).isin(inst)][[ID['PID'], ID['AID'], 'infection_id']]
    elif group == '11':
        return df_data[df_data.vasop_history > 0][[ID['PID'], ID['AID'], 'infection_id']]
    elif group == '12':
        return df_data[df_data.mechanical_ventilation > 0][[ID['PID'], ID['AID'], 'infection_id']]
    elif group == '13':
        return df_data[np.logical_and(df_data.age_yrs < 45 / 121.0, df_data.filter(like='N10', axis=1).sum(axis=1) > 0)][[ID['PID'], ID['AID'], 'infection_id']]

    elif group == '14':
        return df_data[df_data.hospital_id_2574 == 1][[ID['PID'], ID['AID'], 'infection_id']]
    elif group == '15':
        return df_data[df_data.hospital_id_3148 == 1][[ID['PID'], ID['AID'], 'infection_id']]
    elif group == '16':
        return df_data[df_data.hospital_id_5107 == 1][[ID['PID'], ID['AID'], 'infection_id']]
    elif group == '17':
        return df_data[df_data.hospital_id_6729 == 1][[ID['PID'], ID['AID'], 'infection_id']]


def select_hospital(df_data, group='0'):
    if group == '0':
        return df_data[df_data.hospital_id_2572 == 1][[ID['PID'], ID['AID'], 'infection_id']]
    elif group == '1':
        return df_data[df_data.hospital_id_2574 == 1][[ID['PID'], ID['AID'], 'infection_id']]
    elif group == '2':
        return df_data[df_data.hospital_id_3049 == 1][[ID['PID'], ID['AID'], 'infection_id']]
    elif group == '3':
        return df_data[df_data.hospital_id_3148 == 1][[ID['PID'], ID['AID'], 'infection_id']]
    elif group == '4':
        return df_data[df_data.hospital_id_3269 == 1][[ID['PID'], ID['AID'], 'infection_id']]
    elif group == '5':
        return df_data[df_data.hospital_id_4674 == 1][[ID['PID'], ID['AID'], 'infection_id']]
    elif group == '6':
        return df_data[df_data.hospital_id_5107 == 1][[ID['PID'], ID['AID'], 'infection_id']]
    elif group == '7':
        return df_data[df_data.hospital_id_5572 == 1][[ID['PID'], ID['AID'], 'infection_id']]
    elif group == '8':
        return df_data[df_data.hospital_id_6729 == 1][[ID['PID'], ID['AID'], 'infection_id']]
    elif group == '9':
        return df_data[df_data.hospital_id_160559 == 1][[ID['PID'], ID['AID'], 'infection_id']]


def copy_file_to_memory(file_path):
    """Copies the content of a file to memory.

    Args:
        file_path: The path to the file.

    Returns:
        An io.BytesIO object containing the file content, or None if an error occurs.
    """
    try:
        with open(file_path, 'rb') as file:
            file_data = file.read()
            file_in_memory = io.BytesIO(file_data)
        return file_in_memory
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def paste_file_from_memory(file_in_memory, destination_path):
    """Pastes the content of a file from memory to a destination path.

    Args:
        file_in_memory: An io.BytesIO object containing the file content.
        destination_path: The path where the file should be pasted.
    """
    try:
         with open(destination_path, 'wb') as destination_file:
            destination_file.write(file_in_memory.getvalue())
            print(f"File pasted successfully to: {destination_path}")
    except Exception as e:
        print(f"An error occurred while pasting: {e}")
    finally:
        file_in_memory.close()


