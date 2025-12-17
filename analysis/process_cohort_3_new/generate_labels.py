import numpy as np
import pandas as pd
pd.options.mode.copy_on_write = True
import matplotlib.pyplot as plt
from os.path import join
import datetime
from tqdm import tqdm
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


def eval_resistance(row):
    """
    Evaluate microbe level resistance.
    0 -> unknown
    1 -> GNB negative
    2 -> GNB positive, Ceftriaxone-S
    3 -> GNB positive, Ceftriaxone-R, Cefepime-S
    4 -> GNB positive, Ceftriaxone-R, Cefepime-R
    :param row:
    :return:
    """
    if not row.GNB_Positive:
        return 1  # GNB negative
    if row.Ceftriaxone == 0:
        return 0  # Ceftriaxone unknown
    if row.Ceftriaxone == 1:
        return 2  # GNB positive, Ceftriaxone-S
    if row.Cefepime == 0:
        return 0  # GNB positive, Ceftriaxone-R, Cefepime unknown
    if row.Cefepime == 1:
        return 3  # GNB positive, Ceftriaxone-R, Cefepime-S
    return 4  # GNB positive, Ceftriaxone-R, Cefepime-R


def merge_same_microbes(tup_list):
    micro_dict = dict()
    for microbe, score in tup_list:
        if microbe not in micro_dict:
            micro_dict[microbe] = score
            continue
        curr_score = micro_dict[microbe]
        micro_dict[microbe] = max(curr_score, score)
    return list(zip(micro_dict.keys(), micro_dict.values()))


def reset_clock(df_group):
    if_NaN = df_group['infection_id'].isna()
    tmp = df_group.loc[if_NaN]
    df_group.loc[if_NaN, 'time_diff'] = (tmp.groupby('visit_occurrence_id')[
                                             'collection_date'].diff() / np.timedelta64(1, 'h')).fillna(value=0)
    tmp = df_group.loc[if_NaN]
    df_group.loc[if_NaN, 'time_cumsum'] = tmp['time_diff'].cumsum()

    return df_group


# load input
df_cultures = pd.read_csv(join(raw_data_path, '{}_cultures_w_sensitivities_1.csv'.format(prefix)))
Ceftriaxone_self_explain = pd.read_excel(join(manual_data_path, 'Ceftriaxone labels 12.23.22.xlsx'))
Cefepime_self_explain = pd.read_csv(join(manual_data_path, 'gnb_pos_CRO_R_FEP_TBD_MCVG.csv'))
manual_labels = pd.read_csv(join(manual_data_path, 'unknown_instances_manual_label.csv'))
specimen_2 = pd.read_csv(join(manual_data_path, 'specimens_by_frequency_mcvg.csv'))
df_gnb_2 = pd.read_csv(join(manual_data_path, 'microbes_by_frequency_mcvg.csv'))
print("Number of patients: {}".format(df_cultures.person_id.nunique()))
print("Number of admissions: {}".format(df_cultures.visit_occurrence_id.nunique()))

# microbes, GNB +/-, and the sensitivity to ABX
df_gnb_2['GNB'] = df_gnb_2['GNB 1=yes '].fillna(0)
bug2gnb_2 = dict(zip(df_gnb_2['concept_name'], df_gnb_2['GNB']))

bugs_must_be_Ceftriaxone_S = Ceftriaxone_self_explain[Ceftriaxone_self_explain[Ceftriaxone_self_explain.columns[1]] == 'S']['Name'].tolist()
bugs_must_be_Ceftriaxone_R = Ceftriaxone_self_explain[Ceftriaxone_self_explain[Ceftriaxone_self_explain.columns[1]] == 'R']['Name'].tolist()
bugs_must_be_Ceftriaxone_U = Ceftriaxone_self_explain[Ceftriaxone_self_explain[Ceftriaxone_self_explain.columns[1]] == 'U']['Name'].tolist()

bugs_must_be_Cefepime_S = Cefepime_self_explain[Cefepime_self_explain['FEP'] == 'S']['PRINT_NAME'].tolist()
bugs_must_be_Cefepime_R = Cefepime_self_explain[Cefepime_self_explain['FEP'] == 'R']['PRINT_NAME'].tolist()
bugs_must_be_Cefepime_U = Cefepime_self_explain[Cefepime_self_explain['FEP'] == 'U']['PRINT_NAME'].tolist()

bugs_must_be_Ceftriaxone_R_extra = [
    'Achromobacter spp.', 'Acinetobacter spp. (excluding A. calcoaceticus-baumannii complex)',
    'Acinetobacter calcoaceticus-baumannii complex',
    'Alcaligenes faecalis', 'Citrobacter freundii complex', 'Enterobacter aerogenes',
    'Enterobacter cloacae complex', 'Pseudomonas aeruginosa',
    'Pseudomonas aeruginosa (mucoid phenotype)', 'Pseudomonas spp., not P. aeruginosa',
    'Serratia marcescens', 'Stenotrophomonas maltophilia',
    # additionally added on 09/12/2023
    'Bordetella bronchiseptica',
    'Bordetella hinzii',
    'Bordetella pertussis',
    'Bordetella species',
    'Bordetella trematum',
    'Burkholderia cepacia',
    'Burkholderia cepacia complex',
    'Burkholderia gladioli',
    'Acinetobacter baumannii',
    'Acinetobacter calcoaceticus',
    'Acinetobacter calcoaceticus-Acinetobacter baumanii complex',
    'Acinetobacter johnsonii',
    'Acinetobacter lwoffii',
    'Acinetobacter pitii',
    'Acinetobacter species',
    'Acinetobacter ursingii',
    'Helicobacter pylori',
    'Helicobacter species',
    'Achromobacter piechaudii',
    'Achromobacter species',
    'Achromobacter xylosoxidans',
    'Achromobacter xylosoxidans subsp. denitrificans',
    'Achromobacter xylosoxidans subsp. xylosoxidans'
]
bugs_must_be_Cefepime_R_extra = [
    # additionally added on 09/12/2023
    'Bordetella bronchiseptica',
    'Bordetella hinzii',
    'Bordetella pertussis',
    'Bordetella species',
    'Bordetella trematum',
    'Burkholderia cepacia',
    'Burkholderia cepacia complex',
    'Burkholderia gladioli',
    'Acinetobacter baumannii',
    'Acinetobacter calcoaceticus',
    'Acinetobacter calcoaceticus-Acinetobacter baumanii complex',
    'Acinetobacter johnsonii',
    'Acinetobacter lwoffii',
    'Acinetobacter pitii',
    'Acinetobacter species',
    'Acinetobacter ursingii',
    'Helicobacter pylori',
    'Helicobacter species',
    'Achromobacter piechaudii',
    'Achromobacter species',
    'Achromobacter xylosoxidans',
    'Achromobacter xylosoxidans subsp. denitrificans',
    'Achromobacter xylosoxidans subsp. xylosoxidans'
]
bugs_must_be_Ceftriaxone_R = list(set(bugs_must_be_Ceftriaxone_R) | set(bugs_must_be_Ceftriaxone_R_extra))
bugs_must_be_Cefepime_R = list(set(bugs_must_be_Cefepime_R) | set(bugs_must_be_Cefepime_R_extra))

# load culture test
df_cultures = df_cultures[df_cultures.visit_occurrence_id >= 0]
df_cultures['person_id'] = df_cultures['person_id'].astype(int)
df_cultures['visit_occurrence_id'] = df_cultures['visit_occurrence_id'].astype(int)  # exclude dummy rows
df_cultures = df_cultures[
    ['person_id', 'visit_occurrence_id', 'specimen', 'admit_date', 'discharge_date', 'collection_date',
     'concept_name', 'sensitivity', 'abx_string', 'negative_flag']]

def sensitivity2short(sensitivity):
    if sensitivity in ["Sensitive", "Susceptible", "Susceptible Dose-dependent"]:
        return "S"
    elif sensitivity == "Intermediate":
        return "I"
    elif sensitivity in ["Resistant", "Not Susceptible", "NS"]:
        return "R"
    elif sensitivity in ["No Interpretation", "XX", "NaN"]:
        return "NaN"
    return None


# exclude rows with unimportant specimen
specimen_included = specimen_2[specimen_2[specimen_2.columns[2]] > 0]['specimen'].unique().tolist()
specimen_included = [s.capitalize() for s in specimen_included]
df_cultures = df_cultures[df_cultures['specimen'].isin(specimen_included)]
df_cultures['sensitivity'] = df_cultures['sensitivity'].fillna(value='NaN')
df_cultures['sensitivity'] = df_cultures['sensitivity'].apply(lambda x: sensitivity2short(x))
print("Excluded rows with unimportant specimen...")
print("Number of patients: {}".format(df_cultures.person_id.nunique()))
print("Number of admissions: {}".format(df_cultures.visit_occurrence_id.nunique()))

# exclude rows with collection time earlier than 24H before admission time
df_cultures['collection_date'] = pd.to_datetime(df_cultures['collection_date'])
df_cultures['admit_date'] = pd.to_datetime(df_cultures['admit_date'])
df_cultures = df_cultures[df_cultures['collection_date'] >= (df_cultures['admit_date'] - datetime.timedelta(days=1))]
print("Excluded rows with collection time earlier than 24H before admission time...")
print("Number of patients: {}".format(df_cultures.person_id.nunique()))
print("Number of admissions: {}".format(df_cultures.visit_occurrence_id.nunique()))

# negative_names = ['Negative', 'Normal flora', '..', 'Clinically insignificant flora',
#                   'Flora; Additional growth of clinically insignificant bacterial flora',
#                   'Flora; Normal enteric flora isolated', 'Flora; Normal genital flora isolated',
#                   'Flora; Normal upper respiratory flora isolated', 'Mixed microorganisms, gastrointestinal tract',
#                   'Mixed microorganisms, genitourinary tract', 'Mixed microorganisms, skin',
#                   'Mixed microorganisms, upper respiratory tract', 'Negative to date',
#                   'No Growth', 'No organisms seen', 'Normal enteric flora', 'Normal flora',
#                   'Normal upper respiratory flora', 'Neagtive for Normal Flora']
# df_cultures.loc[df_cultures.concept_name.isin(negative_names), 'concept_name'] = 'No Growth'  # 'Negative' = 'No Growth'
df_cultures.loc[df_cultures.negative_flag == 1, 'concept_name'] = 'No Growth'

# setting resistance scores
scores = {
    'S': 1,
    'I': 1,
    'R': 2,
    ' ': 0,
    'NaN': 0,
}
df_cultures['resistance_score'] = df_cultures['sensitivity'].apply(lambda x: scores[x])

# pivot table - ABX test results
idx_cols = ['person_id', 'visit_occurrence_id', 'admit_date', 'discharge_date', 'collection_date', 'specimen',
            'concept_name']
data = df_cultures.pivot_table(index=idx_cols,
                               columns='abx_string',
                               values='resistance_score',
                               aggfunc='max'
                               ).reset_index()
data['GNB_Positive'] = data['concept_name'].apply(lambda x: bug2gnb_2[x] if x in bug2gnb_2 else None).astype(bool)
data = data[idx_cols + ['GNB_Positive', 'Ceftriaxone', 'Cefepime', 'Cefazolin']]
data['Ceftriaxone'] = data['Ceftriaxone'].fillna(value=0)  # 0 -> not tested; 1 -> sensitive; 2 -> resistant
data['Cefepime'] = data['Cefepime'].fillna(value=0)  # 0 -> not tested; 1 -> sensitive; 2 -> resistant

# apply prior knowledge -- use self-explanatory information
data.loc[data['concept_name'].isin(bugs_must_be_Ceftriaxone_S), 'Ceftriaxone'] = 1
data.loc[data['concept_name'].isin(bugs_must_be_Ceftriaxone_R), 'Ceftriaxone'] = 2
data.loc[data['concept_name'].isin(bugs_must_be_Cefepime_S), 'Cefepime'] = 1
data.loc[data['concept_name'].isin(bugs_must_be_Cefepime_R), 'Cefepime'] = 2
data.loc[data['Cefepime'] == 2, 'Ceftriaxone'] = 2  # if Cefepime-R, must be Ceftriaxone-R

data['microbe_level'] = data.apply(lambda row: eval_resistance(row), axis=1)

# Is susceptible to cefazolin (Cefazolin) - then SS
# If resistant to cefazolin, then unknown
helper_a = np.logical_or(data.concept_name.str.contains('Escherichia'), data.concept_name.str.contains('Klebsiella'))
helper_b = np.logical_and(data.microbe_level == 0, data.Cefazolin == 1)
data.loc[np.logical_and(helper_a, helper_b), 'microbe_level'] = 2

# annotating cultures -- same specimen within 1H should be considered SAME CULTURE
data['collection_date'] = pd.to_datetime(data['collection_date'])
data = data.sort_values(['person_id', 'visit_occurrence_id', 'specimen', 'collection_date']).reset_index(drop=True)
specimen_code = dict(zip(data.specimen.unique(), range(data.specimen.nunique())))

data['time_diff'] = (
        data.groupby(['visit_occurrence_id', 'specimen'])['collection_date'].diff() / np.timedelta64(1, 'h')).fillna(
    value=0)
data['if_over_1h'] = data['time_diff'].apply(lambda x: x > 1).astype(int)
data['culture_index'] = data.groupby(['visit_occurrence_id', 'specimen'])['if_over_1h'].cumsum().astype(int)
data['culture_id'] = data.apply(lambda row: "S{}-{}".format(specimen_code[row.specimen], row.culture_index), axis=1)
data = data.drop(columns=['time_diff', 'if_over_1h', 'culture_index'])


# prepare culture records -- each row corresponds to a culture
cultures = data.groupby(['visit_occurrence_id', 'specimen', 'culture_id']).agg(
    {"concept_name": lambda x: list(x), "microbe_level": lambda x: list(x)})
data_by_culture = data.groupby(['visit_occurrence_id', 'specimen', 'culture_id'])[
    ['person_id', 'admit_date', 'discharge_date', 'collection_date']].first()
data_by_culture['microbes'] = cultures.apply(lambda row: list(zip(row.concept_name, row.microbe_level)), axis=1)
data_by_culture['contain_GNB'] = data.groupby(['visit_occurrence_id', 'specimen', 'culture_id'])['GNB_Positive'].max()

data_by_culture['microbes'] = data_by_culture['microbes'].apply(
    lambda x: merge_same_microbes(x))  # drop duplicate microbe names
data_by_culture = data_by_culture.reset_index().sort_values(
    ['person_id', 'visit_occurrence_id', 'collection_date']).reset_index(drop=True)
data_by_culture['culture_order'] = data_by_culture.groupby('visit_occurrence_id').cumcount()
data_by_culture = data_by_culture[
    ['person_id', 'visit_occurrence_id', 'admit_date', 'discharge_date', 'collection_date', 'culture_order', 'culture_id',
     'specimen', 'microbes', 'contain_GNB']]
data_by_culture['culture_level'] = data_by_culture['microbes'].apply(lambda x: max([ele[1] for ele in x]))


## labeling infection instances
data_by_culture['admit_date'] = pd.to_datetime(data_by_culture['admit_date'])
data_by_culture['collection_date'] = pd.to_datetime(data_by_culture['collection_date'])
data_by_culture = data_by_culture.sort_values(
    ['person_id', 'visit_occurrence_id', 'collection_date', 'specimen']).reset_index(drop=True)
data_by_culture['infection_id'] = np.nan
data_result = []

for group_name, df_group in tqdm(data_by_culture.groupby('visit_occurrence_id')):
    # initialize
    infection_id = 0
    microbe_bag = {'No Growth'}
    df_group = reset_clock(df_group)

    # T0 should be within 48H from admission. If not, should start from T1
    adm_time = pd.to_datetime(df_group.admit_date.values[0])
    col_time = pd.to_datetime(df_group.collection_date.values[0])
    if col_time > adm_time + datetime.timedelta(days=2):
        infection_id += 1
        df_group = reset_clock(df_group)
        microbe_bag = {'No Growth'}

    # iterate over rows
    for row_index, row in df_group.iterrows():
        time_cumsum = df_group.loc[row_index, 'time_cumsum']  # might be updated
        microbes = [ele[0] for ele in row['microbes']]

        # for 48H - 120H, if new microbe detected, set as start of new instance
        if time_cumsum > 48 and time_cumsum <= 120:
            if len(set(microbes) - microbe_bag):
                infection_id += 1
                df_group = reset_clock(df_group)
                microbe_bag = {'No Growth'}

        # for > 120H, set as new instance
        if time_cumsum > 120:
            infection_id += 1
            df_group = reset_clock(df_group)
            microbe_bag = {'No Growth'}

        microbe_bag = microbe_bag | set(microbes)
        df_group.loc[row_index, 'infection_id'] = infection_id

    data_result.append(df_group)

# concatenate
data_by_culture = (pd.concat(data_result)
                   .sort_values(['person_id', 'visit_occurrence_id', 'collection_date', 'specimen'])
                   .reset_index(drop=True))
data_by_culture['infection_id'] = data_by_culture['infection_id'].astype(int)
data_by_culture['admission_instance'] = data_by_culture['visit_occurrence_id'].astype(str) + "-" + data_by_culture[
    'infection_id'].astype(str)
data_by_culture.to_pickle(join(tmp_data_path, 'data_by_culture.pickle'))
data.to_pickle(join(tmp_data_path, 'data_by_microbe.pickle'))

# # plot instance histogram
# plt.figure()
# data_by_culture.groupby('visit_occurrence_id')['infection_id'].max().hist(bins=25)
# plt.xlabel('Number of Infection Instances', fontsize=13)
# plt.ylabel('Number of Patient Admissions', fontsize=13)
# plt.yscale('log')
# plt.show()


# evaluate instance-level resistance
APPLY_CONSISTENCY = True
data_by_culture = pd.read_pickle(join(tmp_data_path, 'data_by_culture.pickle'))

if APPLY_CONSISTENCY:
    data_result = []
    dominant = dict()

    # apply consistency rules
    for admission_instance, df_instance in tqdm(data_by_culture.groupby('admission_instance')):
        # setup microbe counter {microbe: (count, max_resistance)}
        cnt = dict()
        repeat = False
        for row_index, row in df_instance.iterrows():
            microbes = row['microbes']
            for microbe, score in microbes:
                # repetition of No Growth does not count
                if microbe == 'No Growth':
                    continue
                # not in counter yet
                if microbe not in cnt:
                    cnt[microbe] = (1, score)
                    continue
                # microbe already existed
                repeat = True
                curr_count, max_score = cnt[microbe]
                cnt[microbe] = (curr_count + 1, max(max_score, score))

        if not len(cnt):
            instance_level = 1
            dominant_microbes = [('No Growth', 1)]
        elif not repeat:
            instance_level = max([value[1] for value in cnt.values()])
            dominant_microbes = list([(key, cnt[key][1]) for key in cnt.keys()])
        else:
            instance_level = max([value[1] for value in cnt.values()])
            dominant_microbes = list([(key, cnt[key][1]) for key in cnt.keys()])
            # instance_level = max([value[1] for value in cnt.values() if value[0] > 1])
            # dominant_microbes = list([(key, cnt[key][1]) for key in cnt.keys() if cnt[key][0] > 1])

        df_instance.loc[:, 'instance_level'] = instance_level
        dominant[admission_instance] = dominant_microbes
        data_result.append(df_instance)

    data_by_culture = (pd.concat(data_result)
                       .sort_values(['person_id', 'visit_occurrence_id', 'collection_date', 'specimen'])
                       .reset_index(drop=True))
    data_by_culture['dominant_microbes'] = data_by_culture['admission_instance'].apply(lambda x: dominant[x])

else:
    # no consistency rule -- highest culture level
    dominant_microbes = data_by_culture.groupby('admission_instance')['microbes'].sum()
    instance_level = data_by_culture.groupby('admission_instance')['culture_level'].max()
    data_by_culture['dominant_microbes'] = data_by_culture['admission_instance'].apply(lambda x: dominant_microbes[x])
    data_by_culture['instance_level'] = data_by_culture['admission_instance'].apply(lambda x: instance_level[x])

    data_by_culture = (data_by_culture
                       .sort_values(['person_id', 'visit_occurrence_id', 'collection_date', 'specimen'])
                       .reset_index(drop=True))

data_by_culture.to_csv(join(tmp_data_path, 'data_by_culture.csv'), index=False)


# analysis
helper = data_by_culture.groupby('admission_instance').culture_level.max() > data_by_culture.groupby(
    'admission_instance').instance_level.max()
df_corrected = data_by_culture[data_by_culture.admission_instance.isin(helper[helper == True].index)]
df_corrected.to_csv(join(tmp_data_path, 'corrected_infections.csv'))

# evaluate instance level resistance
instance_resistance = data_by_culture.groupby('admission_instance')[['instance_level', 'dominant_microbes']].first()
counts = instance_resistance.instance_level.value_counts()
print(
    "GNB Neg -- ", counts[1], "\n",
    "GNB Pos, Ceftriaxone-S -- ", counts[2], "\n",
    "GNB Pos, Ceftriaxone-R, Cefepime-S -- ", counts[3], "\n",
    "GNB Pos, Ceftriaxone-R, Cefepime-R -- ", counts[4], "\n",
    "Unknown -- ", counts[0]
)

# ##################
# complete_cases = pd.read_csv(join(save_data_dir, "labels_complete_cases.csv"))
# complete = complete_cases["admission_instance"].tolist()
# instance_resistance = instance_resistance[instance_resistance.index.isin(complete)]
# ##################


# manual labels
manual_SS = manual_labels[manual_labels.Label == 'SS'].admission_instance.values
manual_RS = manual_labels[manual_labels.Label == 'RS'].admission_instance.values
manual_RR = manual_labels[manual_labels.Label == 'RR'].admission_instance.values
instance_resistance.loc[instance_resistance.index.isin(manual_SS), 'instance_level'] = 2
instance_resistance.loc[instance_resistance.index.isin(manual_RS), 'instance_level'] = 3
instance_resistance.loc[instance_resistance.index.isin(manual_RR), 'instance_level'] = 4
counts = instance_resistance.instance_level.value_counts()
print(
    "Manual labels added..."
    "GNB Neg -- ", counts[1], "\n",
    "GNB Pos, Ceftriaxone-S -- ", counts[2], "\n",
    "GNB Pos, Ceftriaxone-R, Cefepime-S -- ", counts[3], "\n",
    "GNB Pos, Ceftriaxone-R, Cefepime-R -- ", counts[4], "\n",
    "Unknown -- ", counts[0]
)

exploded = instance_resistance.explode('dominant_microbes').reset_index()
exploded['microbes'] = exploded['dominant_microbes'].apply(lambda x: x[0])
exploded['microbe_level'] = exploded['dominant_microbes'].apply(lambda x: x[1])
exploded['hospital_acquired'] = exploded['admission_instance'].apply(lambda x: "-0" not in x)
exploded_community = exploded[~exploded.hospital_acquired]
exploded_hospital = exploded[exploded.hospital_acquired]

# most frequent dominant microbes
for source, exploded in zip(["community", "hospital"], [exploded_community, exploded_hospital]):
    df_aa = exploded[exploded.microbe_level == exploded.instance_level]
    df_aa = df_aa[np.logical_or(df_aa.instance_level == 1, df_aa.instance_level == 2)]
    aa = pd.DataFrame(df_aa['microbes'].value_counts())
    aa['fraction of all instances'] = aa/ df_aa.admission_instance.nunique()
    # aa.to_csv(join(tmp_dir, 'most_frequent_bugs_SS_{}.csv'.format(source)))
    print("------SS------", aa)

    df_bb = exploded[exploded.microbe_level == exploded.instance_level]
    df_bb = df_bb[df_bb.instance_level == 3]
    bb = pd.DataFrame(df_bb['microbes'].value_counts())
    bb['fraction of all instances'] = bb/ df_bb.admission_instance.nunique()
    # bb.to_csv(join(tmp_dir, 'most_frequent_bugs_RS_{}.csv'.format(source)))
    print("------RS------", bb)

    df_cc = exploded[exploded.microbe_level == exploded.instance_level]

    df_cc = df_cc[df_cc.instance_level == 4]
    cc = pd.DataFrame(df_cc['microbes'].value_counts())
    cc['fraction of all instances'] = cc / df_cc.admission_instance.nunique()
    # cc.to_csv(join(tmp_dir, 'most_frequent_bugs_RR_{}.csv'.format(source)))
    print("------RR------", cc)


# label files
df_label = data_by_culture[['person_id', 'visit_occurrence_id', 'admission_instance']].groupby(
    'admission_instance').first().reset_index()
df_label = df_label.rename(columns={'admission_instance': 'instance_id'})
instance_resistance = instance_resistance['instance_level']
df_label['SS'] = df_label['instance_id'].apply(lambda x: instance_resistance[x] == 1 or instance_resistance[x] == 2)
df_label['RS'] = df_label['instance_id'].apply(lambda x: instance_resistance[x] == 3)
df_label['RR'] = df_label['instance_id'].apply(lambda x: instance_resistance[x] == 4)
df_label['UN'] = df_label['instance_id'].apply(lambda x: instance_resistance[x] == 0)
df_label['GNB'] = df_label['instance_id'].apply(lambda x: instance_resistance[x] > 1)

first_dates = data_by_culture.groupby('admission_instance').first()[
    ['person_id', 'admit_date', 'discharge_date', 'collection_date']]
df_label.loc[:, 'person_id'] = df_label['instance_id'].apply(lambda x: first_dates.loc[x, 'person_id'])
df_label.loc[:, 'admit_date'] = df_label['instance_id'].apply(
    lambda x: pd.to_datetime(first_dates.loc[x, 'admit_date']))
df_label.loc[:, 'collection_date'] = df_label['instance_id'].apply(
    lambda x: pd.to_datetime(first_dates.loc[x, 'collection_date']))
# df_label.loc[:, 'discharge_date'] = df_label['instance_id'].apply(
#     lambda x: pd.to_datetime(first_dates.loc[x, 'discharge_date'], format="ISO8601"))
df_label.to_csv(join(processed_data_path, 'df_label.csv'), index=False)


# summary
print('Total instances:{} ({} patients)'.format(len(df_label), df_label.reset_index().person_id.nunique()))
excluded = len(df_label[df_label.UN])
included = len(df_label) - excluded
print('Excluded instances:{}'.format(excluded))
print('Included instances:{}'.format(included))
print('SS:{}, {:0.2f}%'.format(len(df_label[df_label.SS]), len(df_label[df_label.SS]) / included * 100))
print('RS:{}, {:0.2f}%'.format(len(df_label[df_label.RS]), len(df_label[df_label.RS]) / included * 100))
print('RR:{}, {:0.2f}%'.format(len(df_label[df_label.RR]), len(df_label[df_label.RR]) / included * 100))
print('Unknown:{}'.format(len(df_label[df_label.UN])))
print('GNB:{}, {:0.2f}%'.format(len(df_label[df_label.GNB]), len(df_label[df_label.GNB]) / included * 100))

# for deep learning data generator
df_label['infection_id'] = df_label['instance_id'].apply(lambda x: int(x.split('-')[1]))
df_label = df_label.drop(columns=['instance_id']).set_index(['person_id', 'visit_occurrence_id', 'infection_id'])
df_label.to_csv(join(processed_data_path, 'df_label_full.csv'))

