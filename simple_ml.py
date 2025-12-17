import os

import numpy as np
import argparse
from sklearn import preprocessing, metrics
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from utils.utils_data import load_data, RandomizedGroupKFold, select_subgroup
from utils.utils_evaluation import evaluate_multi, plot_prc, plot_roc, evaluate, model_explain_multiclass, \
    evaluate_binary, model_explain_binary
import pandas as pd
from tqdm import tqdm
from paths import ID


def train_model(X_train, y_train, model_name, rs=99):
    # train model
    if model_name == 'xgboost':
        model = XGBClassifier(objective='binary:logistic',
                              booster='gbtree',
                              verbosity=0,
                              random_state=rs,
                              subsample=0.8,
                              use_label_encoder=False)
    elif model_name == 'catboost':
        model = CatBoostClassifier(verbose=0, random_state=rs)
    elif model_name == 'logistic':
        model = LogisticRegression(max_iter=1000,
                                   random_state=rs,
                                   penalty='l1',
                                   solver='liblinear')
    elif model_name == 'svm':
        model = SVC(probability=True, random_state=rs)
    elif model_name == 'mlp':
        model = MLPClassifier(max_iter=1000, random_state=rs)
    else:
        raise NotImplementedError('Please specify the model!!')

    # model.fit(X_train, y_train)
    model.load_model('logs/{}/{}/binary_rep_{}.json'.format("AMR", args.model, rs))

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument('--model', type=str, default='xgboost')  # 'catboost' or 'xgboost'
    parser.add_argument('--full_comorb', type=bool, default=False)
    parser.add_argument('--infection', type=str, default='hospital')
    parser.add_argument('--n_repeat', type=int, default=3)
    args = parser.parse_args()
    print(args)

    # load data
    data_table = load_data(full_comorb=args.full_comorb)

    # use only cases with notes
    notes = pd.read_csv(os.path.join("../cohort_3_new/data_combined", "deep_notes.csv"))
    data_table = data_table.reset_index()
    data_table = data_table[data_table["AID"].isin(notes["AID"].unique())]

    # filter based on infection instances
    if args.infection == 'community':
        data_table = data_table[data_table["infection_id"] == 0]
    elif args.infection == 'hospital':
        data_table = data_table[data_table["infection_id"] > 0]
    data_table = data_table.set_index(["PID", "AID", "infection_id"])

    # get labels
    data_table = data_table[~data_table.UN]  # exclude cases with unknown labels
    data = data_table.drop(columns=['SS', 'RS', 'RR', 'UN', 'GNB']).astype(float)
    admission_ids = data_table.reset_index()[ID['AID']]
    data_table.loc[data_table.SS, 'label'] = 0
    data_table.loc[np.logical_or(data_table.RS, data_table.RR), 'label'] = 1
    labels = data_table['label'].astype(int)

    # train/test split by stratifying so that each split has the same ratio of positive class
    cv = RandomizedGroupKFold(groups=admission_ids.to_numpy(), n_splits=5, random_state=42)
    train_ix, test_ix = cv[0]
    X_train, y_train = data.iloc[train_ix], labels.iloc[train_ix]
    X_test, y_test = data.iloc[test_ix], labels.iloc[test_ix]

    # train model
    models = []
    for i, rs in enumerate(tqdm(range(args.n_repeat))):
        model = train_model(X_train.values, y_train.values, model_name=args.model, rs=rs)
        if i == 0:
            model_explain_binary(model, X_test, y_test, args)
        model.save_model('logs/{}/{}/binary_rep_{}.json'.format("AMR", args.model, rs))
        models.append(model)

    # evaluate model
    evaluate_binary(models, X_test, y_test, args)
