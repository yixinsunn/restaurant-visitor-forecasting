#######################################################################################################
###                             Functions for Training and Data Preparation                         ###
#######################################################################################################
import pandas as pd
import numpy as np
from scipy import stats
import xgboost as xgb
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# This function aggregates min, max, median, mean and mode of visitors, i.e. target encoding.
# Also one-hot encodes categorical features and return training, test/validation sets
def getData(train, test, agg_columns, drop_columns=[], cat_columns=None, ohe=False, seed=0, do_shuffle=True):
    '''
    @params: train: <DataFrame> training set
             test: <DataFrame> test or validation set
             agg_columns: [list] columns that will be aggregated with store_id
             drop_columns: [list] columns that are not features and will be dropped
             cat_columns: [list] columns that will be one-hot encoded
             ohe: boolean, if one-hot encoding will be performed
    @return: X_train, X_test, y_train, y_test: 
             DataFrames where  min_visitors, max_visitors, mean_visitors, 
             median_visitors, mode_visitors, and count_visit are added into
             original train and test. 
             Feature encoding is also performed in this function.
    '''
    unique_stores = test['air_store_id'].unique()        
    values = {}
    for column in agg_columns:
        values[column] = train[column].unique()
    
    stores = pd.DataFrame({'air_store_id': unique_stores})
    for column in agg_columns:
        funcs = {
            ##'min': 'min_visitors' + '_' + column,
            'max': 'max_visitors' + '_' + column,
            'mean': 'mean_visitors' + '_' + column,
            'median': 'median_visitors' + '_' + column,
            ##'count': 'count_visit' + '_' + column,
            stats.mode: 'mode_visitors' + '_' + column,
        }
        # Aggregates min, max, median, mean, mode and std of visitors
        # grouped by each store and column
        if column in ['air_store_id', 'air_genre_name', 'prefecture']:
            for func in funcs:
                tmp = train.groupby(column, as_index=False).agg({'visitors': func}).rename(
                columns={'visitors': funcs[func]})
                train = pd.merge(train, tmp, how='left', on=column)
                test = pd.merge(test, tmp, how='left', on=column)
            train[funcs[stats.mode]] = train[funcs[stats.mode]].map(lambda x:x[0][0], na_action='ignore')
            test[funcs[stats.mode]] = test[funcs[stats.mode]].map(lambda x:x[0][0], na_action='ignore')
        else:
            s = pd.concat([pd.DataFrame({'air_store_id': unique_stores, column: [i] * len(unique_stores)}) \
                           for i in values[column]], ignore_index=True)
            for func in funcs:
                tmp = train.groupby(['air_store_id', column], as_index=False).agg(
                {'visitors': func}).rename(columns={'visitors': funcs[func]})
                s = s.merge(tmp, how='left', on=['air_store_id', column])
            s[funcs[stats.mode]] = s[funcs[stats.mode]].map(lambda x:x[0][0], na_action='ignore')
            stores = stores.merge(s, how='left', on='air_store_id') 
    
    # Merge training and test sets with store information
    key = [col for col in agg_columns if col not in ['air_store_id', 'air_genre_name', 'prefecture']]
    key.append('air_store_id')
    train = pd.merge(train, stores, how='inner', on=key)
    test = pd.merge(test, stores, how='inner', on=key)
    # Shuffle data
    train = shuffle(train, random_state=seed).reset_index(drop=True)
    if do_shuffle:
        test = shuffle(test, random_state=seed)
    else:
        test = test.sort_values(['air_store_id', 'date_int'])
    
    if not ohe:
        X_train, X_test = train.drop(drop_columns, axis=1), test.drop(drop_columns, axis=1)
        y_train, y_test = np.log1p(train['visitors']), np.log1p(test['visitors'])
        return X_train, X_test, y_train, y_test
    assert(cat_columns != None)
    return feature_encoder(train, cat_columns, drop_columns, test=test)
    
    
    
# This function one-hot encodes categorical features
def feature_encoder(train, cat_columns, drop_columns=[], test=None):
    if test is not None:
        train_test = pd.concat([train, test])
        train_test = pd.get_dummies(train_test, columns=cat_columns)
    
        X_train = train_test[:train.shape[0]].drop(drop_columns, axis=1)
        X_test = train_test[train.shape[0]:].drop(drop_columns, axis=1)
        y_train = np.log1p(train_test[:train.shape[0]]['visitors'])
        y_test = np.log1p(train_test[train.shape[0]:]['visitors'])
        return X_train, X_test, y_train, y_test
    else:
        train = pd.get_dummies(train, columns=cat_columns)
        X = train.drop(drop_columns, axis=1)
        y = np.log1p(train['visitors'])
        return X, y



# This function imputes missing values in test/validation set, using pre-defined estimator
def imputer(X_train, X_test, estimator):
    missing_columns = np.append('air_store_id', X_test.columns[X_test.isnull().any()])
    for column in missing_columns[1:]:
        missing_ids = X_test['air_store_id'][X_test[column].isnull()].unique()
        for store_id in missing_ids:
            X_known = X_train[X_train['air_store_id'] == store_id].drop(missing_columns, axis=1)
            y_known = X_train[X_train['air_store_id'] == store_id][column]
            X_unknown = X_test[(X_test['air_store_id'] == store_id) & 
                               (X_test[column].isnull())].drop(missing_columns, axis=1)
            estimator.fit(X_known, y_known)
            X_test[column][(X_test['air_store_id'] == store_id) & 
                           (X_test[column].isnull())] = estimator.predict(X_unknown)
            
            
        
# This function implements a second-level CV that averages target encoding values to
# reduce noise, prevent data leakage and overfitting
def secondLevelCV(KFold, train_level1, valid_level1, agg_columns, drop_columns=[], seed=0):      
    tmp = pd.DataFrame()
    for train_idx_level2, valid_idx_level2 in KFold.split(train_level1):
        train_level2, valid_level2 = train_level1.iloc[train_idx_level2], train_level1.iloc[valid_idx_level2]
        _, X_valid_level2, _, _ = getData(train_level2, valid_level2, seed=seed, 
                                       agg_columns=agg_columns, drop_columns=drop_columns)
        
        new_cols = [] if 'air_store_id' in agg_columns else ['air_store_id']
        for col in X_valid_level2.columns:
            if col.startswith('min_visit') or col.startswith('max_visit') or \
            col.startswith('mean_visit') or col.startswith('median_visit') or \
            col.startswith('mode_visit') or col.startswith('count_visit'):
                new_cols.append(col)
        tmp = pd.concat([tmp, X_valid_level2[agg_columns + new_cols]])    
    for col in agg_columns:
        agg_col = [c for c in new_cols if c.endswith(col)]
        group_col = ['air_store_id', col] if col != 'air_store_id' else ['air_store_id']
        grand_mean = tmp.groupby(group_col, as_index=False)[agg_col].mean()
        valid_level1 = valid_level1.merge(grand_mean, how='left', on=['air_store_id', col])    
    
    valid_level1 = valid_level1.sort_values(['air_store_id', 'date_int'])
    return valid_level1
        
            
# This function adds rolling window into a dataset  
def rolling_window(data, reserve, columns, window=[1, 2, 3, 7, 14, 21, 28, 35]):
    '''
    @params: data: <DataFrame> original data that will be processed
             reserve: <DataFrame> data that will be used to fill the window
             columns: [list] columns that will be aggregated as new features in the window
             window: [list] a list of window size
    @return: data_window: <DataFrame> new data with rolling sums, averages, etc as window
    '''
    if 'mean' in ''.join(columns):
        mean_col = [col for col in columns if 'mean' in col][0]
        sum_col = [col for col in columns if 'sum' in col][0]
        count_col = [col for col in columns if 'count' in col][0]
    data_window = pd.DataFrame()
    
    for store_id in data['air_store_id'].unique():
        if store_id not in reserve['air_store_id'].unique():
            continue
        reserve_id = reserve[reserve['air_store_id'] == store_id]
        for end_date in data[data['air_store_id'] == store_id]['visit_date']:
            data_id_date = pd.DataFrame({'air_store_id': [store_id], 'visit_date': [end_date]})
            for size in window:
                new_columns = {col: col + '_' + str(size) for col in columns}
                start_date = end_date - pd.Timedelta(size, unit='d')
                tmp1 = reserve_id[(reserve_id['visit_date'] >= start_date) & (reserve_id['visit_date'] < end_date)]
                if tmp1.shape[0] == 0:
                    tmp1.loc[0] = 0
                    tmp1['air_store_id'] = store_id
                tmp2 = tmp1.groupby('air_store_id', as_index=False)[columns].sum().rename(columns=new_columns)
                if 'mean' in ''.join(columns):
                    tmp2[new_columns[mean_col]] = tmp2[new_columns[sum_col]] / tmp2[new_columns[count_col]]
                tmp2['visit_date'] = end_date
                data_id_date = data_id_date.merge(tmp2, how='inner', on=['air_store_id', 'visit_date'])
            data_window = pd.concat([data_window, data_id_date])
    
    data_window = data_window.fillna(0)
    return data_window


# This function computes a 'isTest' label to indicates what samples in
# training set has the same population distribution as test set.
# This 'isTest' label can be helpful in adversarial validation!
def get_isTest(train, test, cat_columns, drop_columns, threshold=0.3, seed=0):
    train['row'] = [i for i in range(train.shape[0])]
    test['row'] = -1
    trn, tst, _, _ = feature_encoder(train, cat_columns, drop_columns, test)
    tst = tst.fillna(-1)
    trn['target'] = 0
    tst['target'] = 1
    
    drop = ['date_int']
    trn.drop(drop, axis=1, inplace=True)
    tst.drop(drop, axis=1, inplace=True)
    train_test = pd.concat([trn, tst])
    
    X_train, X_test, y_train, y_test = train_test_split(
        train_test.drop(['air_store_id', 'target', 'row'], axis=1), train_test['target'], 
        stratify=train_test['target'], random_state=seed, test_size=0.3)
    
    params = {
        'eta': 0.1,
        'max_depth': 5,
        'min_child_weight': 5,
        'subsample': 0.85,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'lambda': 2,
        'seed': seed
        }
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dtest = xgb.DMatrix(data=X_test)
    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=200)
    prob_trn = model.predict(dtrain, ntree_limit=200)
    prob_tst = model.predict(dtest, ntree_limit=200)
    
    print('AUC on train: {}, AUC on test: {}'.format(
        roc_auc_score(y_train, prob_trn), roc_auc_score(y_test, prob_tst)))
    
    dtrn = xgb.DMatrix(data=trn.drop(['air_store_id', 'target', 'row'], axis=1))
    prob = model.predict(dtrn)
    trn['prob'] = prob
    res = trn[trn['prob'] > threshold]
    res.drop([col for col in res.columns if col != 'row'], axis=1, inplace=True)
    res['isTest'] = 1
    
    test['isTest'] = 1
    return train.merge(res, on='row', how='left').drop('row', axis=1).fillna(0), test.drop('row', axis=1)
