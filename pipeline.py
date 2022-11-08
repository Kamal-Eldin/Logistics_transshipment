from mimetypes import init
import numpy as np
import pandas as pd
import xgboost as xgb
import datetime as dt
import sklearn
from scipy import stats
from sklearn.metrics import  median_absolute_error
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from utils import *

class pipeline:
    def __init__(self, init_params, outer_cv, inner_cv, search_space = None):
        self.init_params = init_params
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv

        if not search_space:
            self.search_space = { 
                'gamma': stats.betaprime(a = 2,b =6, scale = 10),
                'lambda': stats.expon(scale = 3),
                'lr': stats.loguniform(.0001, .5),
                'maxchild' : stats.nbinom(2, .4, loc = 1),
                'colrow': stats.beta(2.5,1),            
                'treedepth' : stats.nbinom(20, .5)
                }
        else: self.search_space = search_space


    def nested_validation(self, X, y, n_candids = 3, buffer = 10000, amp = 1):
        self.buffer = buffer
        self.amp = amp
        self.n_candids = n_candids
        
        outer_tracker = []

        for fold in self.outer_cv.split(X,y):
            x_fold, y_fold = fold
            xtrn, ytrn = X.iloc[x_fold] , y.iloc[x_fold]
            xtst, ytst = X.iloc[y_fold] , y.iloc[y_fold]

            best_model, scores_df = self.random_search (xtrn, ytrn)
            xtrn_aug, ytrn_aug = apply_smogn(xtrn, ytrn)
            loss_updater = loss(labels = ytrn_aug.discharge, buffer= buffer, amp= amp)
            wtd_loss = loss_updater.updateloss(ytrn_aug)
            # wtd_loss = updateloss (ytrn_aug, buffer = buffer, amplifier= amp)

            best_model.set_params(objective = wtd_loss)
            score_k = self.score_model (best_model, xtrn_aug, ytrn_aug, xtst , ytst)
            
            outer_tracker.append((best_model, score_k))

        scores_df = pd.DataFrame(outer_tracker, columns= ['model', 'score'])
        best_model = scores_df.loc[scores_df['score'] == scores_df.score.min(), 'model'].item()

        return best_model, scores_df


    def random_search(self, x_fold, y_fold): 
        itr_params = self.init_params.copy()
        tracker = []
        for iter in range(self.n_candids):
            itr_params['learning_rate']     = self.search_space['lr'].rvs(size=1).item()
            itr_params['max_depth']         = self.search_space['treedepth'].rvs(size=1).item()
            itr_params['subsample']         = self.search_space['colrow'].rvs(size=1).item()
            itr_params['colsample_bytree']  = self.search_space['colrow'].rvs(size=1).item()
            itr_params['reg_lambda']        = self.search_space['lambda'].rvs(size=1).item()
            itr_params['min_child_weight']  = self.search_space['maxchild'].rvs(size=1).item()
            itr_params['gamma']             = self.search_space['gamma'].rvs(size=1).item()
            
            model, mean_score = self.score_param(x_fold, y_fold, itr_params, cv= self.inner_cv)  # repeated kFlold cv for parameter estimation
            tracker.append((model, mean_score))

        scores_df = pd.DataFrame(tracker, columns= ['model', 'score'])
        best_model = scores_df.loc[scores_df['score'] == scores_df.score.min(), 'model'].item()

        return best_model, scores_df
        
    def score_param(self, X_cv, y_cv, itr_params, cv):

        splits = list(cv.split(X_cv, y_cv))
        scores = np.zeros(len(list(splits)))

        for n_splt, splt in enumerate(splits):
            train_idxs, val_idxs = splt
            splt_xtrn, splt_ytrn = X_cv.iloc[train_idxs] , y_cv.iloc[train_idxs]
            splt_xval, splt_yval = X_cv.iloc[val_idxs] , y_cv.iloc[val_idxs]

            splt_xtrn, splt_ytrn = apply_smogn(splt_xtrn, splt_ytrn)

            # wtd_loss = updateloss (splt_ytrn, buffer = 10000, amplifier= 3)
            loss_updater = loss(labels = splt_ytrn.discharge, buffer= self.buffer, amp= self.amp)
            wtd_loss = loss_updater.updateloss(splt_ytrn)
            # _ = loss_updater.get_weights()
            # wtd_loss = loss_updater.weighted_mse


            itr_params['objective'] = wtd_loss
            model_i = xgb.XGBRegressor(**itr_params)

            score_i = self.score_model (model_i, splt_xtrn, splt_ytrn, splt_xval, splt_yval)

            scores[n_splt] = score_i

        return model_i, scores.mean()

    @staticmethod
    def score_model (model, xt, yt, xv,yv ):
        model = model.fit(xt, yt)
        valpred = model.predict(xv)
        score_i = median_absolute_error(yv, valpred )
        return score_i


class load_model:
    def __init__(self, xgb_instance = xgb.XGBRegressor() ) -> None:
        self.xgb_instance =  xgb_instance

    def load_json(self, json_file):
        model = self.xgb_instance
        model.load_model(json_file)
        return model

class preprocess:
    def __init__(self, dataframe) -> None:
        self.dataframe = dataframe

    @staticmethod
    def _targets(dataframe):
        targets_df = dataframe.filter(regex= r"load\d*|discharge\d*")
        return targets_df

    @staticmethod
    def format_targets(targets_df):

        valid_load1 = targets_df.load1[targets_df.eval('load1 > 0')]
        valid_load2 = targets_df.load2[targets_df.eval('load2 > 0')]
        load_solid = pd.concat([valid_load1, valid_load2] )

        valid_load3 = targets_df.load3[targets_df.eval('load3 > 0')]
        valid_load4 = targets_df.load4[targets_df.eval('load4 > 0')]
        load_liquid = pd.concat([valid_load3, valid_load4])

        valid_disc1 = targets_df.discharge1[targets_df.eval('discharge1 > 0')]
        valid_disc2 = targets_df.discharge2[targets_df.eval('discharge2 > 0')]
        disc_solid= pd.concat([valid_disc1, valid_disc2] )


        valid_disc3 = targets_df.discharge3[targets_df.eval('discharge3 > 0')]
        valid_disc4 = targets_df.discharge4[targets_df.eval('discharge4 > 0')]
        disc_liquid= pd.concat([valid_disc3, valid_disc4] )


        disc_liquid = disc_liquid[~ disc_liquid.index.duplicated()]
        solid_bulk = pd.concat([load_solid, disc_solid], axis = 1)
        solid_bulk.insert(0,'bulk', value ='solid')
        liquid_bulk = pd.concat([load_liquid, disc_liquid], axis = 1)
        liquid_bulk.insert(0,'bulk',value= 'liquid')

        targs = pd.concat([solid_bulk, liquid_bulk], axis= 0)
        targs.columns = ['bulk','load', 'discharge']

        return targs

    def format_features(self):
        targets_df = self._targets(self.dataframe)
        targs = self.format_targets(targets_df)
        df = self.dataframe

        feature_df = df[['vesseldwt', 'vesseltype', 'traveltype']]
        # 2. Engineering a new feature `n_stevs`: num of stevedores per ship
        df['stevedorenames'] = df['stevedorenames'].astype('str')
        feature_df['n_stevs'] = df['stevedorenames'].str.split(',').apply(lambda x : len(x))

        # 3. construct process_time feature
        # ata = pd.to_datetime(df['ata']).dt.date
        # atd = pd.to_datetime(df['atd']).dt.date
        early_eta = pd.to_datetime(df['earliesteta']).dt.date
        late_eta = pd.to_datetime(df['latesteta']).dt.date
        feature_df['process_time'] = (late_eta - early_eta).dt.days

        # Type casting `vesseltype` as type `category`
        feature_df['vesseltype'] = feature_df['vesseltype'].astype('category')

        # Impute the single missing vessel weight value with the median of all type 5 vessel weights  
        vess5_medweight = feature_df.vesseldwt.loc [feature_df.vesseltype == 5].median()
        feature_df['vesseldwt'] = feature_df['vesseldwt'].fillna(vess5_medweight)

        feature_df = feature_df.join(targs, how= 'right')
        return feature_df

    def encode_features(self):
        feature_df = self.format_features()
        encoded_df = pd.get_dummies(feature_df, columns=['vesseltype','traveltype','bulk'])

        load_dataset = encoded_df.dropna(subset=['load']).drop('discharge', axis= 1)
        discharge_dataset = encoded_df.dropna(subset=['discharge']).drop('load', axis= 1)

        return load_dataset, discharge_dataset










    