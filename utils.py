
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, convolve1d
from scipy.signal.windows import gaussian
import smogn


class LDS:

    def __init__(self, labels = None, kernel = None, buffer = 0, maxbin = None, round_factor = 3 ) -> None:
        self.labels = labels
        self.round_factor = round_factor
        self.maxbin = maxbin
        self.buffer = buffer
        self.kernel = gaussian(5, 3, sym=True) if kernel is None else kernel

    @staticmethod
    def _get_bins(labels, buffer = 0, round_factor = None, maxbin = None):

        if round_factor is None :
            round_factor =  str(labels.astype('int').max()).__len__() - 1

        if maxbin is None:
            maxvalue = labels.max() + buffer 
            maxbin = np.round(maxvalue, -round_factor)

        bins = np.arange(0, maxbin, 1000, dtype = 'int')
        bin_labels= bins[1:]

        return bins, bin_labels

    def bin_dataframe (self, labels_df):
        bins, bin_labels = self._get_bins(self.labels, buffer= self.buffer, round_factor= self.round_factor, maxbin = self.maxbin)
        data_binned, bin_edges = pd.cut(self.labels, bins= bins, retbins= True, labels= bin_labels)
        ytrain_bins = pd.concat([labels_df, data_binned], axis= 1)
        ytrain_bins.columns = [ytrain_bins.columns[0], 'bin']

        self.bins = bins
        self.bin_labels = bin_labels
        self.binned_data = data_binned

        return ytrain_bins

    @staticmethod
    def _empirical_dist(binned_data, smooth = 5):

        emp_hist = binned_data.value_counts().sort_index().to_dict()
        emp_hist = {bin: np.clip(freq, a_min= smooth, a_max= 1000) for (bin,freq) in emp_hist.items()}

        return emp_hist

    
    def effective_dist(self, weight = 'inverse', smooth = 5):
        assert weight in ['inverse', 'sqrt_inv'] , "select weight from: {inverse, sqrt_inv}"

        if weight == 'sqrt_inv':
            smooth = 0
            emp_hist =  self._empirical_dist(self.binned_data, smooth = smooth)
            emp_hist =  {bin: np.sqrt(freq) for (bin, freq) in emp_hist.items()}
        else:
            emp_hist =  self._empirical_dist(self.binned_data, smooth = smooth)


        emp_freqs = np.asarray(list(emp_hist.values()))
        eff_freqs = convolve1d(emp_freqs, weights= self.kernel, mode='constant')
        effective_dist = {emp_key: eff_value for emp_key, eff_value in zip(emp_hist, eff_freqs)}

        return effective_dist, emp_hist

    def weight_df(self, dataframe, weight = 'inverse' , smooth = 5):

        eff_dist, _ = self.effective_dist(weight, smooth)
        inv_weights =  self.binned_data.apply(lambda x: 1/eff_dist[x])
        scale = len(inv_weights) / inv_weights.sum()
        inv_weights *=  scale
        dataframe[weight] = inv_weights

        return dataframe



class weightloss:
    def __init__(self, weights = 1) -> None:
        self.weights = weights
    

    def weighted_mse(self, pred_arr, label_arr):
        grad  = (label_arr - pred_arr ) 
        hess  = 1 

        grad *= self.weights
        hess *= self.weights
        return grad, hess


class loss (LDS):

    def __init__(self,amp, labels = None, kernel = None, buffer = 0, maxbin = None, round_factor = 3) -> None:
        super().__init__(labels, kernel, buffer, maxbin, round_factor)
        self.amp = amp
        self.weights = 1

    def get_weights (self):
        # lds= LDS(labels= splt_ytrn.discharge, buffer= buffer)
        binned = self.bin_dataframe(self.labels)
        x_wts = self.weight_df(binned).inverse * self.amp
        self.weights = x_wts
        return self.weights

    def weighted_mse(self, pred_arr, label_arr):
        grad  = (label_arr - pred_arr ) 
        hess  = 1 

        grad *= self.weights
        hess *= self.weights
        return grad, hess

    def updateloss (self, splt_ytrn):
        binned = self.bin_dataframe(splt_ytrn)
        x_wts = self.weight_df(binned).inverse * self.amp
        wtd_loss = weightloss(weights= x_wts)
        loss_func = wtd_loss.weighted_mse
        return loss_func



def apply_smogn (xsplit, ysplit, target_col = 'discharge'):

    train_df= pd.concat([xsplit, ysplit],axis= 1).reset_index(drop= True)
    df_smg = smogn.smoter(data= train_df, y= target_col,
                                    k = 8,
                                    pert = 0.05,
                                    samp_method = 'balance',
                                    drop_na_col = True,
                                    drop_na_row = True,
                                    replace = True,
                                    rel_thres = .8,
                                    rel_method = 'auto',
                                    rel_xtrm_type = 'both',
                                    rel_coef = 1.6
                                    )
        
    x_smg = df_smg.iloc[:,:-1]
    y_smg = df_smg[[target_col]]

    return x_smg, y_smg