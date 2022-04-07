# LOGISTICS: PREDICTING VESSEL DISCHARGE
## INTRODUCTION
This repo shows the data processing and the development of a machine learning model to predict vessel discharge value from a limited dataset ($n\approx1000$) using **XGBoost** regression trees.

## DATASET
The data is a vessel transshipment record from a global port which logs the a set features of each vessel along with some of the vessels' load and discharge values. Every vessel is characterized by a set of ``21`` features that includes: **{arrival eta, arrival type, leave eta, ship weight, vessel type, stevedores' names, HaMIS,...}**. The task is to predict vessel ``discharge`` based on the provided feature set.

### MISSING TARGET DATA
The initial size of the dataset is ``8208`` points with only a total of ``1184`` known vessel ``discharge`` values and a mere ``515`` known vessel load values.
<p align="center">
    <img src= './media/na_matrix.png' width= 800></br>
    <i>Missing data matrix</i></br>
</p>

The record logs ``3`` vessel types encoded numerically from the set ``{2, 3, 5}``. Below is a breakdown of missing data and total target values per vessel type.

<p align="center">
    <img src= './media/na_vessel.png' width= 350>
    <img src= './media/value_vessel.png' width= 350></br>
    <img src= './media/labels_dists.png' width= 700></br>
    <i><b>Top left</b>: Null counts per vessel type | <b>Top right</b>: Discharge/Load sums per vessel type </br> <b>Bottom</b>: KDEs of discharge & load values </i> </br>
</p>

The graphs show that despite vessel type ``5`` harbors most of the missing data values, it also has the greatest values for ``discharge`` and ``load``.
## REGRESSION IMBALANCE
The dataset contains a significant imbalance both in the feature space and the target space, for each vessel type. Both the feature and target spaces over-represent vessel type ``5`` which also has the greatest target values. Below, the graphs show the count distributions over the feature and target spaces.

<p align="center">
    <img src= './media/label_boxplot.png' width= 700>
    <img src= './media/label_vesselcount.png' width= 640></br>
    <i><b>Top left</b>: Load values distribution per vessel type | <b>Top right</b>: Discharge values distribution per vessel type </br> <b>Bottom</b>: Count of targets and rows per vessel type </i> </br>
</p>


### 1. Synthetic Minority Over-Sampling with Gaussian Noise (SMOGN)