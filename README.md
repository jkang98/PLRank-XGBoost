# PLRank-XGBoost

Usage
-------

Raw Learning to Rank dataset cannot be loaded in XGBoost directly, it needs to be processed into XGBoost loadable data files through a preprocessing function. The function we use is the [trans_data.py](https://github.com/dmlc/xgboost/blob/master/demo/rank/trans_data.py) from [XGBoost](https://github.com/dmlc/xgboost).

Call *trans_data.py* to process the original data file, the command is as follows:
```
python trans_data.py [Ranksvm Format Input] [Output Feature File] [Output Group File]
```

For the [Yahoo! Learn to Rank Challenge](https://webscope.sandbox.yahoo.com/catalog.php?datatype=c&did=64):
```
python xgboost/demo/rank/trans_data.py dataset/ltrc_yahoo/set1.train.txt set1.train set1.train.group
python xgboost/demo/rank/trans_data.py dataset/ltrc_yahoo/set1.test.txt set1.test set1.test.group
python xgboost/demo/rank/trans_data.py dataset/ltrc_yahoo/set1.valid.txt set1.valid set1.valid.group
```

The following command compares the NDCG@K results of different algorithms:

1. PL-Rank-2 (optimizes DCG@K with 100 sampled rankings, half for estimating first order derivative, half for estimating second order derivative)

2. LambdaMART (performs pairwise ranking where the pairwise loss is minimized)

3. LambdaMART (performs listwise ranking where NDCG is maximized)
```
python Experiments/compare_for_K.py
```

