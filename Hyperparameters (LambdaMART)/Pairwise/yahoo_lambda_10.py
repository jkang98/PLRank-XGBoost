import xgboost as xgb
import optuna

xgb_train = xgb.DMatrix("set1.train")
xgb_vali = xgb.DMatrix("set1.valid")

xgb_test = xgb.DMatrix("set1.test")

num_round = 50
watchlist = [(xgb_train, "train"), (xgb_vali, "vali")]

def objective(trial):
    param = {
        "eta": trial.suggest_float("eta", 0.05, 0.5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "gamma": trial.suggest_float("gamma", 0, 10),
	"alpha": trial.suggest_float("alpha", 0, 10),
        "objective": "rank:pairwise",
        "eval_metric": "ndcg@10",
    }

    model = xgb.train(
        param,
        xgb_train,
        num_round,
        evals=watchlist,
        early_stopping_rounds=10,
        verbose_eval=False,
    )
    return model.best_score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print("LambdaMART, K=10:")
print(study.best_trial.params)
