import optuna
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import time

xgb_train = xgb.DMatrix("set1.train")
xgb_vali = xgb.DMatrix("set1.valid")

xgb_test = xgb.DMatrix("set1.test")

def multiple_cutoff_rankings(scores, cutoff):
    n_samples = scores.shape[0]
    n_docs = scores.shape[1]
    cutoff = min(n_docs, cutoff)

    ind = np.arange(n_samples)
    partition = np.argpartition(scores, cutoff - 1)
    sorted_partition = np.argsort(scores[ind[:, None], partition[:, :cutoff]])
    rankings = partition[ind[:, None], sorted_partition]

    return rankings

def gumbel_sample_rankings(predict_scores, n_samples, cutoff=None):
    n_docs = len(predict_scores)
    ind = np.arange(n_samples)

    if cutoff:
        ranking_len = min(n_docs, cutoff)
    else:
        ranking_len = n_docs

    gumbel_samples = np.random.gumbel(size=(n_samples, n_docs))
    gumbel_scores = predict_scores + gumbel_samples

    rankings = multiple_cutoff_rankings(-gumbel_scores, ranking_len)

    return rankings

def PL_rank_2_grad(rank_weights, labels, scores, n_samples):
    n_docs = len(labels)
    cutoff = min(rank_weights.shape[0], n_docs)

    sampled_rankings = gumbel_sample_rankings(scores, n_samples, cutoff=cutoff)

    srange = np.arange(n_samples)
    crange = np.arange(cutoff)

    relevant_docs = np.where(np.not_equal(labels, 0))[0]

    weighted_labels = labels[sampled_rankings] * rank_weights[None, :cutoff]
    cumsum_labels = np.cumsum(weighted_labels[:, ::-1], axis=1)[:, ::-1]

    # first order
    result1 = np.zeros(n_docs, dtype=np.float64)
    np.add.at(result1, sampled_rankings[:, :-1], cumsum_labels[:, 1:])
    result1 /= n_samples

    placed_mask = np.zeros((n_samples, cutoff - 1, n_docs), dtype=bool)
    placed_mask[srange[:, None], crange[None, :-1], sampled_rankings[:, :-1]] = True
    placed_mask[:, :] = np.cumsum(placed_mask, axis=1)

    total_denom = np.logaddexp.reduce(scores)
    minus_denom = np.logaddexp.accumulate(scores[sampled_rankings[:, :-1]], axis=1)
    denom_per_rank = (
        np.log(np.maximum(1.0 - np.exp(minus_denom - total_denom), 1e-8)) + total_denom
    )
    prob_per_rank = np.empty((n_samples, cutoff, n_docs), dtype=np.float64)
    prob_per_rank[:, 0, :] = np.exp(scores[None, :] - total_denom)
    prob_per_rank[:, 1:, :] = np.exp(scores[None, None, :] - denom_per_rank[:, :, None])
    prob_per_rank[:, 1:, :][placed_mask] = 0.0

    result1 -= np.mean(
        np.sum(prob_per_rank * cumsum_labels[:, :, None], axis=1),
        axis=0,
        dtype=np.float64,
    )

    result1[relevant_docs] += np.mean(
        np.sum(
            prob_per_rank[:, :, relevant_docs]
            * (rank_weights[None, :cutoff, None] * labels[None, None, relevant_docs]),
            axis=1,
        ),
        axis=0,
        dtype=np.float64,
    )
    return -result1


def PL_rank_2_hess(rank_weights, labels, scores, n_samples):
    n_docs = len(labels)
    cutoff = min(rank_weights.shape[0], n_docs)

    sampled_rankings = gumbel_sample_rankings(scores, n_samples, cutoff=cutoff)

    srange = np.arange(n_samples)
    crange = np.arange(cutoff)

    relevant_docs = np.where(np.not_equal(labels, 0))[0]

    weighted_labels = labels[sampled_rankings] * rank_weights[None, :cutoff]
    cumsum_labels = np.cumsum(weighted_labels[:, ::-1], axis=1)[:, ::-1]

    placed_mask = np.zeros((n_samples, cutoff - 1, n_docs), dtype=bool)
    placed_mask[srange[:, None], crange[None, :-1], sampled_rankings[:, :-1]] = True
    placed_mask[:, :] = np.cumsum(placed_mask, axis=1)

    total_denom = np.logaddexp.reduce(scores)
    minus_denom = np.logaddexp.accumulate(scores[sampled_rankings[:, :-1]], axis=1)
    denom_per_rank = (
        np.log(np.maximum(1.0 - np.exp(minus_denom - total_denom), 1e-8)) + total_denom
    )
    prob_per_rank = np.empty((n_samples, cutoff, n_docs), dtype=np.float64)
    prob_per_rank[:, 0, :] = np.exp(scores[None, :] - total_denom)
    prob_per_rank[:, 1:, :] = np.exp(scores[None, None, :] - denom_per_rank[:, :, None])
    prob_per_rank[:, 1:, :][placed_mask] = 0.0

    # second order
    result2 = np.zeros(n_docs, dtype=np.float64)
    prob_per_doc = np.sum(prob_per_rank, axis=1)
    np.add.at(
        result2,
        sampled_rankings[:, :-1],
        (1 - prob_per_doc[srange[:, None], sampled_rankings[:, :-1]])
        * cumsum_labels[:, 1:],
    )
    result2 /= n_samples

    in_or_not = np.zeros((n_samples, cutoff, n_docs), dtype=np.float64)
    in_or_not[srange[:, None], :, sampled_rankings[:, :]] = 1
    long_item = 1 + in_or_not - prob_per_rank - prob_per_doc[:, None, :]

    result2 -= np.mean(
        np.sum((prob_per_rank * cumsum_labels[:, :, None]) * long_item, axis=1),
        axis=0,
        dtype=np.float64,
    )

    result2[relevant_docs] += np.mean(
        np.sum(
            (
                prob_per_rank[:, :, relevant_docs]
                * (
                    rank_weights[None, :cutoff, None]
                    * labels[None, None, relevant_docs]
                )
            )
            * long_item[:, :, relevant_docs],
            axis=1,
        ),
        axis=0,
        dtype=np.float64,
    )

    return -result2

def plrank2obj(preds, dtrain):
    group_ptr = dtrain.get_uint_info("group_ptr")
    labels = dtrain.get_label()

    # number of rankings
    n_samples = 50

    grad = np.zeros(len(labels), dtype=float)
    hess = np.zeros(len(labels), dtype=float)

    group = np.diff(group_ptr)
    max_query_size = max(group)
    longest_metric_weights = 1.0 / np.log2(np.arange(max_query_size) + 2)

    # number of docs to display
    cutoff = 10

    max_ranking_size = np.min((cutoff, max_query_size))
    metric_weights = longest_metric_weights[:max_ranking_size]

    for q in range(len(group_ptr) - 1):
        q_l = labels[group_ptr[q] : group_ptr[q + 1]]
        scores = preds[group_ptr[q] : group_ptr[q + 1]]

        # first order
        grad[group_ptr[q] : group_ptr[q + 1]] = PL_rank_2_grad(
            metric_weights, q_l, scores, n_samples
        )
        # second order
        hess[group_ptr[q] : group_ptr[q + 1]] = PL_rank_2_hess(
            metric_weights, q_l, scores, n_samples
        )

    return grad, hess

# plrank2
params1 = {
    "eta": 0.2825686907846459,
    "min_child_weight": 2,
    "max_depth": 10,
    "gamma": 0.37694995816176524,
    "alpha": 2.147985171102071,
    "eval_metric": "ndcg@10",
}
num_round = 200
watchlist = [(xgb_train, "train"), (xgb_test, "test")]
evals_result1 = {}

begin1 = time.time()
model1 = xgb.train(
    params1,
    xgb_train,
    num_round,
    evals=watchlist,
    obj=plrank2obj,
    evals_result=evals_result1,
)
end1 = time.time()
print(end1 - begin1)

# lambdamart pairwise
params2 = {
    "objective": "rank:pairwise",
    "eta": 0.31104809709296927,
    "min_child_weight": 2,
    "max_depth": 10,
    "gamma": 2.5443861265864394,
    "alpha": 9.964264316417191,
    "eval_metric": "ndcg@10",
}
evals_result2 = {}

begin2 = time.time()
model2 = xgb.train(
    params2, xgb_train, num_round, evals=watchlist, evals_result=evals_result2,
)
end2 = time.time()
print(end2 - begin2)

# lambdamart listwise
params3 = {
    "objective": "rank:ndcg",
    "eta": 0.49642080741242395,
    "min_child_weight": 9,
    "max_depth": 9,
    "gamma": 0.008883654390199447,
    "alpha": 6.449291637697138,
    "eval_metric": "ndcg@10",
}
evals_result3 = {}

begin3 = time.time()
model3 = xgb.train(
    params3, xgb_train, num_round, evals=watchlist, evals_result=evals_result3,
)
end3 = time.time()
print(end3 - begin3)

# make a plot
t = np.arange(1, 201)
plt.plot(t, evals_result1["test"]["ndcg@10"], label="PL-Rank-2 (N = 100, K = 10)")
plt.plot(t, evals_result2["test"]["ndcg@10"], label="LambdaMART (Pairwise)")
plt.plot(t, evals_result3["test"]["ndcg@10"], label="LambdaMART (Listwise)")

plt.legend()
plt.xlabel("Rounds")
plt.ylabel("NDCG@10")
my_x_ticks = np.arange(0, 201, 20)
plt.xticks(my_x_ticks)

plt.savefig("figures/compare_10.png", dpi=128)