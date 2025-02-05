import torch
import numpy as np

@torch.no_grad()
def compute_mAP(matrix, ranks):
    """
    计算平均精度均值（mAP）

    参数:
    ranks -- 检索结果的排名，形状为 (return_size, query)
    matrix -- 匹配矩阵，形状为 (query, db_size)，1 表示匹配，0 表示不匹配

    返回:
    mAP -- 平均精度均值
    """
    num_queries = ranks.shape[1]
    aps = []

    for query_idx in range(num_queries):
        relevant = np.where(matrix[query_idx] == 1)[0]  # 获取所有相关样本的索引
        if relevant.size == 0:
            continue

        retrieved = ranks[:, query_idx]  # 获取检索结果的排名

        hit_count = 0
        sum_precisions = 0
        for i, retrieved_idx in enumerate(retrieved):
            if retrieved_idx in relevant:
                hit_count += 1
                precision_at_i = hit_count / (i + 1)
                sum_precisions += precision_at_i
        if hit_count > 0:
            ap = sum_precisions / hit_count
        else:
            ap = 0
        aps.append(ap)

    if len(aps) == 0:
        return 0.0  # 如果没有任何匹配样本，返回0
    return np.mean(aps)

@torch.no_grad()
def compute_ndcg(ranks, k):
    db_size, query_size = ranks.shape
    ndcg_k = 0
    for i in range(query_size):
        i_ranks = ranks[:, i]
        ideal_ranks = sorted(i_ranks, reverse=True)
        dcg = np.sum((2 ** np.asarray(i_ranks[:k]) - 1) / np.log2(np.arange(2, k+2)))
        idcg = np.sum((2 ** np.asarray(ideal_ranks[:k]) - 1) / np.log2(np.arange(2, k+2)))
        if idcg > 0:
            ndcg_k += (dcg / idcg)
    ndcg_k /= query_size
    return ndcg_k

@torch.no_grad()
def compute_accuracy(sim_matrix, ranks):
    db_size, query_size = ranks.shape
    accuracy = 0
    for i in range(query_size):
        accuracy_q = 0
        i_ranks = ranks[:, i]
        for rk in i_ranks:
            accuracy_q += sim_matrix[i, rk]
        accuracy += (accuracy_q / len(i_ranks))
    accuracy /= query_size
    return accuracy

@torch.no_grad()
def compute_precision(precision_score_matrix, ranks):
    db_size, query_size = ranks.shape
    precision = 0
    for i in range(query_size):
        precision_q = 0
        i_ranks = ranks[:, i]
        for rk in i_ranks:
            precision_q += precision_score_matrix[i, rk]
        precision += (precision_q / len(i_ranks))
    precision /= query_size
    return precision

@torch.no_grad()
def compute_recall(match_matrix, ranks):
    db_size, query_size = ranks.shape
    recall = 0
    for i in range(query_size):
        hit = 0
        i_ranks = ranks[:, i]
        for rk in i_ranks:
            hit += match_matrix[i, rk]
        recall += (hit / np.sum(match_matrix[i, :]))
    recall /= query_size
    return recall

@torch.no_grad()
def compute_f1(f1_score_matrix, ranks):
    db_size, query_size = ranks.shape
    f1 = 0
    for i in range(query_size):
        f1_q = 0
        i_ranks = ranks[:, i]
        for rk in i_ranks:
            f1_q += f1_score_matrix[i, rk]
        f1 += (f1_q / len(i_ranks))
    f1 /= query_size
    return f1

def compute_noise_rate(args, cfg, ranks):
    noise_number = []
    sim_count = cfg["sim_count"]
    for rk in ranks:
        tmp_cnt = 0
        for out_i in rk:
            if sim_count[out_i] < 1.0:
                tmp_cnt += 1
        noise_number.append(tmp_cnt)
    return np.mean(noise_number) / args.return_size

@torch.no_grad()
def compute_err(sim_matrix, ranks):
    db_size, query_size = ranks.shape
    ERR_k = 0
    for i in range(query_size):
        R_pre = 1.0
        err = 0
        i_ranks = ranks[:, i]
        for j, rk in enumerate(i_ranks):
            r_rk = (2**sim_matrix[i, rk] - 1) / 2
            err += R_pre*r_rk/(j + 1)
            R_pre *= (1-r_rk)
        ERR_k += err
    ERR_k /= query_size
    return ERR_k

@torch.no_grad()
def compute_recall_for166(sim_matrix, ranks):
    db_size, query_size = ranks.shape
    recall = 0
    for i in range(query_size):
        recall_q = 0
        i_ranks = ranks[:, i]
        total_pos = np.sum(sim_matrix[i, :])
        recall_pos = 0
        for rk in i_ranks:
            recall_pos += sim_matrix[i, rk]
        recall += recall_pos / total_pos
    recall /= query_size
    return recall

@torch.no_grad()
def compute_noise_ndcg(ranks):
    db_size, query_size = ranks.shape
    k = db_size
    ndcg_k = 0
    for i in range(query_size):
        i_ranks = ranks[:, i]
        ideal_ranks = sorted(i_ranks, reverse=True)
        dcg = np.sum((2 ** np.asarray(i_ranks) - 1) / np.log2(np.arange(2, k + 2)))
        idcg = np.sum((2 ** np.asarray(ideal_ranks) - 1) / np.log2(np.arange(2, k + 2)))
        if idcg > 0:
            ndcg_k += (dcg / idcg)
    ndcg_k /= query_size
    return ndcg_k