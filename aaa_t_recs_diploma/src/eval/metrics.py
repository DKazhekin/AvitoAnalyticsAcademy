import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def precision_at_k(y_true_list, y_pred_matrix, k: int) -> float:
    precisions = []
    for i in range(len(y_true_list)):
        y_true = set(y_true_list[i])
        y_pred = set(y_pred_matrix[i][:k])
        intersection = len(y_true & y_pred)
        precisions.append(intersection / k)

    return float(np.mean(precisions))


def recall_at_k(y_true_list, y_pred_matrix, k: int) -> float:
    recalls = []
    for i in range(len(y_true_list)):
        y_true = set(y_true_list[i])
        if not y_true:
            continue
        y_pred = set(y_pred_matrix[i][:k])
        intersection = len(y_true & y_pred)
        recalls.append(intersection / len(y_true))

    return float(np.mean(recalls)) if recalls else 0.0


def ndcg_at_k(y_true_list, y_pred_matrix, k: int, relevance_scores=None) -> float:
    ndcgs = []
    for i in range(len(y_true_list)):
        if relevance_scores is None:
            y_true = set(y_true_list[i])
            rel = np.array(
                [1 if item in y_true else 0 for item in y_pred_matrix[i][:k]]
            )
        else:
            rel = np.array(
                [relevance_scores[i].get(item, 0) for item in y_pred_matrix[i][:k]]
            )

        discounts = np.log2(np.arange(2, k + 2))
        dcg = np.sum(rel / discounts)

        ideal_rel = np.sort(rel)[::-1]
        idcg = np.sum(ideal_rel / discounts)

        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return np.mean(ndcgs) if ndcgs else 0.0


def diversity_at_k(recommendations_embeddings, k=10) -> float:
    top_k_embeddings = recommendations_embeddings[:, :k, :]

    similarities = []
    for user_recs in top_k_embeddings:
        sim_matrix = cosine_similarity(user_recs)
        upper_tri = sim_matrix[np.triu_indices(k, 1)]
        similarities.extend(upper_tri)

    avg_similarity = np.mean(similarities)

    return 1 - avg_similarity


def common_metrics(y_true_list, y_pred_matrix, k: int, relevance_scores=None) -> str:
    precision = precision_at_k(y_true_list, y_pred_matrix, k=k)
    recall = recall_at_k(y_true_list, y_pred_matrix, k=k)
    ndcg = ndcg_at_k(y_true_list, y_pred_matrix, k=k, relevance_scores=relevance_scores)

    return f"precision: {precision}\nrecall: {recall}\nndcg: {ndcg}"
