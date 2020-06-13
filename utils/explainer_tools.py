import numpy as np
import subprocess
from itertools import combinations

def rand_row(array,dim_needed):
    """
    randomly get background data
    :param array: the whole train data
    :param dim_needed: how much rows of background data you want
    :return: random background data
    """
    row_total = array.shape[0]
    row_sequence = np.arange(row_total)
    np.random.shuffle(row_sequence)
    return array[row_sequence[0:dim_needed],:]


def get_set_cover_beam(shap_values):
    shap_values =np.array([shap_values])
    sumvalue = np.sum(shap_values,axis=1)
    feature_index=((-sumvalue).argsort())[0]
    return feature_index

def get_set_cover2(shap_values):
    shap_values =np.array([shap_values])
    sumvalue = np.sum(shap_values,axis=1)
    feature_index=((-sumvalue).argsort())[0][:2]
    return feature_index

def get_set_cover(shap_values):
    """
    get scores of the samples of this query and rank them according to the scores,
    we select the 10_top important features
    :param shap_values:
    :return:
    """
    shap_values =np.array([shap_values])
    sumvalue = np.sum(shap_values,axis=1)
    feature_index=((-sumvalue).argsort())[0][0]
    return feature_index

"""
def get_set_cover(shap_values, threshold_flag):

    shap_values =np.array([shap_values])
    #shap_values[shap_values < 0] = 0
    sumvalue = np.sum(shap_values,axis=1)
    mean =  sumvalue/shap_values.shape[1]
    shap_values_std = np.std(shap_values,ddof=1, axis=1)
    top_k = 10
    if threshold_flag == 0:
        top_k_idx=((-sumvalue).argsort())[0][0:top_k]
        return top_k_idx, mean
    elif threshold_flag== 1:
        threshold = 2*mean
    elif threshold_flag== 2:
        threshold = mean - 3*shap_values_std
    elif threshold_flag==3:
        threshold = 15*mean
    elif threshold_flag ==4:
        threshold = [[0 for i in range(shap_values.shape[2])]]
    top_k_idx = []
    for i in range(top_k):
        feature_index=((-sumvalue).argsort())[0][0]
        top_k_idx.append(feature_index)
        bigindex = list(np.where(shap_values[0][:,feature_index]>threshold[0][feature_index]))
        shap_values[0][:,feature_index] = 0
        shap_values[0][bigindex] = 0
        sumvalue = np.sum(shap_values, axis=1)
        if np.all(sumvalue == 0): break
    return top_k_idx, mean
"""

def evaluate(model, restore_path):
    """
    calling the ranklib the evaluate the NDCG of the ranklist
    :param model: the model we choosed
    :param restore_path:
    :return: NDCG@10
    """
    args = ['java', '-jar', 'RankLib-2.12.jar', '-load', model, '-test', restore_path,
            '-metric2T', 'NDCG@10']
    process = subprocess.check_output(args, stderr=subprocess.STDOUT, timeout=5000)
    metric = ((str(process, 'utf-8').splitlines())[-1]).split(' ')[-1]
    return metric

"""


def get_pairsname(ranked_test_data, pairnumbers):

    pairs = np.array([c for c in combinations(range(1, len(ranked_test_data) + 1), 2)])
    maxi = pairs.max(axis = 0)[0]
    sumi = sum([maxi+1-pairs[i,0] for i in range(pairs.shape[0])])
    prob = [(maxi+1-pairs[i,0])/sumi for i in range(pairs.shape[0])]
    pairs = pairs[np.random.choice(pairs.shape[0], pairnumbers , replace=False, p = prob), :]
    pairs = pairs[np.argsort(pairs[:, 0])]
    pairsname = ['{}>{}'.format('d' + str(pairs[i][0]), 'd' + str(pairs[i][1])) for i in range(pairnumbers)]
    return pairsname

"""
def get_weight_pairsname(ranked_test_data, pairnumbers):
    pairs = np.array([c for c in combinations(range(1, len(ranked_test_data) + 1), 2)])
    maxi = pairs.max(axis = 0)[0]
    sumi = sum([maxi+1-pairs[i,0] for i in range(pairs.shape[0])])
    prob = [(maxi+1-pairs[i,0])/sumi for i in range(pairs.shape[0])]
    pairs = pairs[np.random.choice(pairs.shape[0], pairnumbers , replace=False, p = prob), :]
    pairs = pairs[np.argsort(pairs[:, 0])]
    pairsname = ['{}>{}'.format('d' + str(pairs[i][0]), 'd' + str(pairs[i][1])) for i in range(pairnumbers)]
    return pairsname

def get_pairsname(ranked_test_data, pairnumbers):
    """
    randomly get pairs for fulling the matrix
    :param ranked_test_data: the ranked features_matrix according to the scores
    :param pairnumbers:
    :return: the string of pairs
    """
    pairs = np.array([c for c in combinations(range(1, len(ranked_test_data) + 1), 2)])
    pairs = pairs[np.random.choice(pairs.shape[0], pairnumbers , replace=False), :]
    pairs = pairs[np.argsort(pairs[:, 0])]
    pairsname = ['{}>{}'.format('d' + str(pairs[i][0]), 'd' + str(pairs[i][1])) for i in range(pairnumbers)]
    return pairsname


def small_get_pairsname(ranked_test_data):
    """
    randomly get pairs for fulling the matrix
    :param ranked_test_data: the ranked features_matrix according to the scores
    :param pairnumbers:
    :return: the string of pairs
    """
    pairnumbers = int(len(ranked_test_data)*(len(ranked_test_data)-1)/2)
    pairs = np.array([c for c in combinations(range(1, len(ranked_test_data) + 1), 2)])
    pairs = pairs[np.argsort(pairs[:, 0])]
    pairsname = ['{}>{}'.format('d' + str(pairs[i][0]), 'd' + str(pairs[i][1])) for i in range(pairnumbers)]
    return pairsname

def get_rankedduculist(scores,query_index,q_d_len):
    """
    get the ranklist(list of index) according to the scores
    :param scores:
    :param query_index:
    :param q_d_len: how much docus for this query
    :return: list of index
    """
    duculist =np.array([i for i in range(q_d_len[query_index])]).reshape(-1,1)
    doculist_score = np.append(duculist,scores,axis=1)
    rankedduculist  = (doculist_score[(-doculist_score[:,-1]).argsort()])[:,0]
    rankedduculist  = [int(i) for i in rankedduculist]
    return rankedduculist