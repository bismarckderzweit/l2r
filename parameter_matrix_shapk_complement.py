import numpy as np
import subprocess
import os
import shap
import scipy.stats as stats
from multiprocessing import Pool
from utils.rerank import write_average, rerank_ndcg,write_tau,write_ratio
from utils.readdata import get_microsoft_data, rewrite
from utils.separate_set import separate_set
from utils.explainer_tools import rand_row, evaluate, get_rankedduculist, get_set_cover


def score(X):
    A = []
    # this part is for getting shapley values
    scorefile_path = temp_path + 'scorefile_shapk_{}.txt'.format(tmp_test_y_query[0].split(':')[-1].split()[0])
    restore_path = temp_path + 'restore_shapk_{}.txt'.format(tmp_test_y_query[0].split(':')[-1].split()[0])
    rewrite(X, tmp_test_y_query, tmp_test_Query, restore_path)
    args = ['java', '-jar', 'RankLib-2.12.jar', '-rank', restore_path, '-load', model,
            '-indri', scorefile_path]
    subprocess.check_output(args, stderr=subprocess.STDOUT, timeout=2000)

    # rerank the scorefile according the original index
    scorefile_data = ''.join(sorted(open(scorefile_path), key=lambda s: s.split()[1], reverse=False))
    with open(scorefile_path, 'w') as f:
        f.write(scorefile_data)
    with open(scorefile_path, 'r') as f:
        for line in f:
            A.append(float(line.split()[-2]))

    # reset the index to be original otherwise can not get the right NDCG
    restore_context = open(restore_path, 'r').readlines()
    with open(restore_path, 'w') as f:
        for lineindex in range(len(restore_context)):
            split = restore_context[lineindex].split()
            split[1] = 'qid:{}'.format(tmp_test_y_query[0].split(':')[-1].split()[0])
            newline = ''
            for i in range(len(split)):
                newline += (split[i] + ' ')
            f.write(newline + '\n')

    A = np.array(A)
    return A


def loop_query(query_index):
    """
    loop for a query, get scores of the samples of this query and rank them according to the scores
    :param query_index: the index of query
    :return: ranklist file, delta NDCG file
    """
    # get data for this query
    global tmp_test_data
    global tmp_test_y_query
    global tmp_test_Query
    tmp_test_data = test_data[query_index]
    tmp_test_y_query = test_y_query[query_index]
    tmp_test_Query = test_Query[query_index]
    query_id = tmp_test_y_query[0].split(':')[-1].split()[0]

    # calculate the scores for the q-d pairs
    scores = score(tmp_test_data).reshape(-1, 1)
    restore_path = temp_path + 'restore_shapk_{}.txt'.format(query_id)
    scorefile_path = temp_path + 'scorefile_shapk_{}.txt'.format(query_id)

    # reranking the test_data according to the scores and get the list of ranking
    test_data_score = np.append(tmp_test_data, scores, axis=1)
    ranked_test_data = (test_data_score[(-test_data_score[:, -1]).argsort()])[:, :-1]
    rankedduculist1 = get_rankedduculist(scores, query_index, q_d_len)
    NDCG_before = evaluate(model, restore_path)

    NDCG_file_name = NDCGdata_path + '{}_SHAP{}_complement'.format(dataname, k) + modelname + '.txt'
    ranklist_file = NDCGdata_path + '{}_ranklist_SHAP{}_complement'.format(dataname, k) + modelname + '.txt'
    all_features = [i for i in range(46)]
    top_k_idx = feature_set[query_index]
    complement_idx = list(set(all_features) - set(top_k_idx[:5]))
    features_to_change = tmp_test_data
    features_to_change[:, complement_idx] = expected_value[complement_idx]
    # get scores of the changed features
    scores2 = score(features_to_change).reshape(-1, 1)
    NDCG_after = evaluate(model, restore_path)
    delta_NDCG = abs(float(NDCG_before) - float(NDCG_after))
    if float(NDCG_before) == 0:
        ratio_NDCG = 0
    else:
        ratio_NDCG = delta_NDCG / float(NDCG_before)

    rankedduculist2 = get_rankedduculist(scores2, query_index, q_d_len)
    tau, p_value = stats.kendalltau(rankedduculist1, rankedduculist2)
    os.remove(scorefile_path)
    os.remove(restore_path)
    with open(NDCG_file_name, 'a') as NDCG_FILE:
        NDCG_line = tmp_test_y_query[0].split(':')[-1] + '  ' + \
                    'changed feature:' + str(complement_idx) + '  ' + 'kendalltau=' + str(
            round(tau, 4)) + '  ' + 'ratioNDCG:' + str(round(ratio_NDCG, 4)) + '  ' + 'delta_NDCG =' + '  ' + str(
            delta_NDCG) + "\n"
        NDCG_FILE.write(NDCG_line)
    with open(ranklist_file, 'a') as ranklist:
        ranklist_line = tmp_test_y_query[0].split(':')[-1] + '  ' + 'ranklist before:' + str(
            rankedduculist1) + '  ' + 'ranklist after:' + '  ' + str(rankedduculist2) + "\n"
        ranklist.write(ranklist_line)


if __name__ == '__main__':
    # parameters to be set
    model_path = 'model/'
    model = model_path + 'Linearregression_model.txt'
    k_set = [1, 5]  # k: shap k, we select the top kexample to do analysis

    for f in range(1, 2):
        # the path of data
        # datapath = 'MQ2008/Fold1/'
        datapath = 'MQ2008/Fold{}/'.format(f)
        # datapath = 'MSLR-WEB10K/Fold{}/'.format(f)
        train_path = datapath + 'train.txt'
        test_path = datapath + 'test.txt'
        modelname = model.split("_")[0].split("/")[-1]
        dataname = datapath.split('/')[0] + '_' + datapath.split('/')[1].split('Fold')[1]

        # saving path and save files
        NDCGdata_path = 'NDCGdata/'
        temp_path = 'temp_data_shapk_complement/'

        # get train data and test data
        X_train, y_query_train, Query_train = get_microsoft_data(train_path)
        X_train = np.array(X_train)
        X_test, y_query_test, Query_test = get_microsoft_data(test_path)
        X_test = np.array(X_test)
        expected_value = np.mean(X_train, axis=0)

        # separate the test set
        test_data, test_y_query, test_Query, q_d_len = separate_set(y_query_test, X_test, Query_test)

        for k in k_set:
            resultfile_NDCG = 'resultfile/' + '{}_{}_shapk{}_complement_NDCG.txt'.format(dataname, modelname, k)
            resultfile_tau = 'resultfile/' + '{}_{}_shapk{}_complement_tau.txt'.format(dataname, modelname, k)
            resultfile_ratio = 'resultfile/' + '{}_{}_shapk{}_complement_ratio.txt'.format(dataname, modelname, k)
            old_NDCG_file = 'NDCGdata/' + '{}_SHAP{}_10features_threshold0{}.txt'.format(dataname, k, modelname)
            feature_set = []
            with open(old_NDCG_file, 'r') as oldfile:
                for line in oldfile:
                    features = ((line.split('[')[-1].split(']')[0])).split()
                    if len(features) < 10: break
                    this_feature_set = [int(features[i]) for i in range(10)]
                    feature_set.append(this_feature_set)
            with Pool(10) as p:
                print(p.map(loop_query, [query_index for query_index in range(len(test_data))]))

            NDCG_file_name = NDCGdata_path + '{}_SHAP{}_complement'.format(dataname, k) + modelname + '.txt'
            ranklist_file = NDCGdata_path + '{}_ranklist_SHAP{}_complement'.format(dataname, k) + modelname + '.txt'
            rerank_ndcg(NDCG_file_name)
            rerank_ndcg(ranklist_file)
            tau = write_tau(NDCG_file_name)
            NDCG = write_average(NDCG_file_name)
            ratio = write_ratio(NDCG_file_name)
            with open(resultfile_NDCG, 'a') as NDCG_result:
                NDCG_result_line = str(NDCG) + "\n"
                NDCG_result.write(NDCG_result_line)
            with open(resultfile_tau, 'a') as tau_result:
                tau_result_line = str(tau) + "\n"
                tau_result.write(tau_result_line)
            with open(resultfile_ratio, 'a') as ratio_result:
                ratio_result_line = str(ratio) + "\n"
                ratio_result.write(ratio_result_line)