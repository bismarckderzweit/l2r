import numpy as np
import subprocess
import os
import shap
import scipy.stats as stats
from multiprocessing import Pool
from utils.rerank import write_average, rerank_ndcg, rerank_matrix,write_tau,write_ratio
from utils.readdata import get_microsoft_data, rewrite
from utils.separate_set import separate_set
from utils.explainer_tools import rand_row, evaluate, get_pairsname, get_rankedduculist,small_get_pairsname, get_set_cover


def score(X):
    """
    The first if branch is training data, the next is for the single test data. First calling the subprocess of ranklib
    to get the scores, then rerank the scorefile according the original index. We also have to delete the produced
    files which used by the subprocess.
    :param X: input feature matrix
    :return: scores of q-d pairs
    """
    A = []
    scorefile_path = temp_path + 'scorefile_validity_{}.txt'.format(tmp_test_y_query[0].split(':')[-1].split()[0])
    restore_path = temp_path + 'restore_validity_{}.txt'.format(tmp_test_y_query[0].split(':')[-1].split()[0])
    rewrite(X, tmp_test_y_query, tmp_test_Query, restore_path)
    args = ['java', '-jar', 'RankLib-2.12.jar', '-rank', restore_path, '-load', model,
            '-indri', scorefile_path]
    subprocess.check_output(args, stderr=subprocess.STDOUT)

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
    :return: ranklist file, matrix file, delta NDCG file
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
    feature_number = 5
    restore_path = temp_path + 'restore_validity_{}.txt'.format(query_id)
    scorefile_path = temp_path + 'scorefile_validity_{}.txt'.format(query_id)
    scores = score(tmp_test_data).reshape(-1, 1)

    # reranking the test_data according to the scores and get the list of ranking
    test_data_score = np.append(tmp_test_data, scores, axis=1)
    ranked_test_data = np.array((test_data_score[(-test_data_score[:, -1]).argsort()])[:, :-1])
    rankedduculist1 = get_rankedduculist(scores, query_index, q_d_len)
    NDCG_before = evaluate(model, restore_path)

    # get pairsname
    global pairsname
    if q_d_len[query_index] >= 11:
        pairnumbers = 50
        pairsname = get_pairsname(ranked_test_data, pairnumbers)
    else:
        pairsname = small_get_pairsname(ranked_test_data)

    def get_score_matrix(feature_matrix):
        changed_list = []
        for i in range(feature_matrix.shape[0]):
            for m in range(tmp_test_data.shape[1]):
                temp = expected_value.copy()
                temp[m] = feature_matrix[i, m]
                temp[top_k_idx] = feature_matrix[i, top_k_idx]
                changed_list.append(temp)
        changed_list = np.array(changed_list)
        with open(temp_path + 'changed_list_validity{}.txt'.format(query_index), 'w') as f:
            for i in range(feature_matrix.shape[0] * tmp_test_data.shape[1]):
                line = ""
                line += "0 qid:{} ".format(str(i))
                for j in range(len(changed_list[i])):
                    line += ((str(j + 1)) + ":" + str(changed_list[i][j]) + " ")
                line += '#docid = GX008-86-4444840 inc = 1 prob = 0.086622 ' + "\n"
                f.write(line)
        args = ['java', '-jar', 'RankLib-2.12.jar', '-rank',
                temp_path + 'changed_list_validity{}.txt'.format(query_index), '-load', model,
                '-indri', temp_path + 'changed_list_validity_score{}.txt'.format(query_index)]
        subprocess.check_output(args, stderr=subprocess.STDOUT)
        A = ''.join(sorted(open(temp_path + 'changed_list_validity_score{}.txt'.format(query_index)),
                           key=lambda s: int(s.split()[0]), reverse=False))
        with open(temp_path + 'changed_list_validity_score{}.txt'.format(query_index), 'w') as f:
            f.write(A)
        changed_list_score = []
        with open(temp_path + 'changed_list_validity_score{}.txt'.format(query_index), 'r') as f:
            for line in f:
                changed_list_score.append(float(line.split()[-2]))
        changed_list_score = [changed_list_score[i:i + tmp_test_data.shape[1]] for i in
                              range(0, len(changed_list_score), tmp_test_data.shape[1])]
        os.remove(os.path.join(temp_path, 'changed_list_validity{}.txt'.format(query_index)))
        os.remove(os.path.join(temp_path, 'changed_list_validity_score{}.txt'.format(query_index)))
        return changed_list_score

    def get_matrix(ranked_test_data):
        score_values = get_score_matrix(ranked_test_data)
        matrix = []
        for i in range(len(pairsname)):
            index1 = int(pairsname[i][1])
            index2 = int(pairsname[i][-1])
            # row = [(sorted_scores[index1-1]-sorted_scores[index2-1]-score_values[index1-1][j] + score_values[index2-1][j])*(index2 - index1) for j in range(tmp_test_data.shape[1])]
            row = [round((score_values[index1 - 1][j] - score_values[index2 - 1][j]), 4) for j in
                   range(tmp_test_data.shape[1])]
            # row = [(score_values[index2-1][j] - score_values[index1-1][j])*(index2 - index1) for j in range(tmp_test_data.shape[1])]
            matrix.append(row)
        return matrix

    def feature_k_loop(feature_number):
        NDCG_file_name = NDCGdata_path + '{}_validity_{}features'.format(dataname, feature_number) + modelname + '.txt'
        NDCG_file_matrix = NDCGdata_path + '{}_validity_matrix_{}features'.format(dataname,
                                                                                  feature_number) + modelname + '.txt'
        ranklist_file = NDCGdata_path + '{}_ranklist_validity_{}features'.format(dataname,
                                                                                 feature_number) + modelname + '.txt'
        features_to_change = tmp_test_data

        # get the first index
        global top_k_idx
        top_k_idx = []
        matrix = get_matrix(ranked_test_data)
        temp_index = get_set_cover(matrix)
        top_k_idx.append(temp_index)
        temp_tmp_test_data = tmp_test_data

        # get the left 9 indexes
        for i in range(9):
            temp_tmp_test_data[:, temp_index] = expected_value[temp_index]
            scores_temp = score(temp_tmp_test_data).reshape(-1, 1)
            rankedduculist_temp = get_rankedduculist(scores_temp, query_index, q_d_len)
            changedpairs = []

            for i in range(len(rankedduculist1) - 1):
                for j in range(i + 1, len(rankedduculist1)):
                    doc1 = rankedduculist1[i]
                    doc2 = rankedduculist1[j]
                    if rankedduculist_temp.index(doc1) > rankedduculist_temp.index(doc2):
                        changedpairs.append([i, j])

            for i in range(len(changedpairs)):
                if len(pairsname) <= 1: break
                deletedpair = 'd' + str(changedpairs[i][0]) + '>d' + str(changedpairs[i][1])
                if deletedpair in pairsname:
                    pairsname.remove(deletedpair)

            # temp_ranked_test_data[:,temp_index] = expected_value[temp_index]
            temp_matrix = get_matrix(ranked_test_data)

            # delect the features we already selected
            all_features = [i for i in range(46)]
            left_idx = list(set(all_features) - set(top_k_idx))
            temp_matrix = list(map(list, zip(*temp_matrix)))
            matrix = []
            for i in left_idx:
                matrix.append(temp_matrix[i])
            matrix = list(map(list, zip(*matrix)))
            temp_index = get_set_cover(matrix)
            temp_top_k_idx = top_k_idx.copy()
            for i in temp_top_k_idx:
                if i <= temp_index:
                    temp_index += 1
                    while temp_index in temp_top_k_idx:
                        temp_top_k_idx.remove(temp_index)
                        temp_index += 1

            top_k_idx.append(temp_index)

            with open(NDCG_file_matrix, 'a') as matrix_FILE:
                matrix_line = 'matrix for {}'.format(tmp_test_y_query[0].split(':')[-1].split()[0]) \
                              + '  ' + str(matrix) + '  ' + "\n"
                matrix_FILE.write(matrix_line)

        if len(top_k_idx) <= feature_number:
            feature_number = len(top_k_idx)
        features_to_change[:, top_k_idx[0:feature_number]] = expected_value[top_k_idx[0:feature_number]]
        restore_path = temp_path + 'restore_validity_{}.txt'.format(query_id)
        scorefile_path = temp_path + 'scorefile_validity_{}.txt'.format(query_id)
        # get scores of the changed features
        scores2 = score(features_to_change).reshape(-1, 1)
        rankedduculist2 = get_rankedduculist(scores2, query_index, q_d_len)
        NDCG_after = evaluate(model, restore_path)
        delta_NDCG = abs(float(NDCG_before) - float(NDCG_after))
        if float(NDCG_before) == 0:
            ratio_NDCG = 0
        else:
            ratio_NDCG = delta_NDCG / float(NDCG_before)
        tau, p_value = stats.kendalltau(rankedduculist1, rankedduculist2)

        with open(NDCG_file_name, 'a') as NDCG_FILE:
            NDCG_line = tmp_test_y_query[0].split(':')[-1] + '  ' \
                        + 'changed feature:' + str(top_k_idx) + ' ' + 'kendalltau=' + str(
                round(tau, 4)) + '  ' + 'ratioNDCG:' + str(round(ratio_NDCG, 4)) + '  ' + 'pairnames:' + ' ' + str(
                pairsname) + \
                        '   ' + 'delta_NDCG =' + '  ' + str(round(delta_NDCG, 4)) + "\n"
            NDCG_FILE.write(NDCG_line)
        with open(NDCG_file_matrix, 'a') as matrix_FILE:
            matrix_line = 'matrix for {}'.format(tmp_test_y_query[0].split(':')[-1].split()[0]) \
                          + '  ' + str(matrix) + '  ' + "\n"
            matrix_FILE.write(matrix_line)
        with open(ranklist_file, 'a') as ranklist:
            ranklist_line = tmp_test_y_query[0].split(':')[-1] + '  ' + 'ranklist before:' + str(
                rankedduculist1) + '  ' + 'ranklist after:' + '  ' + str(rankedduculist2) + "\n"
            ranklist.write(ranklist_line)
        os.remove(scorefile_path)
        os.remove(restore_path)

    feature_k_loop(5)
    feature_k_loop(10)


if __name__ == '__main__':
    # parameters to be set
    model_path = 'model/'
    model_set = ['LambdaMART_model.txt']
    # model_set  =['Linearregression_model.txt','coordinateascent_model.txt','Randomforest_model.txt','Listnet_model.txt']
    for MODEL in model_set:
        model = model_path + MODEL

        for f in range(1, 2):
            # the path of data
            datapath = 'MQ2008/Fold{}/'.format(f)
            train_path = datapath + 'train.txt'
            test_path = datapath + 'test.txt'
            modelname = model.split("_")[0].split("/")[-1]
            dataname = datapath.split('/')[0] + '_' + datapath.split('/')[1].split('Fold')[1]
            # saving path and save files
            NDCGdata_path = 'NDCGdata/'
            temp_path = 'temp_data_validity/'

            # get train data and test data
            X_train, y_query_train, Query_train = get_microsoft_data(train_path)
            X_train = np.array(X_train)
            X_test, y_query_test, Query_test = get_microsoft_data(test_path)
            X_test = np.array(X_test)
            expected_value = np.mean(X_train, axis=0)

            # separate the test set
            test_data, test_y_query, test_Query, q_d_len = separate_set(y_query_test, X_test, Query_test)

            resultfile_NDCG = 'resultfile/' + '{}_{}_validity_NDCG.txt'.format(dataname, modelname)
            resultfile_tau = 'resultfile/' + '{}_{}_validity_tau.txt'.format(dataname, modelname)
            resultfile_ratio = 'resultfile/' + '{}_{}_validity_ratio.txt'.format(dataname, modelname)

            with Pool(10) as p:
                # print(p.map(loop_query, [query_index for query_index in range(10)]))
                print(p.map(loop_query, [query_index for query_index in range(len(test_data))]))
            for feature_number in (5, 10):
                NDCG_file_name = NDCGdata_path + '{}_validity_{}features'.format(dataname,
                                                                                 feature_number) + modelname + '.txt'
                NDCG_file_matrix = NDCGdata_path + '{}_validity_matrix_{}features'.format(dataname,
                                                                                          feature_number) + modelname + '.txt'
                ranklist_file = NDCGdata_path + '{}_ranklist_validity_{}features'.format(dataname,
                                                                                         feature_number) + modelname + '.txt'
                rerank_ndcg(NDCG_file_name)
                NDCG = write_average(NDCG_file_name)
                rerank_ndcg(ranklist_file)
                rerank_ndcg(NDCG_file_matrix)
                ratio = write_ratio(NDCG_file_name)
                tau = write_tau(NDCG_file_name)
                with open(resultfile_NDCG, 'a') as NDCG_result:
                    NDCG_result_line = str(NDCG) + "\n"
                    NDCG_result.write(NDCG_result_line)
                with open(resultfile_tau, 'a') as tau_result:
                    tau_result_line = str(tau) + "\n"
                    tau_result.write(tau_result_line)
                with open(resultfile_ratio, 'a') as ratio_result:
                    ratio_result_line = str(ratio) + "\n"
                    ratio_result.write(ratio_result_line)