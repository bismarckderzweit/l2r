import numpy as np
import subprocess
import os
import shap
import scipy.stats as stats
from multiprocessing import Pool
from utils.rerank import write_average, rerank_ndcg, rerank_matrix
from utils.readdata import get_microsoft_data, rewrite
from utils.separate_set import separate_set
from utils.explainer_tools import rand_row, get_set_cover, evaluate, get_pairsname, get_rankedduculist


def score(X):
    """
    The first if branch is training data, the next is for the single test data. First calling the subprocess of ranklib
    to get the scores, then rerank the scorefile according the original index. We also have to delete the produced
    files which used by the subprocess.
    :param X: input feature matrix
    :return: scores of q-d pairs
    """
    A = []
    if X.shape[0] == background_datasize:
        # this part is for the training
        scorefile_path = temp_path + 'scorefile_matrix.txt'
        restore_path = temp_path + 'restore_matrix.txt'
        rewrite(X, y_query_train, Query_train, restore_path)
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
        os.remove(scorefile_path)
        os.remove(restore_path)
    else:
        scorefile_path = temp_path + 'scorefile_matrix_{}.txt'.format(tmp_test_y_query[0].split(':')[-1].split()[0])
        restore_path = temp_path + 'restore_matrix_{}.txt'.format(tmp_test_y_query[0].split(':')[-1].split()[0])
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
    tmp_test_data =test_data[query_index]
    tmp_test_y_query = test_y_query[query_index]
    tmp_test_Query = test_Query[query_index]
    query_id = tmp_test_y_query[0].split(':')[-1].split()[0]

    # calculate the scores for the q-d pairs
    scores = score(tmp_test_data).reshape(-1, 1)
    restore_path = temp_path +  'restore_matrix_{}.txt'.format(query_id)
    scorefile_path = temp_path + 'scorefile_matrix_{}.txt'.format(query_id)

    # reranking the test_data according to the scores and get the list of ranking
    test_data_score = np.append(tmp_test_data,scores,axis=1)
    ranked_test_data = (test_data_score[(-test_data_score[:,-1]).argsort()])[:,:-1]
    rankedduculist1 = get_rankedduculist(scores, query_index, q_d_len)
    NDCG_before = evaluate(model, restore_path)

    # get shapley value for the all the q_d pairs
    query1_shap_values = explainer.shap_values(ranked_test_data, nsamples=200)

    # get pairsname
    pairnumbers = 50
    pairsname = get_pairsname(ranked_test_data, pairnumbers)

    # create the matrix
    matrix = []
    for i in range(len(pairsname)):
        index1 = int(pairsname[i][1])
        index2 = int(pairsname[i][-1])
        row = [(query1_shap_values[index1-1][j]-query1_shap_values[index2-1][j]) for j in range(46)]
        matrix.append(row)

    # get top 5 features and set them to 0
    top_k_idx,_ = get_set_cover(matrix)
    features_to_change = tmp_test_data
    features_to_change[:,top_k_idx] = 0

    # get scores of the changed features
    scores2 = score(features_to_change).reshape(-1,1)
    rankedduculist2 = get_rankedduculist(scores2, query_index, q_d_len)
    NDCG_after = evaluate(model, restore_path)
    delta_NDCG = float(NDCG_before) - float(NDCG_after)
    tau, p_value = stats.kendalltau(rankedduculist1, rankedduculist2)
    os.remove(scorefile_path)
    os.remove(restore_path)

    with open(NDCG_file_name, 'a') as NDCG_FILE:
        NDCG_line =  tmp_test_y_query[0].split(':')[-1]+'  ' \
                    + 'changed feature:'+ str(top_k_idx)+' '+'kendalltau='+str(round(tau,4))+ ' '+ 'pairnames:'+' '+str(pairsname) + \
                    '   ' + 'delta_NDCG ='+'  '+str(delta_NDCG)+ "\n"
        NDCG_FILE.write(NDCG_line)
    with open(NDCG_file_matrix, 'a') as matrix_FILE:
        matrix_line = 'matrix for {}'.format(tmp_test_y_query[0].split(':')[-1].split()[0]) \
                      + '  ' + str(matrix) + '  ' + "\n"
        matrix_FILE.write(matrix_line)
    with open(ranklist_file, 'a') as ranklist:
        ranklist_line = tmp_test_y_query[0].split(':')[-1] + '  ' + 'ranklist before:' + str(
            rankedduculist1) + '  ' + 'ranklist after:' + '  ' + str(rankedduculist2) + "\n"
        ranklist.write(ranklist_line)


if __name__ == '__main__':
    # the path of data
    datapath = 'MQ2008/Fold1/'
    model_path = 'model/'
    train_path = datapath + 'train.txt'
    test_path = datapath + 'test.txt'
    model = model_path + 'MART_model.txt'
    modelname = model.split("_")[0].split("/")[-1]

    # saving path and save files
    NDCGdata_path = 'NDCGdata/'
    temp_path = 'temp_data/'
    NDCG_file_name = NDCGdata_path + 'SHAP_matrix' + modelname + '.txt'
    NDCG_file_matrix = NDCGdata_path + 'SHAP_matrix_matrix' + modelname + '.txt'
    ranklist_file = NDCGdata_path + 'ranklist_SHAP_matrix' + modelname + '.txt'

    # get train data and test data
    X_train, y_query_train, Query_train = get_microsoft_data(train_path)
    X_train = np.array(X_train)
    X_test, y_query_test, Query_test = get_microsoft_data(test_path)
    X_test = np.array(X_test)

    # separate the test set
    test_data, test_y_query, test_Query, q_d_len = separate_set(y_query_test, X_test, Query_test)

    # creat a explainer
    background_datasize = 500
    X_train = rand_row(X_train,background_datasize)
    explainer = shap.KernelExplainer(score, X_train)

    with Pool(10) as p:
        print(p.map(loop_query, [query_index for query_index in range(len(test_data))]))

    # reranking the NDCG file
    rerank_ndcg(NDCG_file_name)

    # reranking the matrix file
    rerank_matrix(NDCG_file_matrix)

    # write the average value
    write_average(NDCG_file_name)

    # reranking the ranklist file
    rerank_ndcg(ranklist_file)