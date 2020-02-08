import numpy as np
import subprocess
import os
import scipy.stats as stats
from multiprocessing import Pool
from utils.rerank import write_average, rerank_ndcg, rerank_matrix
from utils.readdata import get_microsoft_data, rewrite
from utils.separate_set import separate_set
from utils.explainer_tools import get_set_cover, evaluate, get_pairsname, get_rankedduculist


def score(X):
    """
    Get the scores for the batch-changed-features. First calling the subprocess of ranklib
    to get the scores, then rerank the scorefile according the original index. We also have to delete the produced
    files which used by the subprocess.
    :param X: batch-changed-features
    :return: scores of q-d pairs
    """
    A = []
    scorefile_path = temp_path + 'scorefile_matrix_coverage{}.txt'.format(tmp_test_y_query[0].split(':')[-1].split()[0])
    restore_path = temp_path + 'restore_matrix_coverage{}.txt'.format(tmp_test_y_query[0].split(':')[-1].split()[0])
    rewrite(X, tmp_test_y_query, tmp_test_Query,restore_path)
    args = ['java', '-jar', 'RankLib-2.12.jar', '-rank', restore_path, '-load', model,
            '-indri', scorefile_path]
    subprocess.check_output(args, stderr=subprocess.STDOUT)

    # rerank the scorefile according the original index
    scorefile_data = ''.join(sorted(open(scorefile_path), key=lambda s: s.split()[1],reverse=False))
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
                newline+=(split[i]+' ')
            f.write(newline+'\n')
    A = np.array(A)
    return A

# get scores of the samples of this query and rank them according to the scores
def loop_query(query_index):
    """
    loop for a query
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
    scores = score(tmp_test_data).reshape(-1, 1)

    # reranking the test_data according to the scores and get the list of ranking
    test_data_score = np.append(tmp_test_data, scores, axis=1)
    ranked_test_data = (test_data_score[(-test_data_score[:, -1]).argsort()])[:, :-1]
    rankedduculist1 = get_rankedduculist(scores, query_index, q_d_len)
    NDCG_before = evaluate(model, 'temp_data/restore_matrix_coverage{}.txt'.format(query_id))

    # get pairsname
    pairnumbers = 20
    pairsname = get_pairsname(ranked_test_data,pairnumbers)

    # create the matrix
    def get_coverage(feature_index,docu_index1,docu_index2):
        step_num = 11
        temp_pair_data = ranked_test_data[[docu_index1-1,docu_index2-1],:]
        changed_list = []
        original1 =  temp_pair_data[0,feature_index]
        original2 = temp_pair_data[1,feature_index]
        if original1 == 0 or original2 == 0:
            return 0
        step_len = min(original1,original2)/((step_num-1)/2)
        support_count = 0
        for m in range(step_num):
            temp1 =  temp_pair_data[0].copy()
            temp1[feature_index] = original1 +(m - (step_num-1)/2)*step_len
            for n in range(step_num):
                changed_list.append(temp1)
                temp2 =  temp_pair_data[1].copy()
                temp2[feature_index] = original2 +(n - (step_num-1)/2)*step_len
                changed_list.append(temp2)
        changed_list = np.array(changed_list)
        with open('temp_data/changed_list{}.txt'.format(query_index),'w') as f:
            for i in range(2*step_num**2):
                line = ""
                line += "0 qid:{} ".format(str(i))
                for j in range(len(changed_list[i])):
                    line += ((str(j+1))+":"+str(changed_list[i][j])+" ")
                line += '#docid = GX008-86-4444840 inc = 1 prob = 0.086622 ' + "\n"
                f.write(line)
        args = ['java', '-jar', 'RankLib-2.12.jar', '-rank', 'temp_data/changed_list{}.txt'.format(query_index), '-load', model,
                '-indri', 'temp_data/changed_list_score{}.txt'.format(query_index)]
        subprocess.check_output(args, stderr=subprocess.STDOUT)
        """
        rewrite the scores according to the index we set, there is a trick here, for the changed data of first docu, the indexes should be even
        number, the indexes for the second are odd like 1,3,5,7 ...., the reason for that is for the following pair-comparing, if the score of even_i
        is greater than the one of odd_i, then this changed data is support for the preference "d1>d2", if smaller, then in the contrary.  
        """
        A = ''.join(sorted(open('temp_data/changed_list_score{}.txt'.format(query_index)), key=lambda s: int(s.split()[0]), reverse=False))
        with open('temp_data/changed_list_score{}.txt'.format(query_index),'w') as f:
            f.write(A)
        changed_list_score = []
        with open('temp_data/changed_list_score{}.txt'.format(query_index),'r') as f:
            for line in f:
                changed_list_score.append(float(line.split()[-2]))
        for i in range(0,len(changed_list_score),2):
            if changed_list_score[i]>=changed_list_score[i+1]:
                support_count+=1
        support_ratio = support_count/step_num**2
        return support_ratio

    # get coverage matrix
    matrix = []
    for i in range(len(pairsname)):
        index1 = int(pairsname[i][1])
        index2 = int(pairsname[i][-1])
        row = [get_coverage(j, index1, index2) for j in range(46)]
        matrix.append(row)

    # change the features of the top k
    top_k_idx,_ = get_set_cover(matrix)
    features_to_change = tmp_test_data
    features_to_change[:, top_k_idx] = 0

    # get scores for the changed features
    scores2 = score(features_to_change).reshape(-1, 1)
    rankedduculist2 = get_rankedduculist(scores2, query_index, q_d_len)
    NDCG_after = evaluate(model, 'temp_data/restore_matrix_coverage{}.txt'.format(query_id))
    delta_NDCG = float(NDCG_before) - float(NDCG_after)
    tau, p_value = stats.kendalltau(rankedduculist1, rankedduculist2)
    NDCG_file_name = 'NDCGdata/' + 'coverage' + model.split("_")[0].split("/")[-1] + '.txt'
    NDCG_file_matrix = 'NDCGdata/' + 'coverage_matrix' + model.split("_")[0].split("/")[-1] + '.txt'
    os.remove(os.path.join(temp_path, 'restore_matrix_coverage{}.txt'.format(query_id)))
    os.remove(os.path.join(temp_path, 'scorefile_matrix_coverage{}.txt'.format(query_id)))
    os.remove(os.path.join(temp_path, 'changed_list{}.txt'.format(query_index)))
    os.remove(os.path.join(temp_path, 'changed_list_score{}.txt'.format(query_index)))

    with open(NDCG_file_name, 'a') as NDCG_FILE:
        NDCG_line = tmp_test_y_query[0].split(':')[-1]+'  ' + 'changed feature:'+ str(top_k_idx) \
                    + '  ' + 'kendalltau=' + str(round(tau,4)) +'' + 'pairnames:'+' '+str(pairsname) +' ' + 'delta_NDCG =' + '  ' + str(delta_NDCG)+ "\n"
        NDCG_FILE.write(NDCG_line)

    with open(NDCG_file_matrix, 'a') as matrix_FILE:
        matrix_line = 'matrix for {}'.format(tmp_test_y_query[0].split(':')[-1].split()[0]) + '  ' + str(matrix) + '  ' + "\n"
        matrix_FILE.write(matrix_line)

    with open(ranklist_file, 'a') as ranklist:
        ranklist_line = tmp_test_y_query[0].split(':')[-1] + '  ' + 'ranklist before:' + str(
            rankedduculist1) + '  ' + 'ranklist after:' + '  ' + str(rankedduculist2) + "\n"
        ranklist.write(ranklist_line)


if __name__ == '__main__':
    # the path of data and model
    datapath = 'MQ2008/Fold1/'
    train_path = datapath + 'train.txt'
    test_path = datapath + 'test.txt'
    model_path = 'model/'
    model = model_path + 'MART_model.txt'
    modelname = model.split("_")[0].split("/")[-1]

    # saving path and save files
    NDCGdata_path = 'NDCGdata/'
    temp_path = 'temp_data/'
    NDCG_file_name = NDCGdata_path + 'coverage' + modelname + '.txt'
    NDCG_file_matrix = NDCGdata_path + 'coverage_matrix' + modelname + '.txt'
    ranklist_file = NDCGdata_path + 'ranklist_coverage' + modelname + '.txt'

    # get train data and test data
    X_train, y_query_train, Query_train = get_microsoft_data(train_path)
    X_train = np.array(X_train)
    X_test, y_query_test, Query_test = get_microsoft_data(test_path)
    X_test = np.array(X_test)

    # separate the test set
    test_data, test_y_query, test_Query, q_d_len = separate_set(y_query_test, X_test, Query_test)

    with Pool(10) as p:
        print(p.map(loop_query, [query_index for query_index in range(len(test_data))]))

    # reranking the NDCG file
    rerank_ndcg(NDCG_file_name)

    # reranking the NDCG_matrix file
    rerank_matrix(NDCG_file_matrix)

    # write the average value
    write_average(NDCG_file_name)

    # reranking the ranklist file
    rerank_ndcg(ranklist_file)
