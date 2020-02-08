import math
def write_average(NDCG_file_name):
    """
    write the mean value of delta NDCG to the last line of NDCG file.
    :param NDCG_file_name:
    :return:
    """
    with open( NDCG_file_name ,'r+') as f:
        a = []
        for line in f:
            a.append(float(line.split()[-1]))
        average_mean = sum(a)/len(a)
        #f.write("average delta NDCG: " + str(round(average_mean,4)))
    return round(average_mean,4)
    #return average_mean


def write_tau(NDCG_file_name):
    with open( NDCG_file_name ,'r+') as f:
        a = []
        for line in f:
            if len(line.split())< 10:
                break
            else:
                a.append(float(line.split('kendalltau=')[-1].split()[0]))
        a = list(map(lambda x: 0.0 if math.isnan(x) else x, a))         
        average_mean = sum(a)/len(a)
        #f.write("average tau: " + str(round(average_mean,4)))
    return round(average_mean,4)  
    #return average_mean



def write_ratio(NDCG_file_name):
    with open( NDCG_file_name ,'r+') as f:
        a = []
        for line in f:
            if len(line.split())< 10:
                break
            else:
                a.append(float(line.split('ratioNDCG:')[-1].split()[0]))
        average_mean = sum(a)/len(a)
    return round(average_mean,4)
    #return average_mean

def rerank_ndcg(NDCG_file_name):
    """
    rerank the NDCG file according to the query index
    :param NDCG_file_name:
    :return:
    """
    a = ''.join(sorted(open(NDCG_file_name), key=lambda s: int(s.split()[0]), reverse=False))
    with open(NDCG_file_name,'w') as f:
        f.write(a)


def rerank_matrix(NDCG_file_matrix):
    """
    rerank the NDCG matrix file according to the query index.
    :param NDCG_file_matrix:
    :return:
    """
    a = ''.join(sorted(open(NDCG_file_matrix), key=lambda s: int(s.split()[2]),reverse=False))
    with open(NDCG_file_matrix,'w') as f:
        f.write(a)