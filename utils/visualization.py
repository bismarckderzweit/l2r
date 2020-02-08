import pandas as pd
import numpy as np
from pytablewriter import UnicodeTableWriter

def show_shap_in_table(matrix,pairsname):
    """
    :param matrix: shapley value matrix
    :param pairsname: name of docu pairs
    :return: a table for visualization
    """
    feature_names = ['Body TF','Anchor TF','Title TF','URL TF','Whole docu TF',
                    'Body IDF','Anchor IDF','Title IDF','URL IDF','Whole docu IDF',
                    'Body TF*IDF','Anchor TF*IDF','Title TF*IDF','URL TF*IDF','Whole docu TF*IDF',
                    'Body DL','Anchor DL','Title DL','URL DL','Whole docu DL',
                    'Body BM25','Anchor BM25','Title BM25','URL BM25','Whole docu BM25',
                    'Body LMIR.ABS','Anchor LMIR.ABS','Title LMIR.ABS','URL LMIR.ABS','Whole docu LMIR.ABS',
                    'Body LMIR.DIR','Anchor LMIR.DIR','Title LMIR.DIR','URL LMIR.DIR','Whole docu LMIR.DIR',
                    'Body LMIR.JM','Anchor LMIR.JM','Title LMIR.JM','URL LMIR.JM','Whole docu LMIR.JM',
                    'PageRank','Inlink number','Outlink number','Number of slash in URL',
                    'URL Length','Number of child page']
    table = pd.DataFrame(matrix, index=feature_names, columns = pairsname)
    return table


def shap_tablewriter(matrix,pairsname):
    """
    use python tool tablewrite to write the shapely value into a more nice table
    :param matrix: shapley value matrix
    :param pairsname: name of docu pairs
    :return:
    """
    feature_names = ['Body TF', 'Anchor TF', 'Title TF', 'URL TF', 'Whole docu TF',
                     'Body IDF', 'Anchor IDF', 'Title IDF', 'URL IDF', 'Whole docu IDF',
                     'Body TF*IDF', 'Anchor TF*IDF', 'Title TF*IDF', 'URL TF*IDF', 'Whole docu TF*IDF',
                     'Body DL', 'Anchor DL', 'Title DL', 'URL DL', 'Whole docu DL',
                     'Body BM25', 'Anchor BM25', 'Title BM25', 'URL BM25', 'Whole docu BM25',
                     'Body LMIR.ABS', 'Anchor LMIR.ABS', 'Title LMIR.ABS', 'URL LMIR.ABS', 'Whole docu LMIR.ABS',
                     'Body LMIR.DIR', 'Anchor LMIR.DIR', 'Title LMIR.DIR', 'URL LMIR.DIR', 'Whole docu LMIR.DIR',
                     'Body LMIR.JM', 'Anchor LMIR.JM', 'Title LMIR.JM', 'URL LMIR.JM', 'Whole docu LMIR.JM',
                     'PageRank', 'Inlink number', 'Outlink number', 'Number of slash in URL',
                     'URL Length', 'Number of child page']
    feature_names2 = np.array(feature_names).reshape(-1,1)
    matrix2 = np.array(matrix)
    matrix2 = np.append(feature_names2,matrix2,axis=1)
    writer = UnicodeTableWriter()
    writer.table_name = "parameter_matrix"
    pairsname.insert(0,'feature')
    writer.headers = pairsname
    matrix3 = []
    for i in range(matrix2.shape[0]):
        row = list(matrix2[i])
        matrix3.append(row)
    writer.value_matrix = matrix3
    writer.write_table()
