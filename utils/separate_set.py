import numpy as np
def separate_set(y_query_test,X_test,Query_test):
    """
    separate the test data according to the query index, for every sub has several docus for the same query
    :param y_query_test:
    :param X_test:
    :param Query_test:
    :return:
    """
    y_query_test = np.array(y_query_test)
    queryname = (y_query_test[0]).split()[1]
    test_data = []
    test_y_query = []
    test_Query = []
    q_d_len = []
    j = 0
    for i in range(X_test.shape[0]):
        if (y_query_test[i]).split()[1] != queryname:
            test_data.append(X_test[j:i])
            test_y_query.append(y_query_test[j:i])
            test_Query.append(Query_test[j:i])
            queryname = (y_query_test[i]).split()[1]
            q_d_len.append(i - j)
            j = i
    test_data.append(X_test[j:X_test.shape[0]])
    test_y_query.append(y_query_test[j:X_test.shape[0]])
    test_Query.append(Query_test[j:X_test.shape[0]])
    q_d_len.append(X_test.shape[0] - j)
    return test_data, test_y_query, test_Query, q_d_len