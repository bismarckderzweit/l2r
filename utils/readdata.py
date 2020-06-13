def extractYquery(split):
    """
    like "0 qid:18219 "
    :param split:
    :return:
    """
    y_query = ''
    y_query += (split[0] + ' ' + split[1] + ' ')
    return y_query


def extractQuery(split):
    """
    like "#docid = GX004-93-7097963 inc = 0.0428115405134536 prob = 0.860366 "
    :param split:
    :return:
    """
    if len(split) < 60:
        Query = ''
        for i in range(48, 57):
            Query += (split[i] + ' ')
    else:
        Query = " "
    return Query


def extractFeatures(split):
    """
    get the features
    :param split:
    :return:
    """
    features = []
    if len(split) < 60:
        for i in range(2, 48):
            features.append(float(split[i].split(':')[1]))
    else:
        for i in range(2, 138):
            features.append(float(split[i].split(':')[1]))
    return features


def get_microsoft_data(file_path):
    """
    :param file_path:
    :return:
    """
    with open(file_path, 'r') as fp:
        X_train = []
        y_query = []
        Query = []
        for data in fp:
            split = data.split()
            y_query.append(extractYquery(split))
            Query.append(extractQuery(split))
            X_train.append(extractFeatures(split))
    return X_train, y_query, Query


def rewrite(X, y_query, Query, restore_path):
    """
    we must rewrite the data into a txt file otherwise can not use ranklib.
    :param X: the feature matrix
    :param y_query: like "0 qid:18219 "
    :param Query: like "#docid = GX004-93-7097963 inc = 0.0428115405134536 prob = 0.860366 "
    :param restore_path: the saving path
    :return:
    """
    with open(restore_path, 'w') as f:
        for i in range(len(X)):
            line = ""
            line += y_query[(i % len(y_query))].split(':')[0] + ':' + str(i) + ' '
            for j in range(len(X[i])):
                line += ((str(j + 1)) + ":" + str(X[i][j]) + " ")
            line += Query[(i % len(Query))] + "\n"
            f.write(line)

