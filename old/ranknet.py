# import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
import csv
BATCH_SIZE = 100
feature_num = 46
h1_num = 10
ndcglist = []

train_path = 'data/train.txt'
test_path = 'data/test.txt'
vali_path = 'data/vali.txt'
csv_path = 'data/data.csv'
ranking_method = 'pairwise'

tf.disable_v2_behavior()


def extractFeatures(split):
    features = []
    for i in range(2, 48):
        features.append(float(split[i].split(':')[1]))
    return features


def extractQueryData(split):
    queryFeatures = [split[1].split(':')[1]]
    queryFeatures.append(split[50])
    queryFeatures.append(split[53])
    queryFeatures.append(split[56])
    return queryFeatures


def get_microsoft_data(file_path):
    with open(file_path, 'r') as fp:
        y_train = []
        X_train = []
        Query = []
        for data in fp:
            split = data.split()
            y_train.append(int(split[0]))
            X_train.append(extractFeatures(split))
            Query.append(extractQueryData(split))
    return y_train, X_train, Query
# X_train[0] Query-doc pair 特征
# Query[0]:  ['10', 'GX000-00-0000000', '1', '0.0246906']  query号和文章号


def write_csv(csv_path, datasize,x0_features, x1_features):
    with open(csv_path, "w") as file:
        writer = csv.writer(file)
        firstrow = ['label']
        for i in range(46):
            firstrow.append("x0_feature{}".format(i))
        for j in range(46):
            firstrow.append("x1_feature{}".format(j))
        writer.writerow(map(lambda x: x, firstrow))
        b = np.array([[1] * datasize])
        c = np.c_[b.T, array_train_x0]
        for k in range(datasize-1):
            this_row = list(c[k]) + list(array_train_x1[k])
            writer.writerow(map(lambda x: x, this_row))


def dcg(predicted_order):
    i = np.log(1. + np.arange(1,len(predicted_order)+1))
    l = 2 ** (np.array(predicted_order)) - 1
    return np.sum(l/i)


def ndcg(score, top_ten=True):
    end = 10 if top_ten else len(score)
    sorted_score = np.sort(score)[::-1]
    dcg_ = dcg(score[:end])
    if dcg_ == 0:
        return 0
    dcg_max = dcg(sorted_score[:end])
    return dcg_/dcg_max


def get_pair_feature(y_train,X_train, Query):
    pairs = []
    tmp_x0 = []
    tmp_x1 = []
    for i in range(0, len(Query)):
        for j in range(i + 1, len(Query)):
            if (Query[i][0] != Query[j][0]):
                break
            if (Query[i][0] == Query[j][0] and y_train[i] != y_train[j]):
                if (y_train[i] > y_train[j]):
                    pairs.append([i, j])
                    tmp_x0.append(X_train[i])
                    tmp_x1.append(X_train[j])
                else:
                    pairs.append([j, i])
                    tmp_x0.append(X_train[j])
                    tmp_x1.append(X_train[i])
    array_train_x0 = np.array(tmp_x0)
    array_train_x1 = np.array(tmp_x1)
    print('Found %d document pairs' % (len(pairs)))
    return pairs, len(pairs), array_train_x0, array_train_x1


with tf.name_scope("input"):
    x1 = tf.placeholder(tf.float32, [None, feature_num], name="x1")
    x2 = tf.placeholder(tf.float32, [None, feature_num], name="x2")

# 添加隐层节点
with tf.name_scope("layer1"):
    with tf.name_scope("w1"):
        w1 = tf.Variable(tf.random_normal([feature_num, h1_num]), name="w1")
        tf.summary.histogram('w1', w1)
    with tf.name_scope("b1"):
        b1 = tf.Variable(tf.random_normal([h1_num]), name="b1")
    with tf.name_scope("h1_o1"):
        h1_o1 = tf.matmul(x1, w1) + b1
        h1_o1 = tf.nn.relu(h1_o1)
    with tf.name_scope("h2_o1"):
        h1_o2 = tf.matmul(x2, w1) + b1
        h1_o2 = tf.nn.relu(h1_o2)

# 添加输出节点
with tf.name_scope("output"):
    with tf.name_scope("w2"):
        w2 = tf.Variable(tf.random_normal([h1_num, 1]), name="w2")
    with tf.name_scope("b2"):
        b2 = tf.Variable(tf.random_normal([1]))

    h2_o1 = tf.matmul(h1_o1, w2) + b2
    h2_o2 = tf.matmul(h1_o2, w2) + b2
    h2_o1 = tf.sigmoid(h2_o1)
    h2_o2 = tf.sigmoid(h2_o2)

with tf.name_scope("loss"):

    h_o12 = h2_o1 - h2_o2
    pred = tf.sigmoid(h_o12)

    lable_p = 1

    cross_entropy = -lable_p * tf.log(pred) - (1 - lable_p) * tf.log(1 - pred)

    reduce_sum = tf.reduce_sum(cross_entropy)
    loss = tf.reduce_mean(reduce_sum)
    tf.summary.scalar('loss', loss)

with tf.name_scope("train_op"):
    train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

saver = tf.train.Saver()

merged_summary = tf.summary.merge_all()
with tf.Session() as sess:
    y_train, X_train, Query = get_microsoft_data(train_path)
    pairs, datasize, array_train_x0, array_train_x1 = get_pair_feature(y_train,X_train, Query)
    # write_csv(csv_path=csv_path,datasize=datasize, x0_features=array_train_x0,x1_features=array_train_x1)
    init = tf.global_variables_initializer()
    sess.run(init)

    writer = tf.summary.FileWriter("./logs/", sess.graph)
    for epoch in range(0, 200000):
        start = (epoch * BATCH_SIZE) % datasize
        end = min(start + BATCH_SIZE, datasize)
        sess.run(train_op, feed_dict={x1: array_train_x0[start:end, :], x2: array_train_x1[start:end, :]})
        summary = sess.run(merged_summary, feed_dict={x1: array_train_x0[start:end, :], x2: array_train_x1[start:end, :]})
        writer.add_summary(summary, epoch)
        if epoch % 1000 == 0:
            l_v = sess.run(loss, feed_dict={x1: array_train_x0, x2: array_train_x1})
            result_0 = sess.run(h2_o1, feed_dict={x1: array_train_x0, x2: array_train_x1})
            result_1 = sess.run(h2_o2, feed_dict={x1: array_train_x0, x2: array_train_x1})
            # print(sess.run(cross_entropy, feed_dict={x1:array_train_x0, x2:array_train_x1}))
            # print("------ epoch[%d] loss_v[%f] ------ " % (epoch, l_v))
            print("train_data accuracy is:", np.sum(result_0 > result_1) * 1.0 / datasize)
    saver.save(sess, 'save/'+ranking_method+'/model.ckpt', global_step=epoch)
    y_test, X_test, test_Query = get_microsoft_data(test_path)
    test_pairs, test_datasize, array_test_x0, array_test_x1 = get_pair_feature(y_test, X_test, test_Query)
    l_v = sess.run(loss, feed_dict={x1: array_test_x0, x2: array_test_x1})
    result_0 = sess.run(h2_o1, feed_dict={x1: array_test_x0, x2: array_test_x1})
    result_1 = sess.run(h2_o2, feed_dict={x1: array_test_x0, x2: array_test_x1})
    epoch_ndcg1 = (ndcg([0, 1])) * np.sum(result_0 > result_1)
    epoch_ndcg2 = (ndcg([1, 0])) * np.sum(result_1 > result_0)
    minndcg = (epoch_ndcg1 + epoch_ndcg2) / test_datasize
    print("the mean-ndcg is {}".format(minndcg))
    print("test_data accuracy is:", np.sum(result_0 > result_1) * 1.0 / test_datasize)


