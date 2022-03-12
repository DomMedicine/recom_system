import numpy as np
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import argparse
from scipy.stats.mstats import gmean


class Appendix:
    def __init__(self, data_set_, x_, number):
        self.data_set = data_set_
        self.x = x_
        self.number = number
        self.geometrical_mean = gmean(self.data_set['rating_train'])
        self.mean = (sum(self.data_set['rating_train']))/(len(self.data_set['rating_train']))
        self.creative_matrix = creative_matrix(self.data_set)

    def zeros(self):
        return creative_Z(self.creative_matrix, self.data_set)

    def arithmetical_mean(self):
        return creative_Z(self.creative_matrix+self.mean, self.data_set)

    def geometric_mean(self):
        return creative_Z(self.creative_matrix+self.geometrical_mean, self.data_set)

    def aggregation_userId_mean(self):
        aggregation_ = self.data_set['data_train'].groupby('userId')[['rating']].mean()
        agg_values = aggregation_.values
        agg_key = aggregation_.index
        help_dict = {}
        for j in range(len(agg_key)):
            help_dict[agg_key[j]] = float(agg_values[j])
        Z = self.creative_matrix.copy()
        for i in self.data_set['dict_userId']:
            if i in help_dict:
                Z[self.data_set['dict_userId'][i], ] = help_dict[i]
            else:
                Z[self.data_set['dict_userId'][i], ] = self.mean
        return creative_Z(Z, self.data_set)

    def aggregation_movieId_mean(self):
        aggregation_ = self.data_set['data_train'].groupby('movieId')[['rating']].mean()
        agg_values = aggregation_.values
        agg_key = aggregation_.index
        help_dict = {}
        for j in range(len(agg_key)):
            help_dict[agg_key[j]] = float(agg_values[j])
        Z = self.creative_matrix.copy()
        Z = Z.transpose()
        for i in self.data_set['dict_movieId']:
            if i in help_dict:
                Z[self.data_set['dict_movieId'][i], ] = help_dict[i]
            else:
                Z[self.data_set['dict_movieId'][i], ] = self.mean
        return creative_Z(Z.transpose(), self.data_set)

    def aggregation_userId_median(self):
        aggregation_ = self.data_set['data_train'].groupby('userId')[['rating']].median()
        agg_values = aggregation_.values
        agg_key = aggregation_.index
        help_dict = {}
        for j in range(len(agg_key)):
            help_dict[agg_key[j]] = float(agg_values[j])
        Z = self.creative_matrix.copy()
        for i in self.data_set['dict_userId']:
            if i in help_dict:
                Z[self.data_set['dict_userId'][i], ] = help_dict[i]
            else:
                Z[self.data_set['dict_userId'][i], ] = self.mean
        return creative_Z(Z, self.data_set)

    def aggregation_movieId_median(self):
        aggregation_ = self.data_set['data_train'].groupby('movieId')[['rating']].median()
        agg_values = aggregation_.values
        agg_key = aggregation_.index
        help_dict = {}
        for j in range(len(agg_key)):
            help_dict[agg_key[j]] = float(agg_values[j])
        Z = self.creative_matrix.copy()
        Z = Z.transpose()
        for i in self.data_set['dict_movieId']:
            if i in help_dict:
                Z[self.data_set['dict_movieId'][i], ] = help_dict[i]
            else:
                Z[self.data_set['dict_movieId'][i], ] = self.mean
        return creative_Z(Z.transpose(), self.data_set)

    def aggregation_userId_position(self):
        aggregation_ = self.data_set['data_train'].groupby('userId')[['rating']].agg(['max', 'min'])
        agg_values = aggregation_.values
        agg_key = aggregation_.index
        help_dict = {}
        for j in range(len(agg_key)):
            help_dict[agg_key[j]] = float((agg_values[j][0]+agg_values[j][1])/2)
        Z = self.creative_matrix.copy()
        for i in self.data_set['dict_userId']:
            if i in help_dict:
                Z[self.data_set['dict_userId'][i], ] = help_dict[i]
            else:
                Z[self.data_set['dict_userId'][i], ] = self.mean
        return creative_Z(Z, self.data_set)

    def aggregation_movieId_position(self):
        aggregation_ = self.data_set['data_train'].groupby('movieId')[['rating']].agg(['max', 'min'])
        agg_values = aggregation_.values
        agg_key = aggregation_.index
        help_dict = {}
        for j in range(len(agg_key)):
            help_dict[agg_key[j]] = float((agg_values[j][0]+agg_values[j][1])/2)
        Z = self.creative_matrix.copy()
        Z = Z.transpose()
        for i in self.data_set['dict_movieId']:
            if i in help_dict:
                Z[self.data_set['dict_movieId'][i], ] = help_dict[i]
            else:
                Z[self.data_set['dict_movieId'][i], ] = self.mean
        return creative_Z(Z.transpose(), self.data_set)

    def number_matrix(self):
        return creative_Z(self.creative_matrix+self.number, self.data_set)


def ParseArguments():
    parser = argparse.ArgumentParser(description="Project ")
    parser.add_argument('--train', required=True, default='', help='The collection of train data')
    parser.add_argument('--test', required=True, default='', help='The collection of test data')
    parser.add_argument('--alg', required=True, default='',
                        help='The algorithm of decomposition i.e. NMF, NMF_mod, SVD1, SVD2, SGD')
    parser.add_argument('--results', required=False, default='results_RMSE.txt',
                        help='The name of saving file with results')
    return vars(parser.parse_args())


def data(set_):
    data_train_ = pd.read_csv(set_['train'])
    data_test_ = pd.read_csv(set_['test'])
    userId_train = data_train_['userId']
    movieId_train = data_train_['movieId']
    rating_train = data_train_['rating']
    userId_test = data_test_['userId']
    movieId_test = data_test_['movieId']
    rating_test = data_test_['rating']
    both_data = data_train_.append(data_test_)
    #
    set_of_movieId = set(both_data['movieId'])
    set_of_userId = set(both_data['userId'])
    dict_movieId = {}
    dict_userId = {}
    b = 0
    for i in set_of_movieId:
        dict_movieId[i] = b
        b = b + 1
    b = 0
    for i in set_of_userId:
        dict_userId[i] = b
        b = b + 1
    n = len(set_of_userId)
    p = len(set_of_movieId)
    return dict(userId_train=userId_train, movieId_train=movieId_train, rating_train=rating_train, n=n, p=p,
                userId_test=userId_test, movieId_test=movieId_test, rating_test=rating_test, data_train=data_train_,
                data_test=data_test_, dict_userId=dict_userId, dict_movieId=dict_movieId)


def creative_matrix(set_):
    return np.zeros(shape=(set_['n'], set_['p']))


def creative_Z(mat, set_):
    for i in range(len(set_['userId_train'])):
        a = set_['dict_userId'][set_['userId_train'][i]]
        b = set_['dict_movieId'][set_['movieId_train'][i]]
        mat[a, b] = set_['rating_train'][i]
    return mat


def nmf(mat):
    model = NMF(n_components=10, init='random', random_state=0)
    W = model.fit_transform(mat)
    H = model.components_
    return np.dot(W, H)


def nmf_mod(mat, set_):
    first_step = nmf(mat)
    first_RMSE = RMSE(set_, first_step)
    len_train = len(set_['userId_train'])
    for k in range(30):
        second_step = first_step.copy()
        for i in range(len_train):
            a = set_['dict_userId'][set_['userId_train'][i]]
            b = set_['dict_movieId'][set_['movieId_train'][i]]
            second_step[a, b] = mat[a, b]
        second_step = nmf(second_step)
        second_RMSE = RMSE(set_, second_step)
        if (first_RMSE - second_RMSE) > 0.00005:
            print(second_RMSE)
        else:
            break
        first_step = second_step.copy()
        first_RMSE = second_RMSE
    return first_step


def svd_1(mat):
    svd = TruncatedSVD(n_components=10, random_state=42)
    svd.fit(mat)
    Sigma2 = np.diag(svd.singular_values_)
    VT = svd.components_
    W = svd.transform(mat)/svd.singular_values_
    H = np.dot(Sigma2, VT)
    return np.dot(W, H)


def svd_2_help(mat):
    svd = TruncatedSVD(n_components=10, random_state=42)
    svd.fit(mat)
    sigma = np.diag([y**0.5 for y in svd.singular_values_])
    VT = svd.components_
    U = np.dot(svd.transform(mat)/svd.singular_values_, sigma)
    H = np.dot(sigma, VT)
    return np.dot(U, H)


def svd_2(mat, set_):
    first_step = svd_1(mat)
    first_RMSE = RMSE(set_, first_step)
    len_train = len(set_['userId_train'])
    for k in range(20):
        second_step = first_step.copy()
        for i in range(len_train):
            a = set_['dict_userId'][set_['userId_train'][i]]
            b = set_['dict_movieId'][set_['movieId_train'][i]]
            second_step[a, b] = mat[a, b]
        second_step = svd_1(second_step)
        second_RMSE = RMSE(set_, second_step)
        if (first_RMSE - second_RMSE) > 0.0001:
            print(second_RMSE)
        else:
            break
        first_step = second_step.copy()
        first_RMSE = second_RMSE
    return first_step


def sgd(mat, set_):
    W = np.zeros(shape=(10, set_['n']))+0.6
    H = np.zeros(shape=(10, set_['p']))+0.6
    matrix_component = creative_matrix(set_)
    for i in range(len(set_['userId_train'])):
        a = set_['dict_userId'][set_['userId_train'][i]]
        b = set_['dict_movieId'][set_['movieId_train'][i]]
        matrix_component[a, b] = 1
    step = -0.00005
    first_RMSE = RMSE(set_, np.dot(W.transpose(), H))
    for i in range(400):
        M_WtH = mat - np.dot(W.transpose(), H)
        M_WtH = M_WtH * matrix_component
        matrix_diff_w = np.dot(M_WtH, H.transpose())*step
        matrix_diff_h = np.dot(W, M_WtH)*step
        W = W - matrix_diff_w.transpose()
        H = H - matrix_diff_h
        second_RMSE = RMSE(set_, np.dot(W.transpose(), H))
        if (first_RMSE - second_RMSE) < 0.00005:
            break
        else:
            print(second_RMSE)
        first_RMSE = second_RMSE
    return np.dot(W.transpose(), H)


def RMSE(data_set_, mat):
    v = [0]*len(data_set_['rating_test'])
    for i in range(len(data_set_['rating_test'])):
        a = data_set_['dict_userId'][data_set_['userId_test'][i]]
        b = data_set_['dict_movieId'][data_set_['movieId_test'][i]]
        v[i] = (mat[a, b] - data_set_['rating_test'][i])**2
    return (sum(v)/len(data_set_['rating_test']))**0.5


def haupt(x_, data_set_, number=5):
    APP = Appendix(data_set_, x_, number)
    if 'NMF' == '{}'.format(x_['alg']):
        a_ = APP.aggregation_userId_mean()
        return RMSE(data_set_, a_)
    elif 'NMF_mod' == '{}'.format(x_['alg']):
        a_ = APP.aggregation_userId_mean()
        return RMSE(data_set_, nmf_mod(a_, data_set_))
    elif 'SVD1' == '{}'.format(x_['alg']):
        a_ = APP.aggregation_userId_mean()
        return RMSE(data_set_, svd_1(a_))
    elif 'SVD2' == '{}'.format(x_['alg']):
        a_ = APP.aggregation_userId_mean()
        return RMSE(data_set_, svd_2(a_, data_set_))
    elif 'SGD' == '{}'.format(x_['alg']):
        a_ = APP.aggregation_userId_mean()
        return RMSE(data_set_, sgd(a_, data_set_))
    else:
        raise Exception("The alg in not in set NMF, NMF_mod, SVD1, SVD2, SGD")


if __name__ == "__main__":
    x = ParseArguments()
    data_set = data(x)
    temp = haupt(x, data_set)
    f = open('{}'.format(x['results']), 'w')
    f.write('{}'.format(temp))
    f.close()
