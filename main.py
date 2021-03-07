import pandas as pd
import numpy as np


class LFM(object):
    def __init__(self, alpha, beta, K=10, epochs=10, columns=["uid", "iid", "rating"]):
        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.epochs = epochs
        self.columns = columns

    def fit(self, dataset):
        self.dataset = pd.DataFrame(dataset)

        self.users_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.items_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]

        self.mean = self.dataset[self.columns[2]].mean()
        self.P, self.Q = self.train()

    def init_matrix(self):
        P = dict(zip(
            self.users_ratings.index,
            np.random.rand(len(self.users_ratings), self.K).astype(np.float64)
        ))
        Q = dict(zip(
            self.items_ratings.index,
            np.random.rand(len(self.items_ratings), self.K).astype(np.float64)
        ))
        return P, Q

    def train(self):
        # 使用随机梯度下降法
        P, Q = self.init_matrix()

        current_err, last_err = 0, 1000
        err_limit = 0.001
        for i in range(self.epochs):
            print("第%d次迭代"%i)
            error_list = []
            for uid, iid, r_ui in self.dataset.itertuples(index=False):
                v_pu = P[uid]
                v_qi = Q[iid]
                err = np.float64(r_ui - np.dot(v_pu, v_qi))

                v_pu += self.alpha * (err * v_qi - self.beta * v_pu)
                v_qi += self.alpha * (err * v_pu - self.beta * v_qi)

                P[uid] = v_pu
                Q[iid] = v_qi

                error_list.append(err ** 2)
            current_err = np.sqrt(np.mean(error_list))
            print(current_err)
            if current_err > last_err: #or abs(current_err - last_err) < err_limit:
                break
            last_err = current_err
            self.alpha *= 0.9

        return P, Q

    def predict(self, uid, iid):
        # 如果uid或iid不在，使用全剧平均分作为预测结果返回
        if uid not in self.users_ratings.index or iid not in self.items_ratings.index:
            return self.mean

        p_u = self.P[uid]
        q_i = self.Q[iid]

        return np.dot(p_u, q_i)


if __name__ == '__main__':
    train_data_path = './data/training.dat'
    test_data_path = './data/testing.dat'
    result_path = './result.txt'

    datatype = [("userId", np.int64), ("movieId", np.int64), ("rating", np.float64)]
    col_names = ['userId', 'movieId', 'rating']

    dataset = pd.read_csv(train_data_path, header=None, usecols=range(3), delimiter=',',
                          dtype=dict(datatype), names=col_names)

    lfm = LFM(0.02, 0.01, K=30, epochs=30, columns=["userId", "movieId", "rating"])
    lfm.fit(dataset)

    test_data = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        txt = f.readline()
        while txt:
            test_data.append(txt.split(',')[:2])
            txt = f.readline()

    result = []
    for user_id, movie_id in test_data:
        result.append(lfm.predict(int(user_id), int(movie_id)))

    with open(result_path, 'w') as f:
        for ele in result:
            a = int(float(ele))
            if a > 5:
                a = 5
            if a < 0:
                a = 0
            f.write(str(a) + '\n')
