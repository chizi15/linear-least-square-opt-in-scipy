# 重要：对于scipy.optimize中的优化方法，不能通过改变目标函数的方式，将含有联合自变量的约束条件加入目标函数；
# 要增加带有联合自变量的约束条件，必须在方法中传入constraints关键字参数；若该方法没有constraints参数，则不能定义带有联合自变量的约束条件。
# the loss function of linear least square optimization is a convex function, so local or global optimizations are equal
from time import process_time
from pandas import DataFrame
from scipy import optimize
import numpy as np
import random
from warnings import filterwarnings
filterwarnings("ignore")

# random.seed(10)
Acc_Start = process_time()

# 代表每个序列的长度，这里真实值序列长度为90，即代表90天真实销量；其他预测序列长度递减，用于模拟有的算法在部分点处无法生成预测值。
k_mul = np.array([i for i in range(90, -1, -10)])
y_all = [[]] * len(k_mul)  # 生成用于存放真实值序列和预测值序列的二维列表，以便后续生成dataframe

weights = []
for i in range(-99, 1):
    weights.append(0.94 ** i)  # 设置随机序列的权重呈递减指数分布
weights = np.array(weights)

for i in range(0, 10):
    if i == 0:
        y_all[i] = np.array(random.choices(range(0, 100), k=k_mul[i], weights=weights))  # 生成随机序列，用来代表真实值序列
    else:
        # 生成正弦随机序列，用于代表各种算法的预测序列与真实序列的偏差，各序列的非nan值长度递减。
        y_all[i] = 3 * np.sin(random.choices(range(0, 100), k=k_mul[i]))

y_all = DataFrame(y_all).T
y_all.fillna(value=0, inplace=True)  # 将dataframe中所有nan用0代替
y_all[3] = 0  # 假定第四列（即第3种算法的预测值序列）全为0，用于测试下面两个for循环是否robust

for i in range(10):
    if (y_all[i] == 0).sum() / len(y_all[i]) < 0.9 and i > 0:
        y_all[i][y_all[i] != 0] = y_all[0][y_all[i] != 0] + y_all[i][
            y_all[i] != 0]  # 只让每条预测值序列中不为0的点（即有预测值的点）与第0列（即真实值序列）相加，生成最终的预测序列。
    elif (y_all[i] == 0).sum() / len(y_all[i]) >= 0.9:  # 若预测值序列中零点个数与真实值序列总长之比大于等于90%，则认为对应算法此时不适用，不进行动态加权。
        y_all.drop(columns=[i], axis=1, inplace=True)

for i in range(len(y_all.columns)):
    y_all.columns.values[i] = i  # 对y_all的列名按顺序重命名

y_all = y_all.sample(frac=1).reset_index(drop=True)  # 打乱列次序同时保持列属性，并重置索引；此操作不改变最小二乘的目标函数。
# 设置梯度下降时权重的初始值，在生产程序中取MAPE动态加权后生成的值
w = np.array([1 / (len(y_all.columns) - 1) for _ in range(len(y_all.columns) - 1)])

Block_End = process_time() - Acc_Start
Acc_End = process_time() - Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End, 2) + "secs")
print("Accumulate use " + "%0.2f" % round(Acc_End, 2) + "secs\n")


class LeastSqure(object):
    """docstring for ClassName"""

    def __init__(self, y_all, w):
        self.y_all = y_all
        self.w = w

    def ols(self, alpha):
        yhat = np.zeros(len(self.y_all))
        for i in range(len(alpha)):
            yhat += alpha[i] * self.y_all[i + 1]  # yhat是多条预测序列加权后的预测值序列
        residual = yhat - self.y_all[0]
        return residual

    def loss_function(self, alpha):
        yhat = np.zeros(len(self.y_all))
        distance = []
        for i in range(len(yhat)):
            for j in range(len(alpha)):
                yhat[i] += alpha[j] * self.y_all[j + 1][i]
            distance.append((yhat[i] - self.y_all[0][i]) ** 2)
        distance = 0.5 * sum(distance)
        return np.array(distance)

    def equa_cons(self, alpha):
        return sum(alpha) - 1  # == 0

    def compute(self):
        # ordinary linear least square optimization
        para_ols = optimize.least_squares(self.ols, self.w, jac='2-point', bounds=([1e-5] * len(self.w), 1 - 1e-5),
                                          method='dogbox')
        weights_sum = sum(para_ols.x)
        for i in range(len(para_ols.x)):
            para_ols.x[i] = para_ols.x[i] / weights_sum

        bounds = [(1e-5, 1 - 1e-5)] * len(self.w)
        cons = ({'type': 'eq', 'fun': self.equa_cons})

        # Sequential Least SQuares Programming optimization
        para_slqsp = optimize.minimize(self.loss_function, self.w, method='SLSQP', bounds=bounds, constraints=cons)

        # Differential Evolution with stochastic in nature
        para_diff = optimize.differential_evolution(self.loss_function, bounds, updating='deferred', workers=-1, seed=1)
        weights_sum = sum(para_diff.x)
        for i in range(len(para_diff.x)):
            para_diff.x[i] = para_diff.x[i] / weights_sum

        final_weight = (para_ols.x + para_slqsp.x + para_diff.x) / 3

        return final_weight


final_weights = LeastSqure(y_all, w).compute()
final_pred_1 = np.sum(final_weights * y_all[y_all.columns[1:len(y_all.columns)]], axis=1)


def ols(alpha):
    yhat = np.zeros(len(y_all))
    for i in range(len(alpha)):
        yhat += alpha[i] * np.array(y_all[i + 1])  # yhat是多条预测序列加权后的预测值序列
    residual = yhat - np.array(y_all[0])
    return residual


def loss_function(alpha):
    yhat = np.zeros(len(y_all))
    distance = []
    for i in range(len(yhat)):
        for j in range(len(alpha)):
            yhat[i] += alpha[j] * y_all[j + 1][i]
        distance.append((yhat[i] - y_all[0][i]) ** 2)
    distance = 0.5 * sum(distance)
    return np.array(distance)


def equa_cons(alpha):
    return sum(alpha) - 1  # == 0


# ordinary linear least square optimization
Block_Start = process_time()
para_ols = optimize.least_squares(ols, w, jac='2-point', bounds=([1e-5] * len(w), 1 - 1e-5), method='dogbox')
weights_sum = sum(para_ols.x)
for i in range(len(para_ols.x)):
    para_ols.x[i] = para_ols.x[i] / weights_sum
print('local minimized point is:', para_ols.x)
print(sum(para_ols.x))
Block_End = process_time() - Block_Start
Acc_End = process_time() - Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End, 2) + "secs")
print("Accumulate use " + "%0.2f" % round(Acc_End, 2) + "secs\n")

# Sequential Least SQuares Programming optimization
bounds = [(1e-5, 1 - 1e-5)] * len(w)
cons = ({'type': 'eq', 'fun': equa_cons})
Block_Start = process_time()
para_slqsp = optimize.minimize(loss_function, w, method='SLSQP', bounds=bounds, constraints=cons)
print('local minimized point is:', para_slqsp.x)
print(sum(para_slqsp.x))
Block_End = process_time() - Block_Start
Acc_End = process_time() - Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End, 2) + "secs")
print("Accumulate use " + "%0.2f" % round(Acc_End, 2) + "secs\n")

# quickest global optimization of a multivariate function
Block_Start = process_time()
para_diff = optimize.differential_evolution(loss_function, bounds, updating='deferred', workers=-1, seed=1)
weights_sum = sum(para_diff.x)
for i in range(len(para_diff.x)):
    para_diff.x[i] = para_diff.x[i] / weights_sum
print(para_diff.x)
print(sum(para_diff.x))
Block_End = process_time()-Block_Start
Acc_End = process_time()-Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End, 2) + "secs")
print("Accumulate use " + "%0.2f" % round(Acc_End, 2) + "secs\n")

final_pred_2 = (np.sum(para_ols.x * y_all[y_all.columns[1:len(y_all.columns)]], axis=1) +
                np.sum(para_slqsp.x * y_all[y_all.columns[1:len(y_all.columns)]], axis=1) +
                np.sum(para_diff.x * y_all[y_all.columns[1:len(y_all.columns)]], axis=1)) / 3

print(sum((final_pred_1 - final_pred_2) < 1e-10) == len(y_all))
