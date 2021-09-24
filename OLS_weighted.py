"""
构造某时序的历史真实值和对应期间的几条预测值，将这(n+1)条时序，用不同方式构造最小二乘的损失函数，再用scipy.optimize中各局部优化及全局优化方法
来求解这些损失函数，以测试各种优化方法求解各种损失函数时的速度、精度、稳定性。

重要：对于scipy.optimize中的优化方法，不能通过改变目标函数的方式，将含有联合自变量的约束条件加入目标函数；否则对初值极敏感，混沌性极强。
要增加带有联合自变量的约束条件，必须在方法中传入constraints关键字参数；若该方法没有constraints参数，则不能定义带有联合自变量的约束条件。
In optimize.minimize, 'SLSQP' is faster than other methods and has 'constraints',
and is also faster than optimize.least_squares(method = 'dogbox'), which is faster than 'trf' with 'bounds'
in optimize.least_squares and has no 'constraints'. 'lm' method doesn't support 'bounds'.
In arithmetic operation, firstly transforming data type into numpy.array, then using numpy operation,
that is faster than python original loop, which is faster than calculations in pandas.
"""


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

Block_End = process_time()-Acc_Start
Acc_End = process_time()-Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End, 2) + "secs")
print("Accumulate use " + "%0.2f" % round(Acc_End, 2) + "secs\n")


w = np.array([1/(len(y_all.columns)-1) for _ in range(len(y_all.columns)-1)])  # 设置梯度下降时权重的初始值，在生产程序中取MAPE动态加权后生成的值
bounds = [(1e-5, 1-1e-5)]*len(w)
y_all = np.array(y_all)


def ols(alpha):
    # ols is much faster than ols_vector, and ols_vector is the same with ols_vector_1.
    yhat = np.zeros(len(y_all))
    for i in range(len(alpha)):
        yhat += alpha[i] * y_all[:, i+1]  # yhat是多条预测序列加权后的预测值序列
    residual = yhat - y_all[:, 0]  # the type of residual is ndarray.
    return residual


def ols_vector(alpha):
    # the type of alpha is np.array, and y_all is pd.DataFrame.
    yhat = (alpha * np.delete(y_all, 0, axis=1)).sum(axis=1)
    residual = yhat - y_all[:, 0]
    return residual


def ols_vector_df(alpha):
    y_all_df = DataFrame(y_all)
    # the type of alpha is np.array, and y_all is pd.DataFrame.
    yhat = (alpha*y_all_df.drop(columns=[0])).sum(axis=1)
    residual = yhat - y_all_df[0]
    return np.array(residual)


def ols_union_variate_1(alpha):
    y_all_df = DataFrame(y_all)
    yhat = np.zeros(len(y_all))
    part = 0
    for i in range(1, len(alpha)):
        yhat += alpha[i]*np.array(y_all_df[i+1])  # yhat是多条预测序列加权后的预测值序列
        part += alpha[i]
    yhat += (1 - part)*np.array(y_all_df[1])
    residual = yhat - np.array(y_all_df[0])
    return residual


def ols_union_variate_2(alpha):
    y_all_df = DataFrame(y_all)
    yhat = np.zeros(len(y_all))
    part = 0
    for i in range(1, len(alpha)):
        yhat += alpha[i]*np.array(y_all_df[i+1])  # yhat是多条预测序列加权后的预测值序列
        part += alpha[i]
    yhat += (1.0000000001 - part)*np.array(y_all_df[1])
    residual = yhat - np.array(y_all_df[0])
    return residual


def loss_function(alpha):
    yhat = np.zeros(len(y_all))
    distance = np.empty(shape=[0])
    for i in range(len(yhat)):
        for j in range(len(alpha)):
            yhat[i] += alpha[j] * y_all[i, j+1]
        distance = np.append(distance, (yhat[i] - y_all[i, 0])**2)
    distance = 0.5*sum(distance)
    return distance


def loss_function_vector(alpha):
    yhat = (alpha * np.delete(y_all, 0, axis=1)).sum(axis=1)
    distance = 0.5*sum((yhat - y_all[:, 0])**2)
    return distance


def loss_function_union_variate_1(alpha):
    y_all_df = DataFrame(y_all)
    yhat = np.zeros(len(y_all))
    part = 0
    distance = []
    for i in range(len(yhat)):
        for j in range(1, len(alpha)):
            yhat[i] += alpha[j] * y_all_df[j + 1][i]
            part += alpha[j]
        distance.append((yhat[i] + (1-part)*y_all_df[1][i] - y_all_df[0][i])**2)
    distance = 0.5*sum(distance)
    return np.array(distance)


def loss_function_union_variate_2(alpha):
    y_all_df = DataFrame(y_all)
    yhat = np.zeros(len(y_all))
    part = 0
    distance = []
    for i in range(len(yhat)):
        for j in range(1, len(alpha)):
            yhat[i] += alpha[j] * y_all_df[j + 1][i]
            part += alpha[j]
        distance.append((yhat[i] + (1.0000000001-part)*y_all_df[1][i] - y_all_df[0][i])**2)
    distance = 0.5*sum(distance)
    return np.array(distance)


def h1(alpha):
    return sum(alpha) - 1  # == 0


cons = ({'type': 'eq', 'fun': h1})
n = 1000

# ordinary linear least square optimization using optimize.least_squares
Block_Start = process_time()
for _ in range(n):
    para = optimize.least_squares(ols, w, jac='2-point', bounds=([1e-5]*len(w), 1-1e-5), method='dogbox')
print('optimize.least_squares, ols, method=dogbox')
print(para.x)
print(sum(para.x))
Block_End = process_time()-Block_Start
Acc_End = process_time()-Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End, 2) + "secs")
print("Accumulate use " + "%0.2f" % round(Acc_End, 2) + "secs\n")


Block_Start = process_time()
for _ in range(n):
    para = optimize.least_squares(ols_vector, w, jac='2-point', bounds=([1e-5]*len(w), 1-1e-5), method='dogbox')
print('optimize.least_squares, ols_vector, method=dogbox')
print(para.x)
print(sum(para.x))
Block_End = process_time()-Block_Start
Acc_End = process_time()-Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End, 2) + "secs")
print("Accumulate use " + "%0.2f" % round(Acc_End, 2) + "secs\n")


Block_Start = process_time()
for _ in range(n):
    para = optimize.least_squares(ols_vector, w, jac='2-point', method='lm')
print('optimize.least_squares, ols_vector, method=lm')
print(para.x)
print(sum(para.x))
Block_End = process_time()-Block_Start
Acc_End = process_time()-Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End, 2) + "secs")
print("Accumulate use " + "%0.2f" % round(Acc_End, 2) + "secs\n")


Block_Start = process_time()
for _ in range(n):
    para = optimize.least_squares(ols_vector, w, jac='2-point', bounds=([1e-5]*len(w), 1-1e-5), method='trf')
print('optimize.least_squares, ols_vector, method=trf')
print(para.x)
print(sum(para.x))
Block_End = process_time()-Block_Start
Acc_End = process_time()-Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End, 2) + "secs")
print("Accumulate use " + "%0.2f" % round(Acc_End, 2) + "secs\n")


Block_Start = process_time()
for _ in range(n):
    para = optimize.least_squares(ols_vector_df, w, jac='2-point', bounds=([1e-5]*len(w), 1-1e-5), method='dogbox')
print('optimize.least_squares, ols_vector_df, method=dogbox')
print(para.x)
print(sum(para.x))
Block_End = process_time()-Block_Start
Acc_End = process_time()-Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End, 2) + "secs")
print("Accumulate use " + "%0.2f" % round(Acc_End, 2) + "secs\n")


# ordinary linear least square optimization
Block_Start = process_time()
para = optimize.least_squares(ols_union_variate_1, w, jac='2-point', bounds=([1e-5]*len(w), 1-1e-5), method='dogbox')
print('optimize.least_squares, ols_union_variate_1, method=dogbox')
print(para.x)
print(sum(para.x))
Block_End = process_time()-Block_Start
Acc_End = process_time()-Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End, 2) + "secs")
print("Accumulate use " + "%0.2f" % round(Acc_End, 2) + "secs\n")


Block_Start = process_time()
para = optimize.least_squares(ols_union_variate_2, w, jac='2-point', bounds=([1e-5]*len(w), 1-1e-5), method='dogbox')
print('optimize.least_squares, ols_union_variate_2, method=dogbox')
print(para.x)
print(sum(para.x))
Block_End = process_time()-Block_Start
Acc_End = process_time()-Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End, 2) + "secs")
print("Accumulate use " + "%0.2f" % round(Acc_End, 2) + "secs\n")
# # 由上可知，使用optimize.least_squares时，当有自变量间的联合约束条件时，算法对初值极敏感；所以只能用自变量的独立约束条件bounds


# ordinary linear least square optimization using optimize.minimize
Block_Start = process_time()
for _ in range(n):
    para = optimize.minimize(loss_function, w, method='SLSQP', bounds=bounds, constraints=cons)
print('optimize.minimize, loss_function, method=SLSQP')
print('local minimized point is:', para.x)
print(sum(para.x))
Block_End = process_time()-Block_Start
Acc_End = process_time()-Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End, 2) + "secs")
print("Accumulate use " + "%0.2f" % round(Acc_End, 2) + "secs\n")


Block_Start = process_time()
for _ in range(n):
    para = optimize.minimize(loss_function_vector, w, method='SLSQP', bounds=bounds, constraints=cons)
print('optimize.minimize, loss_function_vector, method=SLSQP')
print('local minimized point is:', para.x)
print(sum(para.x))
Block_End = process_time()-Block_Start
Acc_End = process_time()-Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End, 2) + "secs")
print("Accumulate use " + "%0.2f" % round(Acc_End, 2) + "secs\n")


Block_Start = process_time()
para = optimize.minimize(loss_function_union_variate_1, w, method='SLSQP', bounds=bounds)
print('optimize.minimize, loss_function_union_variate_1, method=SLSQP')
print('local minimized point is:', para.x)
print(sum(para.x))
Block_End = process_time()-Block_Start
Acc_End = process_time()-Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End, 2) + "secs")
print("Accumulate use " + "%0.2f" % round(Acc_End, 2) + "secs\n")


Block_Start = process_time()
para = optimize.minimize(loss_function_union_variate_2, w, method='SLSQP', bounds=bounds)
print('optimize.minimize, loss_function_union_variate_2, method=SLSQP')
print('local minimized point is:', para.x)
print(sum(para.x))
Block_End = process_time()-Block_Start
Acc_End = process_time()-Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End, 2) + "secs")
print("Accumulate use " + "%0.2f" % round(Acc_End, 2) + "secs\n")


Block_Start = process_time()
for _ in range(n):
    para = optimize.minimize(loss_function_vector, w, method='L-BFGS-B', bounds=bounds)
print('optimize.minimize, loss_function_vector, method=L-BFGS-B')
print('local minimized point is:', para.x)
print(sum(para.x))
Block_End = process_time()-Block_Start
Acc_End = process_time()-Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End, 2) + "secs")
print("Accumulate use " + "%0.2f" % round(Acc_End, 2) + "secs\n")


Block_Start = process_time()
para = optimize.minimize(loss_function_union_variate_1, w, method='L-BFGS-B', bounds=bounds)
print('optimize.minimize, loss_function_union_variate_1, method=L-BFGS-B')
print('local minimized point is:', para.x)
print(sum(para.x))
Block_End = process_time()-Block_Start
Acc_End = process_time()-Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End, 2) + "secs")
print("Accumulate use " + "%0.2f" % round(Acc_End, 2) + "secs\n")


Block_Start = process_time()
para = optimize.minimize(loss_function_union_variate_2, w, method='L-BFGS-B', bounds=bounds)
print('optimize.minimize, loss_function_union_variate_2, method=L-BFGS-B')
print('local minimized point is:', para.x)
print(sum(para.x))
Block_End = process_time()-Block_Start
Acc_End = process_time()-Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End, 2) + "secs")
print("Accumulate use " + "%0.2f" % round(Acc_End, 2) + "secs\n")


Block_Start = process_time()
for _ in range(n):
    para = optimize.differential_evolution(loss_function_vector, bounds, updating='deferred', workers=-1)
print('optimize.differential_evolution, loss_function_vector')
print(para.x)
print(sum(para.x))
Block_End = process_time()-Block_Start
Acc_End = process_time()-Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End, 2) + "secs")
print("Accumulate use " + "%0.2f" % round(Acc_End, 2) + "secs\n")


Block_Start = process_time()
for _ in range(n):
    para = optimize.shgo(loss_function_vector, bounds, iters=3, constraints=cons)
print('optimize.shgo, loss_function_vector')
print(para.x)
print(sum(para.x))
Block_End = process_time()-Block_Start
Acc_End = process_time()-Acc_Start
print("Execute block use " + "%0.2f" % round(Block_End, 2) + "secs")
print("Accumulate use " + "%0.2f" % round(Acc_End, 2) + "secs\n")
