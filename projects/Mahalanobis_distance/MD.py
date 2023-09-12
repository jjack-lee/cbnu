# %%
from scipy.stats import chi2  # 카이제곱 분포 라이브러리
import numpy as np
import pandas as pd
import scipy.stats as sps
import random

from sympy import print_glsl


row = 12000
col = 15

mu = np.full(col, 2)
cov = np.full((col, col), 1)
np.fill_diagonal(cov, 3)
# print(mu)
# print("="*50)
# print(cov)

rv = sps.multivariate_normal(mu, cov)
data = rv.rvs(row)

df = pd.DataFrame(data)
df['outlier'] = 0

row = 100
col = 15

mu = np.full(col, 11)
cov = np.zeros((col, col))
np.fill_diagonal(cov, 1)

# print(mu)
# print("="*50)
# print(cov)

rv = sps.multivariate_normal(mu, cov)
out = rv.rvs(row)
# print(out)
# 랜덤하게 정상 데이터와 이상치 데이터를 섞음

out = np.c_[out, np.ones(row)]
out_index = random.sample(range(0, 12000), row)
for i, val in enumerate(out_index):
    df.loc[df.index[val], :] = out[i]
# display(df)

df_y = df.outlier
df_x = df.drop(df[['outlier']], axis=1)
df_x = np.array(df_x)
realIndex = np.where(df_y == 1)

# 공분산의 역행렬 구하기
covariance = np.cov(df_x, rowvar=False)
covariance_reverse = np.linalg.matrix_power(covariance, -1)

# 중심점 찾기 (평균 이용)
centerpoint = np.mean(df_x, axis=0)

distances = []
for i, val in enumerate(df_x):
    x1 = val
    x2 = centerpoint
    distance = (x1-x2).T.dot(covariance_reverse).dot(x1-x2)
    distances.append(distance)
distances = np.array(distances)

cutoff = chi2.ppf(0.99, df_x.shape[1])

outlierIndex = np.where(distances > cutoff)

print(realIndex)
print(outlierIndex)

print("True Positive = ", len(set(outlierIndex[0]).intersection(
    set(realIndex[0])))/len(outlierIndex[0]))
# True Positive =  0.5607476635514018
len(set(realIndex[0])-(set(outlierIndex[0])))

# %%
