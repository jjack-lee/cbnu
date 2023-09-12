# %%

from aem import app
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
import random

# start input parameter
dataCount = 150
init_count = 20
clusters = 5
# end input parameter

marker = ['o', 'v', '^', '<', '>', '8', 's',
          'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
color = ['green', 'blue', 'black',
         'Cyan', 'Magenta', 'orange', 'lightblue', 'yellow']

checkCount = dataCount/clusters

deviation = dataCount

chooseResultX = []
chooseResultY = []
chooseCentrooids = []

print("========== sample data ==========")
X, y = make_blobs(n_samples=dataCount,  # 데이터 수
                  n_features=2,  # 독립 변수 수
                  centers=1,  # 클러스터 수
                  cluster_std=4,  # 클러스터의 표준 편차
                  shuffle=True,  # 숫자를 랜덤으로 섞을 것인지
                  )
print(X[0])

plt.scatter(X[:, 0],
            X[:, 1],
            c='white',
            marker='o',
            edgecolors='black',
            s=50)

plt.grid()
plt.tight_layout()
plt.show()
print("")


def orgKmeans():
    kmean = KMeans(n_clusters=clusters,
                   init='random',
                   n_init=init_count,
                   max_iter=300,
                   tol=1e-04,
                   random_state=0)

    y_km = kmean.fit_predict(X)
    sum = 0

    for i in range(0, clusters):
        sum += abs(checkCount - len(X[y_km == i, 0]))

    print("편차 합 : ", sum)

    for i in range(0, clusters):
        print("c ", i, " 점 갯수 : ", len(X[y_km == i, 0]))

    for i in range(0, clusters):
        label = 'c' + str(i)
        marker_num = i % len(marker)
        color_num = i % len(color)
        plt.scatter(
            X[y_km == i, 0],
            X[y_km == i, 1],
            s=50, c=color[color_num],
            marker=marker[marker_num], edgecolors='black',
            label=label)

    plt.scatter(
        kmean.cluster_centers_[:, 0],
        kmean.cluster_centers_[:, 1],
        s=250, c='red',
        marker='*', edgecolors='black',
        label='Centrooids')

    plt.legend(scatterpoints=1)
    plt.grid()
    plt.tight_layout()
    plt.show()


def distance(x1, y1, x2, y2):
    result = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    return result


def findCenter(Xs, Ys):

    minX = min(Xs)
    maxX = max(Xs)
    minY = min(Ys)
    maxY = max(Ys)

    centerX = minX+((maxX-minX)/2)
    centerY = minY+((maxY-minY)/2)
    return [centerX, centerY]


def findMinDistance(distances):
    tempD = [999999999, 0]

    for d in distances:
        if (tempD[0] > d[0]):
            tempD = d

    return tempD


def excute():
    global deviation
    global chooseCentrooids
    global chooseResultX
    global chooseResultY

    randomV = randomValue()

    centrooids = []

    for i in range(0, clusters):
        centrooids.append([X[randomV[i]][0], X[randomV[i]][1]])

    flag = True

    while flag:
        resultX = []
        resultY = []
        for i in range(0, clusters):
            resultX.append([])
            resultY.append([])
        for x in X:
            distances = []
            for i in range(0, clusters):
                distances.append([distance(centrooids[i][0],
                                 centrooids[i][1], x[0], x[1]), i])

            minD = findMinDistance(distances)

            resultX[minD[1]].append(x[0])
            resultY[minD[1]].append(x[1])

        findCentrooids = []

        for i in range(0, clusters):
            findCentrooids.append(findCenter(resultX[i], resultY[i]))

        if (findCentrooids != centrooids):
            centrooids = findCentrooids
        else:
            flag = False
            sum = 0
            for i in range(0, clusters):
                sum += abs(checkCount - len(resultX[i]))

            # print('==========')
            # print(centrooids)
            # print(findCentrooids)
            # print(deviation, "   ", sum)

            if (deviation > sum):
                # print("change")
                deviation = sum
                chooseResultX = resultX
                chooseResultY = resultY
                chooseCentrooids = findCentrooids

            # print('==========')


def randomValue():
    value = set()
    list = []
    while len(value) < clusters:
        value.add(random.randrange(1, dataCount))

    for v in range(0, len(value)):
        list.append(value.pop())

    return list


def makeKmeans():
    for i in range(0, init_count):
        excute()

    print("편차 합 : ", deviation)
    for i in range(0, clusters):
        print("c ", i, " 점 갯수 : ", len(chooseResultX[i]))

    for i in range(0, clusters):
        label = 'c' + str(i)
        marker_num = i % len(marker)
        color_num = i % len(color)
        plt.scatter(
            chooseResultX[i],
            chooseResultY[i],
            s=50, c=color[color_num],
            marker=marker[marker_num], edgecolors='black',
            label=label)

    chooseCentrooidsX = []
    chooseCentrooidsY = []

    for i in range(0, clusters):
        chooseCentrooidsX.append(chooseCentrooids[i][0])
        chooseCentrooidsY.append(chooseCentrooids[i][1])

    plt.scatter(
        chooseCentrooidsX,
        chooseCentrooidsY,
        s=250, c='red',
        marker='*', edgecolors='black',
        label='Centrooids')

    plt.legend(scatterpoints=1)
    plt.grid()
    plt.tight_layout()
    plt.show()


print('========== func orignal Kmeans ==========')
orgKmeans()
print('')

print('========== func make Kmeans ==========')
makeKmeans()
print('')

# %%
