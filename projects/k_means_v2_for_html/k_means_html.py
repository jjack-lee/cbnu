# %%

from aem import app
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
import random
from flask import Flask, send_file, Response, render_template, request

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io

app = Flask(__name__)

datas = []

# # start input parameter
# dataCount = 150
# init_count = 20
# clusters = 5
# # end input parameter

# checkCount = dataCount/clusters

# deviation = dataCount

# start input parameter
dataCount = 0
init_count = 10
clusters = 3
# end input parameter

checkCount = dataCount/clusters

deviation = 99999

chooseResultX = []
chooseResultY = []
chooseCentrooids = []


marker = ['o', 'v', '^', '<', '>', '8', 's',
          'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
color = ['green', 'blue', 'black',
         'Cyan', 'Magenta', 'orange', 'lightblue', 'yellow']


def distance(x1, y1, x2, y2):
    result = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    return result


def findCenter(Xs, Ys, beforeCenter):
    if(len(Xs) < 1):
        return beforeCenter

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
        if(tempD[0] > d[0]):
            tempD = d

    return tempD


def excute():
    global deviation
    global chooseCentrooids
    global chooseResultX
    global chooseResultY
    global datas
    global clusters
    global checkCount

    randomV = randomValue()
    centrooids = []

    for i in range(0, clusters):
        centrooids.append([datas[randomV[i]][0], datas[randomV[i]][1]])

    flag = True

    while flag:
        resultX = []
        resultY = []
        for i in range(0, clusters):
            resultX.append([])
            resultY.append([])

        for data in datas:
            distances = []
            for i in range(0, clusters):
                distances.append([distance(centrooids[i][0],
                                 centrooids[i][1], data[0], data[1]), i])

            minD = findMinDistance(distances)

            resultX[minD[1]].append(data[0])
            resultY[minD[1]].append(data[1])

        findCentrooids = []

        for i in range(0, clusters):
            findCentrooids.append(findCenter(
                resultX[i], resultY[i], centrooids[i]))

        if(findCentrooids != centrooids):
            centrooids = findCentrooids
        else:
            flag = False
            sum = 0
            for i in range(0, clusters):
                sum += abs(checkCount - len(resultX[i]))
                # print("i : ", i, "sum : ", sum)

            # print('==========')
            # print(centrooids)
            # print(findCentrooids)
            # print(deviation, "   ", sum)
            # print(resultX)
            # print(chooseResultX)

            if(deviation > sum):
                # print("change")
                deviation = sum
                chooseResultX = resultX
                chooseResultY = resultY
                chooseCentrooids = findCentrooids

            # print('==========')


def randomValue():
    global dataCount
    global clusters

    value = set()
    list = []
    while len(value) < clusters:
        value.add(random.randrange(1, dataCount))

    for v in range(0, len(value)):
        list.append(value.pop())

    return list


def makeKmeans():
    global dataCount
    global clusters
    global checkCount
    global deviation
    global datas
    global init_count
    global chooseResultX
    global chooseResultY
    global chooseCentrooids

    deviation = 99999
    chooseResultX = []
    chooseResultY = []
    chooseCentrooids = []

    checkCount = dataCount/clusters

    f = plt.figure()

    for i in range(0, init_count):
        print("init makeKmeans : ", i)
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

    return f


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/setP')
def set_p():
    global init_count
    global clusters
    init_countP = int(request.args.get('init_count'))
    clustersP = int(request.args.get('clusters'))
    init_count = init_countP
    clusters = clustersP
    return "setP true"


@app.route('/getP')
def get_p():
    print(init_count)
    print(clusters)
    return "getP true"


@app.route('/setData')
def set_data():
    global datas
    global dataCount
    global deviation
    datas.append([float(request.args.get('x_value')),
                 float(request.args.get('y_value'))])

    dataCount = len(datas)
    deviation = 99999
    print(datas)
    print("setData =", dataCount)
    return "setData true"


@app.route('/makeKmeans')
def plot_png():
    global dataCount
    global makeKmeans

    if(dataCount < 5):
        return "data가 5개 이상이여야 표시됩니다."

    fig = makeKmeans()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


if __name__ == '__main__':
    app.run()

# %%
