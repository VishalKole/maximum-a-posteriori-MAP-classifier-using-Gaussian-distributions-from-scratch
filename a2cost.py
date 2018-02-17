from pylab import *
import csv

"""
@Author - Vishal V Kole - vvk3025@rit.edu
For partial completion of the course Pattern Recognition
"""

"""
compute the covariance of the given X and Y values and the mean of the data points and return the matrix
"""


def covariance(x, y, mean):
    sum = 0
    for i in range(len(x)):
        sum = sum + ((x[i] - mean[0]) * (y[i] - mean[1]))
    sum = sum / (len(x) - 1)
    return sum


"""
plots the bayesian classifier and stores the plot on disk for the given data set
"""


def Question2(flag):
    # load the data
    data = np.ndarray(shape=(94, 3))
    with open('nuts_bolts.csv') as file:
        spamreader = csv.reader(file, delimiter=',', quotechar='|')
        i = 0
        for row in spamreader:
            data.put(indices=i, values=float(row[0]))
            data.put(indices=i + 1, values=(float(row[1])))
            data.put(indices=i + 2, values=(int(row[2])))
            i = i + 3

    # plotting the given data
    for row in data:
        if row[2] == 1:
            plt.scatter(row[0], row[1], marker='s', color='blue', s=9)
        elif row[2] == 2:
            plt.scatter(row[0], row[1], marker='o', color='green', s=9)
        elif row[2] == 3:
            plt.scatter(row[0], row[1], marker='x', color='brown', s=9)
        elif row[2] == 4:
            plt.scatter(row[0], row[1], marker='D', color='red', s=9)

    # setting the initial arrays
    xc0 = data[:20, 0];
    yc0 = data[:20, 1]
    xc1 = data[20:48, 0];
    yc1 = data[20:48, 1]
    xc2 = data[48:75, 0];
    yc2 = data[48:75, 1]
    xc3 = data[75:, 0];
    yc3 = data[75:, 1]

    # mean of the classes
    meanxc0 = 0;
    meanxc1 = 0;
    meanxc2 = 0;
    meanxc3 = 0
    meanyc0 = 0;
    meanyc1 = 0;
    meanyc2 = 0;
    meanyc3 = 0

    for i in range(len(xc0)):
        meanxc0 = meanxc0 + xc0[i]
        meanyc0 = meanyc0 + yc0[i]

    for i in range(len(xc1)):
        meanxc1 = meanxc1 + xc1[i]
        meanyc1 = meanyc1 + yc1[i]

    for i in range(len(xc2)):
        meanxc2 = meanxc2 + xc2[i]
        meanyc2 = meanyc2 + yc2[i]

    for i in range(len(xc3)):
        meanxc3 = meanxc3 + xc3[i]
        meanyc3 = meanyc3 + yc3[i]

    meanxc0 = meanxc0 / len(xc0);
    meanxc1 = meanxc1 / len(xc1);
    meanxc2 = meanxc2 / len(xc2);
    meanxc3 = meanxc3 / len(xc3);
    meanyc0 = meanyc0 / len(xc0);
    meanyc1 = meanyc1 / len(xc1);
    meanyc2 = meanyc2 / len(xc2);
    meanyc3 = meanyc3 / len(xc3);

    # covariances of the classes
    cov_class0 = [[0, 0],
                  [0, 0]]
    cov_class1 = [[0, 0],
                  [0, 0]]
    cov_class2 = [[0, 0],
                  [0, 0]]
    cov_class3 = [[0, 0],
                  [0, 0]]

    cov_class0[0][0] = covariance(xc0, xc0, [meanxc0, meanxc0])
    cov_class0[0][1] = covariance(xc0, yc0, [meanxc0, meanyc0])
    cov_class0[1][0] = cov_class0[0][1]
    cov_class0[1][1] = covariance(yc0, yc0, [meanyc0, meanyc0])

    cov_class1[0][0] = covariance(xc1, xc1, [meanxc1, meanxc1])
    cov_class1[0][1] = covariance(xc1, yc1, [meanxc1, meanyc1])
    cov_class1[1][0] = cov_class1[0][1]
    cov_class1[1][1] = covariance(yc1, yc1, [meanyc1, meanyc1])

    cov_class2[0][0] = covariance(xc2, xc2, [meanxc2, meanxc2])
    cov_class2[0][1] = covariance(xc2, yc2, [meanxc2, meanyc2])
    cov_class2[1][0] = cov_class2[0][1]
    cov_class2[1][1] = covariance(yc2, yc2, [meanyc2, meanyc2])

    cov_class3[0][0] = covariance(xc3, xc3, [meanxc3, meanxc3])
    cov_class3[0][1] = covariance(xc3, yc3, [meanxc3, meanyc3])
    cov_class3[1][0] = cov_class3[0][1]
    cov_class3[1][1] = covariance(yc3, yc3, [meanyc3, meanyc3])

    # cost matrix
    cost_matrix = [[-0.20, 0.07, 0.07, 0.07],
                   [0.07, -0.15, 0.07, 0.07],
                   [0.07, 0.07, -0.05, 0.07],
                   [0.03, 0.03, 0.03, 0.03]]

    # constant cost matrix
    cont_cost_matrix = [[0, 1, 1, 1],
                        [1, 0, 1, 1],
                        [1, 1, 0, 1],
                        [1, 1, 1, 0]]

    if flag == 0:
        cost_matrix = cont_cost_matrix

    # priors and other consolidations
    priors = [1040 / 1114, 28 / 1114, 27 / 1114, 19 / 1114]
    cov_class = [cov_class0, cov_class1, cov_class2, cov_class3]
    mean_class = [[meanxc0, meanyc0], [meanxc1, meanyc1], [meanxc2, meanyc2], [meanxc3, meanyc3]]

    # gap = 0.02
    gap = 0.01
    gi_class = [0, 0, 0, 0]
    v = plt.axis()

    # computing for each point in the grid
    for x in np.arange(v[0] - 0.1, v[1] + 0.03, gap):
        for y in np.arange(v[2] - 0.1, v[3] + 0.3, gap):
            for point_class in range(4):
                point = np.array([x, y])
                cov_inverse = np.linalg.inv(cov_class[point_class])
                gi_class[point_class] = (np.matmul(np.matmul(point, (-0.5 * cov_inverse)), point.T)) + \
                                        np.matmul(
                                            np.matrix(np.matmul(cov_inverse, np.matrix(mean_class[point_class]).T)).T,
                                            point.T) + \
                                        (np.matmul(np.matmul((-0.5 * np.asarray(mean_class[point_class])), cov_inverse),
                                                   np.matrix(mean_class[point_class]).T)) - \
                                        (0.5 * (log(np.linalg.det(cov_class[point_class])))) + \
                                        (np.log(priors[point_class]))

            # computiong a persentage value from the discriminant values
            minv = min(gi_class)
            minv = abs(minv)
            gi_class[0] = gi_class[0] + minv + 1
            gi_class[1] = gi_class[1] + minv + 1
            gi_class[2] = gi_class[2] + minv + 1
            gi_class[3] = gi_class[3] + minv + 1

            total = gi_class[0] + gi_class[1] + gi_class[2] + gi_class[3]
            gi_class[0] = gi_class[0] / total
            gi_class[1] = gi_class[1] / total
            gi_class[2] = gi_class[2] / total
            gi_class[3] = gi_class[3] / total

            # multiplying with the cost
            final = [0, 0, 0, 0]
            for w in range(4):
                sum = 0
                for k in range(4):
                    a = cost_matrix[w][k]
                    b = gi_class[k]
                    c = priors[k]
                    vv = (a * b)
                    sum = sum + vv
                final[w] = sum

            class_ind = final.index(min(final))

            # plotting the values
            if class_ind == 0:
                plt.scatter(x, y, marker='o', color='blue', s=0.7)
            elif class_ind == 1:
                plt.scatter(x, y, marker='o', color='green', s=0.7)
            elif class_ind == 2:
                plt.scatter(x, y, marker='o', color='yellow', s=0.7)
            elif class_ind == 3:
                plt.scatter(x, y, marker='o', color='red', s=0.7)

    c1 =[0, 0, 0, 0]
    c2 = [0, 0, 0, 0]
    c3 =[0, 0, 0, 0]
    c4 = [0, 0, 0, 0]
    i = 0
    while (i < len(data[:, 1]) - 1):
        i = i + 1
        for point_class in range(4):
            point = np.array([data[i, 0], data[i, 1]])
            cov_inverse = np.linalg.inv(cov_class[point_class])
            gi_class[point_class] = (np.matmul(np.matmul(point, (-0.5 * cov_inverse)), point.T)) + \
                                    np.matmul(np.matrix(np.matmul(cov_inverse, np.matrix(mean_class[point_class]).T)).T,
                                              point.T) + \
                                    (np.matmul(np.matmul((-0.5 * np.asarray(mean_class[point_class])), cov_inverse),
                                               np.matrix(mean_class[point_class]).T)) - \
                                    (0.5 * (log(np.linalg.det(cov_class[point_class])))) + \
                                    (np.log(priors[point_class]))

            # multiplying with the cost
            final = [0, 0, 0, 0]
            for w in range(4):
                sum = 0
                for k in range(4):
                    a = cost_matrix[w][k]
                    b = gi_class[k]
                    c = priors[k]
                    vv = (a * b)
                    sum = sum + vv
                final[w] = sum

            class_ind = final.index(min(final))

        # plotting the values
        if class_ind == 0:
            if data[i, 2] == 1:
                c1[0] = c1[0] + 1
            if data[i, 2] == 2:
                c1[1] = c1[1] + 1
            if data[i, 2] == 3:
                c1[2] = c1[2] + 1
            if data[i, 2] == 4:
                c1[3] = c1[3] + 1

        elif class_ind == 1:
            if data[i, 2] == 1:
                c2[0] = c2[0] + 1
            if data[i, 2] == 2:
                c2[1] = c2[1] + 1
            if data[i, 2] == 3:
                c2[2] = c2[2] + 1
            if data[i, 2] == 4:
                c2[3] = c2[3] + 1
        elif class_ind == 2:
            if data[i, 2] == 1:
                c3[0] = c3[0] + 1
            if data[i, 2] == 2:
                c3[1] = c3[1] + 1
            if data[i, 2] == 3:
                c3[2] = c3[2] + 1
            if data[i, 2] == 4:
                c3[3] = c3[3] + 1
        elif class_ind == 3:
            if data[i, 2] == 1:
                c4[0] = c4[0] + 1
            if data[i, 2] == 2:
                c4[1] = c4[1] + 1
            if data[i, 2] == 3:
                c4[2] = c4[2] + 1
            if data[i, 2] == 4:
                c4[3] = c4[3] + 1

    if flag == 0:
        print("___________________________Uniform Cost")
    else:
        print("___________________________Non-uniform Cost")
    print("          Classification rate for Bayesian classifier for question two is - " + str(
        ((c1[0] + c2[1] + c3[2] + c4[3]) / 94) * 100) + "%")
    print("          Confusion matrix for linear classifier is as follows:")
    print("                   Predicted    Predicted   Predicted   Predicted")
    print("                   bolt         nut         ring        scrap")
    print("   Actual bolt    " + str(c1[0]) + "             " + str(c1[1]) + "           " + str(c1[2]) + "            " + str(c1[3]))
    print("   Actual nut      " + str(c2[0]) + "            " + str(c2[1]) + "           " + str(c2[2]) + "            " + str(c2[3]))
    print("   Actual ring     " + str(c3[0]) + "             " + str(c3[1]) + "           " + str(c3[2]) + "           " + str(c3[3]))
    print("   Actual scrap    " + str(c4[0]) +  "             " + str(c4[1]) + "           " + str(c4[2]) +  "            " + str(c4[3]))
    print();
    print()

    if flag == 0:
        plt.savefig("bayes, nuts and bolts, constant cost.png")
    else:
        plt.savefig("bayes, nuts and bolts, with cost.png")


if __name__ == "__main__":
    Question2(0)
    Question2(1)
