from pylab import *

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
    sum = sum / (len(x)-1)
    return sum

"""
plots the bayesian classifier and stores the plot on disk for the given data set
"""
def plot_bayes():
    #loading the data
    data = load("data.npy")

    # plotting the given data
    i = 0
    while (i < len(data[:, 1]) - 1):
        i = i + 1
        if (data[i, 2] == 1):
            plt.scatter(data[i, 0], data[i, 1], marker='o', color='g')
        else:
            plt.scatter(data[i, 0], data[i, 1], marker='o', color='b')

    #splitting the array into seperate arrays
    xc0 = data[:100, 0]
    yc0 = data[:100, 1]
    xc1 = data[100:, 0]
    yc1 = data[100:, 1]

    #computing the scale for the matrix
    scalex = [min(min(xc0), min(xc1)), max(max(xc0), max(xc1))]
    scaley = [min(min(yc0), min(yc1)), max(max(yc0), max(yc1))]

    meanxc0 = 0
    meanxc1 = 0
    meanyc1 = 0
    meanyc0 = 0

    #computing the mean of the points
    for i in range(len(xc0)):
        meanxc0 = meanxc0 + xc0[i]
        meanxc1 = meanxc1 + xc1[i]
        meanyc0 = meanyc0 + yc0[i]
        meanyc1 = meanyc1 + yc1[i]

    meanxc0 = meanxc0 / len(xc0)
    meanxc1 = meanxc1 / len(xc0)
    meanyc1 = meanyc1 / len(xc0)
    meanyc0 = meanyc0 / len(xc0)

    cov_class0 = [[0, 0],
                  [0, 0]]
    cov_class1 = [[0, 0],
                  [0, 0]]

    #computing the covariances of the classes by passing the data to covariance function
    cov_class0[0][0] = covariance(xc0, xc0, [meanxc0, meanxc0])
    cov_class0[0][1] = covariance(xc0, yc0, [meanxc0, meanyc0])
    cov_class0[1][0] = cov_class0[0][1]
    cov_class0[1][1] = covariance(yc0, yc0, [meanyc0, meanyc0])

    cov_class1[0][0] = covariance(xc1, xc1, [meanxc1, meanxc1])
    cov_class1[0][1] = covariance(xc1, yc1, [meanxc1, meanyc1])
    cov_class1[1][0] = cov_class1[0][1]
    cov_class1[1][1] = covariance(yc1, yc1, [meanyc1, meanyc1])

    #combining the covariances and means into single variable
    cov_class = [cov_class0, cov_class1]
    mean_class = [[meanxc0, meanyc0], [meanxc1, meanyc1]]


    gap = 0.08
    gi_class = [0, 0]

    #iterating for all the points and checking the posterior probability of the point being in class 0 or 1
    for x in np.arange(scalex[0], scalex[1], gap):
        for y in np.arange(scaley[0], scaley[1], gap):
            for point_class in range(2):
                point = np.array([x, y])
                cov_inverse = np.linalg.inv(cov_class[point_class])

                #computing the probability of Z being in the class 0 or 1
                gi_class[point_class] = (np.matmul(np.matmul(point, (-0.5 * cov_inverse)), point.T)) + \
                                        np.matmul(
                                            np.matrix(np.matmul(cov_inverse, np.matrix(mean_class[point_class]).T)).T,
                                            point.T) + \
                                        (np.matmul(np.matmul((-0.5 * np.asarray( mean_class[point_class])), cov_inverse),
                                                   np.matrix(mean_class[point_class]).T)) - \
                                        (0.5 * (log(np.linalg.det(cov_class[point_class])))) + \
                                        (np.log(0.5))


            #modifying to get the plot boundaries
            a = gi_class[0]*6
            b = gi_class[1]*6

            #printing the plot
            if int(a) == int(b):
                plt.scatter(x, y, marker='o', color='black', s=10)
            elif gi_class[0] < gi_class[1]:
                plt.scatter(x, y, marker='o', color='g', s=0.7)
            else:
                plt.scatter(x, y, marker='o', color='b', s=0.7)

    g=0;b=0

    i = 0
    while (i < len(data[:, 1]) - 1):
        i = i + 1

        for point_class in range(2):
            point = np.array([data[i,0], data[i,1]])
            cov_inverse = np.linalg.inv(cov_class[point_class])

            # computing the probability of Z being in the class 0 or 1
            gi_class[point_class] = (np.matmul(np.matmul(point, (-0.5 * cov_inverse)), point.T)) + \
                                    np.matmul(
                                        np.matrix(np.matmul(cov_inverse, np.matrix(mean_class[point_class]).T)).T,
                                        point.T) + \
                                    (np.matmul(np.matmul((-0.5 * np.asarray(mean_class[point_class])), cov_inverse),
                                               np.matrix(mean_class[point_class]).T)) - \
                                    (0.5 * (log(np.linalg.det(cov_class[point_class])))) + \
                                    (np.log(0.5))

        # printing the plot
        if (gi_class[0] > gi_class[1]) and (data[i,2]!=1):
            b = b+1
        elif gi_class[0] < gi_class[1]and (data[i,2]==1):
            g=g+1

    # printing the results
    print("          Classification rate for Bayesian classifier is - " + str(((g + b) / 200) * 100) + "%")
    print("          Confusion matrix for linear classifier is as follows:")
    print("                   Predicted       Predicted")
    print("                   green           blue")
    print("   Actual Green     " + str(g) + "              " + str(100 - g) + "")
    print("   Actual Blue      " + str(100 - b) + "              " + str(b) + "")
    print();
    print()


    #saving the plot file
    plt.savefig("bayes_classify.png")


if __name__ == "__main__":
    plot_bayes()
