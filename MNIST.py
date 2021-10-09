from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random
import math


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_data = np.concatenate([X_train, X_test])

y_labels = np.concatenate([y_train, y_test])

indices = np.arange(X_data.shape[0])
np.random.shuffle(indices)

X_data = X_data[indices]
y_labels = y_labels[indices]



x = random.randint(0, y_labels.shape[0])

X_final = []
y_final = []
for i in range(0, len(y_labels)):
  if y_labels[i] == 2 or y_labels[i] == 4 or y_labels[i] == 6 or y_labels[i] == 8:
    X_final.append(X_data[i])
    y_final.append(y_labels[i])
X_final = np.array(X_final)
y_final = np.array(y_final)




X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.3, random_state=4)

Mi = 0
Ltr = []
for number in y_train:
    if number != 0 and number % 2 == 0:
        Mi += 1 
        Ltr.append(number)
print('The number of train data is: ',Mi)
Ni = 0
Lte = [] 
for number in y_test:
    if number != 0 and number % 2 == 0:
        Ni += 1 
        Lte.append(number)
print('The number of test data is: ',Ni)

# In this function i convert the images of the mnist dataset to a 1d array.
def convertMatrixToArray(A):
    result = []
    for i in range(0, 28):
        for j in range(0, 28):
            result.append(A[i][j])
    return result

#In this function insert a image to a 2D matrix.
def insertImageToMatrix(matrix, image, index):
    for i in range(0, 784):
        matrix[index][i] = image[i]
    return matrix

M = np.zeros((Mi, 784))
N = np.zeros((Ni, 784))


i = 0
for id, img in enumerate(X_train):
    if y_train[id] != 0 and y_train[id] % 2 == 0:
        M = insertImageToMatrix(M, convertMatrixToArray(img), i)
        i += 1

i = 0
for id, img in enumerate(X_test):
    if y_test[id] != 0 and y_test[id] % 2 == 0:
        N = insertImageToMatrix(N, convertMatrixToArray(img), i) 
        i += 1











def convertArrayToMatrix(A, index):
    image = np.zeros((28, 28))
    r = c = 0
    for i in range(0, 784):
        image[r][c] = A[index][i]
        c += 1
        if c == 28:
            c = 0
            r += 1
    return image











#In this function i compute the mean value of the brightness of the image in the rows which are even numbers.
def meanEvenBrightness(image):
    brightness = 0
    rowSum = 0
    for i in range(0, 28, 2):
        for j in range(0, 28):
            rowSum += image[i][j]
        brightness += rowSum / 28
        rowSum = 0

    return brightness / 14

#In this function i compute the mean value of the brightness of the image in the rows which are odd numbers.
def meanOddBrightness(image):
    brightness = 0
    colSum = 0
    for i in range(1, 28, 2):
        for j in range(0, 28):
            colSum += image[j][i]
        brightness += colSum / 28
        colSum = 0

    return brightness / 14


M_2 = np.zeros((Mi, 2))
for i in range(0, Mi):
    M_2[i][0] = meanEvenBrightness(convertArrayToMatrix(M, i))
    M_2[i][1] = meanOddBrightness(convertArrayToMatrix(M, i))

x_two = []
y_two = []

x_four = []
y_four = []

x_six = []
y_six = []

x_eight = []
y_eight = []

for i in range(0, Mi):
    if Ltr[i] == 2:
        x_two.append(M_2[i][0])
        y_two.append(M_2[i][1])
    if Ltr[i] == 4:
        x_four.append(M_2[i][0])
        y_four.append(M_2[i][1])
    if Ltr[i] == 6:
        x_six.append(M_2[i][0])
        y_six.append(M_2[i][1])
    if Ltr[i] == 8:
        x_eight.append(M_2[i][0])
        y_eight.append(M_2[i][1])  









def plotCluster(x1, y1, x2, y2, x3, y3, x4, y4, labelX, labelY, titleOfPlot):
    fig, ax = plt.subplots(1, figsize=(10, 6))
    plt.title(titleOfPlot)
    plt.scatter(x1, y1, c = 'red')
    plt.scatter(x2, y2, c = 'green')
    plt.scatter(x3, y3, c = 'blue')
    plt.scatter(x4, y4, c = 'yellow')
    ax.set_xlabel(labelX)
    ax.set_ylabel(labelY)


plotCluster(x_two, y_two, x_four, y_four, x_six, y_six, x_eight, y_eight, "Mean Even Rows", "Mean Odd Columns", "Brightness Features")











#Euclidean distance
def EuclideanDistance(a, b):
    return pow(a.x - b.x, 2) + pow(a.y - b.y, 2)


def EuclideanDistance_N_Dimension(a, b):
    return np.linalg.norm(a-b)






# Maximin algorithm
def MaximinAlgorithm(vectors, classes, ltr):
    centers = []
    center1 = vectors[0,:]
    centers.append(center1)
    total1 = ltr[0]
    
    maxDistance = 0
    k=0
    for i in range(1, Mi):
        point = vectors[i,:]
        if maxDistance < EuclideanDistance_N_Dimension(point, center1):
            maxDistance = EuclideanDistance_N_Dimension(point, center1)
            k = i

    center2 = vectors[k,:]
    centers.append(center2)
    total2 = ltr[k]


    index=0
    total34 = []
    minDistances = []
    for i in range(1, Mi):
        point = vectors[i,:]
        if i!=index and i != k and EuclideanDistance_N_Dimension(point, center1) > EuclideanDistance_N_Dimension(point, center2):
            minDistances.append(EuclideanDistance_N_Dimension(point, center2))
        if i!=index and i != k and EuclideanDistance_N_Dimension(point, center1) < EuclideanDistance_N_Dimension(point, center2):
            minDistances.append(EuclideanDistance_N_Dimension(point, center1))

    index = minDistances.index(max(minDistances))
    new_center = vectors[index+1,:]
    centers.append(new_center)
    total34.append(ltr[index])

    minDistances = []
    for i in range(1, Mi):
        point = vectors[i,:]
        if i!=index and i != k:
            minDistances.append(min(EuclideanDistance_N_Dimension(point, center1), EuclideanDistance_N_Dimension(point, center2), EuclideanDistance_N_Dimension(point, centers[0])))

    index = minDistances.index(max(minDistances))
    new_center = vectors[index+1,:]
    centers.append(new_center)
    total34.append(ltr[index])




    return centers, total1, total2, total34[0], total34[1]







def pointInSet(point, set):
    check = 0
    for i in range(0, len(set)):
        for j in range(0, len(point)):
            if point[j] != set[i][j]:
              break
            else:
              check += 1
        if check==len(point):
            return True
        check = 0
    return False

    
def calculateCentriods(a):
    axis = []
    for i in range(0, len(a[0])):
      axis.append(a[0][i])

    for i in range(0, len(a[0])):
      for j in range(1, len(a)):
        axis[i] = a[j][i]

    newCenter = [number / len(a) for number in axis]
    return newCenter
  

def KMeans(centers, num1, num2, num3, num4, matrix, ltr):
    firstSet= []
    firstNumbers = []
    firstSet.append(centers[0])
    firstNumbers.append(num1)

    secondSet = []
    secondNumbers = []
    secondSet.append(centers[1])
    secondNumbers.append(num2)

    thirdSet = []
    thirdNumbers = []
    thirdSet.append(centers[2])
    thirdNumbers.append(num3)

    fourthSet = []
    fourthNumbers = []
    fourthSet.append(centers[3])
    fourthNumbers.append(num4)


    flag = 1
    while flag!=0:
        flag = 0
        for i in range(0, Mi):     
            point = matrix[i,:]
            distance1 = EuclideanDistance_N_Dimension(point, centers[0])
            distance2 = EuclideanDistance_N_Dimension(point, centers[1])
            distance3 = EuclideanDistance_N_Dimension(point, centers[2])
            distance4 = EuclideanDistance_N_Dimension(point, centers[3])

            if distance1 == min(distance1, distance2, distance3, distance4) and (not pointInSet(point, firstSet)):
                firstSet.append(point)
                firstNumbers.append(ltr[i])
                flag = 1
            if distance2 == min(distance1, distance2, distance3, distance4) and (not pointInSet(point, secondSet)):
                secondSet.append(point)
                secondNumbers.append(ltr[i])
                flag = 1
            if distance3 == min(distance1, distance2, distance3, distance4) and (not pointInSet(point, thirdSet)):
                thirdSet.append(point)
                thirdNumbers.append(ltr[i])
                flag = 1
            if distance4 == min(distance1, distance2, distance3, distance4) and (not pointInSet(point, fourthSet)):
                fourthSet.append(point)
                fourthNumbers.append(ltr[i])
                flag = 1
        firstCenter = calculateCentriods(firstSet)
        secondCenter = calculateCentriods(secondSet)
        thirdCenter = calculateCentriods(thirdSet)
        fourthCenter = calculateCentriods(fourthSet)


    return firstSet, firstNumbers, secondSet, secondNumbers, thirdSet, thirdNumbers, fourthSet, fourthNumbers

def calculatePurity(num1, num2, num3, num4, set1X, set2X, set3X, set4X):
    purity = 0
    twoTimes = 0
    fourTimes = 0
    sixTimes = 0
    eightTimes = 0
    for i in range(0, len(set1X)):
        if num1[i] == 2:
            twoTimes += 1
        if num1[i] == 4:
            fourTimes += 1
        if num1[i] == 6:
            sixTimes += 1
        if num1[i] == 8:
            eightTimes += 1


    purity += max(twoTimes, fourTimes, sixTimes, eightTimes) / len(set1X)

    twoTimes = 0
    fourTimes = 0
    sixTimes = 0
    eightTimes = 0
    for i in range(0, len(set2X)):
        if num2[i] == 2:
            twoTimes += 1
        if num2[i] == 4:
            fourTimes += 1
        if num2[i] == 6:
            sixTimes += 1
        if num2[i] == 8:
            eightTimes += 1

            
    purity += max(twoTimes, fourTimes, sixTimes, eightTimes) / len(set2X)



    twoTimes = 0
    fourTimes = 0
    sixTimes = 0
    eightTimes = 0
    for i in range(0, len(set3X)):
        if num3[i] == 2:
            twoTimes += 1
        if num3[i] == 4:
            fourTimes += 1
        if num3[i] == 6:
            sixTimes += 1
        if num3[i] == 8:
            eightTimes += 1

            
    purity += max(twoTimes, fourTimes, sixTimes, eightTimes) / len(set3X)


    twoTimes = 0
    fourTimes = 0
    sixTimes = 0
    eightTimes = 0
    for i in range(0, len(set4X)):
        if num4[i] == 2:
            twoTimes += 1
        if num4[i] == 4:
            fourTimes += 1
        if num4[i] == 6:
            sixTimes += 1
        if num4[i] == 8:
            eightTimes += 1

            
    purity += max(twoTimes, fourTimes, sixTimes, eightTimes) / len(set4X)
    purity /= 4
    return purity

def plot2(a, b, c, d, e, f, g):
    fig, ax = plt.subplots(1, figsize=(10, 6))
    plt.title(g)
    a = list(zip(*a))
    b = list(zip(*b))
    c = list(zip(*c))
    d = list(zip(*d))
    plt.scatter(a[0], a[1], c = 'red')
    plt.scatter(b[0], b[1], c = 'green')
    plt.scatter(c[0], c[1], c = 'blue')
    plt.scatter(d[0], d[1], c = 'yellow')
    ax.set_xlabel(e)
    ax.set_ylabel(f)

K = 4
centers, num1, num2, num3, num4 = MaximinAlgorithm(M_2, K, Ltr)
for i in range(0, len(centers)):
  print('Center {} is {}'.format(i+1, centers[i]))
set1, num1, set2, num2, set3, num3, set4, num4 = KMeans(centers, num1, num2, num3, num4, M_2, Ltr)
print('The purity of the Maximin K-means algorithm is: ', calculatePurity(num1, num2, num3, num4, set1, set2, set3, set4))
plot2(set1, set2, set3, set4, "Mean Even Rows", "Mean Odd Columns", "Clustered Brightness Features")






def PCA(X , dim):
    X_meaned = X - np.mean(X , axis = 0)
    cov_mat = np.cov(X_meaned , rowvar = False)
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    eigenvector_subset = sorted_eigenvectors[:,0:dim]
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
     
    return X_reduced


M_reduced = PCA(M, 2)
firstDimension_two = []
secondDimension_two = []
firstDimension_four = []
secondDimension_four = []
firstDimension_six = []
secondDimension_six = []
firstDimension_eight = []
secondDimension_eight = []
for i in range(0, Mi):
    if Ltr[i] == 2:
        firstDimension_two.append(M_reduced[i][0])
        secondDimension_two.append(M_reduced[i][1])
    if Ltr[i] == 4:
        firstDimension_four.append(M_reduced[i][0])
        secondDimension_four.append(M_reduced[i][1])
    if Ltr[i] == 6:
        firstDimension_six.append(M_reduced[i][0])
        secondDimension_six.append(M_reduced[i][1])
    if Ltr[i] == 8:
        firstDimension_eight.append(M_reduced[i][0])
        secondDimension_eight.append(M_reduced[i][1])

plotCluster(firstDimension_two,secondDimension_two, firstDimension_four,secondDimension_four, firstDimension_six,secondDimension_six, firstDimension_eight,secondDimension_eight, "", "", "Data After PCA algorithm")


centers1, number1, number2, number3, number4 = MaximinAlgorithm(M_reduced, K, Ltr)

set1, num1, set2, num2, set3, num3, set4, num4 = KMeans(centers1, number1, number2, number3, number4, M_reduced, Ltr)
#set1X, set1Y, num1, set2X, set2Y, num2, set3X, set3Y, num3, set4X, set4Y, num4 = MaximinAlgorithm(M_reduced, K, Ltr)
purity2 = calculatePurity(num1, num2, num3, num4, set1, set2, set3, set4)
print('The purity of the Maximin K-means after PCA algorithm is: ', purity2)
plot2(set1, set2, set3, set4, "", "", "Clustered PCA data")





M_reduced25 = PCA(M, 25)
M_reduced50 = PCA(M, 50)
M_reduced100 = PCA(M, 100)
centers, number1, number2, number3, number4 = MaximinAlgorithm(M_reduced25, K, Ltr)

set1, num1, set2, num2, set3, num3, set4, num4 = KMeans(centers, number1, number2, number3, number4, M_reduced25, Ltr)
purity25 = calculatePurity(num1, num2, num3, num4, set1, set2, set3, set4)
print('The purity of the data with V = 25 is: ',purity25)




centers2, number1, number2, number3, number4 = MaximinAlgorithm(M_reduced50, K, Ltr)
set1, num1, set2, num2, set3, num3, set4, num4 = KMeans(centers2, number1, number2, number3, number4, M_reduced50, Ltr)
purity50 = calculatePurity(num1, num2, num3, num4, set1, set2, set3, set4)
print('The purity of the data with V = 50 is: ',purity50)




centers3, num1, num2, num3, num4 = MaximinAlgorithm(M_reduced100, K, Ltr)
set1, num1, set2, num2, set3, num3, set4, num4 = KMeans(centers3, num1, num2, num3, num4, M_reduced100, Ltr)
purity100 = calculatePurity(num1, num2, num3, num4, set1, set2, set3, set4)
print('The purity of the data with V = 100  is; ',purity100)




def gaussianDistributionVectors(x, mean, standardDeviation):
    exponent = np.exp(-(np.power(x-np.array(mean), 2) / (2*np.power(np.array(standardDeviation),2))))
    result = exponent / (np.sqrt(2*np.pi)*np.array(standardDeviation))
    return np.sum(np.log(result))



# This function help me to found how many times appear a number in the training dataset
def determineTimes(matrix, ltr):
    two = four = six = eight = 0
    for i in range(0, matrix.shape[0]):
        if ltr[i] == 2:
            two += 1
        if ltr[i] == 4:
            four += 1
        if ltr[i] == 6:
            six += 1
        if ltr[i] == 8:
            eight +=1
    return two, four, six, eight



#In this function i calculate the prediction for a set of characteristics to belong to a class 2 or 4 or 6 or 8
def prediction(index, meanTwo, meanFour, meanSix, meanEight, deviationTwo, deviationFour, deviationSix, deviationEight, matrix_test, matrix_train, ltr):
    prob2 = 0 #probability to be number 2
    prob4 = 0 #probability to be number 4
    prob6 = 0 #probability to be number 6
    prob8 = 0 #probability to be number 8

    two, four, six, eight = determineTimes(matrix_train, ltr)

    prob2 += math.log(two / (two + four + six + eight))
    prob4 += math.log(four / (two + four + six + eight))
    prob6 += math.log(six / (two + four + six + eight))
    prob8 += math.log(eight / (two + four + six + eight))
   

    prob2 += gaussianDistributionVectors(matrix_test[index][:], meanTwo, deviationTwo)
    prob4 += gaussianDistributionVectors(matrix_test[index][:], meanFour, deviationFour)
    prob6 += gaussianDistributionVectors(matrix_test[index][:], meanSix, deviationSix)
    prob8 += gaussianDistributionVectors(matrix_test[index][:], meanEight, deviationEight)
    

    if prob2 == max(prob2, prob4, prob6, prob8):
        return 2
    if prob4 == max(prob2, prob4, prob6, prob8):
        return 4
    if prob6 == max(prob2, prob4, prob6, prob8):
        return 6
    if prob8 == max(prob2, prob4, prob6, prob8):
        return 8


#Calculate the accuracy of the Gaussian Naive Bayes Classifier
def percentageAccuracy(y_predictions, y_test):
    right = 0
    for i in range(0, len(y_predictions)):
        if y_predictions[i] == y_test[i]:
            right += 1
    return (right * 100) / len(y_predictions)


#Function that calculate the mean value of a column of a 2D matrix
def calculateMean(matrix, ltr, number):
    result = []
    column = 0
    totalRows = 0
    for i in range(0, matrix.shape[0]):
        if ltr[i] == number:
            totalRows += 1

    for i in range(0, matrix.shape[1]):
        for j in range(0, matrix.shape[0]):
            if ltr[j] == number:
                column += matrix[j][i]    
        result.append(column / totalRows)
        column = 0
    return result

#Function that calculate the standard deviation of the column of a 2D matrix
def calculateStandardDeviation(matrix, mean, column, number, ltr):
    result = 0
    totalRows = 0
    for i in range(0, matrix.shape[0]):
        if ltr[i] == number:
            totalRows += 1

    for i in range(0, matrix.shape[0]):
        if ltr[i] == number:
            result += pow(matrix[i][column] - mean[column], 2)
    
    return math.sqrt(result / totalRows)






if purity2 == max(purity2, purity25, purity50, purity100):
     print("For V = 2 we have the maximum purity = ", purity2)
if purity25 == max(purity2, purity25, purity50, purity100):
    print("For V = 25 we have the maximum purity = ", purity25)
if purity50 == max(purity2, purity25, purity50, purity100):
    print("For V = 50 we have the maximum purity = ", purity50)
if purity100 == max(purity2, purity25, purity50, purity100):
    print("For V = 100 we have the maximum purity = ", purity100)


"""
Pattern Recognition
Exercise 5
"""
if purity2 == max(purity2, purity25, purity50, purity100):
    M_Vmax = M_reduced
if purity25 == max(purity2, purity25, purity50, purity100):
    M_Vmax = M_reduced25
if purity50 == max(purity2, purity25, purity50, purity100):
     M_Vmax = M_reduced50
if purity100 == max(purity2, purity25, purity50, purity100):
     M_Vmax = M_reduced100


meanTwo = calculateMean(M_Vmax, Ltr, 2) # list that has the mean value of the Vmax characteristics of number 2
deviationTwo = [] # list that has the standerd deviation value of the Vmax characteristics of number 2

meanFour = calculateMean(M_Vmax, Ltr, 4) # list that has the mean value of the Vmax characteristics of number 4
deviationFour = [] # list that has the standerd deviation value of the Vmax characteristics of number 4

meanSix = calculateMean(M_Vmax, Ltr, 6) # list that has the mean value of the Vmax characteristics of number 6
deviationSix = [] # list that has the standerd deviation value of the Vmax characteristics of number 6

meanEight = calculateMean(M_Vmax, Ltr, 8) # list that has the mean value of the Vmax characteristics of number 8
deviationEight = [] # list that has the standerd deviation value of the Vmax characteristics of number 8




for i in range(0, M_Vmax.shape[1]):
    deviationTwo.append(calculateStandardDeviation(M_Vmax, meanTwo, i, 2, Ltr))
    deviationFour.append(calculateStandardDeviation(M_Vmax, meanFour, i, 4, Ltr))
    deviationSix.append(calculateStandardDeviation(M_Vmax, meanSix, i, 6, Ltr))
    deviationEight.append(calculateStandardDeviation(M_Vmax, meanEight, i, 8, Ltr))


N_reduced = PCA(N, M_Vmax.shape[1])
predictions = np.zeros(N_reduced.shape[0])

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(M_Vmax, Ltr).predict(N_reduced)
print("#### The Accuracy of the Gaussian Naive Bayes Classifier algorithm is: ", percentageAccuracy(y_pred, Lte), "####")
for i in range(0, N_reduced.shape[0]):
    predictions[i] = prediction(i, meanTwo, meanFour, meanSix, meanEight, deviationTwo, deviationFour, deviationSix, deviationEight, N_reduced, M_Vmax, Ltr)
print("#### The Accuracy of the Gaussian Naive Bayes Classifier algorithm is: ", percentageAccuracy(predictions, Lte), "####")

plt.show()



