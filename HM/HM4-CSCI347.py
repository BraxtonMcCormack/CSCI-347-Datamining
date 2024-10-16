import math as m
import matplotlib.pyplot as plt
import numpy as np

def q1(dataset):
    print("Question 1 : Scatter plot of Dataset")
    x = [i[0] for i in dataset]
    y = [i[1] for i in dataset]
    plt.scatter(x,y)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Scatterplot")
    plt.show()
def q2(dataset, A):
    print("Question 2 : Linear Transformation of matrix A")
    result = []
    for num, i in enumerate(dataset, 1):
        v_i = np.array([[i[0],i[1]]]).reshape((2,1))
        ans = np.dot(A, v_i)
        print("v_"+str(num)+": ")
        print(ans)
        result.append([ans[0],ans[1]])
    return result        
    
def q3(dataset, transformedDS):
    print("Question 3 : Scatter plot of Dataset against Transformed Dataset")
    #original
    x = [i[0] for i in dataset]
    y = [i[1] for i in dataset]
    #transformed
    xt = [i[0] for i in transformedDS]
    yt = [i[1] for i in transformedDS]

    plt.scatter(x,y, color = "b",label = "Original")
    plt.scatter(xt,yt, color = "r",label = "Transformed")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("OG Data vs Transformed Data")
    plt.legend(loc = "upper left")
    plt.show()

def q4(dataset):
    print("Question 4 : Multi-variance mean of Dataset")
    col1 = sum([i[0] for i in dataset])/len(dataset)
    col2 = sum([i[1] for i in dataset])/len(dataset)
    print(col1,col2)
    return [col1,col2]

def q5(dataset, multiVarMean):
    print("Question 5 : Mean Centered data of Dataset")
    result = []
    for i in dataset:
        x1 = i[0]-multiVarMean[0]
        x2 = i[1]-multiVarMean[1]
        result.append([x1,x2])
    print(result)
    return result

def q6(dataset, mcds):
    print("Question 6 : Scatter plot of Dataset against Mean-Centered Dataset")
    #original
    x = [i[0] for i in dataset]
    y = [i[1] for i in dataset]
    #mean-Centered
    xt = [i[0] for i in mcds]
    yt = [i[1] for i in mcds]

    plt.scatter(x,y, color = "b",label = "Original")
    plt.scatter(xt,yt, color = "r",label = "mean-Centered")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Mean Centered Scatterplot")
    plt.legend(loc = "upper left")
    plt.show()
    
def q7(dataset):
    print("Question 7 : Covariance of Dataset")
    ds = np.array(dataset)
    dsCov = np.cov(dataset, rowvar=False)
    #rowvar=false uses columns for calculations instead of rows (default)
    print(dsCov)
    return dsCov

def q8(centeredDataset):
    print("Question 8 : Covariance of Mean-Centered Dataset")
    cds = np.array(centeredDataset)
    cdsCov = np.cov(centeredDataset, rowvar=False)
    #rowvar=false uses columns for calculations instead of rows (default)
    print(cdsCov)
    return cdsCov
def q9(dataset):
    print("Question 9 : Covariance of Normalized Dataset")
    ds = np.array(dataset)
    normalizedDSCov = np.cov(dataset, rowvar=False, bias=True)
    #rowvar=false uses columns for calculations instead of rows (default)
    #bias=True changes denominator of calculation from N-1 to N, which normalizes
    print(normalizedDSCov)
    return normalizedDSCov

def main():
    A = np.array([[m.sqrt(3)/2, -.5],[.5, m.sqrt(3)/2]])
    dataset = [[1,1.5],[1,2],[3,4],[-1,-1],[-1, 1], [1, -2], [2,2],[2,3]]
    q1(dataset)
    tDS = q2(dataset, A)
    q3(dataset,tDS)
    multiVarMean = q4(dataset)
    meanCenteredDS = q5(dataset,multiVarMean)
    q6(dataset, meanCenteredDS)
    q7(dataset)
    q8(meanCenteredDS)
    q9(dataset)

main()
