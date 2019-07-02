# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 12:09:23 2018

@author: YLC
"""
import numpy as np #用于矩阵运算
import matplotlib.pyplot as plt #用于画图
import math #用于指数运算
h = np.array([0.5,0.75,1,1.25,1.5,1.75,1.75,2,2.25,2.5,2.75,3,3.25,3.5,4,4.25,4.5,4.75,5,5.50]) #学习小时数
Y = np.array([0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1]) #是否通过标记
matrix = np.vstack((h,Y)) #原始数据
print("原始数据如下：")
for i in range(0,len(h)):
    print("h="+str(matrix[0][i])+"\t p="+str(matrix[1][i]))
n = 0 #迭代次数
arr_len = len(h) #数据量大小
theta0 = 0.5 #第一个要学习的参数
theta1 = 0.5 #第二个要学习的参数
theta = np.array([[theta0],[theta1]])#将两个参数放到参数矩阵，不保留参数的历史记录，只存当前参数值
epsilon = 0.001 # 定义阈值
print("学习前,初始参数:theta0="+str(theta[0])+", theta1="+str(theta[1]))
def p(x,theta): #定义Logistic回归函数
    return 1/(1+math.exp(-theta[0]-theta[1]*x))
def grad(x,y,theta): #定义梯度
    sum1 = sum2 = 0
    for i in range(0,arr_len):
        sum1 = sum1 + y[i]-p(x[i],theta) #相当于theta0参数沿当前梯度的变化量
        sum2 = sum2 + (y[i]-p(x[i],theta))*x[i] #相当于theta1参数沿当前梯度的变化量
    return np.array([[sum1],[sum2]])
#梯度的二范数大于或等于阈值或小于迭代次数则进入循环，小于跳出循环
while (np.linalg.norm(grad(h,Y,theta),ord=2)>=epsilon):#梯度的范数可理解为二维中y=kx+b的斜率，高维中的下降最快的坡度
    if(n >= 1000) : break #迭代超过阈值
    alpha = 0.08 #初始步长通常在0.01~0.3之间取值
    thetaT = theta #临时存放参数
    n = n + 1 #迭代次数+1
    theta = thetaT + alpha * grad(h,Y,theta) #更新参数 
    print("误差(梯度的二范数)为：",np.linalg.norm(grad(h,Y,theta),ord=2))
    #print("第"+str(n)+"次迭代，梯度的二范数为："+str(np.linalg.norm(grad(h,Y,theta),ord=2)))
print("迭代"+str(n)+"次后结束，学习后,theta0="+str(theta[0])+",theta1="+str(theta[1]))
print("模型为：y = 1/(1+exp("+str(float(-theta[0]))+str(float(-theta[1]))+"*x))")
pp = np.array([])
for i in range(0,len(h)):
    pp = np.append(pp,p(h[i],theta)) 
pmatrix = np.vstack((h,pp))
print("将学习小时数进行输入，获得课程通过概率为：")
for i in range(0,len(h)):
    print("h="+str(pmatrix[0][i])+"\t p="+str(pmatrix[1][i]))
plt.scatter(h,pp,c = 'r',marker = 'o') 
print("\n其散点图为：")
plt.show()
''' 整个模型的图像 '''
print("整个模型的图像")
x=np.linspace(0,5,1000)  #这个表示在-5到5之间生成1000个x值
y=[1/(1+np.exp(-theta[0]-theta[1]*i)) for i in x]  #对上述生成的1000个数循环用sigmoid公式求对应的y
plt.plot(x,y)  #用上述生成的1000个xy值对生成1000个点
plt.show()  #绘制图像 
