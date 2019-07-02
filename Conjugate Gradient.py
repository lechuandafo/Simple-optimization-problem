# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 13:39:12 2018

@author: YLC
"""
import numpy as np
x = np.array([0,0,0,0]).T #.T表示转置，下同
H = np.array([[158,20,90,101],[20,36,46,61],[90,46,306,156],[101,61,156,245]])
g = np.array([8,-5,1,6]).T
def grad(H,x,g): #梯度计算公式，由原方程求导得到
    return np.dot(H,x)-g
eta = grad(H,x,g) #梯度
d = -eta #梯度方向
i = 1 #迭代次数
while(np.linalg.norm(eta,ord=2) > 1e-10):
    alpha = -np.dot(eta.T,d)/np.dot(np.dot(d.T,H),d)
    x = x + np.dot(alpha,d)
    eta = grad(H,x,g)
    d = -eta + np.dot(np.dot(np.dot(eta.T,H),d)/np.dot(np.dot(d.T,H),d),d)
    #print("========================================")
    #print("迭代第"+str(i)+"次||eta||的值为:",np.linalg.norm(eta,ord=2))    
    #print("迭代第"+str(i)+"次alpha的值为:\n",alpha)
    #print("迭代第"+str(i)+"次eta的值为:\n",eta)
    #print("迭代第"+str(i)+"次d的值为:\n",d)    
    print("迭代第"+str(i)+"次x的值为:\n",x)
    i = i + 1