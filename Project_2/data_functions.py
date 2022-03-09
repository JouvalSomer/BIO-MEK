#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 10:23:59 2022

@author: jacob
"""
import numpy as np


def pressure(G,pa,pv):
    n = int((np.sqrt(np.size(G)))/2)
    #A = np.zeros((n,n))
    A1 = A2 = np.zeros((n,n))

    A3 = A4 = np.zeros((n,n))
    for i in range(1,int(n/2)):
        for j in range(0,n):
            if G[i,j] != 0:
                A1[i,j] = -1/G[i,j]
                A1[i,i] += 1/G[i,j]
                
    for i in range(int(n/2), n):
        A1[i,i] = 1
        
    for i in range(0,n):
        A3[i,i] = 1/G[n+i,i]
    A1[0,0] = 1
    A1[n-1, n-1] = 1
    A = np.bmat([[A1, A3], [A4, A2]])
    b = np.zeros(2*n)
    b[0] = pa
    b[int(n/2):] = pv
    p = np.linalg.solve(A,b)
    return p,A,b



def flow_rates(G,p,R):
    n = np.size(p)
    Q = np.zeros(n)
    for i in range(0,n+1):
        for j in range(0,n+1):
            if G[i,j] != 0:
                Q[i,j] = np.abs(p[i]- p[j])/(R[i,j]*G[i,j])       
    return Q




def average_speeds(G,Q,r):
    n = np.size(G)
    v = np.zeros(n)
    for i in range(0,n+1):
        for j in range(0,n+1):
            if G[i,j] != 0:
                v[i,j] = Q[i,j]/(np.pi*r[i,j]**2)
    return v