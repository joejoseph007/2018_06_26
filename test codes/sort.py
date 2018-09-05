import csv, random, re, sys, os, math, numpy, time, subprocess, shutil
import matplotlib.pyplot as plt 
from multiprocessing import Pool
from distutils.dir_util import copy_tree
import scipy.interpolate as si
import operator
from mpl_toolkits.mplot3d import Axes3D


def randf(x,y):
    a=10
    return float(random.randrange(x*a,y*a))/a

def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

#Function to sort by values
def sort_by_values(list1, value):
    values=value
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
            #del values[index_of(min(values),values)]
        values[index_of(min(values),values)]=float('inf')
    return sorted_list

#To make fronts
def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front


def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance

def Cost1(x,y):
    Cost1=(4*x**2+4*y**2)
    #Cost1=0.5*(x**2+y**2)+math.sin(math.radians(x**2+y**2))
    #Cost1=randf(0,10)
    return -Cost1



def Cost2(x,y):
    Cost2=(x-5)**2+(y-5)**2
    #Cost2=(3*x-2*y+4)**2/8+(x-y+1)**2/27+15
    #Cost2=randf(15,40)
    return -Cost2


def constraint1(x,y):
    a=(x-5)**2+y**2
    return a

def constraint2(x,y):
    a=(x-8)**2+(y+3)**2
    return a


max1=5
max2=3
pop_size=10
z1=[];z2=[]
qwe=0
while qwe<50:
    i=0
    a1=[];a2=[]
    
    while i<500:
        x1=(randf(0,max1))
        x2=(randf(0,max2))
        if (constraint1(x1,x2)<=25) and (constraint2(x1,x2)>=7.7):
            z1.append(x1)
            z2.append(x2)
            a1.append(Cost1(z1[i],z2[i]))
            a2.append(Cost2(z1[i],z2[i]))
            i+=1    
    #print a1

    a1=[round(a1[i],1)for i in range(len(a1))]

    a2=[round(a2[i],1)for i in range(len(a2))]
    #plt.ylim(-10,140)
    F=fast_non_dominated_sort(a1,a2)
    #'''
    CD=[]
    for i in range(len(F)):
        CD.append(crowding_distance(a1,a2,F[i][:]))
        
    #'''

    new_solution= []

    for i in range(0,len(F)):
        F_NS = [index_of(F[i][j],F[i] ) for j in range(0,len(F[i]))]
        F1 = sort_by_values(F_NS[:], CD[i][:])
        front = [F[i][F1[j]] for j in range(0,len(F[i]))]
        front.reverse()
        for value in front:
            new_solution.append(value)
            if(len(new_solution)==pop_size):
                break
        if (len(new_solution) == pop_size):
            break

    Z_1 = [z1[i] for i in new_solution]
    Z_2 = [z2[i] for i in new_solution]
    z1=Z_1
    z2=Z_2
    '''
    for i in range(len(CD)):
        print [round(CD[i][j],2) for j in range(len(CD[i]))]
        print F[i]
    '''



    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlim(-140,10)
    plt.ylim(-80,10)
    n=[z for z in range(len(Z_1))]

    for j in range(len(Z_1)):
        ax.scatter(Cost1(Z_1[j],Z_2[j]),Cost2(Z_1[j],Z_2[j]),s=20,c='Red',alpha=1,edgecolor=None)
        
        #print F[i][j]
    ax.scatter(a1,a2,s=0.5,c='black')
    for j, txt in enumerate(n):
        ax.annotate(txt,(Cost1(Z_1[j],Z_2[j]),Cost2(Z_1[j],Z_2[j])))

    plt.savefig('Pics/Final%i.svg'%qwe)
    plt.close()
    qwe+=1
    

'''
for i in range(len(F)):
    c=[randf(0,1),randf(0,1),randf(0,1),randf(0,1)]
    #print c
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlim(-140,10)
    plt.ylim(-80,10)

    for j in range(len(F[i])):
        ax.scatter(a1[F[i][j]],a2[F[i][j]],s=20,c=c,alpha=1,edgecolor=None)
        #print F[i][j]
    ax.scatter(a1,a2,s=0.5,c='black')
    plt.savefig('Pics/%i.svg'%i)
    plt.close()
    #print F[i]
'''
'''
for j in range(len(F[0])):
    ax.scatter(a1[F[0][j]],a2[F[0][j]],s=20,c='red')


for j in range(len(F[len(F)-2])):
    ax.scatter(a1[F[len(F)-2][j]],a2[F[len(F)-2][j]],s=20,c='blue')


for j in range(len(F[len(F)-1])):
    ax.scatter(a1[F[len(F)-1][j]],a2[F[len(F)-1][j]],s=50,c='yellow')
'''            