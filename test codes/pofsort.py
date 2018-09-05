import csv, random, re, sys, os, math, numpy, time, subprocess, shutil
import matplotlib.pyplot as plt 
from multiprocessing import Pool
from distutils.dir_util import copy_tree
import scipy.interpolate as si
import operator
from mpl_toolkits.mplot3d import Axes3D

def randf(x,y):
    a=1000
    return float(random.randrange(x*a,y*a))/a

class Pop(object):
    
    def __init__(self, R=[],A=[], Cost=[]) :
        self.R = R
        self.A = A
        self.Cost = Cost

def Cost1(x,y):
	Cost1=4*x**2+4*y**2
	#Cost1=0.5*(x**2+y**2)+math.sin(math.radians(x**2+y**2))
	#Cost1=randf(0,10)
	return Cost1



def Cost2(x,y):
	Cost2=(x-5)**2+(y-5)**2
	#Cost2=(3*x-2*y+4)**2/8+(x-y+1)**2/27+15
	#Cost2=randf(15,40)
	return Cost2



def Cost3(x,y):
	#Cost2=(x-5)**2+(y-5)**2
	Cost3=1/(x**2+y**2+1)-1.1*math.exp(-(x**2+y**2))
	#Cost3=randf(0,10)
	return Cost3

def insertionSortRecursive(arr,n):
    # base case
    if n<=1:
        return
     
    # Sort first n-1 elements
    insertionSortRecursive(arr,n-1)
    last = arr[n-1]
    j = n-2
    while (j>=0 and arr[j]>last):
        arr[j+1] = arr[j]
        j = j-1
 
    arr[j+1]=last



def POFsort(p):

    def Arr(X):
    	return X[1]
    def Arr2(X):
    	return X[2]
    arr=[]
    for j in range(len(p[0].Cost)):
    	arr.append([(i,p[i].Cost[0]) for i in range(len(p))])
    	arr[j].sort(key=Arr)
    
    #arr2=[(i,p[i].Cost[1]) for i in range(len(p))]
    #arr3=[(i,p[i].Cost[2]) for i in range(len(p))]
    
    arr_main=[]
    for x in range(len(p)):
    	#print x,'\t' ,arr1[x][0],arr2[x][0],arr3[x][0],'\t\t',arr1[x][0]+arr2[x][0]+arr3[x][0]
    	
    	#arr_main.append((arr[0][x][0],arr[1][x][0],arr[2][x][0],arr[0][x][0]+arr[1][x][0]+arr[2][x][0],x))
    	arr_main.append((arr[0][x][0],arr[1][x][0],arr[0][x][0]+arr[1][x][0],x))
    
    arr_main.sort(key=Arr2) 
    #for x in range(len(arr_main)):
    	#print x,'\t' ,arr_main[x]
    #print arr_main
    a=[]
    for i in range(len(p)):
    	x=arr_main[i][3]

    	a.append(Pop(p[x].R,p[x].A,[p[x].Cost[0],p[x].Cost[1]]))
    	#print i, a[i].Cost,[arr_main[i][0],arr_main[i][1],arr_main[i][2]],arr_main[i][3]
    
    b=[]
    for i1 in range(len(p)):
    	#print i1
    	for i in range(len(p)):
	    	b1=[]
	    	for j in range(len(p[i].Cost)):
	    		if i1==arr_main[i][j]:
	    			b1=[arr_main[i]]
	    	
	    	if b1!=[]:
		    	#print "b1",b1
		    	if len(b)==0:
		    		b=b+b1
		    		#print 'b ',b,len(b)
		    	elif all(b[j1][3]!=b1[0][3] for j1 in range(len(b))):
		    	   	b+=b1
		    	   	#print 'b ',b,len(b)
	z=[]
    for i in range(len(p)):
    	x=b[i][3]

    	z.append(Pop(p[x].R,p[x].A,[p[x].Cost[0],p[x].Cost[1]]))
    	print i, z[i].Cost,[b[i][0],b[i][1]],b[i][2]
    

    return z


#for i in range(len(p)):
#	print i,"\t\t",p[i].R,p[i].A,"\t\t",p[i].Cost1,p[i].Cost2,p[i].Cost2

#plt.show()

#print Random

#insertionSortRecursive(Random[1],len(Random[1]))
#print Random


'''

cost=[[]]
cost=[[[]] for x in xrange(1,10)]
	
cost[1][0].append(1)
cost[1][0].append(2)
cost[1].append(1)

print cost

for x in xrange(1,10):
	for x1 in xrange(10,100):
		cost[x].append(randf(x,x1))
		print cost[x][x1]
'''



'''
for j in range(50):

	p1=[]
	for i in range(100):
		R=randf(Rmin,Rmax)
		A=randf(Amin,Amax)
		p1.append(Pop(R,A,[Cost1(R,A),Cost2(R,A)]))
	

	
	p=p+p1
	p=POFsort(p)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	p2=[]
	if len(p)>nPop:
	    for i in range(nPop):
	        p2.append(p[i])
	    p=[]
	    p=p2    
	print "Sorted"
	
	for i in range(len(p)):
		#print i,"\t\t",p[i].R,p[i].A,"\t\t",p[i].Cost
		ax.scatter(p[i].Cost[0],p[i].Cost[1],s=5,c='red')
	plt.xlim(0,150)
	plt.ylim(0,150)
	plt.savefig('Fig%i.svg'%j)
'''	

def constraint1(x,y):
	a=(x-5)**2+y**2
	return a

def constraint2(x,y):
	a=(x-8)**2+(y+3)**2
	return a




def POFsort1(p):

    def Arr(X):
    	return X[1]
    def Arr2(X):
    	return X[2]
    arr=[]
    for j in range(len(p[0].Cost)):
    	arr.append([(i,p[i].Cost[0]) for i in range(len(p))])
    	arr[j].sort(key=Arr)
    
    #arr2=[(i,p[i].Cost[1]) for i in range(len(p))]
    #arr3=[(i,p[i].Cost[2]) for i in range(len(p))]
    
    arr_main=[]
    for x in range(len(p)):
    	#print x,'\t' ,arr1[x][0],arr2[x][0],arr3[x][0],'\t\t',arr1[x][0]+arr2[x][0]+arr3[x][0]
    	
    	#arr_main.append((arr[0][x][0],arr[1][x][0],arr[2][x][0],arr[0][x][0]+arr[1][x][0]+arr[2][x][0],x))
    	arr_main.append((arr[0][x][0],arr[1][x][0],arr[0][x][0]+arr[1][x][0],x))
    
    arr_main.sort(key=Arr2) 
    #for x in range(len(arr_main)):
    	#print x,'\t' ,arr_main[x]
    #print arr_main
    a=[]
    for i in range(len(p)):
    	x=arr_main[i][3]

    	a.append(Pop(p[x].R,p[x].A,[p[x].Cost[0],p[x].Cost[1]]))
    	print i, a[i].Cost,[arr_main[i][0],arr_main[i][1],arr_main[i][2]]
    '''
    b=[]
    for i1 in range(len(p)):
    	#print i1
    	for i in range(len(p)):
	    	b1=[]
	    	for j in range(len(p[i].Cost)):
	    		if i1==arr_main[i][j]:
	    			b1=[arr_main[i]]
	    	
	    	if b1!=[]:
		    	#print "b1",b1
		    	if len(b)==0:
		    		b=b+b1
		    		#print 'b ',b,len(b)
		    	elif all(b[j1][3]!=b1[0][3] for j1 in range(len(b))):
		    	   	b+=b1
		    	   	#print 'b ',b,len(b)
	z=[]
    for i in range(len(p)):
    	x=b[i][3]

    	z.append(Pop(p[x].R,p[x].A,[p[x].Cost[0],p[x].Cost[1]]))
    	print i, z[i].Cost,[b[i][0],b[i][1]],b[i][2]
    '''

    return a


nPop=100
p=[]

Rmax=5
Rmin=0
Amax=3
Amin=0

p1=[]
i=0
while i <(1000):
	R=randf(Rmin,Rmax)
	A=randf(Amin,Amax)
	if (constraint1(R,A)<=25) and (constraint2(R,A)>=7.7):
		
		p1.append(Pop(R,A,[Cost1(R,A),Cost2(R,A)]))
		i+=1

	if not(constraint1(R,A)<=25):
		print i,'1 Yes'
	if not(constraint2(R,A)>=7.7):
		print i,"2 Yes"
	
	else:
		continue

fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(len(p1)):
	print i,"\t\t",[p1[i].R,p1[i].A],"\t\t",'[%.1f,\t%.1f]'%(p1[i].Cost[0],p1[i].Cost[1])
	ax.scatter(p1[i].Cost[0],p1[i].Cost[1],s=1,c='red')

#plt.show()
p1=POFsort1(p1)


p2=[]
if len(p1)>nPop:
    for i in range(nPop):
        p2.append(p1[i])
    p1=[]
    p1=p2    
print "Sorted"




for i in range(len(p1)):
	print i,"\t\t",[p1[i].R,p1[i].A],"\t\t",'[%.1f,\t%.1f]'%(p1[i].Cost[0],p1[i].Cost[1])
	ax.scatter(p1[i].Cost[0],p1[i].Cost[1],s=1,c='blue')

plt.xlim(0,150)
plt.ylim(0,50)


plt.savefig('Fig.svg')
