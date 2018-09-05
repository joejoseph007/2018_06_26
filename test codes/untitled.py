import csv, random, re, sys, os, math, numpy, time, subprocess, shutil
import matplotlib.pyplot as plt 
from multiprocessing import Pool
from distutils.dir_util import copy_tree
import scipy.interpolate as si
import operator
from mpl_toolkits.mplot3d import Axes3D


a=[1,2,3,4,float('inf'),6,7,8,9,10]

b=[1,3,5,7,9]
a1=[]
for i in range(len(a)):
	if all(b[j]!=a[i] for j in range(len(b))):
		print "yes"
		a1.append(a[i])
print float('inf')
print sorted(a) 