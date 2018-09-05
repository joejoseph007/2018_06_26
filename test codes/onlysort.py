import csv, random, re, sys, os, math, numpy, time, subprocess, shutil
import matplotlib.pyplot as plt 
from multiprocessing import Pool
from distutils.dir_util import copy_tree
import scipy.interpolate as si

p=[]
r=read_integers('../Generation',"i")

def read_integers(filename,t):
    with open(filename) as f:
        if t=='f':
            z=[float(x) for x in f]
            if len(z)==1:
                return z[0]
            else:
                return z
        if t=='i':
            z=[int(x) for x in f]
            if len(z)==1:
                return z[0]
            else:
                return z
        if t=='fxy':
            x=[];y=[]
            for i in f:
                row = i.split()
                x.append(float(row[0]))
                y.append(float(row[1]))

            return x,y
def randf(x,y):
    a=10
    return float(random.randrange(x*a,y*a))/a
def myRound(x,base,prec=2):
    
    return round(base * round(float(x)/base),prec)
def cartesian(R,A):
    coorArrX=[];coorArrY=[]
    for i in range(len(R)):
        coorArrX.append(R[i]*math.cos(math.radians(A[i])))
        coorArrY.append(R[i]*math.sin(math.radians(A[i])))
    return coorArrX,coorArrY
def polar(X,Y):
    R=[];A=[]
    for i in range(len(X)):
        R.append((X[i]**2+Y[i]**2)**0.5)
        A.append(math.degrees(math.atan(float(Y[i])/X[i])))
    return R,A
def bspline(cv, n=100, degree=2, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
                  False - Curve is open
    """

    # If periodic, extend the point array by count+degree+1
    cv = numpy.asarray(cv)
    count = len(cv)
    if periodic:
        factor, fraction = divmod(count+degree+1, count)
        cv = numpy.concatenate((cv,) * factor + (cv[:fraction],))
        count = len(cv)
        degree = numpy.clip(degree,1,degree)
    # If opened, prevent degree from exceeding count-1
    else:
        degree = numpy.clip(degree,1,count-1)
    # Calculate knot vector
    kv = None
    if periodic:
        kv = numpy.arange(0-degree,count+degree+degree-1,dtype='int')
    else:
        kv = numpy.array([0]*degree + range(count-degree+1) + [count-degree]*degree,dtype='int')
    # Calculate query range
    u = numpy.linspace(periodic,(count-degree),n)
    # Calculate result
    arange = numpy.arange(len(u))
    points = numpy.zeros((len(u),cv.shape[1]))
    for i in xrange(cv.shape[1]):
        points[arange,i] = si.splev(u, (kv,cv[:,i],degree))
    return points
def graph(coorArrX,coorArrY,r,I,f='No'):
    from Constants import *
    #Rmax,Rmin,Amax,Amin,R_Lc,A_Lc,per,l,t,Type=Constants(["Rmax","Rmin","Amax","Amin","R_Lc","A_Lc","per",'l','t',"Type"])
    n=len(coorArrX)
    numSteps = 500    
    XY=bspline(numpy.array(zip(coorArrX,coorArrY)),numSteps,2)
    #print XY
    x1=[XY[i][0] for i in range(len(XY))]
    y1=[XY[i][1] for i in range(len(XY))]
    
    #print R,A


    #print x,y
    plt.xlim(-100,500)  
    plt.ylim(-300,300) 
    plt.scatter(coorArrX,coorArrY,s=2)
    plt.scatter(coorArrX[0],coorArrY[0],c='k',s=2)
    plt.scatter(coorArrX[n-1],coorArrY[n-1],c='k',s=2)
    #n=["%s" %i for i in range(len(coorArrX))]
    #for i, txt in enumerate(n):
    #    plt.annotate(txt, (coorArrX[i],coorArrY[i]),size="10")
    
    x=[];y=[]
    res=1000
    R=[Rmin for i in range(res+1)]
    A=[(Amax-Amin)*z/res+Amin for z in range(res+1)]

    
    for i in range(len(R)):
        x.append(R[i]*math.cos(math.radians(A[i])))
        y.append(R[i]*math.sin(math.radians(A[i])))
    plt.plot(x,y,c='r', linewidth=0.5)
    
    x=[];y=[]
    R=[Rmax for i in range(res+1)]
    A=[(Amax-Amin)*i/res+Amin for i in range(res+1)]
    for i in range(len(R)):
        x.append(R[i]*math.cos(math.radians(A[i])))
        y.append(R[i]*math.sin(math.radians(A[i])))
    plt.plot(x,y,c='r', linewidth=0.5)
    
    x=[];y=[]
    R=[(Rmax-Rmin)*i/res+Rmin for i in range(res+1)]
    A=[Amin for i in range(res+1)]
    for i in range(len(R)):
        x.append(R[i]*math.cos(math.radians(A[i])))
        y.append(R[i]*math.sin(math.radians(A[i])))
    plt.plot(x,y,c='r', linewidth=0.5)

    x=[];y=[]
    R=[(Rmax-Rmin)*i/res+Rmin for i in range(res+1)]
    A=[Amax for i in range(res+1)]
    for i in range(len(R)):
        x.append(R[i]*math.cos(math.radians(A[i])))
        y.append(R[i]*math.sin(math.radians(A[i])))
    plt.plot(x,y,c='r', linewidth=0.5)
        

    plt.plot(x1,y1,c='k', linewidth=1)
    if f=='No':
        thefile = open('../Results/Generation_0/Specie_%i/Points'%I, 'a+')
        for i in range(len(x1)):
            
            thefile.write("%.6f %.6f\n" %(x1[i],y1[i]))#randf(random.randrange(random.randrange(-150,-145),-70),0))
            
        thefile.close()
        
        plt.savefig('../Results/Generation_%i/Fig%i.svg'%(r,I))
        plt.close()
    elif f=='Yes':
        plt.savefig('../GA/Figure%i.svg'%I)
        plt.close()

class Pop(object):
    def __init__(self, R=[],A=[], Cost=[]) :
        self.R = R
        self.A = A
        self.Cost = Cost
    

def Cost(e):
    copy_tree("../CFD", "../Results/Generation_%d/Specie_%d/CFD" %(r,e))
    os.chdir("../Results/Generation_%d/Specie_%d/CFD" %(r,e))
    #subprocess.call(['./All'])
    os.chdir("../")

    Cost1=randf(-300,300)
    Cost2=randf(-30,30)
    thefile=open('Temperature',"w")
    thefile.write("%f" %(Cost1))
    thefile=open('Area',"w")
    thefile.write("%f" %(Cost2))
    thefile.close()
    thefile.close()

    os.chdir("../../../GA")
    shutil.rmtree("../Results/Generation_%d/Specie_%d/CFD" %(r,e))    
    Cost[0]=read_integers("../Results/Generation_%d/Specie_%d/Temperature" %(r,e),"f")
    #Cost2=read_integers("../Results/Generation_%d/Specie_%d/Area" %(r,e),"f")
    return Cost
    
def Gen1(nPop0):
    def insert(coorArrX,coorArrY,R,A):
        #n+=1
        coorArrX.append(R*math.cos(math.radians(A)))
        coorArrY.append(R*math.sin(math.radians(A)))


    from Constants import Rmax,Rmin, Angle
    coorArrX = []
    coorArrY = []
    R=randf(Rmin,Rmax)
    A=0
    insert(coorArrX,coorArrY,R,A)
    for i in range(180/Angle):
        R=randf(Rmin,Rmax)
        A=(i+randf(0,1))*Angle
        #print R,(A-i*Angle)
        insert(coorArrX,coorArrY,R,A)
    R=randf(Rmin,Rmax)
    A=180
    insert(coorArrX,coorArrY,R,A)
    '''
    thefile = open('../Genes', 'w')
    for i in range(len(coorArrX)):
        thefile.write("%.2f %.2f\n" %(coorArrX[i],coorArrY[i]))
    thefile.close()
    
    '''
    n=len(coorArrX)
    os.makedirs('../Results/Generation_0/Specie_%i'%I)
    thefile = open('../Results/Generation_0/Specie_%i/Genes'%I, 'a+')
    [thefile.write("%.2f %.2f\n" %(coorArrX[i],coorArrY[i])) for i in range(len(coorArrX))]#randf(random.randrange(random.randrange(-150,-145),-70),0))
    thefile.close()

    def Gen1_run(nPop0):
    
        if(not os.path.isdir("../Results")):
            os.mkdir('../Results')
        else:
            shutil.rmtree('../Results')
            os.mkdir('../Results')
        
        #os.makedirs('../Results/Generation_0/Specie_%i'%(1))
        y = Pool()
        result = y.map(run,range(nPop0))
        y.close()
        y.join() 
    
    Gen1_run(nPop0)
    #return read_integers("../Results/Generation_%d/Specie_%d/Pitch" %(r,e))

def Constraints(R,A,i):
    from Constants import *
    
    #Can add function here
    R=max(R,Lim[i][0][1])
    R=min(R,Lim[i][0][0])
    R=myRound(R,R_Lc)
    A=max(A,Lim[i][1][1])
    A=min(A,Lim[i][1][0])
    A=myRound(A,A_Lc)
    
    return R,A
def Seed(W,Wmax,Wmin,Sigma):
    W=W+(Wmax-Wmin)*Sigma*randf(-1,1)
    return W
def new(R,A,sigma):
    
    R,A=polar(R,A)
    from Constants import *
    for i in range(len(R)):
        R[i]=Seed(R[i],Lim[i][0][0],Lim[i][0][1],sigma)
        A[i]=Seed(A[i],Lim[i][1][0],Lim[i][1][1],sigma)
        R[i],A[i]=Constraints(R[i],A[i],i)    
    return cartesian(R,A)

def index_of(a,list1):
    for i in range(0,len(list1)):
        if list1[i] == a:
            return i
    return -1

#Function to sort by values
def sort_by_values(list1, F):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(F),F) in list1:
            sorted_list.append(index_of(min(F),F))
        F[index_of(min(F),F)] = float('inf')
    return sorted_list

def fast_non_dominated_sort(F1, F2):
    S=[[] for i in range(0,len(F1))]
    front = [[]]
    n=[0 for i in range(0,len(F1))]
    rank = [0 for i in range(0, len(F1))]

    for p in range(0,len(F1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(F1)):
            if (F1[p] > F1[q] and F2[p] > F2[q]) or (F1[p] >= F1[q] and F2[p] > F2[q]) or (F1[p] > F1[q] and F2[p] >= F2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (F1[q] > F1[p] and F2[q] > F2[p]) or (F1[q] >= F1[p] and F2[q] > F2[p]) or (F1[q] > F1[p] and F2[q] >= F2[p]):
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
def crowding_distance(F1, F2, front):
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, F1[:])
    sorted2 = sort_by_values(front, F2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (F1[sorted1[k+1]] - F2[sorted1[k-1]])/(max(F1)-min(F1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (F1[sorted2[k+1]] - F2[sorted2[k-1]])/(max(F2)-min(F2))
    return distance



if __name__ == "__main__":
    t2=time.time()
    
    global g,r,p
    from Constants import *
    if r==0:
        t1=time.time()
        Gen1(nPop0)
        x=[];y=[]
        for i in range(nPop0):
            x1,y1=read_integers('../Results/Generation_0/Specie_%d/Genes' %i, "fxy")
            x.append(x1) #float(random.randrange((VarMin*a),(VarMax*a)))/a)
            y.append(y1)
        
        ypool= Pool(nPop0)
        result = ypool.map(Cost,range(nPop0))
        ypool.close()
        ypool.join()
        p=[(Pop(x[i],y[i],result[i][0],result[i][1]))for i in range(nPop0)]
        
        for i in range(len(p)):
            print i,p[i].Cost1,p[i].Cost2
        
        thefile=open('../Results/Generation_%d/Costs'%(r),"w") 
        [thefile.write("%f\t%f\n" %(p[i].Cost1,p[i].Cost2)) for i in range(len(p))]
        thefile.close()
        
        print "\nGeneration=",r,"\nNet Time=",time.time()-t1
        r+=1
        #Write_Iteration()


    #IWO Main Loop
    
    BestCosts=[]
    while r<(MaxIt+1):
        t1=time.time()
        #Popread()
        
        #Update Standard Deviation
        sigma = (((MaxIt - float(r))/(MaxIt - 1))**Exponent )* (sigma_initial - sigma_final) + sigma_final;
        
        #Get Best and Worst Cost Values
        Costs=[]
        Costs=[(p[t].Cost1) for t in range(len(p))]

        BestCost = min(Costs);
        
        WorstCost = max(Costs);
        
        #Initialize Offsprings Population
        
        pn=[]
        X=[]
        Y=[]
        #Reproduction
        g=0

        def reproduction(i):            
            global g
            ratio = (p[i].Cost1 - WorstCost)/(BestCost - WorstCost)
            S = int(Smin + (Smax - Smin)*ratio)
            if S>0:
                for j in range(S):
                    
                    #print r,S,j,g
                    #Initialize Offspring
                    if(not os.path.isdir("../Results/Generation_%i/Specie_%i" %(r,g))):
                        os.makedirs("../Results/Generation_%i/Specie_%i/" %(r,g))
                    thefile = open('../Results/Generation_%i/Specie_%i/Genes' %(r,g), 'w')
                    #import Graph
                    
                    #Generate Random Location
                    #print "\nSpecie%i,%i"%(i,j)

                    X1,Y1=new(p[i].R,p[i].A,sigma)
                    [thefile.write("%.1f\t%.1f\n" %(X1[t],Y1[t])) for t in range(len(X1))]
                    '''
                    while t<nVar: # in range(nVar):
                        a=(p[i].Genes[t]+sigma*randf(-1,1))  
                        a=max(a,VarMin)
                        a=min(a,VarMax)
                        A.append(round(a,2))
                        t+=1
                    '''
                    #print A            
                    X.append(X1)
                    Y.append(Y1)
                    #Graph.Graph(A,r,g)
                    thefile.close()
                    g+=1

        [reproduction(x) for x in range(len(p))]


        y = Pool(g)
        result = y.map(Cost,range(g))
        y.close()
        y.join()    
        
        #print "\n",g,result
        #print Z[j]
        pn=[(Pop(X[j],Y[j],result[j][0],result[j][1])) for j in range(g)]
        #Add Offpsring to the New Population
        #for i in range(len(pn)):
        #    print r,g,i,"Genes",pn[i].Cost
        p=p+pn
        #Merge Populations
        #Sort Population
        p=POFsort(p)
        if len(p)>nPop:
            p2=[(p[i])for i in range(nPop)]
            p=[]
            p=p2    
        subprocess.call(['clear'])
        print "\nGeneration=",r,g,"\nNet Time=",time.time()-t1,"\t%f"%sigma
        for i in range(len(p)):
            print i,"Genes",p[i].Cost1,p[i].Cost2
            os.makedirs("../Results/Generation_%i/Population/Specie_%i" %(r,i))
            thefile = open('../Results/Generation_%i/Population/Specie_%i/Genes' %(r,i), 'w')
            [thefile.write("%.1f\t%.1f\n" %(p[i].R[j],p[i].A[j]))for j in range(len(p[i].R))]
            thefile = open('../Results/Generation_%i/Population/Specie_%i/Temperature' %(r,i), 'w')
            thefile.write("%.3f" %(p[i].Cost1))
            thefile = open('../Results/Generation_%i/Population/Specie_%i/Area' %(r,i), 'w')
            thefile.write("%.3f" %(p[i].Cost2))
            thefile.close()
            thefile.close()
            thefile.close()
            #graph(p[i].R,p[i].A,r,i)
        
        #Store Best Solution Ever Found
        #Store Best Cost History
        BestCosts.append(p[0].Cost1)
        
        #Display Iteration Information
        #print("Iteration--- %s\n" %r)
        #print("Best Cost--- %s" %BestCosts[r])
        #thefile = open('../BestCosts', 'w')
        #thefile.write("%s\t" %BestCosts[r])
        
        thefile=open('../Results/Generation_%i/Costs'%(r),"w")
        [thefile.write("%f\t%f\n" %(p[i].Cost1,p[i].Cost2)) for i in range(len(p))]
        thefile.close()
        #Write_Iteration()
        
        r+=1
        #Write_Iteration()
        

    #print "actual Cost",Cost1(0,0,f)
    print "Function             ---",
    #print "Minima at Coordinates---", p[0].Genes
    #BestCosts.sort()
    print "Solution at minima   ---", min(BestCosts)
    #print BestCosts
    Total=0
    for i in range(len(BestCosts)):

        Cost1,Cost2=read_integers('../Results/Generation_%d/Costs' %i, 'fxy')
        s=0;s1=0;Cost3=[]
        if i>=1:
            while s==0:
                if (os.path.isdir("../Results/Generation_%d/Specie_%d/" %(i,s1))):
                    Cost3=(read_integers('../Results/Generation_%d/Specie_%d/Temperature' %(i,s1), 'f'))
                    s1+=1
                    plt.scatter(i,Cost3,s=0.1,label='Species',c='blue')
                else:
                    s+=1
            print "number of Species",i,s1
            Total=Total+s1
        x=[i for j in range(len(Cost1))]
        plt.scatter(x,Cost1,s=2,label='Species',c='yellow')
    
    plt.title('Objective Function')
    plt.xlabel("Generations")
    plt.ylabel("Cost")
    plt.savefig('../Figure1.svg')
    
    plt.close()
    print "Total Species:",Total
    
    '''
    for i in range(len(BestCosts)):
        x=[]
        Cost=read_integers('../Results/Generation_%d/Cost' %i, 'f')
        for j in range(len(Cost)):
            x.append(i)    
        plt.scatter(x,Cost,label='Species')
    plt.title('Objective Function')
    plt.savefig('Figure1.svg')
    
    plt.close()
    '''
    '''
    thefile = open('../GA/FinalGenes', 'w')
    for i in range(len(p[0].R)):
        thefile.write("%.1f\t%.1f\n" %(p[0].R[i],p[0].A[i]))
    thefile.close()    
    graph(p[0].R,p[0].A,0,0,'Yes')
    x,y=read_integers('../GA/Genes','fxy')
    graph(x,y,0,1,'Yes')
    '''
    '''
    plt.plot(numpy.fabs(BestCosts))
    plt.savefig('Figure.svg')
    plt.show()
    '''

'''
def Popread():
    r=read_integers('../Generation',"i")
    r=r-1
    global p
    p=[]
    g=0
    if r==0:
        while os.path.isdir("../Results/Generation_%i/Specie_%i" %(r,g)):
            x,y =read_integers("../Results/Generation_%i/Specie_%i/Genes" %(r,g),'fxy')
            z=read_integers("../Results/Generation_%i/Specie_%i/Temperature" %(r,g),'f')
            p.append(Pop(x,y,z))                
    else:
        while os.path.isdir("../Results/Generation_%i/Population/Specie_%i" %(r,g)):
            x,y =read_integers("../Results/Generation_%i/Population/Specie_%i/Genes" %(r,g),'fxy')
            z=read_integers("../Results/Generation_%i/Population/Specie_%i/Temperature" %(r,g),'f')
            p.append(Pop(x,y,z))            

def Write_Iteration():
    global r
    thefile=open('../Generation','w')
    thefile.write('%i'%r)
    thefile.close()
'''







