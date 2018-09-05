import csv, random, re, sys, os, math, numpy, time, subprocess, shutil
import matplotlib.pyplot as plt 
from multiprocessing import Pool
from distutils.dir_util import copy_tree
import scipy.interpolate as si




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
    if x==y:
        return x
    
    else:
        a=5
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
    plt.xlim(-50,50)  
    plt.ylim(100,250) 
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
    def __init__(self, R=[],A=[], Cost1=0,Cost2=0,Rank=float('inf')) :
        self.R = R
        self.A = A
        self.Cost1 = Cost1
        self.Cost2 = Cost2
        self.Rank = Rank
        

def Cost(e):
    #x1,y1=read_integers('Genes','fxy')
    copy_tree("../CFD", "../Results/Generation_%d/Specie_%d/CFD" %(r,e))
    os.chdir("../Results/Generation_%d/Specie_%d/CFD" %(r,e))
    #subprocess.call(['./All'])
    os.chdir("../")
    #x,y=read_integers('Genes','fxy')
    x=randf(0,5)
    y=randf(0,3)
    Cost1=4*x**2+4*y**2
    Cost2=(x-5)**2+(y-5)**2
    
    thefile=open('Temperature',"w")
    thefile.write("%f" %(-Cost1))
    thefile=open('Area',"w")
    thefile.write("%f" %(-Cost2))
    thefile.close()
    thefile.close()

    os.chdir("../../../GA")
    shutil.rmtree("../Results/Generation_%d/Specie_%d/CFD" %(r,e))    
    Cost1=read_integers("../Results/Generation_%d/Specie_%d/Temperature" %(r,e),"f")
    Cost2=read_integers("../Results/Generation_%d/Specie_%d/Area" %(r,e),"f")
    return Cost1,Cost2
    
def Gen1(nPop0):
    def insert(coorArrX,coorArrY,R,A):
        #n+=1
        coorArrX.append(R*math.cos(math.radians(A)))
        coorArrY.append(R*math.sin(math.radians(A)))

    def run(I): 

        from Constants import Rmin,Rmax,Lim
        coorArrX = []
        coorArrY = []
        R1=randf(Rmin,Rmax)
        A=Lim[0][1][1]
        #print R,(A-i*Angle)
        insert(coorArrX,coorArrY,R1,A)
        for i in range(1,nVar-1):
            R=randf(Rmin,Rmax)
            A=randf(Lim[i][1][1],Lim[i][1][0])
            #print R,(A-i*Angle)
            insert(coorArrX,coorArrY,R,A) 
        insert(coorArrX,coorArrY,R1,A)
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


    if(not os.path.isdir("../Results")):
        os.mkdir('../Results')
    else:
        shutil.rmtree('../Results')
        os.mkdir('../Results')
    #print "yes"
    #os.makedirs('../Results/Generation_0/Specie_%i'%(1))
    [run(i) for i in range(nPop0)]
    '''
    y = Pool()
    result = y.map(run,range(nPop0))
    y.close()
    y.join() 
    '''

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
    distance = [0.0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, F1[:])
    sorted2 = sort_by_values(front, F2[:])
    distance[0] = float('inf')
    distance[len(front) - 1] = float('inf')
    #print "max",max(F1),"min",min(F1)
    for k in range(1,len(front)-1):
        #print distance[k]
        distance[k] = distance[k]+ math.fabs((F1[sorted1[k+1]] - F2[sorted1[k-1]])/(max(F1)-min(F1)))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ math.fabs((F1[sorted2[k+1]] - F2[sorted2[k-1]])/(max(F2)-min(F2)))
    return distance

def POFSort(p):
    
    def takeRank(elem):
        return elem.Rank
    front=fast_non_dominated_sort([p[i].Cost1 for i in range(len(p))],[p[i].Cost2 for i in range(len(p))])
    #print '\n\nFront\n',front
    CD=[crowding_distance([p[j].Cost1 for j in range(len(p))],[p[j].Cost2 for j in range(len(p))],front[i][:]) for i in range(len(front))]
    for i in range(len(front)):
        for j in range(len(front[i])):
            #print 'yes',front[i][j],(float(i)+1/(CD[i][j]+2))
            
            p[(front[i][j])].Rank=1+float(i)/2+1/(CD[i][j]+3)
    #print [p[i].Rank for i in range(len(p))]
            #print [i,front[i][j],CD[i][j],p[(front[i][j])].Rank]
    plt.xlim(-150,10)
    plt.ylim(-60,10)
    for z in range(len(front)):
        plt.scatter([p[front[z][i]].Cost1 for i in range(len(front[z]))] ,[p[front[z][i]].Cost2 for i in range(len(front[z]))],s=z+1,c='black')
        #plt.scatter([p[i].Cost1 for i in range(len(p))] ,[p[i].Cost2 for i in range(len(p))],s=3,c='black') 
        #plt.show()
    
    p.sort(key=takeRank)
    
    #print [p[i].Rank for i in range(len(p))]
    return p


p=[]
r=read_integers('../Generation',"i")


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
        
        ypool= Pool()
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

        #Ranking
        #print 
        #plt.scatter([p[i].Cost1 for i in range(len(p))],[p[i].Cost2 for i in range(len(p))],s=5,c='black')
        p=POFSort(p)
        #Get Best and Worst Cost Values
        Ranks=[(p[t].Rank) for t in range(len(p))]

        Rankmin = min(Ranks);
        
        Rankmax = max(Ranks);
        
        #Initialize Offsprings Population
        
        pn=[]
        X=[]
        Y=[]
        #Reproduction
        g=0
        def reproduction(i):            
            global g
 
            S= int(Smin + (Smax - Smin)*(p[i].Rank - Rankmax)/(Rankmin - Rankmax))
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

        p=POFSort(p)
        #plt.scatter([p[i].Cost1 for i in range(len(p))],[p[i].Cost2 for i in range(len(p))],s=5,c='black')
                
        if len(p)>nPop:
            p2=[(p[i])for i in range(nPop)]
            p=[]
            p=p2    
        plt.scatter([p[i].Cost1 for i in range(len(p))],[p[i].Cost2 for i in range(len(p))],s=20,c='red')
        n=[(int(p[i].Rank),round((100/(p[i].Rank-int(p[i].Rank)+1)-1),1)) for i in range(len(p))]
        for j, txt in enumerate(n):
            plt.annotate(txt,((p[j].Cost1),(p[j].Cost2)),fontsize=5)
   
        plt.savefig('Pics/%i.svg'%r)
        plt.close()
        
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
        
    '''
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
    
    thefile = open('../GA/FinalGenes', 'w')
    for i in range(len(p[0].R)):
        thefile.write("%.1f\t%.1f\n" %(p[0].R[i],p[0].A[i]))
    thefile.close()    
    graph(p[0].R,p[0].A,0,0,'Yes')
    x,y=read_integers('../GA/Genes','fxy')
    '''
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







