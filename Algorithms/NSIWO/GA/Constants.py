R_Lc=0.5
A_Lc=0.5
#per=0.1
l=8
t=8
nVar=10
MaxIt = 100 
nPop0 = 30    
nPop = 15
Smin = 1     
Smax = 5      
Exponent = 2
sigma_initial = 1
sigma_final = 0.001





#Constraints on variables

Rmax=170
Rmin=140
Amax=12
Amin=-12
Angle=(Amax-Amin)*2/(nVar-2)
#print Angle
Lim=[[[0,0],[0,0]] for i in range(nVar)] # [(Rmax,Rmin),(Amin,Amax)], constraint matrix

for x in range(nVar/2+1):
	Lim[x][0][0]=Rmax
	Lim[x][0][1]=Rmin
	Lim[x][1][0]=Amin+x*Angle
	Lim[x][1][1]=Amin+(x-1)*Angle



for i in range(0,nVar/2):
	Lim[i+x][0][0]=Rmax
	Lim[i+x][0][1]=Rmin
	Lim[i+x][1][0]=Lim[x-i][1][0]
	Lim[i+x][1][1]=Lim[x-i][1][1]

Lim[0][1][1]=Amin
Lim[0][1][0]=Amin

Lim[x][1][0]=Amax

#Lim[x+i][1][0]=Amin

'''
for i in range(len(Lim)):
	print Lim[i]
'''
'''
Values=[Rmax,Rmin,Amax,Amin,R_Lc,A_Lc,per,l,t,Type,MaxIt,nPop0,nPop,Smin,Smax,Exponent,sigma_initial,sigma_final]
Variables=['Rmax','Rmin','Amax','Amin','R_Lc','A_Lc','per','l','t','Type','MaxIt','nPop0','nPop','Smin','Smax','Exponent','sigma_initial','sigma_final']

#print Values,Variables
R=[]
for i in range(len(name)):
    for j in range(len(Values)):
        if name[i]==Variables[j]:
            R.append(Values[j])
'''
