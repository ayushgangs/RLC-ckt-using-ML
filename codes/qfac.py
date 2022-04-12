import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

data=pd.read_csv("C:\\Users\\rajesh\\Desktop\\qfactor.csv")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge


def datapreprocess(x1,values):
   
    
    label=data[x1]
    data2=data.drop(x1,axis=1)
    wltrain, wltest, Rtrain, Rtest = train_test_split(data2, label, test_size=0.2)
    
    scalar=StandardScaler().fit(wltrain,Rtrain)
    wltrain=scalar.transform(wltrain)
    values=scalar.transform(values)
    Rtrain=np.array(Rtrain).reshape(-1,1)
    
    
    
    wltest=scalar.transform(wltest)
   
    Rtest=np.array(Rtest).reshape(-1,1)
    
    return wltrain,wltest,Rtest,Rtrain,values


def mlpredict(x1,values):
    
    wltrain,wltest,Rtest,Rtrain,values=datapreprocess(x1,values)
    y=mlcalc(wltrain,wltest,Rtest,Rtrain,values)
    return y
    
        
def mlcalc(wltrain,wltest,Rtest,Rtrain,values):    

    poly= PolynomialFeatures(4).fit(wltrain)
    wltrain=poly.transform(wltrain)
    wltest=poly.transform(wltest)
    values=poly.transform(values)
    #Using Simple Linear Regression
    i = 1
    while i <10:
        reg = LinearRegression().fit(wltrain,Rtrain)
        
        y=reg.predict(wltrain)
        y2=np.absolute(np.subtract(Rtrain,y))
        y3=np.argsort(y2,axis=0)
        y3=y3[-5:]
        Rtrain=np.delete(Rtrain,y3).reshape(-1,1)
        wltrain=np.delete(wltrain,y3,axis=0)
        i=i+1
       
    scor=reg.score(wltest,Rtest)
    print("The accuracy score for Linear Regression is",scor,"/1")


#Using Ridge Rigression



    i = 1
    while i <10:
        reg = Ridge().fit(wltrain,Rtrain)
        
        y=reg.predict(wltrain)
        y2=np.absolute(np.subtract(Rtrain,y))
        y3=np.argsort(y2,axis=0)
        y3=y3[-5:]
        Rtrain=np.delete(Rtrain,y3).reshape(-1,1)
        wltrain=np.delete(wltrain,y3,axis=0)
        i=i+1
       
    scor2=reg.score(wltest,Rtest)
    print("The accuracy score for Ridge Regression is",scor2,"/1")  
  
    y=reg.predict(values)
   
    return y



    
d=data.drop(['Inductance','Quality factor'],axis=1)

w3=float(input("Enter qfactor b/w 2 & 100: "))
x1="Inductance"             



n=np.ones(10000)*w3
d.insert(2, "Q", n, True) 
y=mlpredict(x1,d)

y2=np.where(y>0)[0]
y=y[y2]
d=d.drop(['Q'],axis=1)
d=d.iloc[y2]
d.insert(2,"Inductance",y,True)



x1="Quality factor"
y=mlpredict(x1,d)
y2=np.abs(np.subtract(y,w3))

y2=np.argsort(y2,axis=None)[0:5]
d=d.iloc[y2]
y=y[y2]

d.insert(3, "Q-Factor", y, True) 
print(d)



             





