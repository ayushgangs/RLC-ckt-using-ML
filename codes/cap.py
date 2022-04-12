import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

data=pd.read_csv("C:\\Users\\rajesh\\Desktop\\cap1.csv")

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

    poly= PolynomialFeatures(3).fit(wltrain)
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


print("1. Length\n2. Width\n3. Thickness\n4. Capacitance")
x1= input("Enter string choice: ")   

if(x1=='Length'):
    w = float(input("Enter width b/w 2.29e-08 & 1.76e-07: "))
    w2= float(input("Enter thickness b/w 8.43e-10 & 1.14e-09: "))
    w3= float(input("Enter capacitance b/w 1.77e-16 & 2.34e-15: "))
    t="the length is:"
elif(x1=='Width'):
    w = float(input("Enter length b/w 8.79e-07 & 8.85e-06: "))
    w2= float(input("Enter thickness b/w 8.43e-10 & 1.14e-09: "))
    w3= float(input("Enter capacitance 1.77e-16 & 2.34e-15: "))
    t="the width is:"
elif(x1=='Thickness'):     
    w = float(input("Enter length b/w 8.79e-07 & 8.85e-06: "))
    w2= float(input("Enter width b/w 2.29e-08 & 1.76e-07: "))
    w3= float(input("Enter capacitance b/w 1.77e-16 & 2.34e-15: "))
    t="the thickness is:"
elif(x1=='Capacitance'):    
    w = float(input("Enter length b/w 8.79e-07 & 8.85e-06: "))
    w2= float(input("Enter width b/w 2.29e-08 & 1.76e-07: "))
    w3= float(input("Enter thickness b/w 8.43e-10 & 1.14e-09: "))
    t="the capacitance is:"
             
                 
n=np.array([w,w2,w3]).reshape(1,-1)
y=mlpredict(x1,n)
print(t,y)