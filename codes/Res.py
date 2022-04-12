import pandas as pd
import numpy as np

train=pd.read_csv("C:\\Users\\rajesh\\Desktop\\training dataset.csv")
test=pd.read_csv("C:\\Users\\rajesh\\Desktop\\testing dataset.csv")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.preprocessing import PolynomialFeatures

def datapreprocess(x1,values):
   
    
    wltrain=train.drop(x1,axis=1)
    Rtrain=train[x1]
    
    scalar=StandardScaler().fit(wltrain,Rtrain)
    wltrain=scalar.transform(wltrain)
    values=scalar.transform(values)
    Rtrain=np.array(Rtrain).reshape(-1,1)
    
    wltest=test.drop(x1,axis=1)
    
    wltest=scalar.transform(wltest)
    Rtest=test[x1]
    Rtest=np.array(Rtest).reshape(-1,1)
    
    return wltrain,wltest,Rtest,Rtrain,values


def mlpredict(x1,values):
    
    wltrain,wltest,Rtest,Rtrain,values=datapreprocess(x1,values)
    y=mlcalc(wltrain,wltest,Rtest,Rtrain,values)
    return y
    
        
def mlcalc(wltrain,wltest,Rtest,Rtrain,values):    

    #Using Simple Linear Regression
    poly= PolynomialFeatures(5).fit(wltrain)
    wltrain=poly.transform(wltrain)
    wltest=poly.transform(wltest)
    values=poly.transform(values)
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


print("1. Wire-Length\n2. Wire-Width\n3. Temperature\n4. Resistance")
x1= input("Enter output string choice: ")   

if(x1=='Wire-Length'):
    w = float(input("Enter wire width b/w 2e-08 & 1.78e-07: "))
    w2= float(input("Enter temperature b/w 9.91 & 101.51: "))
    w3= float(input("Enter resistance b/w 22.68 & 486.27: "))
    t="the wire length is:"
elif(x1=='Wire-Width'):
    w = float(input("Enter wire length b/w 1.4e-06 & 8.84e-06: "))
    w2= float(input("Enter temperature b/w 9.91 & 101.51: "))
    w3= float(input("Enter resistance b/w 22.68 & 486.27: "))
    t="the wire width is:"
elif(x1=='Temperature'):     
    w = float(input("Enter wire length b/w 1.4e-06 & 8.84e-06: "))
    w2= float(input("Enter wire width b/w 2e-08 & 1.78e-07: "))
    w3= float(input("Enter resistance b/w 22.68 & 486.27: "))
    t="the temperature is:"
else:    
    w = float(input("Enter wire length b/w 1.4e-06 & 8.84e-06: "))
    w2= float(input("Enter wire width b/w 2e-08 & 1.78e-07: "))
    w3= float(input("Enter temperature 9.91 & 101.51: "))
    t="the resistance is:"
             
                    
n=np.array([w,w2,w3]).reshape(1,-1)
y=mlpredict(x1,n)
print(t,y)