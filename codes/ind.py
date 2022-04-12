import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

data=pd.read_csv("C:\\Users\\rajesh\\Desktop\\induc.csv")


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge

def datapreprocess(x1,values):
   
    
    label=data[x1]
    data2=data.drop(x1,axis=1)
    wltrain, wltest, Rtrain, Rtest = train_test_split(data2, label, test_size=0.2)
    
    scalar=StandardScaler().fit(wltrain,Rtrain)
    wltrain=scalar.transform(wltrain)
    
    print("the values is",values)
    values=scalar.transform(values)
    Rtrain=np.array(Rtrain).reshape(-1,1)
    
    
    
    wltest=scalar.transform(wltest)
   
    Rtest=np.array(Rtest).reshape(-1,1)
    
    return wltrain,wltest,Rtest,Rtrain,values


def mlpredict(x1,values):
    
    wltrain,wltest,Rtest,Rtrain,values=datapreprocess(x1,values)
    y,scor=mlcalc(wltrain,wltest,Rtest,Rtrain,values)
    return y,scor
    
        
def mlcalc(wltrain,wltest,Rtest,Rtrain,values):    

    #Using Simple Linear Regression
    
    poly= PolynomialFeatures(4).fit(wltrain)
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

    y=reg.predict(values)

    
  
    
    
    return y,scor


print("1. Wire-Length\n2. Wire-Width\n3. Thickness\n4. Inductance")
x1= input("Enter string choice: ")   

if(x1=='Wire-Length'):
    w = float(input("Enter wire width b/w 2e-08 & 1.78e-07: "))
    w2= float(input("Enter thickness b/w 8.43e-10 & 1.14e-09: "))
    w3= float(input("Enter inductance b/w 9.42e-10 & 1.28e-08: "))
    t="the wire length is:"
elif(x1=='Wire-Width'):
    w = float(input("Enter wire length b/w 1.4e-06 & 8.83e-06: "))
    w2= float(input("Enter thickness b/w 8.43e-10 & 1.14e-09: "))
    w3= float(input("Enter inductance b/w 9.42e-10 & 1.28e-08: "))
    t="the wire width is:"
elif(x1=='Thickness'):     
    w = float(input("Enter wire length b/w 1.4e-06 & 8.83e-06: "))
    w2= float(input("Enter wire width b/w 2e-08 & 1.78e-07: "))
    w3= float(input("Enter inductance b/w 9.42e-10 & 1.28e-08: "))
    t="the thickness is:"
elif(x1=='Inductance'):
    w = float(input("Enter wire length b/w 1.4e-06 & 8.83e-06: "))
    w2= float(input("Enter wire width b/w 2e-08 & 1.78e-07: "))
    w3= float(input("Enter thickness b/w 8.43e-10 & 1.14e-09: "))
    t="the inductance is:"
             
                    
n=np.array([w,w2,w3]).reshape(1,-1)
y,scor=mlpredict(x1,n)
print(t,y)