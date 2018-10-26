import numpy as np
import matplotlib.pyplot as plt

k=2

tk=np.array([9,2])
ck=np.array([4,2])

tk_=np.array([0,0])
ck_=np.array([1,1])

t=np.arange(10)
c=np.arange(5)

def h(x):
      return np.exp(-x*x/0.5)
e=0

def z(x):
       temp=0
       for i in range(0,len(tk)):
                temp=temp+ck[i]*h(x-tk[i])
       return temp

q=np.zeros(10)
for i in t:
      q[i]=z(i)+np.random.uniform(0,3)
plt.plot(t,q)
plt.show()

print(tk_)
print(ck_)

def z_(x,y):
             return ck_[0]*h(y-x)+ck_[1]*h(y-tk_[1])

def z_2(x,y):
             return ck_[0]*h(y-tk_[0])+ck_[1]*h(y-x)

def z_1(x,y):
             return x*h(y-tk_[0])+ck_[1]*h(y-tk_[1])

def z_3(x,y):
             return ck_[0]*h(y-tk_[0])+x*h(y-tk_[1])

def f(x):
       temp=0
       for i in range(10):
           temp=temp+(q[i]-z_(x,i))*(q[i]-z_(x,i))
       return -1*temp

def f2(x):
       temp=0
       for i in range(10):
           temp=temp+(q[i]-z_2(x,i))*(q[i]-z_2(x,i))
       return -1*temp

def f3(x):
       temp=0
       for i in range(10):
           temp=temp+(q[i]-z_3(x,i))*(q[i]-z_3(x,i))
       return -1*temp

def f1(x):
       temp=0
       for i in range(10):
           temp=temp+(q[i]-z_1(x,i))*(q[i]-z_1(x,i))
       return -1*temp

for i in range(1,2000):
         tk_[0]=t[np.argmax(f(t))]
         ck_[0]=c[np.argmax(f1(c))]
         tk_[1]=t[np.argmax(f2(t))]
         ck_[1]=c[np.argmax(f3(c))]

u=np.zeros(10)

print(tk_)
print(ck_)
for i in range(0,k):
         u[tk_[i]]=ck_[i]
 
plt.stem(t,u)
plt.show()




