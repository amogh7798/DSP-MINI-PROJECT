import numpy as np
import matplotlib.pyplot as plt
import time




k=int(input("ENTER THE k:"))
M=int(input("ENTER THE M:"))

tk_temp=input("ENTER tk VALUES:")
ck_temp=input("ENTER ck VALUES:")

tk_data=list(map(int,tk_temp.split(',')))
ck_data=list(map(float,ck_temp.split(',')))

tk=np.array(tk_data)
ck=np.array(ck_data)

ack=input("ERROR TO BE ADDED?:")
tk_=np.zeros(k,np.int8)
ck_=np.ones(k,np.float32)

t=np.arange(M)
c=np.arange(0,5,0.001)

def h(x):
      return np.exp(-x*x/(M*0.05))
e=0

def z(x):
       temp=0
       for i in range(0,k):
                temp=temp+ck[i]*h(x-tk[i])
       return temp

q=np.zeros(M)
for i in t:
      if(ack=="yes" or ack=="YES"):
             q[i]=z(i)+np.random.uniform(-1,1)
      else:
             q[i]=z(i)
plt.plot(t,q)
plt.xlabel("TIME")
plt.ylabel("AMPLITUDE")
plt.title("IMPULSE PASSED THROUGH GAUSSIAN KERNEL")
plt.show()

start=time.time()
print("TIMER STARTED.....")
print("\n\n\n")
e=0

def z_(x,y):
             temp=0
             for i in range(0,k):
                     if(i==e):
                           temp=temp+ck_[i]*h(y-x)
                     else:
                           temp=temp+ck_[i]*h(y-tk_[i])
             return temp

def z_1(x,y):
             temp=0
             for i in range(0,k):
                     if(i==e):
                           temp=temp+x*h(y-tk_[i])
                     else:
                           temp=temp+ck_[i]*h(y-tk_[i])
             return temp

def f(x):
       temp=0
       for i in range(M):
           temp=temp+(q[i]-z_(x,i))*(q[i]-z_(x,i))
       return -1*temp

def f1(x):
       temp=0
       for i in range(M):
           temp=temp+(q[i]-z_1(x,i))*(q[i]-z_1(x,i))
       return -1*temp

#ITERML LOOP
for i in range(1,2000):
         for j in range(0,k):
               tk_[j]=t[np.argmax(f(t))]
               ck_[j]=c[np.argmax(f1(c))]
               e=e+1 
         e=0

u=np.zeros(M,np.float32)


print(tk_)
print(ck_)
for i in range(0,k):
         u[tk_[i]]=ck_[i]

print("TOTAL TIME ELASPED:"+str(time.time()-start)+" SECONDS")

plt.stem(t,u,label="Reconstructed Output")
plt.xlabel("TIME")
plt.ylabel("AMPLITUDE")
plt.title("RECONSTRUCTED IMPULSE")
plt.stem(tk,ck,'r',markerfmt='ro',label="Original impulse")
plt.legend()
plt.show()

def z_final(x):
       temp=0
       for i in range(0,k):
                temp=temp+u[tk_[i]]*h(x-tk_[i])
       return temp

temperr=0
temperr2=0
sd1=0
for i in range(0,M):
       temperr=temperr+(abs(z_final(i)-z(i))*abs(z_final(i)-z(i)))
       temperr2=temperr2+(abs(z(i))*abs(z(i)))
       sd1=sd1+((q[i]-z_final(i))*(q[i]-z_final(i)))

w=z_final(t)
plt.plot(t,q,label="Original kernel Output")
plt.plot(t,w,'r',label="Reconstructed kernel Output")
plt.legend()
plt.show()

print('\n')
errA=temperr/temperr2
print("THE ERROR IN AMPLITUDE IS:"+str(errA))

temperrt=0
tk_.sort()
for i in range(0,k):
        temperrt=temperrt+(abs(tk_[i]-tk[i])*abs(tk_[i]-tk[i]))

errT=np.sqrt((1/k)*temperrt)
print("THE ERROR IN TIME IS:"+str(errT))

print("THE STANDARD DEVIATION IS:"+str(np.sqrt((1/M)*sd1)))