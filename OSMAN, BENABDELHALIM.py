# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np            
import matplotlib.pyplot as plt 
import os                     
import scipy
#pour éviter un repliement du spectre,
#le théorème de Shanon stipule:fe>2*fmax,
#la fe minimal est donc de 200Hz



donnée = np.loadtxt("C:\\Users\\nouar\\Downloads\\Data.csv", delimiter=';')
 
def lfic(nom):
    x=[]
    y=[]
    for i in range (len(nom)):
        x.append((donnée[i][0]))
        y.append((donnée[i][1]))
    return x,y
        
        
X,Y=lfic(donnée)

print(lfic(donnée))



plt.figure()
plt.plot(X,Y)
plt.title('Raw signal')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude (mV)')
plt.show()


numer=[1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1]
denom=[1, -2, 1]
ypb=scipy.signal.lfilter(numer,denom,Y)
# scipy.signal.lfilter 

plt.figure()
plt.plot(X,ypb)
plt.title('Passe bas')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude (mV)')
plt.show()
nume=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1]
deno=[1,-1]
nume2=[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,32,]
deno2=[1]
yph1=scipy.signal.lfilter(nume,deno,ypb)
yph2=scipy.signal.lfilter(nume2,deno2,ypb)
yph= yph2-yph1


plt.figure()
plt.plot(X,yph)
plt.title('Passe haut')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude (mV)')
plt.show()

##on nous donne la fonction de transfert pour avoir avoir la dérivé pour T=1
yd = np.convolve(yph, np.array([-1, -2, 0, 2, 1])*1/8,mode="same")
yd2=yd**2
plt.figure()
plt.plot(X, yd2)
plt.title('Signal dérivé mis au carré')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude (mV)')

plt.show()
#nous avons un filtre FIR car d'aprés l equation aux differences y(n) ne dépend que de x(i)
# donc y(n) se calcul comme somme de produits simplement car le dénominateur vaut 1
#et une somme de produit peut se mettre sous la forme d'un produit de convolution


H=np.ones(30)/30
#on utilise d'abord la moyenne avec la convolution
#on definit H comme la somme des x(n-i) i allant de 0 à 29
#une deuxième approche aurait été de ne pas utiliser la convolution
#si notre systéme n'était pas FIR , on aurait utiliser lfilter alors car
#H aurait eu un denominateur plus complexe
ymg=np.convolve(yd2,H,mode="same")
plt.figure()
plt.plot(X, ymg)
plt.title('Signal dérivé mis au carré')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude (mV)')
plt.show()
ymgnorm=ymg/np.max(ymg)
plt.figure()
plt.plot(X, ymgnorm)
plt.title('Signal dérivé mis au carré')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude (mV)')
plt.show()
#on a un décalage de 18 environ 
decalage=18
seuil=0.7
I=[]
i=0
while i<len(ymgnorm):
    if ymgnorm[i]>seuil:
        I.append(i)
        i=i+30
        i=i+1
    else:
        i=i+1
# for i in range(len(ymgnorm)): 
#     if ymgnorm[i]>seuil:
#         I.append(i)
#les fronts montants sont pairs et descendants impairs
#on enléve les impairs et on met le retard
Ind=[I[k]-decalage for k in range(len(I))]
#on prend le max entre X[i] et X[i+30]
Indmax=[]
for i in Ind:
    L=[i+k for k in range(0,30)]
    Yi=[Y[L[j]] for j in range(len(L)) ]
    a=np.argmax(Yi)+i
    Indmax.append(a)
Xmax=[X[Indmax[i]]for i in range(len(Indmax))]
Ymax=[Y[k] for k in Indmax]
plt.figure()
plt.plot(X,Y)
plt.plot(Xmax,Ymax,'rs')

plt.title('Raw signal')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude (mV)')
plt.show()

print(Ymax)
