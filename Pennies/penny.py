import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

mass = np.loadtxt('Class_Data.csv', delimiter=',', skiprows=1)
plt.hist(mass,bins=20)

light = np.array(mass[mass<mass.mean()])
heavy = np.array(mass[mass>mass.mean()])

l = np.mean(light)
dl = np.std(light, ddof=1)/np.sqrt(len(light))

h = np.mean(heavy)
dh = np.std(heavy, ddof=1)/np.sqrt(len(heavy))

def gaussian(x, X, dX):
    return 1/(dX*np.sqrt(2*np.pi))*np.exp(-(x-X)**2/(2*dX**2))

def error(t):
    return 1/np.sqrt(2*np.pi)*np.exp(-(t**2/2))

def prob_in(t):
    return quad(error,-t,t)


def t(x,X,dX):
    return np.abs(x-X)/dX




print(f'Any penny weight: {np.mean(mass)} ± {np.std(mass, ddof=1)/np.sqrt(len(mass))}')
print(f'Light penny weight: {l} ± {dl}')
print(f'Heavy penny weight: {h} ± {dh}')
print(f'Number of light: {len(light)}')
print(f'Number of heavy: {len(heavy)}')



points = np.linspace(np.min(mass), np.max(mass), num=500)

l_gauss = np.array([gaussian(x, l, dl) for x in points])
h_gauss = np.array([gaussian(x, h, dh) for x in points])
plt.xlabel('Mass (g)')
plt.ylabel('Number of Pennies')
plt.title('Distribution of Penny Masses')

light.sort()
print(light)

t_light = np.array([t(i,l,dl) for i in light])
print(t_light)



#plt.plot(points, l_gauss)
#plt.plot(points, h_gauss)
plt.savefig('penny.png')
plt.show()
