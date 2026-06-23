import numpy as np

nx = 257
ny = 257

def interface(z):
    return 0.5 * (1.0 + np.sin(np.pi * max(-0.5, min(0.5, z / l_int))))

amplitude = 0.5
wavenumber = 4.0
l_int = 4.0
Ly = 51.2
x = np.linspace(0,1,nx)
ycos = amplitude * np.cos(2.0*np.pi*x*wavenumber)

y = np.linspace(0,Ly, ny)
dist = 7.0/8.0*Ly + ycos

field = np.zeros((nx,ny))
for i in range(nx):
    for j in range(ny):
        field[i,j] = min(max(interface(dist[i] - y[j]), 1e-5), 1.0 - 1e-5)


f = open("dealloy_phi.in",'wb')
f.write(field.tobytes(order='F'))
f.close()