import numpy as np
import matplotlib.pylab as plt
N_X = 1000
X = np.linspace(-1, 1, N_X)
Y = 1 * np.sin(X*2*np.pi*50)
Y += 2 * np.sin(X*2*np.pi*70)
Y += 5 * np.sin(X*2*np.pi*90)
Y += 7 * np.sin(X*2*np.pi*30)
F = np.abs(np.fft.fft(Y)) / N_X * 2
freq = np.fft.fftfreq(N_X) * N_X / 2
plt.subplot(211)
plt.plot(Y)
plt.subplot(212)
plt.plot(freq, F, 'k.')
print("""
Y = 1 * np.sin(X*2*np.pi*50)
Y += 2 * np.sin(X*2*np.pi*70)
Y += 5 * np.sin(X*2*np.pi*90)
Y += 7 * np.sin(X*2*np.pi*30)
""")
plt.show()
