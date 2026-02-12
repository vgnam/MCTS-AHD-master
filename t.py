import numpy as np
import matplotlib.pyplot as plt

T = 1.0
dt = 0.001
t = np.arange(0, 4*T, dt)

# Xây dựng rho(t)
rho = np.zeros_like(t)
# n=0: s2 -> +1 trên [2T/3, T)
rho[(t >= 2*T/3) & (t < T)] = 1
# n=1: s3 -> -1 trên [T, T+T/3)
rho[(t >= T) & (t < T + T/3)] = -1
# n=2: s1 -> +1 trên [2T, 2T+T/3)
rho[(t >= 2*T) & (t < 2*T + T/3)] = 1
# n=3: s4 -> -1 trên [3T+2T/3, 4T)
rho[(t >= 3*T + 2*T/3) & (t < 4*T)] = -1

# h1(t) = rect trên (2T/3, T]
h1 = np.where((t > 2*T/3) & (t <= T), 1, 0)
# h2(t) = rect trên [0, T/3)
h2 = np.where((t >= 0) & (t < T/3), 1, 0)

# Tích chập (dùng np.convolve, chú ý scaling)
yA = np.convolve(rho, h1, mode='full') * dt
yC = np.convolve(rho, h2, mode='full') * dt
t_conv = np.arange(0, len(yA))*dt

plt.figure(figsize=(10,6))
plt.plot(t_conv, yA, label='MF output for b1 (y_A)')
plt.plot(t_conv, yC, label='MF output for b2 (y_C)')
plt.axvline(T, color='k', linestyle='--', alpha=0.5)
plt.axvline(2*T, color='k', linestyle='--', alpha=0.5)
plt.axvline(3*T, color='k', linestyle='--', alpha=0.5)
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()