# Specifikáció:
# 5G rendszer downlink irányú PSS detekció
# SNR függvényében

import numpy as np
import math
import matplotlib.pyplot as plt
import cmath

PI = 3.141592653589793


# Creating X vector
def PSSgenX():
    X = np.zeros(127)
    X[0] = 0
    X[1] = 1
    X[2] = 1
    X[3] = 0
    X[4] = 1
    X[5] = 1
    X[6] = 1
    for bit in range(127 - 7):
        X[bit + 7] = (X[bit + 4] + X[bit]) % 2
    return X


# creating PSS signal
def PSSgen(x, Nid):
    Pss = np.zeros(127)
    for bit in range(127):
        bit = int(bit)
        m = (bit + 43 * Nid) % 127
        Pss[bit] = 1 - 2 * x[m]
        # print(str(bit) + " " + str(m) + " " + str(Pss[bit]))
    return Pss


def IFFT(symbols_frequency):
    ifft_results = np.fft.ifft(symbols_frequency)
    return ifft_results


def calculateNoisePower(time_domain, SNRdB):
    signal_power = np.mean(abs(time_domain**2))
    sigma2 = signal_power * 10 ** (-SNRdB / 10)  # zajteljesítmény
    return sigma2


def generateNoise(sigma2):
    # komplex zaj sigma2 teljesítménnyel, kétdimenziós normális eloszlás
    noise_real = np.sqrt(sigma2) / 2 * np.random.randn(2 * NOISE_LENGTH + CARRIERNO)
    noise_imag = (
        np.sqrt(sigma2) / 2 * 1j * np.random.randn(2 * NOISE_LENGTH + CARRIERNO)
    )
    noise = noise_real + noise_imag
    return noise


# adding frequency error to the signal
def add_error(signal_time, normalized_frequency_offset):
    size = signal_time.size
    error = np.zeros_like(signal_time)
    for index in range(size):
        error[index] = cmath.exp(1j * 2 * PI * normalized_frequency_offset * index)
    signal_error = signal_time * error
    return signal_error


# finding PSS in the time domain
def findPss(signal_error, Pss_time):
    Pss_size = 256
    correlation = np.zeros(signal_error.size - Pss_size)
    for i in range(correlation.size):
        sub_arr = signal_error[i : i + Pss_size]
        # print(sub_arr.size)
        # print(np.conj(Pss_time).size)
        # correlation[i] = np.correlate(sub_arr, np.conj(Pss_time))[0]
        correlation[i] = np.correlate(sub_arr, Pss_time)[0]
    # print(correlation)
    return np.argmax(np.abs(correlation))


CARRIERNO = 256  # no. of subcarriers
MU = 4  # bits / symbol
SNRdB = 5
NOISE_LENGTH = 100

Nid = 0
x = PSSgenX()
Pss = PSSgen(x, Nid)
begin = np.zeros(56)  # TODO: don't hardcode it
end = np.zeros(73)  # TODO: don't hardcode it
Pss = np.append(begin, Pss)
Pss = np.append(Pss, end)
Pss_time = IFFT(Pss)

Pss_time_zeros = np.zeros(NOISE_LENGTH)
Pss_time_extended = np.append(Pss_time_zeros, Pss_time)
Pss_time_extended = np.append(Pss_time_extended, Pss_time_zeros)

p_vector = np.array([])
SNR_vector = np.array([])

for x in range(50):
    SNRdB = x - 45  # simulating from -45 dB to 5 dB
    increment = 0
    NoisePower = calculateNoisePower(Pss_time, SNRdB)
    for simulation in range(400):
        Noise_time = generateNoise(NoisePower)
        signal_time = Noise_time + Pss_time_extended
        signal_error = add_error(signal_time, 0.0007)
        index = findPss(signal_error, Pss_time)
        # print(f"findPss által megtalált index:{index}")
        if index == NOISE_LENGTH:
            increment = increment + 1
    p = increment / 400
    print(f"{x} {p}")
    p_vector = np.append(p_vector, p)
    SNR_vector = np.append(SNR_vector, SNRdB)

# print(p_vector)
plt.figure(figsize=(10, 6))  # Set the figure size
plt.plot(
    SNR_vector, p_vector, marker="o", linestyle="-", color="b"
)  # Set marker style, line style, and color

plt.title(
    "Downlink irányú PSS detekció 10 ppm frekvenciahiba esetén"
)  # Set the title of the plot
plt.xlabel("SNR [dB]")  # Set the label for the x-axis
plt.ylabel("Helyes Pss megtalálásának valószínűsége")  # Set the label for the y-axis

plt.grid("minor")  # Add a grid
# plt.legend(['Data'])  # Add a legend

plt.show()


"""
ppm szerint pásztázni.

pss után sss-ből frekihibát becsülni.

jel zaj viszony függvényében mennyire pontos a frekvenciabecslés (mennyi a varianciája).
"""

# Nid = 1
# x = PSSgenX()
# dpss = PSSgen(x, Nid)
# print(dpss)

# Nid = 2
# x = PSSgenX()
# dpss = PSSgen(x, Nid)
# print(dpss)
