# Specifikáció:
# 5G rendszer downlink irányú PSS detekció
# SNR függvényében

import numpy as np
import math
import matplotlib.pyplot as plt
import cmath

PI = 3.141592653589793


# PSS szimbólumhoz szükséges X vektor létrehozása
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


# PSS szimbólum létrehozása
def PSSgen(x, Nid):
    Pss = np.zeros(127)
    for bit in range(127):
        bit = int(bit)
        m = (bit + 43 * Nid) % 127
        Pss[bit] = 1 - 2 * x[m]
        # print(str(bit) + " " + str(m) + " " + str(Pss[bit]))
    return Pss

# IFFT
def IFFT(symbols_frequency):
    ifft_results = np.fft.ifft(symbols_frequency)
    return ifft_results

# Időtartzományibeli jelhez számít zajteljesítményt
def calculateNoisePower(time_domain, SNRdB):
    signal_power = np.mean(abs(time_domain**2))
    sigma2 = signal_power * 10 ** (-SNRdB / 10)  # zajteljesítmény
    return sigma2

# Gaussi zajt ad a jelhez
def generateNoise(sigma2):
    # komplex zaj sigma2 teljesítménnyel, kétdimenziós normális eloszlás
    noise_real = np.sqrt(sigma2) / 2 * np.random.randn(2 * NOISE_LENGTH + CARRIERNO)
    noise_imag = (
        np.sqrt(sigma2) / 2 * 1j * np.random.randn(2 * NOISE_LENGTH + CARRIERNO)
    )
    noise = noise_real + noise_imag
    return noise


# Frekvenciahibát ad a jelhez
def add_error(signal_time, normalized_frequency_offset):
    size = signal_time.size
    error = np.zeros_like(signal_time)
    for index in range(size):
        error[index] = cmath.exp(1j * 2 * PI * normalized_frequency_offset * index)
    signal_error = signal_time * error
    return signal_error


# Megkeresi a PSS szimbólumot időtartományban
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

# SSS szimbólumhoz szükséges X vektor létrehozása
def SSSgenX12():
    X = np.zeros(127)
    X[0] = 1
    X[1] = 0
    X[2] = 0
    X[3] = 0
    X[4] = 0
    X[5] = 0
    X[6] = 0
    for bit in range(127 - 7):
        X[bit + 7] = (X[bit + 4] + X[bit]) % 2
    return X

# SSS szimbólum létrehozása
def SSSgen(X, Nid1, Nid2):
    m1 = Nid1 % 112
    m0 = 15 * (Nid1 / 112) + 5 * Nid2
    dsss = np.zeros(127)
    for i in range(127):
        X_index_1 = int((i + m0) % 127)
        X_index_2 = int((i + m1) % 127)
        dsss[i] = (1 - 2 * X[X_index_1]) * (1 - 2 * X[X_index_2])
    return dsss

# SSS szimbólum evaluálása
def evaluateSSS(SSS_hat, Nid2, X):
    corr = np.zeros(336)
    for i in range(335):
        SSS = SSSgen(X, i, Nid2)
        SSS_time = np.fft.ifft(SSS)
        corr[i] = np.correlate(SSS_time, SSS_hat)[0]
    Nid1 = np.argmax(np.abs(corr))
    return Nid1


CARRIERNO = 256  # no. of subcarriers
MU = 4  # bits / symbol
SNRdB = 5
NOISE_LENGTH = 100

# Szimulációs paraméterek
Nid2 = 0
Nid1 = 120

x = PSSgenX()
X_sss = SSSgenX12()
Pss = PSSgen(x, Nid2)
begin = np.zeros(56)  # TODO: don't hardcode it
end = np.zeros(73)  # TODO: don't hardcode it
Pss = np.append(begin, Pss)
Pss = np.append(Pss, end)
Pss_time = IFFT(Pss)
Pss_time_length = Pss_time.size
# print(Pss_time_length)

Pss_time_zeros = np.zeros(NOISE_LENGTH)
Pss_time_extended = np.append(Pss_time_zeros, Pss_time)
Pss_time_extended = np.append(Pss_time_extended, Pss_time_zeros)

p_vector = np.array([])
SNR_vector = np.array([])

for x in range(50):
    SNRdB = x - 45  # simulating from -45 dB to 5 dB
    Pss_found = 0
    Nid1_found = 0
    NoisePower = calculateNoisePower(Pss_time, SNRdB)
    for simulation in range(400):
        Noise_time = generateNoise(NoisePower)
        signal_time = Noise_time + Pss_time_extended
        signal_error = add_error(signal_time, 0.0001)
        index = findPss(signal_error, Pss_time)
        # print(f"findPss által megtalált index:{index}")
        if index == NOISE_LENGTH:
            Pss_found = Pss_found + 1

            # Miután megvan az index, a Pss utáni Sss-ből megpróbáljuk kitalálni, hogy milyen CellID-t kapott a telefon
            # Keressük a legnagyobb korrelációs együtthatóval rendelkező Nid1 SSS-t
            SSS_hat = signal_error[index : index + Pss_time.size]
            Nid1_hat = evaluateSSS(SSS_hat, Nid2, X_sss)
            if Nid1_hat == Nid1:
                Nid1_found = Nid1_found + 1;
    
    p1 = Pss_found / 400
    p2 = Nid1_found / Pss_found

    print(f"{x} {p1} {p2}")
    p_vector = np.append(p_vector, p1)
    SNR_vector = np.append(SNR_vector, SNRdB)

# print(p_vector)
plt.figure(figsize=(10, 6))  # Set the figure size
plt.plot(
    SNR_vector, p_vector, marker="o", linestyle="-", color="b"
)  # Set marker style, line style, and color

plt.title(
    "Downlink irányú PSS detekció frekvenciahibával terhelve"
)  # Set the title of the plot
plt.xlabel("SNR [dB]")  # Set the label for the x-axis
plt.ylabel("Helyes Pss megtalálásának valószínűsége")  # Set the label for the y-axis

plt.grid("minor")  # Add a grid
# plt.legend(['Data'])  # Add a legend

# plt.show()


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
