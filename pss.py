import numpy as np
import math
import matplotlib.pyplot as plt
import cmath
# 5G rendszer szinkronizaciojanak 1. lepese: PSS megtalalasa

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

def gen_all_PSS(X):
    PSS_all = np.zeros((3, 127))
    for Nid2 in range(2):
        PSS_all[Nid2] = PSSgen(X, Nid2)
    return PSS_all

# IFFT
def IFFT(symbols_frequency):
    ifft_results = np.fft.ifft(symbols_frequency)
    return ifft_results

# Időtartományibeli jelhez számít zajteljesítményt
def calculateNoisePower(time_domain, SNRdB):
    signal_power = np.mean(abs(time_domain**2))
    sigma2 = signal_power * 10 ** (-SNRdB / 10)  # zajteljesítmény
    return sigma2


# Gaussi zajt ad a jelhez
def generateNoise(sigma2, Symbol_time_extended):
    # komplex zaj sigma2 teljesítménnyel, kétdimenziós normális eloszlás
    noise_real = np.sqrt(sigma2) / 2 * np.random.randn(Symbol_time_extended.size)
    noise_imag = np.sqrt(sigma2) / 2 * 1j * np.random.randn(Symbol_time_extended.size)
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
def findPss(signal_error, Pss_all):
    Pss_size = 256
    # TODO: IFFT
    correlation = np.zeros(signal_error.size - Pss_size)
    for i in range(correlation.size):
        sub_arr = signal_error[i : i + Pss_size]
        # print(sub_arr.size)
        # print(np.conj(Pss_time).size)
        # correlation[i] = np.correlate(sub_arr, np.conj(Pss_time))[0]
        correlation[i] = np.correlate(sub_arr, Pss_time)[0]
    # print(correlation)
    return np.argmax(np.abs(correlation))

NOISE_LENGTH = 100

Nid2 = 0
PSS_all = gen_all_PSS(PSSgenX())

# Kezdoinformaciok
X_pss = PSSgenX()
begin = np.zeros(56)  # 256 alvivő miatt
end = np.zeros(73)

# Pss generalasa
Pss = PSSgen(X_pss, Nid2)
Pss = np.append(begin, Pss)
Pss = np.append(Pss, end)
Pss_time = IFFT(Pss)  # Pss időtartományban
Pss_time_length = Pss_time.size  # debug

Symbol_time_zeros = np.zeros(NOISE_LENGTH)
Symbol_time_extended = np.append(Symbol_time_zeros, Pss_time)
Symbol_time_extended = np.append(
    Symbol_time_extended, Symbol_time_zeros
)  # Előtte és utána üres hely

Pss_probability_vector = np.array([])

for x in range(50):
    SNRdB = x - 45
    Pss_found = 0
    NoisePower = calculateNoisePower(Pss_time, SNRdB)
    sim_number = 500

    for simulation in range(sim_number):
        Noise_time = generateNoise(NoisePower, Symbol_time_extended)
        signal_time = Noise_time + Symbol_time_extended
        signal_error = (signal_time, 0.0001)

        index = findPss(signal_error)

        if index == NOISE_LENGTH:
            Pss_found = Pss_found + 1

    Pss_probability = Pss_found / sim_number
    print(f"x: {x}, SNR: {SNRdB} Pss_found: {Pss_found}->{Pss_probability}")
    # print(f"{x} {Pss_probability} {Sss_probability}")
    Pss_probability_vector = np.append(Pss_probability_vector, Pss_probability)
    SNR_vector = np.append(SNR_vector, SNRdB)

# Eredmények megjelenítése
plt.figure(figsize=(10, 6))
plt.plot(SNR_vector, Pss_probability_vector, marker="o", linestyle="-", color="b")

plt.title("Downlink irányú PSS detekció frekvenciahibával terhelve")
plt.xlabel("SNR [dB]")
plt.ylabel("Helyes Pss megtalálásának valószínűsége")

plt.grid("minor")

plt.savefig("output.png")