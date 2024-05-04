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


def FFT(symbols_time):
    fft_results = np.fft.fft(symbols_time)
    return fft_results


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


# Összes lehetséges SSS generálása
def gen_all_SSS(X, Nid2):
    SSS_all = np.zeros((356, 256))
    begin = np.zeros(56)  # 256 alvivő miatt
    end = np.zeros(73)
    for Nid1 in range(355):
        SSS_current = SSSgen(X, Nid1, Nid2)
        SSS_extended = np.append(begin, SSS_current)
        SSS_extended = np.append(SSS_extended, end)
        SSS_all[Nid1] = SSS_extended
    return SSS_all


# SSS szimbólum evaluálása
def evaluateSSS(SSS_hat, Nid2, X, SSS_all):
    corr = np.zeros(336)
    for i in range(335):
        SSS_time_actual = IFFT(SSS_all[i])
        corr[i] = np.correlate(SSS_hat, SSS_time_actual)[0]
    Nid1 = np.argmax(np.abs(corr))
    return Nid1


CARRIERNO = 256  # no. of subcarriers
MU = 4  # bits / symbol
SNRdB = 5
NOISE_LENGTH = 100

# Szimulációs paraméterek
Nid2 = 0
Nid1 = 120

SSS_all = gen_all_SSS(SSSgenX12(), Nid2)

# Kezdőinformációk:
x = PSSgenX()
X_sss = SSSgenX12()
begin = np.zeros(56)  # 256 alvivő miatt
end = np.zeros(73)

# Pss generálása
Pss = PSSgen(x, Nid2)
Pss = np.append(begin, Pss)
Pss = np.append(Pss, end)
Pss_time = IFFT(Pss)  # Pss időtartományban
Pss_time_length = Pss_time.size  # debug

# Sss
Sss = SSSgen(X_sss, Nid1, Nid2)
Sss = np.append(begin, Sss)
Sss = np.append(Sss, end)
Sss_time = IFFT(Sss)  # SSS időtartományban
Sss_time_length = Sss_time.size  # debug

# Pss és Sss időtartományban egymás után
Symbol_time = np.append(Pss_time, Sss_time)

Symbol_time_zeros = np.zeros(NOISE_LENGTH)
Symbol_time_extended = np.append(Symbol_time_zeros, Symbol_time)
Symbol_time_extended = np.append(
    Symbol_time_extended, Symbol_time_zeros
)  # Előtte és utána üres hely

Pss_probability_vector = np.array([])
Sss_probability_vector = np.array([])
Sss_conditional_probability_vector = np.array([])
SNR_vector = np.array([])

for x in range(50):
    SNRdB = x - 45  #  -45 dB-től 5 dB-ig szimulálok
    Pss_found = 0
    Nid1_found = 0
    NoisePower = calculateNoisePower(Pss_time, SNRdB)
    sim_number = 1000

    for simulation in range(sim_number):
        Noise_time = generateNoise(NoisePower, Symbol_time_extended)
        signal_time = Noise_time + Symbol_time_extended
        # hozzáadjuk a frekvenciahibát
        signal_error = add_error(signal_time, 0.0001)
        index = findPss(signal_error, Pss_time)  # Pss kezdete

        if index == NOISE_LENGTH:
            Pss_found = Pss_found + 1

            # Miután megvan az index, a Pss utáni Sss-ből megpróbáljuk
            # kitalálni, hogy milyen CellID-t kapott a telefon

            # Keressük a legnagyobb korrelációs együtthatóval
            # rendelkező Nid1 SSS-t
            SSS_hat = signal_error[
                index + Pss_time.size : index + Pss_time.size + Sss_time.size
            ]

            # if x > 35:
            # print(f"{index + Pss_time.size}
            # {index + Pss_time.size + Sss_time.size}" ) # debug

            # becsült Nid1
            Nid1_hat = evaluateSSS(SSS_hat, Nid2, X_sss, SSS_all)
            if Nid1_hat == Nid1:
                Nid1_found = Nid1_found + 1

    # Valószínűségek:
    Pss_probability = Pss_found / sim_number
    Sss_probability = Nid1_found / sim_number

    if Pss_found != 0:
        Sss_conditional_probability = Nid1_found / Pss_found
    else:
        Sss_conditional_probability = 0

    print(
        f"""x: {x}, SNR: {SNRdB}
        Pss_found:  {Pss_found}->{Pss_probability},
        correct Cell ID: {Nid1_found}->{Sss_probability}"""
    )
    # print(f"{x} {Pss_probability} {Sss_probability}")
    Pss_probability_vector = np.append(Pss_probability_vector, Pss_probability)
    Sss_probability_vector = np.append(Sss_probability_vector, Sss_probability)
    Sss_conditional_probability_vector = np.append(
        Sss_conditional_probability_vector, Sss_conditional_probability
    )
    SNR_vector = np.append(SNR_vector, SNRdB)

# Eredmények megjelenítése
plt.figure(figsize=(10, 6))
plt.plot(SNR_vector, Pss_probability_vector, marker="o", linestyle="-", color="b")
plt.plot(SNR_vector, Sss_probability_vector, marker="o", linestyle="-", color="r")
# plt.plot(SNR_vector, Sss_conditional_probability_vector, marker="o", linestyle="-", color="g")

plt.title("Downlink irányú PSS és SSS detekció frekvenciahibával terhelve")
plt.xlabel("SNR [dB]")
plt.ylabel("Helyes Pss és Sss megtalálásának valószínűsége")

plt.grid("minor")
# plt.legend(["Pss", "Sss", "Conditional"])
plt.legend(["Pss", "Sss"])

plt.savefig("output.png")


"""
ppm szerint pásztázni.

pss után sss-ből frekihibát becsülni.

jel zaj viszony függvényében mennyire pontos a frekvenciabecslés (mennyi a varianciája).
"""
