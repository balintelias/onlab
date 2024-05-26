# Specifikáció:
# 5G rendszer downlink irányú PSS detekció
# SNR függvényében

import numpy as np
import math
import matplotlib.pyplot as plt
import cmath
import csv
import sys

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
        error[index] = cmath.exp(
            1j * 2 * np.pi * normalized_frequency_offset / 256 * index
        )
    signal_error = signal_time * error
    return signal_error


# Megkeresi a PSS szimbólumot időtartományban
def findPss(signal_error, Pss_time, correlation_limit):
    Pss_size = 256
    correlation = np.zeros(signal_error.size - Pss_size)
    for i in range(correlation.size):
        sub_arr = signal_error[i : i + Pss_size]
        correlation[i] = np.correlate(sub_arr, Pss_time)[0]

        # correlation[i] = scipy.signal.correlate(sub_arr, Pss_time, mode='valid')[0]
        correlation[i] /= np.sqrt(
            np.sum(np.abs(sub_arr) ** 2) * np.sum(np.abs(Pss_time) ** 2)
        )
        if correlation[i] < correlation_limit:
            correlation[i] = 0

    index = np.argmax(np.abs(correlation))
    corr_value = correlation[index]
    return_tuple = (index, corr_value)
    return return_tuple


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


def frequency_offset_estimation(signal_error, index, expected_pss):
    received_pss = signal_error[index : index + Nfft]
    # phase_difference = np.angle(np.dot(received_pss, np.conj(expected_pss)))
    # frequency_offset = np.pi * phase_difference /
    phase_1 = np.angle(
        np.dot(received_pss[0 : int(Nfft / 2 - 1)], np.conj(expected_pss[0 : int(Nfft / 2 - 1)]))
    )
    phase_2 = np.angle(
        np.dot(received_pss[int(Nfft / 2) : Nfft - 1], np.conj(expected_pss[int(Nfft / 2) : Nfft - 1]))
    )
    phase_difference = phase_2 - phase_1
    frequency_offset = 1 / np.pi * phase_difference
    return frequency_offset


def measure_error(signal_error, index, Pss_time):  # TODO: this might be wrong
    sub_arr = signal_error[index : index + Nfft]
    first_sum = 0
    second_sum = 0
    interval = int(Nfft / 2 - 1)
    for indice in range(interval):
        i = indice
        first_sum += sub_arr[i] * np.conjugate(Pss_time[i])

    for indice in range(interval):
        i = indice + int(Nfft / 2)
        second_sum += sub_arr[i] * np.conjugate(Pss_time[i])

    error_angle = np.angle(np.conjugate(first_sum) * second_sum)
    error = 1 / np.pi * error_angle
    # error = 1 / np.pi * cmath.exp(error_angle)
    # print(error)
    return error


def compensate_error(signal_error, freq_error):
    size = signal_error.size
    compensate = np.zeros_like(signal_time)
    for index in range(size):
        compensate[index] = cmath.exp(1j * 2 * np.pi * (-1) * freq_error / 256 * index)
    signal_reset = signal_error * compensate
    return signal_reset


CARRIERNO = 256  # no. of subcarriers
MU = 4  # bits / symbol
SNRdB = 5
NOISE_LENGTH = 100
Nfft = 256

# Szimulációs paraméterek
Nid2 = 0
Nid1 = 120
deltaF = float(sys.argv[1])
correlation_limit = float(sys.argv[2])
reset = int(sys.argv[3])

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
Pss_false_probability_vector = np.array([])
Pss_notfound_probability_vector = np.array([])
Sss_probability_vector = np.array([])
Sss_conditional_probability_vector = np.array([])
freq_error_variance_vector = np.array([])
SNR_vector = np.array([])

data_pairs = []

for x in range(50):
    data_entry = []
    SNRdB = x - 45 #  -45 dB-től 5 dB-ig szimulálok
    # SNRdB = 5
    Pss_found = 0
    Pss_notfound = 0
    Pss_false = 0
    Nid1_found = 0
    NoisePower = calculateNoisePower(Pss_time, SNRdB)
    sim_number = 1000

    correlations = np.array([])
    freq_error_vector = np.array([])

    for simulation in range(sim_number):
        Noise_time = generateNoise(NoisePower, Symbol_time_extended)
        signal_time = Noise_time + Symbol_time_extended
        # hozzáadjuk a frekvenciahibát
        signal_error = add_error(signal_time, deltaF)
        index_tuple = findPss(signal_error, Pss_time, correlation_limit)  # Pss kezdete
        index = index_tuple[0]
        corr = index_tuple[1]

        if index == NOISE_LENGTH:
            Pss_found = Pss_found + 1
            correlations = np.append(correlations, corr)

            # Frekvenciahiba visszaállítása
            # freq_error = measure_error(signal_error, index, Pss_time)
            freq_error = frequency_offset_estimation(signal_error, index, Pss_time)
            # print(f"Frekvenciahiba: {deltaF} becsült: {np.real(freq_error)}")

            if np.abs(freq_error) < 100:
                signal_reset = compensate_error(signal_error, freq_error)
            else:
                signal_reset = signal_error  # nem lehet kompenzálni

            if reset == 0:
                signal = signal_error
            else:
                signal = signal_reset

            freq_error_vector = np.append(freq_error_vector, freq_error)

            # Miután megvan az index, a Pss utáni Sss-ből megpróbáljuk
            # kitalálni, hogy milyen CellID-t kapott a telefon

            # Keressük a legnagyobb korrelációs együtthatóval
            # rendelkező Nid1 SSS-t
            SSS_hat = signal[
                index + Pss_time.size : index + Pss_time.size + Sss_time.size
            ]

            # becsült Nid1
            Nid1_hat = evaluateSSS(SSS_hat, Nid2, X_sss, SSS_all)
            if Nid1_hat == Nid1:
                Nid1_found = Nid1_found + 1
        else:
            if index == 0:
                Pss_notfound += 1
            else:
                Pss_false += 1

    # Valószínűségek:
    Pss_probability = Pss_found / sim_number
    Pss_notfound_probability = Pss_notfound / sim_number
    Pss_false_probability = Pss_false / sim_number
    Sss_probability = Nid1_found / sim_number

    freq_error_var = np.var(freq_error_vector)
    # print(np.var(np.abs(freq_error_vector)))
    # print(np.var(freq_error_vector))
    freq_error_variance_vector = np.append(freq_error_variance_vector, freq_error_var)

    # print(correlations)
    if Pss_found != 0:
        Sss_conditional_probability = Nid1_found / Pss_found
    else:
        Sss_conditional_probability = 0

    if correlations.size != 0:
        corr_mean = np.mean(correlations)
    else:
        corr_mean = 0

    print(
        f"""x: {x}, SNR: {SNRdB}
        Pss_found:  {Pss_found}->{Pss_probability},
        correct Cell ID: {Nid1_found}->{Sss_probability},
        Correlation mean: {corr_mean},
        Variance: {freq_error_var}
        """
    )

    data_entry.extend([x])
    data_entry.extend([SNRdB])
    data_entry.extend([Pss_found])
    data_entry.extend([Pss_probability])
    data_entry.extend([Nid1_found])
    data_entry.extend([Sss_probability])
    data_entry.extend([corr_mean])
    data_entry.extend([freq_error_var])
    data_entry.extend([Pss_false_probability])
    data_entry.extend([Pss_notfound_probability])
    

    Pss_probability_vector = np.append(Pss_probability_vector, Pss_probability)
    Pss_false_probability_vector = np.append(
        Pss_false_probability_vector, Pss_false_probability
    )
    Pss_notfound_probability_vector = np.append(
        Pss_notfound_probability_vector, Pss_notfound_probability
    )
    Sss_probability_vector = np.append(Sss_probability_vector, Sss_probability)
    SNR_vector = np.append(SNR_vector, SNRdB)

    data_pairs.append(data_entry)


# Eredmények megjelenítése
plt.figure(figsize=(10, 6))
plt.plot(SNR_vector, Pss_probability_vector, marker="o", linestyle="-", color="b")
plt.plot(SNR_vector, Sss_probability_vector, marker="o", linestyle="-", color="r")
plt.plot(SNR_vector, Pss_false_probability_vector, marker="o", linestyle="-", color="g")
plt.plot(SNR_vector, Pss_notfound_probability_vector, marker="o", linestyle="-", color="c")

plt.title(f"Downlink irányú PSS és SSS detekció f={deltaF} frekvenciahibával terhelve")
plt.xlabel("SNR [dB]")
plt.ylabel("Helyes Pss és Sss megtalálásának valószínűsége")

plt.grid("minor")
# plt.legend(["Pss", "Sss", "Conditional"])
plt.legend(["Pss", "Sss", "Pss_false", "Pss_notfound"])

plt.savefig("output.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(SNR_vector, freq_error_variance_vector)
plt.title(f"Becsült frekvenciahiba varianciája, f={deltaF}")
plt.xlabel("SNR [dB]")
plt.ylabel("Variancia")
plt.grid("minor")
plt.semilogy()
plt.savefig("variance.png")

# print(SNR_vector)
# print(freq_error_variance_vector)


"""
ppm szerint pásztázni.

pss után sss-ből frekihibát becsülni.

jel zaj viszony függvényében mennyire pontos a frekvenciabecslés (mennyi a varianciája).
"""


"""
kuszob a detekciora PIPA
frekvenciahelyreallitas: elejen tul nagy hibak, fura, de ertheto
hibaarany (hibas detekcio) (kuszob feletti hibak)

relativ primes kereses a pss ismetlodes szerint (mennyi ido lehet??)
bejovo teljesitmenybol allithatjuk a kuszobot
"""

# Specify the file name
file_name = f'output-F{deltaF}-L{correlation_limit}-R{reset}.csv'

# Writing the list to a CSV file
with open(file_name, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data_pairs)

print(f"The list has been written to '{file_name}' successfully.")