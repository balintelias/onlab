import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd

# Read the CSV file into a Pandas DataFrame
df_00 = pd.read_csv(f"output-F0.0-L0.15-R0.csv")
df_005 = pd.read_csv(f"output-F0.05-L0.15-R0.csv")
df_01 = pd.read_csv(f"output-F0.1-L0.15-R0.csv")
df_02 = pd.read_csv(f"output-F0.2-L0.15-R0.csv")
df_03 = pd.read_csv(f"output-F0.3-L0.15-R0.csv")
df_04 = pd.read_csv(f"output-F0.4-L0.15-R0.csv")

# 1. meres
# PSS detekcio frekihiba fuggvenyeben
SNR_Pss_prob_00 = list(zip(df_00.iloc[:, 1], df_00.iloc[:, 3]))
SNR_Pss_prob_005 = list(zip(df_005.iloc[:, 1], df_005.iloc[:, 3]))
SNR_Pss_prob_01 = list(zip(df_01.iloc[:, 1], df_01.iloc[:, 3]))
SNR_Pss_prob_02 = list(zip(df_02.iloc[:, 1], df_02.iloc[:, 3]))
SNR_Pss_prob_03 = list(zip(df_03.iloc[:, 1], df_03.iloc[:, 3]))
SNR_Pss_prob_04 = list(zip(df_04.iloc[:, 1], df_04.iloc[:, 3]))

x1, y1 = zip(*SNR_Pss_prob_00)
x2, y2 = zip(*SNR_Pss_prob_005)
x3, y3 = zip(*SNR_Pss_prob_01)
x4, y4 = zip(*SNR_Pss_prob_02)
x5, y5 = zip(*SNR_Pss_prob_03)
x6, y6 = zip(*SNR_Pss_prob_04)

plt.figure(figsize=(10, 6))
plt.plot(x1, y1)
plt.plot(x2, y2)
plt.plot(x3, y3)
plt.plot(x4, y4)
plt.plot(x5, y5)
plt.plot(x6, y6)
plt.title(f"Downlink irányú PSS detekció frekvenciahibával terhelve")
plt.xlabel("SNR [dB]")
plt.ylabel("Helyes PSS megtalálásának valószínűsége")

plt.grid("minor")
# plt.legend(["Pss", "Sss", "Conditional"])
plt.legend(["f=0", "f=0.05", "f=0.1", "f=0.2", "f=0.3", "f=0.4"])
plt.savefig("feladat1.png")
plt.close()

# 2. feladat
# kuszob
df_00 = pd.read_csv(f"output-F0.2-L0.0-R0.csv")
df_005 = pd.read_csv(f"output-F0.2-L0.05-R0.csv")
df_01 = pd.read_csv(f"output-F0.2-L0.1-R0.csv")
df_02 = pd.read_csv(f"output-F0.2-L0.2-R0.csv")
df_03 = pd.read_csv(f"output-F0.2-L0.3-R0.csv")

SNR_Pss_prob_00 = list(zip(df_00.iloc[:, 1], df_00.iloc[:, 3]))
SNR_Pss_prob_005 = list(zip(df_005.iloc[:, 1], df_005.iloc[:, 3]))
SNR_Pss_prob_01 = list(zip(df_01.iloc[:, 1], df_01.iloc[:, 3]))
SNR_Pss_prob_02 = list(zip(df_02.iloc[:, 1], df_02.iloc[:, 3]))
SNR_Pss_prob_03 = list(zip(df_03.iloc[:, 1], df_03.iloc[:, 3]))

x1, y1 = zip(*SNR_Pss_prob_00)
x2, y2 = zip(*SNR_Pss_prob_005)
x3, y3 = zip(*SNR_Pss_prob_01)
x4, y4 = zip(*SNR_Pss_prob_02)
x5, y5 = zip(*SNR_Pss_prob_03)

plt.figure(figsize=(10, 6))
plt.plot(x1, y1)
plt.plot(x2, y2)
plt.plot(x3, y3)
plt.plot(x4, y4)
plt.plot(x5, y5)
plt.title(f"Downlink irányú PSS detekció frekvenciahibával terhelve")
plt.xlabel("SNR [dB]")
plt.ylabel("Helyes PSS megtalálásának valószínűsége")

plt.grid("minor")
plt.legend(["limit=0", "limit=0.05", "limit=0.1", "limit=0.2", "limit=0.3"])
plt.savefig("feladat2.png")
plt.close()

# 3. feladat SSS
# df_00 = pd.read_csv(f"output-F0.0L0.15.csv")
# df_005 = pd.read_csv(f"output-F0.05L0.15.csv")
# df_01 = pd.read_csv(f"output-F0.1L0.15.csv")
df_020 = pd.read_csv(f"output-F0.1-L0.15-R0.csv")
df_021 = pd.read_csv(f"output-F0.1-L0.15-R1.csv")
# df_03 = pd.read_csv(f"output-F0.3L0.15.csv")
# df_04 = pd.read_csv(f"output-F0.4L0.15.csv")

# SNR_Pss_prob_00 = list(zip(df_00.iloc[:, 1], df_00.iloc[:, 3]))
# SNR_Pss_prob_005 = list(zip(df_005.iloc[:, 1], df_005.iloc[:, 3]))
# SNR_Pss_prob_01 = list(zip(df_01.iloc[:, 1], df_01.iloc[:, 3]))
SNR_Pss_prob_020 = list(zip(df_020.iloc[:, 1], df_020.iloc[:, 3]))
SNR_Pss_prob_021 = list(zip(df_021.iloc[:, 1], df_021.iloc[:, 3]))

# SNR_Pss_prob_03 = list(zip(df_03.iloc[:, 1], df_03.iloc[:, 3]))
# SNR_Pss_prob_04 = list(zip(df_04.iloc[:, 1], df_04.iloc[:, 3]))

# SNR_Sss_prob_00 = list(zip(df_00.iloc[:, 1], df_00.iloc[:, 5]))
# SNR_Sss_prob_005 = list(zip(df_005.iloc[:, 1], df_005.iloc[:, 5]))
# SNR_Sss_prob_01 = list(zip(df_01.iloc[:, 1], df_01.iloc[:, 5]))
SNR_Sss_prob_020 = list(zip(df_020.iloc[:, 1], df_020.iloc[:, 5]))
SNR_Sss_prob_021 = list(zip(df_021.iloc[:, 1], df_021.iloc[:, 5]))
# SNR_Sss_prob_03 = list(zip(df_03.iloc[:, 1], df_03.iloc[:, 5]))
# SNR_Sss_prob_04 = list(zip(df_04.iloc[:, 1], df_04.iloc[:, 5]))

# x1, y1 = zip(*SNR_Pss_prob_00)
# x2, y2 = zip(*SNR_Pss_prob_005)
# x3, y3 = zip(*SNR_Pss_prob_01)
x4, y4 = zip(*SNR_Pss_prob_020)
# x5, y5 = zip(*SNR_Pss_prob_021)

# x5, y5 = zip(*SNR_Pss_prob_03)

# xx1, yy1 = zip(*SNR_Sss_prob_00)
# xx2, yy2 = zip(*SNR_Sss_prob_005)
# xx3, yy3 = zip(*SNR_Sss_prob_01)
xx4, yy4 = zip(*SNR_Sss_prob_020)
xx5, yy5 = zip(*SNR_Pss_prob_021)
# xx5, yy5 = zip(*SNR_Sss_prob_03)

plt.figure(figsize=(10, 6))
# plt.plot(x1, y1)
# plt.plot(x2, y2)
# plt.plot(x3, y3)
plt.plot(x4, y4)
# plt.plot(x5, y5)
# plt.plot(xx1, yy1)
# plt.plot(xx2, yy2)
# plt.plot(xx3, yy3)
plt.plot(xx4, yy4)
plt.plot(xx5, yy5)
plt.title(f"Downlink irányú PSS detekció frekvenciahibával terhelve")
plt.xlabel("SNR [dB]")
plt.ylabel("Helyes PSS és SSS megtalálásának valószínűsége")

plt.grid("minor")
plt.legend(["PSS visszaállítás nélkül", "SSS visszaállítás nélkül", "SSS visszaállítva"])
plt.savefig("feladat3.png")
plt.close()

# Extract data pairs for plotting
# SNR_Pss_prob = list(zip(df.iloc[:, 1], df.iloc[:, 3]))
# SNR_Sss_prob = list(zip(df.iloc[:, 1], df.iloc[:, 5]))
# SNR_equalization = list(zip(df.iloc[:, 1], df.iloc[:, 3]))
# SNR_estimation = list(zip(df.iloc[:, 1], df.iloc[:, 4]))
# SNR_theorethical = list(zip(df.iloc[:, 1], df.iloc[:, 5]))

# x1, y1 = zip(*SNR_Pss_prob)
# x2, y2 = zip(*SNR_Sss_prob)
# x3, y3 = zip(*SNR_equalization)
# x4, y4 = zip(*SNR_estimation)
# x5, y5 = zip(*SNR_theorethical)

# plt.plot(x1, y1)
# plt.plot(x2, y2)
# plt.xlabel("SNR (in dB)")
# plt.ylabel("Probability of finding the correct symbols")
# plt.title(f"5G NR System PSS and SSS synchronisation, error: f={error}")
# plt.grid(True)
# plt.legend(["PSS", "SSS"])
# # plt.show()
# plt.savefig(f"PSS_SSS_f{error}.eps")
# plt.close()


# # ideal vs theoretical
# plt.scatter(x1, y1)
# plt.scatter(x5, y5)
# plt.plot(x1, y1)
# plt.plot(x5, y5)
# plt.xlabel('SNR (in dB)')
# plt.ylabel('Bit Error Rate')
# plt.title('Bit Error Rate Simulation of an OFDM System with QPSK modulation')
# plt.grid(True)
# plt.legend(['Ideal Channel', 'Theoretical'])
# plt.semilogy()
# # plt.show()
# plt.savefig('ideal_theoretical.png')
# plt.close()

# #ideal vs channel
# plt.scatter(x1, y1)
# plt.scatter(x2, y2)
# plt.plot(x1, y1)
# plt.plot(x2, y2)
# plt.xlabel('SNR (in dB)')
# plt.ylabel('Bit Error Rate')
# plt.title('Bit Error Rate Simulation of an OFDM System with QPSK modulation')
# plt.grid(True)
# plt.legend(['Ideal Channel', 'Channel'])
# plt.semilogy()
# # plt.show()
# plt.savefig('ideal_channel.png')
# plt.close()

# #ideal vs equalization
# plt.scatter(x1, y1)
# plt.scatter(x3, y3)
# plt.plot(x1, y1)
# plt.plot(x3, y3)
# plt.xlabel('SNR (in dB)')
# plt.ylabel('Bit Error Rate')
# plt.title('Bit Error Rate Simulation of an OFDM System with QPSK modulation')
# plt.grid(True)
# plt.legend(['Ideal Channel', 'Equalized Channel'])
# plt.semilogy()
# # plt.show()
# plt.savefig('ideal_equalized.png')
# plt.close()

# #ideal vs estimation
# plt.scatter(x1, y1)
# plt.scatter(x4, y4)
# plt.plot(x1, y1)
# plt.plot(x4, y4)
# plt.xlabel('SNR (in dB)')
# plt.ylabel('Bit Error Rate')
# plt.title('Bit Error Rate Simulation of an OFDM System with QPSK modulation')
# plt.grid(True)
# plt.legend(['Ideal Channel', 'Estimated Channel'])
# plt.semilogy()
# # plt.show()
# plt.savefig('ideal_estimated.png')
# plt.close()


# # plt.scatter(x2, y2)
# # plt.scatter(x3, y3)
# # plt.scatter(x4, y4)
# # plt.plot(x2, y2)
# # plt.plot(x3, y3)
# # plt.plot(x4, y4)
# # plt.xlabel('SNR (in dB)')
# # plt.ylabel('Bit Error Rate')
# # plt.title('Bit Error Rate Simulation of an OFDM System with QPSK modulation')
# # plt.grid(True)
# # plt.legend(['Channel', 'Channel and equalized', 'Channel and estimated'])
# # plt.semilogy()
# # # plt.show()
# # #plt.savefig()
# # plt.close()
