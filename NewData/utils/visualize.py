import numpy as np
import matplotlib.pyplot as plt


CHANNEL_RESOLUTION_MV = 78e-6

lead1 = np.loadtxt("txt_files/V1.Session 0 - Page 1.7.TXT")
lead2 = np.loadtxt("txt_files/V2.Session 0 - Page 1.8.TXT")
lead3 = np.loadtxt("txt_files/V3.Session 0 - Page 1.9.TXT")
lead4 = np.loadtxt("txt_files/V4.Session 0 - Page 1.10.TXT")
lead5 = np.loadtxt("txt_files/V5.Session 0 - Page 1.11.TXT")
lead6 = np.loadtxt("txt_files/V6.Session 0 - Page 1.12.TXT")


def plot_signal(ecg, title):
    plt.figure(figsize=(12, 4))
    plt.plot(ecg, linewidth=1)

    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(title, dpi=300)
    plt.close()


def plot_all():
    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(lead1)
    axes[0].set_title("V1")
    axes[0].grid(True)

    axes[1].plot(lead2)
    axes[1].set_title("V2")
    axes[1].grid(True)

    axes[2].plot(lead3)
    axes[2].set_title("V3")
    axes[2].grid(True)

    axes[3].plot(lead4)
    axes[3].set_title("V4")
    axes[3].grid(True)

    axes[4].plot(lead5)
    axes[4].set_title("V5")
    axes[4].grid(True)

    axes[5].plot(lead6)
    axes[5].set_title("V6")
    axes[5].grid(True)

    axes[5].set_xlabel("Sample")

    plt.tight_layout()
    plt.savefig("vis/all_leads.png", dpi=300)
    plt.close()


plot_signal(lead1, "vis/V1")
plot_signal(lead2, "vis/V2")
plot_signal(lead3, "vis/V3")
plot_signal(lead4, "vis/V4")
plot_signal(lead5, "vis/V5")
plot_signal(lead6, "vis/V6")
plot_all()