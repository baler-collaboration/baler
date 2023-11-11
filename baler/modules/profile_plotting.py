import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import seaborn as sns
import os


def plot(profiling_path, func):
    """
    Visualizes the data that is generated from .CSV logs
    of codecarbon by plotting graphs .The codecarbon package
    is utilized for tracking amount of electricity that is
    consumed by the given function and the amount of CO2 emission.

    Args:
        path: Specifies the path where the .CSV logs are generated
        func (callable): The function to be profiled.

    Returns:
        result: Void. The plots are stored in the profiling_pathlocation
    """

    # Load CSV data into a DataFrame
    emission_csv_path = os.path.join(profiling_path, "emissions.csv")
    data = pd.read_csv(emission_csv_path)

    # Define the scaling factor (adjust this value according to your needs)
    scaling_factor = 10**6

    # List of columns to scale up (you can modify this as per your requirement)
    columns_to_scale = [
        "duration",
        "emissions",
        "emissions_rate",
        "cpu_power",
        "gpu_power",
        "ram_power",
        "cpu_energy",
        "gpu_energy",
        "ram_energy",
        "energy_consumed",
    ]

    # Scale up the values in the selected columns
    data[columns_to_scale] *= scaling_factor

    # Convert the 'timestamp' column to datetime
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors='coerce')

    # Plot 1: Time series graph for 'timestamp' vs 'duration'
    plt.figure(figsize=(10, 6))
    plt.plot(data["timestamp"], data["duration"], marker="o")
    plt.xlabel("Timestamp")
    plt.ylabel("Duration (in seconds) x 1e-6")
    plt.title("Time Series " + f"{func.__name__}" + " : Duration vs Timestamp")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(profiling_path, f"{func.__name__}" + "_" + "plot1.png"))
    plt.close()
    # plt.show()

    # Plot 2 and 3: Time series graph for 'timestamp' vs 'emissions' and 'emissions_rate'
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ax1.plot(data["timestamp"], data["emissions"], marker="o")
    ax1.set_ylabel("Emissions in CO₂eq (in kg) x 1e-6")
    ax1.set_title("Time Series " + f"{func.__name__}" + " : Emissions vs Timestamp")
    ax1.grid(True)

    ax2.plot(data["timestamp"], data["emissions_rate"], marker="o")
    ax2.set_xlabel("Timestamp")
    ax2.set_ylabel("Emissions Rate in CO₂eq in (kg/second) x 1e-6")
    ax2.set_title(
        "Time Series " + f"{func.__name__}" + " : Emissions Rate vs Timestamp"
    )
    ax2.grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(profiling_path, f"{func.__name__}" + "_" + "plot2.png"))
    plt.close()
    # plt.show()

    # Plot 4, 5, 6, and 7: Time series graph for 'timestamp' vs 'ram_power', 'cpu_energy', 'ram_energy', and 'energy_consumed'
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True)

    axs[0, 0].plot(data["timestamp"], data["ram_power"], marker="o")
    axs[0, 0].set_ylabel("RAM Power (in watt)")
    axs[0, 0].set_title(
        "Time Series " + f"{func.__name__}" + " : RAM Power vs Timestamp"
    )
    axs[0, 0].grid(True)

    axs[0, 1].plot(data["timestamp"], data["cpu_energy"], marker="o")
    axs[0, 1].set_ylabel("CPU Energy in (kilo-watt) x 1e-6")
    axs[0, 1].set_title(
        "Time Series " + f"{func.__name__}" + " : CPU Energy vs Timestamp"
    )
    axs[0, 1].grid(True)

    axs[1, 0].plot(data["timestamp"], data["ram_energy"], marker="o")
    axs[1, 0].set_xlabel("Timestamp")
    axs[1, 0].set_ylabel("RAM Energy (in kilo-watt) x 1e-6")
    axs[1, 0].set_title(
        "Time Series " + f"{func.__name__}" + " : RAM Energy vs Timestamp"
    )
    axs[1, 0].grid(True)

    axs[1, 1].plot(data["timestamp"], data["energy_consumed"], marker="o")
    axs[1, 1].set_xlabel("Timestamp")
    axs[1, 1].set_ylabel("Energy Consumed (in kilo-watt) x 1e-6")
    axs[1, 1].set_title(
        "Time Series " + f"{func.__name__}" + " : Energy Consumed vs Timestamp"
    )
    axs[1, 1].grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(profiling_path, f"{func.__name__}" + "_" + "plot3.png"))
    plt.close()
    # plt.show()

    # Measure central tendencies of the data for each metric
    emissions_central = data["emissions"].describe()
    emissions_rate_central = data["emissions_rate"].describe()
    ram_power_central = data["ram_power"].describe()
    cpu_energy_central = data["cpu_energy"].describe()
    ram_energy_central = data["ram_energy"].describe()
    energy_consumed_central = data["energy_consumed"].describe()

    # Plot histograms for each metric with central tendencies
    plt.figure(figsize=(15, 10))

    # Emissions
    plt.subplot(2, 3, 1)
    sns.histplot(data["emissions"], bins=20, kde=True, color="skyblue")
    plt.axvline(emissions_central["mean"], color="red", linestyle="--", label="Mean")
    plt.axvline(
        emissions_central["50%"], color="orange", linestyle="--", label="Median (50%)"
    )
    plt.axvline(
        emissions_central["25%"], color="green", linestyle="--", label="Q1 (25%)"
    )
    plt.axvline(
        emissions_central["75%"], color="blue", linestyle="--", label="Q3 (75%)"
    )
    plt.xlabel("Emissions")
    plt.title("Histogram " + f"{func.__name__}" + " : Emissions")
    plt.legend()

    # Emissions Rate
    plt.subplot(2, 3, 2)
    sns.histplot(data["emissions_rate"], bins=20, kde=True, color="salmon")
    plt.axvline(
        emissions_rate_central["mean"], color="red", linestyle="--", label="Mean"
    )
    plt.axvline(
        emissions_rate_central["50%"],
        color="orange",
        linestyle="--",
        label="Median (50%)",
    )
    plt.axvline(
        emissions_rate_central["25%"], color="green", linestyle="--", label="Q1 (25%)"
    )
    plt.axvline(
        emissions_rate_central["75%"], color="blue", linestyle="--", label="Q3 (75%)"
    )
    plt.xlabel("Emissions Rate")
    plt.title("Histogram " + f"{func.__name__}" + " Emissions Rate")
    plt.legend()

    # RAM Power
    plt.subplot(2, 3, 3)
    sns.histplot(data["ram_power"], bins=20, kde=True, color="lightgreen")
    plt.axvline(ram_power_central["mean"], color="red", linestyle="--", label="Mean")
    plt.axvline(
        ram_power_central["50%"], color="orange", linestyle="--", label="Median (50%)"
    )
    plt.axvline(
        ram_power_central["25%"], color="green", linestyle="--", label="Q1 (25%)"
    )
    plt.axvline(
        ram_power_central["75%"], color="blue", linestyle="--", label="Q3 (75%)"
    )
    plt.xlabel("RAM Power")
    plt.title("Histogram " + f"{func.__name__}" + " : RAM Power")
    plt.legend()

    # CPU Energy
    plt.subplot(2, 3, 4)
    sns.histplot(data["cpu_energy"], bins=20, kde=True, color="lightcoral")
    plt.axvline(cpu_energy_central["mean"], color="red", linestyle="--", label="Mean")
    plt.axvline(
        cpu_energy_central["50%"], color="orange", linestyle="--", label="Median (50%)"
    )
    plt.axvline(
        cpu_energy_central["25%"], color="green", linestyle="--", label="Q1 (25%)"
    )
    plt.axvline(
        cpu_energy_central["75%"], color="blue", linestyle="--", label="Q3 (75%)"
    )
    plt.xlabel("CPU Energy")
    plt.title("Histogram " + f"{func.__name__}" + " : CPU Energy")
    plt.legend()

    # RAM Energy
    plt.subplot(2, 3, 5)
    sns.histplot(data["ram_energy"], bins=20, kde=True, color="lightblue")
    plt.axvline(ram_energy_central["mean"], color="red", linestyle="--", label="Mean")
    plt.axvline(
        ram_energy_central["50%"], color="orange", linestyle="--", label="Median (50%)"
    )
    plt.axvline(
        ram_energy_central["25%"], color="green", linestyle="--", label="Q1 (25%)"
    )
    plt.axvline(
        ram_energy_central["75%"], color="blue", linestyle="--", label="Q3 (75%)"
    )
    plt.xlabel("RAM Energy")
    plt.title("Histogram " + f"{func.__name__}" + " : RAM Energy")
    plt.legend()

    # Energy Consumed
    plt.subplot(2, 3, 6)
    sns.histplot(data["energy_consumed"], bins=20, kde=True, color="lightyellow")
    plt.axvline(
        energy_consumed_central["mean"], color="red", linestyle="--", label="Mean"
    )
    plt.axvline(
        energy_consumed_central["50%"],
        color="orange",
        linestyle="--",
        label="Median (50%)",
    )
    plt.axvline(
        energy_consumed_central["25%"], color="green", linestyle="--", label="Q1 (25%)"
    )
    plt.axvline(
        energy_consumed_central["75%"], color="blue", linestyle="--", label="Q3 (75%)"
    )
    plt.xlabel("Energy Consumed")
    plt.title("Histogram " + f"{func.__name__}" + " : Energy Consumed")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(profiling_path, f"{func.__name__}" + "_" + "plot4.png"))
    plt.close()
    # plt.show()

    print(f"Your codecarbon profiling plots are available at {profiling_path}")
