import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import scipy.optimize as opt
import pickle

# data = "http://opendata.cern.ch/record/21856#"


def analysis(project_name, data_path_before, data_path_after):
    print(data_path_before, data_path_after)
    project_path = f"projects/{project_name}/"

    with open(
        "./projects/example_CMS/compressed_output/column_names.pickle", "rb"
    ) as handle:
        column_names = pickle.load(handle)
    with open(data_path_before, "rb") as handle:
        before = pickle.load(handle)
    with open(data_path_after, "rb") as handle:
        after = pickle.load(handle)
        after = pd.DataFrame(after, columns=column_names)

    # plot_all(project_path, before, after)
    variable = "recoGenJets_slimmedGenJets__PAT.obj.m_state.p4Polar_.fCoordinates.fM"
    plot_peak(project_path, before[variable], after[variable])


def fit(x, a, b, c, k, m):
    return a * np.exp(-((x - b) ** 2) / (2 * c**2)) + m * x + k


def plot_peak(project_path, before, after):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8 * 2.5 * (1 / 2.54), 6 * 2.5 * (1 / 2.54)), sharex=True
    )

    x_min = min(before + after)
    x_max = max(before + after)
    x_diff = abs(x_max - x_min)

    with PdfPages(project_path + "/plotting/analysis.pdf") as pdf:

        # Before Histogram
        counts_before, bins_before = np.histogram(
            before, bins=np.linspace(100, 225, 200)
        )
        hist1 = ax1.hist(
            bins_before[:-1],
            bins_before,
            weights=counts_before,
            label="Before",
            histtype="step",
            color="black",
        )
        before_bin_centers = hist1[1][:-1] + (hist1[1][1:] - hist1[1][:-1]) / 2
        before_bin_centers_error = (hist1[1][1:] - hist1[1][:-1]) / 2
        before_bin_counts = hist1[0]
        before_count_error = np.sqrt(hist1[0])
        ax1.errorbar(
            before_bin_centers,
            before_bin_counts,
            yerr=before_count_error,
            xerr=None,
            marker="",
            linewidth=0.75,
            markersize=1,
            linestyle="",
            color="black",
        )
        optimizedParameters1, pcov1 = opt.curve_fit(
            fit, before_bin_centers, before_bin_counts, p0=[1, 173, 1, 1, 1]
        )
        perr1 = np.sqrt(np.diag(pcov1))
        ax1.plot(
            before_bin_centers,
            fit(before_bin_centers, *optimizedParameters1),
            linewidth=1,
            label="Fit",
            color="red",
        )
        leg1 = ax1.legend(
            borderpad=0.5,
            loc=1,
            ncol=2,
            frameon=True,
            facecolor="white",
            framealpha=1,
            fontsize="medium",
        )
        leg1._legend_box.align = "left"
        leg1.set_title(
            f"Mass  : {round(optimizedParameters1[1],2)} +/- {round(perr1[1],2)}"
            + f"Width : {round(optimizedParameters1[2],2)} +/- {round(perr1[2],2)}"
        )
        ax1.set_ylabel("Counts", fontsize=14, ha="right", y=1.0)
        ax1.set_xlabel("Mass [GeV]", fontsize=14, ha="right", x=1.0)
        ax1.set_title("Before Compression")
        ax1.set_ylim(0, 1500)
        print(f"Before compression:")
        print(f"Mass  : {round(optimizedParameters1[1],2)} +/- {round(perr1[1],2)}")
        print(f"Width : {round(optimizedParameters1[2],2)} +/- {round(perr1[2],2)}")

        # After Histogram
        counts_after, bins_after = np.histogram(after, bins=np.linspace(100, 225, 200))
        hist2 = ax2.hist(
            bins_after[:-1],
            bins_after,
            weights=counts_after,
            label="After",
            histtype="step",
            color="black",
        )
        after_bin_centers = hist2[1][:-1] + (hist1[1][1:] - hist1[1][:-1]) / 2
        after_bin_centers_error = (hist2[1][1:] - hist1[1][:-1]) / 2
        after_bin_counts = hist2[0]
        after_count_error = np.sqrt(hist2[0])
        ax2.errorbar(
            after_bin_centers,
            after_bin_counts,
            yerr=after_count_error,
            xerr=None,
            marker="",
            linewidth=0.75,
            markersize=1,
            linestyle="",
            color="black",
        )
        optimizedParameters2, pcov2 = opt.curve_fit(
            fit, after_bin_centers, after_bin_counts, p0=[1, 173, 1, 1, 1]
        )
        perr2 = np.sqrt(np.diag(pcov2))
        ax2.plot(
            after_bin_centers,
            fit(after_bin_centers, *optimizedParameters2),
            linewidth=1,
            label="Fit",
            color="red",
        )
        leg2 = ax2.legend(
            borderpad=0.5,
            loc=1,
            ncol=2,
            frameon=True,
            facecolor="white",
            framealpha=1,
            fontsize="medium",
        )
        leg2._legend_box.align = "left"
        leg2.set_title(
            f"Mass  : {round(optimizedParameters2[1],2)} +/- {round(perr2[1],2)}"
            + f"Width : {round(optimizedParameters2[2],2)} +/- {round(perr2[2],2)}"
        )
        ax2.set_ylabel("Counts", fontsize=14, ha="right", y=1.0)
        ax2.set_xlabel("Mass [GeV]", fontsize=14, ha="right", x=1.0)
        ax2.set_title("After Decompression")
        ax2.set_ylim(0, 1500)

        print(f"After compression:")
        print(f"Mass  : {round(optimizedParameters2[1],2)} +/- {round(perr2[1],2)}")
        print(f"Width : {round(optimizedParameters2[2],2)} +/- {round(perr2[2],2)}")

        diff = round(
            (
                abs(optimizedParameters1[1] - optimizedParameters2[1])
                / optimizedParameters2[1]
            )
            * 100,
            1,
        )
        fig.suptitle(f"Relative Mass Difference = {diff} %", fontsize=16)

        pdf.savefig()


project_name = "example_CMS"
data_path_before = "./data/example_CMS/example_CMS.pickle"
data_path_after = "./projects/example_CMS/decompressed_output/decompressed.pickle"
analysis(project_name, data_path_before, data_path_after)
