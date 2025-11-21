import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import arviz as az
from arviz.labels import MapLabeller
import json
import pandas as pd
current_file_path = Path(__file__).resolve()
auxl_path = current_file_path.parent.parent / "auxl"
pltsyle = auxl_path / "matplotlib.mplstyle"
plt.style.use(pltsyle)


def save_arviz_data(az_data, psr_name, output_base):

    output_dir = Path(output_base) / psr_name
    output_dir.mkdir(parents=True, exist_ok=True)
    az.to_netcdf(az_data, output_dir / f"{psr_name}_outlier_analysis_arviz_data.nc")


def make_summary(ds_psr, az_posterior, efac_params, equad_params, ecorr_params, output_base, dataset_path):

    summary = az.summary(
    az_posterior,  # or idata_single
    var_names=["efacs", "equads", "ecorrs"],
    hdi_prob=0.95)

    rename_map = {}
    for i, name in enumerate(efac_params):
        rename_map[f"efacs[{i}]"] = name
    for i, name in enumerate(equad_params):
        rename_map[f"equads[{i}]"] = name
    for i, name in enumerate(ecorr_params):
        rename_map[f"ecorrs[{i}]"] = name

    wndict = json.load(open(f"{Path(dataset_path).parent}/ng20_v1p1_dmx_noise_dict.json", 'r'))

    summary.rename(index=rename_map, inplace=True)
    summary["true_value"] = summary.index.map(lambda name: wndict.get(name))

    cols = list(summary.columns)
    mean_idx = cols.index("mean")
    cols.insert(mean_idx + 1, cols.pop(cols.index("true_value")))

    summary = summary[cols]
    output_dir = Path(output_base) / ds_psr.name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / f"{ds_psr.name}_outliers_summary.csv")


def plot_outliers(ds_psr, output_base, outlier_threshold=0.1):

    output_dir = Path(output_base) / ds_psr.name
    output_dir.mkdir(parents=True, exist_ok=True)
    # read in numpyro from output directory given from args
    df = pd.read_feather(output_dir / f"{ds_psr.name}-numpyro-samples.feather")
    df_z = df.filter(like='z_i')
    means = df_z.mean().values

    plt.plot(means, zorder = 1)
    plt.title(f"{ds_psr.name} Outlier probabilities")
    plt.xlabel("TOA Number")
    plt.ylabel("$<z_i>$")
    plt.axhline(y=outlier_threshold, color='r', linestyle='--', label='Outlier Threshold')
    #plt.legend()

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 5))
    ax.spines['bottom'].set_position(('outward', 5))

    plt.ylim(0, 1)
    plt.savefig(output_dir / f"outlier_prb_vs_TOA_t{outlier_threshold:.1f}.pdf", bbox_inches="tight")
    plt.close()

    bad_toa_mask = means > outlier_threshold
    plt.errorbar(ds_psr.toas / 86400, ds_psr.residuals, yerr=ds_psr.toaerrs,
                 alpha=0.1, fmt='o', label='TOAs', zorder = 1)
    plt.scatter(ds_psr.toas[bad_toa_mask] / 86400, ds_psr.residuals[bad_toa_mask],
                facecolor='none', edgecolor='r', label = 'Outliers',
                zorder = 2)
    plt.ylabel(f"Residuals (s)")
    plt.xlabel("Time (days)")
    plt.title(f"{ds_psr.name} ")

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 5))
    ax.spines['bottom'].set_position(('outward', 5))

    plt.savefig(output_dir / f"outlier_highlighted_{outlier_threshold:.1f}.pdf", bbox_inches="tight")
    plt.close()


def plot_wnp(ds_psr, efac_params, equad_params, ecorr_params, posterior, output_base, dataset_path):

    output_dir = Path(output_base) / ds_psr.name
    output_dir.mkdir(parents=True, exist_ok=True)

    for dim in posterior["efacs"].coords["efacs_dim_0"].values:
        efac_single   = posterior["efacs"].isel(efacs_dim_0=dim)
        equad_single  = posterior["equads"].isel(equads_dim_0=dim)
        ecorr_single  = posterior["ecorrs"].isel(ecorrs_dim_0=dim)

        idata_single = az.from_dict(
        posterior={
            "efacs": efac_single.values,
            "equads": equad_single.values,
            "ecorrs": ecorr_single.values,
        }
        )

        labeller = MapLabeller(var_name_map={
            "efacs": "EFAC",
            "equads": "EQUAD",
            "ecorrs": "ECORR"
        })

        # Make the pair plot
        axes = az.plot_pair(
            idata_single,
            var_names=["efacs", "equads", "ecorrs"],
            kind="kde",
            marginals=True,
            labeller=labeller
        )


        fig = axes.ravel()[0].figure
        clean_name = "_".join(efac_params[dim].split("_")[1:-1])
        fig.suptitle(f"{clean_name} Posterior Samples", fontsize=20)

        wndict = json.load(open(f"{Path(dataset_path).parent}/ng20_v1p1_dmx_noise_dict.json", 'r'))
        ax = axes[0,0]
        ax.axvline(wndict[efac_params[dim]], color="red", linestyle="--", linewidth=2)
        ax = axes[1,1]
        ax.axvline(wndict[equad_params[dim]], color="red", linestyle="--", linewidth=2)
        ax = axes[2,2]
        ax.axvline(wndict[ecorr_params[dim]], color="red", linestyle="--", linewidth=2)

        fig.savefig(output_dir / f"{clean_name}_wnp_posterior.pdf", bbox_inches="tight")

