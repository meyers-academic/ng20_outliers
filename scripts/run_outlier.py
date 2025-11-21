from glob import glob
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
module_path = current_file_path.parent.parent / "modules"
sys.path.append(str(module_path))
import utils as ou
from loguru import logger
import discovery as ds
import numpyro
from numpyro import infer
import jax
import pprocess as pp
import arviz as az
#import discovery.samplers.numpyro as ds_numpyro
import pandas as pd
import inspect
from numpyro import infer


def makesampler_hmcgibbs(numpyro_model, jgibbs, gibbs_sites,
                         num_warmup, num_samples, num_chains, **kwargs):

    hmcargs = dict(max_tree_depth=8, dense_mass=False,
                   forward_mode_differentiation=False, target_accept_prob=0.8,
                   **{arg: val for arg, val in kwargs.items()
                      if arg in inspect.getfullargspec(infer.NUTS).args})

    mcmcargs = dict(num_warmup=num_warmup, num_samples=num_samples,
                    num_chains=num_chains, chain_method='parallel',
                    progress_bar=True,
                    **{arg: val for arg, val in kwargs.items()
                       if arg in inspect.getfullargspec(infer.MCMC).kwonlyargs})

    hmc_kernel = infer.NUTS(numpyro_model, **hmcargs)

    kernel = infer.HMCGibbs(hmc_kernel, gibbs_fn=jgibbs, gibbs_sites=gibbs_sites)

    sampler = infer.MCMC(kernel, **mcmcargs)

    sampler.to_df = lambda: numpyro_model.to_df(sampler.get_samples())

    return sampler


def run(args):

    logger.info(f"Attempting to run outliers for {args.pulsar_name}")
    logger.info(jax.devices())
    logger.info([jax.devices()[ii].device_kind for ii in range(len(jax.devices()))])

    ng20_v1p1_path = Path(args.dataset_path)
    ng20_v1p1_pulsar_files = sorted(list(ng20_v1p1_path.glob("*_v1p1_dmx.feather")))

    ndevices = len(jax.devices())
    psrfile = [f for f in ng20_v1p1_pulsar_files if args.pulsar_name in f.name]
    print(ng20_v1p1_pulsar_files)
    print(psrfile)
    if len(psrfile)>1:
        raise ValueError("Give exact pulsar J or B name, please. Pulsar name matches multiple files.")
    elif len(psrfile)==0:
        raise ValueError("Could not find requested pulsar")

    logger.info(f"Loading data for {args.pulsar_name}")
    psr = ds.Pulsar.read_feather(psrfile[0])

    logger.info(f"Making likelihood for gibbs...")
    psrl_gibbs = ou.make_psrl_for_outlier_gibbs_step(psr, num_frequencies=args.n_frequencies)
    logger.info("Making likelihood for HMC...")
    psrl_hmc = ou.make_psrl_for_outlier_hmc_step(psr, num_frequencies=args.n_frequencies)     # TODO: the psrl_hmc not used anywhere?

    logger.info("Making gibbs fuction for outlier analysis")
    gibbs_function = ou.make_gibbs_fn(psrl_gibbs)
    logger.info("Making numpyro model...")

    numpyro_hmc_model, ecorr_params, efac_params, equad_params = ou.make_numpyro_model(psrl_gibbs, num_frequencies=args.n_frequencies)

    jgibbs = jax.jit(gibbs_function)

    logger.info(f"Running MCMC {args.num_warmup} warmup steps and {args.num_samples} sample steps using checkpointing every {args.checkpoint_steps} steps")


   # hmc_kernel = infer.NUTS(numpyro_hmc_model, max_tree_depth=args.max_tree_depth,
                       #     target_accept_prob=args.target_accept)
    #kernel = infer.HMCGibbs(hmc_kernel, gibbs_fn=jgibbs, gibbs_sites=['theta','z_i', 'alpha_i', 'coeffs', 'q'])

   # sampler = infer.MCMC(kernel, num_warmup=args.num_warmup, num_samples=args.num_samples,
                #      num_chains=ndevices)# , chain_method=jax.vmap)

    #sampler.to_df = lambda: numpyro_model.to_df(sampler.get_samples())

    npsampler = makesampler_hmcgibbs(
        numpyro_model=numpyro_hmc_model,
        jgibbs=jgibbs,
        gibbs_sites=['theta','z_i', 'alpha_i', 'coeffs', 'q'],
        num_samples=args.num_samples,
        num_warmup=args.num_warmup,
        num_chains=ndevices,)

    #npsampler = ds_numpyro.makesampler_nuts(
    #numpyro_model=mcmc_obj,
    #num_samples=args.num_samples,
    #num_warmup=args.num_warmup,
    #num_chains=ndevices,)

    # TODO: under resume, check whether input parameters like frequencies -n match those in checkpoint file

    #mcmc.run(jax.random.key(0))
    #logger.info("Finished sampling")

    ou.run_nuts_with_checkpoints(
        sampler=npsampler,
        num_samples_per_checkpoint = args.checkpoint_steps,
        rng_key = jax.random.key(0),
        psr_name = args.pulsar_name,
        outdir = args.outdir,
        resume = args.resume)
    logger.info("Finished sampling")

    # we have to check so that this works with the checkpointing too
    # proceed with plotting and summary
    #logger.info("Exporting numpyro data to arviz and saving mcmc samples...")
    #az_data = az.from_numpyro(mcmc)

    # Must fix the arviz data as well, now it only saves it from last checkpoint
    #pp.save_arviz_data(az_data, args.pulsar_name, args.outdir)

    #mcmc.run(jax.random.key(0))

    #mcmc.print_summary()

    #logger.info(f"Exporting numpyro data to arviz and saving mcmc samples...")
    #az_data = az.from_numpyro(mcmc)
    #pp.save_arviz_data(az_data, args.pulsar_name, args.outdir)

    # now load the arviz data from saved file to make sure it's complete
    # combine all checkpointed samples into one arviz dataset
    logger.info("Loading arviz data from each checkpoint and combining by concatenation...")
    
    # glob the checkpoint files
    file = Path(args.outdir) / args.pulsar_name / f"{args.pulsar_name}_outlier_analysis_arviz_data_checkpoint_*.nc"
    files = sorted(glob(str(file)))
    idata_parts = [az.from_netcdf(f) for f in files]
    #idata_parts = [az.from_netcdf(f) for f in files]

    az_data = az.concat(idata_parts, dim="chain", copy=False)

    logger.info("Plotting outlier results...")
    # TODO: for the plot_otliers, read in the feather file instead (easy fix, see read.ipynb)
   # pp.plot_outliers(psr, mcmc.get_samples(), outlier_threshold=args.outlier_threshold, output_base=args.outdir)
    pp.plot_wnp(psr, efac_params, equad_params, ecorr_params, az_data.posterior, output_base=args.outdir, dataset_path=args.dataset_path)

    logger.info("Storing summary for EFAC, EQUAD, ECORR as csv...")
    pp.make_summary(psr, az_data.posterior, efac_params, equad_params, ecorr_params, output_base=args.outdir, dataset_path=args.dataset_path)

    print(f"Pulsar {psr.name} done!")


if __name__=="__main__":
    # parse arguments from command line
    parser = ou.get_parser()
    args = parser.parse_args()
    # run the analysis
    run(args)





