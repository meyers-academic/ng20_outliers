from pathlib import Path
import discovery as ds
import argparse
import jax
import jax.random
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pyarrow
import pyarrow.feather
import json
import pandas as pd
import pickle
import arviz as az
import os


def chain_to_df(chain):

    flat = {}
    for name, value in chain.items():
        arr = np.asarray(value)

        if arr.ndim == 1:
            flat[name] = arr
        else:
           sample_shape = arr.shape[1:]

           for idx in np.ndindex(sample_shape):
                col_name = f"{name}_" + "_".join(map(str, idx))
                flat[col_name] = arr[(slice(None),) + idx]

    return pd.DataFrame(flat)

def make_numpyro_model(psrl, priordict=ds.prior.priordict_standard, num_frequencies=30):
    """
    Make the numpyro model for sampling.

    Parameters:
    -----------
    psrl : `ds.PulsarLikelihood`
        Pulsar likelihood for the marginalized HMC sampling.
    priordict: `dict`, optional, default=discovery.prior.priordiect_standard
        prior dictionary used to set efac/equad/ecorr priors.

    Returns:
    model : `numpyro.model`
        A numpyro model that can be used for sampling.
    """


    # manually set these priors right now. efac prior too restrictive at the moment.
    ecorr_params = [p for p in psrl.logL.params if 'ecorr' in p]
    ecorr_range = [v for k, v in priordict.items() if 'ecorr' in k][0]

    efac_params = [p for p in psrl.logL.params if 'efac' in p]
    efac_range = [v for k, v in priordict.items() if 'efac' in k][0]
    # reset this!!
    efac_range = [0.1, 10] # TODO: should this be included?

    equad_params = [p for p in psrl.logL.params if 'equad' in p]
    equad_range = [v for k, v in priordict.items() if 'equad' in k][0]

    jlogl = jax.jit(psrl.logL)

    residuals = psrl.y

    def model(rng_key=None): # TODO: what about the rng_key??
        # wn params
        pardict = {}
        efacs = numpyro.sample('efacs', dist.Uniform(efac_range[0], efac_range[1]).expand([len(efac_params)]), rng_key=rng_key)
        equads = numpyro.sample('equads', dist.Uniform(equad_range[0], equad_range[1]).expand([len(equad_params)]), rng_key=rng_key)
        ecorrs = numpyro.sample('ecorrs', dist.Uniform(ecorr_range[0], ecorr_range[1]).expand([len(ecorr_params)]), rng_key=rng_key)
        pardict = {efn: ef for ef, efn in zip(efacs, efac_params)}
        pardict.update({eqn: eq for eq, eqn in zip(equads, equad_params)})
        pardict.update({ecn: ec for ec, ecn in zip(ecorrs, ecorr_params)})

        # coefficients
        coeffs = numpyro.sample("coeffs", dist.Uniform(-1e-4, 1e-4).expand([psrl.N.F.shape[-1]]), rng_key=rng_key)

        # rn rhos
        log10_rhos = numpyro.sample(f'{psrl.name}_red_noise_log10_rho({num_frequencies})', dist.Uniform(-9, -4).expand([num_frequencies]), rng_key=rng_key)
        pardict[f'{psrl.name}_red_noise_log10_rho({num_frequencies})'] = log10_rhos

        # theta
        theta = numpyro.sample("theta", dist.Uniform(0, 1), rng_key=rng_key)     # TODO?

        # dof
        nu = numpyro.sample("nu", dist.Uniform(1, 40), rng_key=rng_key)

        # z_i
        z_i = numpyro.sample("z_i", dist.Binomial(1, 0.5).expand([residuals.size]), rng_key=rng_key)
        q = numpyro.sample("q", dist.Uniform(0, 1).expand([residuals.size]), rng_key=rng_key)
        # z_i = numpyro.sample("z_i", dist.Binomial(1, .5), rng_key=rng_key)
        # alpha_i
        alpha_i = numpyro.sample("alpha_i", dist.Uniform(0, 100).expand([residuals.size]), rng_key=rng_key)
        pardict.update({f'{psrl.name}_alpha_scaling({residuals.size})': alpha_i**z_i})
        pardict.update({k: coeffs[slc] for k, slc in psrl.N.index.items()})

        logl = numpyro.deterministic("loglike", jlogl(pardict)) # Uses the one from psrl_faster!! Bad notebook coding, I'm sorry
        numpyro.factor('logl', logl)

    model.to_df = lambda chain: chain_to_df(chain)


    return model, ecorr_params, efac_params, equad_params

def make_psrl_for_outlier_gibbs_step(ds_psr, spectrum_prior=ds.freespectrum, num_frequencies=30):
    """
    make likelihood for the Gibbs step.
    so we use gp ECORR and make the timing model variable,
    so that the noise matrix is truly diagonal. We will
    use this large GP representation to draw all of the relevant
    coefficients when we need them to whiten the data.

    Parameters:
    -----------
    ds_psr: `ds.Pulsar`
        Discovery pulsar object. Contains data we want to run on.
    spectrum_prior: Python function, optional, default=ds.freespectrum
        Function that implements the spectrum we want. Defaults to a free spectrum.
    num_frequencies: int, optional, default=30
        Number of frequencies for the spectrum.
    """

    psrl = ds.PulsarLikelihood([ds_psr.residuals,
                                ds.makegp_timing(ds_psr, svd=True, variable=True),
                                ds.makegp_ecorr(ds_psr, noisedict={}),
                                ds.makenoise_measurement(ds_psr, noisedict={}, outliers=True),
                                ds.makegp_fourier(ds_psr, spectrum_prior, num_frequencies, name='red_noise')], concat=True)
    return psrl

def make_psrl_for_outlier_hmc_step(ds_psr, spectrum_prior=ds.freespectrum, num_frequencies=30):
    """
    make marginalized likelihood for HMC step.
    so we use kernel ECORR now. We concat things, and we
    use variable timing model because these are faster than
    the alternative.

    Parameters:
    -----------
    ds_psr: `ds.Pulsar`
        Discovery pulsar object. Contains data we want to run on.
    spectrum_prior: Python function, optional, default=ds.freespectrum
        Function that implements the spectrum we want. Defaults to a free spectrum.
    num_frequencies: int, optional, default=30
        Number of frequencies for the spectrum.
    """
    psrl_faster = ds.PulsarLikelihood([ds_psr.residuals,
                            ds.makegp_timing(ds_psr, svd=True, variable=True),
                            ds.makenoise_measurement(ds_psr, noisedict={}, outliers=True, ecorr=True),
                            ds.makegp_fourier(ds_psr, ds.freespectrum, num_frequencies, name='red_noise')])
    return psrl_faster

def make_gibbs_fn(psrl):
    """
    make the gibbs function
    for our pulsars

    psrl : `ds.PulsarLikelihood`
        Pulsar likelihood object.
    """

    # extract efac/equad/ecorr parameters from the likelihood
    # bake them into the closure.
    ecorr_params = [p for p in psrl.logL.params if 'ecorr' in p]
    efac_params = [p for p in psrl.logL.params if 'efac' in p]
    equad_params = [p for p in psrl.logL.params if 'equad' in p]



    make_Nalpha = psrl.N.N.getN
    Nalpha = psrl.N.N
    Nalpha_solve_2d = Nalpha.make_solve_2d()
    Nalpha_solve_1d = Nalpha.make_solve_1d()
    num_resids = psrl.y.size
    ones = jnp.ones(num_resids)
    Tmat = psrl.N.F
    mval = 0.01 # from Tak, Ellis and Ghosh??
    k = num_resids
    y = psrl.y

    jcond = jax.jit(psrl.sample_conditional)
    cvars = list(psrl.N.index.keys())
    def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        # draw coefficients
        pardict = hmc_sites.copy()
        pardict.update({efn: ef for ef, efn in zip(hmc_sites['efacs'], efac_params)})
        pardict.update({eqn: eq for eq, eqn in zip(hmc_sites['equads'], equad_params)})
        pardict.update({ecn: ec for ec, ecn in zip(hmc_sites['ecorrs'], ecorr_params)})


        nu_dof = hmc_sites['nu']

        # get all the keys at once.
        coeff_key, theta_key, z_i_key, alpha_key = jax.random.split(rng_key, 4)

        # update with alpha scaling
        pardict.update({f'{psrl.name}_alpha_scaling({y.size})': gibbs_sites['alpha_i']**gibbs_sites['z_i']})

        # draw coefficients for whitening
        nkey, coeffs = jcond(coeff_key, pardict)
        
        # turn into a single arary, store
        pardict.update(coeffs)
        coeffs = jnp.hstack([coeffs[c] for c in cvars])

        means = Tmat @ coeffs # for whitening

        yprime = y - means # do the whitening...

        # store GP coefficients
        gibbs_sites['coeffs'] = coeffs

        # theta (population mixture parameter)
        gibbs_sites['theta'] = numpyro.sample("theta", dist.Beta(k * mval + jnp.sum(gibbs_sites['z_i']),
                                                      k * (1-mval) + num_resids - jnp.sum(gibbs_sites['z_i'])),
                                           rng_key=theta_key)

        # z_i's (indicators)
        norm_prob_alpha = jnp.exp(dist.Normal(means, jnp.sqrt(make_Nalpha(pardict))).log_prob(y))
        pardict.update({f'{psrl.name}_alpha_scaling({y.size})': ones})
        norm_prob_reg = jnp.exp(dist.Normal(means, jnp.sqrt(make_Nalpha(pardict))).log_prob(y))



        theta = gibbs_sites['theta']

        # p_i's in Tak, Ellis, Ghosh --
        # TOA-level "observed" mixture probability (TOA-level version of theta)
        q = theta * norm_prob_alpha / (theta * norm_prob_alpha + (1 - theta)*norm_prob_reg)

        # is this still needed? I think it's a legacy from when I had a bug above.
        q = jnp.where(q<1, q, 1)
        gibbs_sites['q'] = q

        q = numpyro.deterministic('q', q)

        n = jnp.array(1, dtype=jnp.int64)
        q = jnp.asarray(q, dtype=jnp.float64)

        # draw z_i based on q probabilities
        gibbs_sites['z_i'] = numpyro.sample("z_i", dist.Binomial(n, q), rng_key=z_i_key)

        # for alpha update
        # Wang & Taylor Eq. 15 -- Unscaled noise matrix is used.
        # We unscaled it above when calculating z's
        tot = yprime @ Nalpha_solve_1d(pardict, yprime)[0]
        top = 0.5 * (nu_dof + gibbs_sites['z_i'] * tot)

        # this is how Gabe and Aaron do it in enterprise outliers code.
        bot = numpyro.sample("alpha_i", dist.Gamma(0.5 * (nu_dof + gibbs_sites['z_i'])), rng_key=alpha_key)
        gibbs_sites['alpha_i'] = top / bot
        return gibbs_sites

    return gibbs_fn


def save_chain(df, filename):
    """Saves Pandas chain table to Feather, preserving `attrs` in `schema.metadata['json']`."""

    table = pyarrow.Table.from_pandas(df)
    table = table.replace_schema_metadata({**table.schema.metadata, 'json': json.dumps(df.attrs)})
    pyarrow.feather.write_feather(table, filename)

    
def run_nuts_with_checkpoints(
    sampler,
    num_samples_per_checkpoint,
    rng_key,
    psr_name,
    outdir="chains",
    resume=False,

):
    """Run NumPyro MCMC and save checkpoints.
    This function performs multiple iterations of MCMC sampling, saving checkpoints
    after each iteration. It saves samples to feather files and the NumPyro MCMC
    state to JSON.
    Parameters
    ----------
    sampler : numpyro.infer.MCMC
        A NumPyro MCMC sampler object.
    num_samples_per_checkpoint : int
        The number of samples to save in each checkpoint.
    rng_key : jax.random.PRNGKey
        The random number generator key for JAX.
    outdir : str | Path
        The directory for output files.
    resume : bool
        Whether to look for a state to resume from.
    Returns
    -------
    None
        This function doesn't return any value but saves the results to disk.
    Side Effects
    ------------
    - Runs the MCMC sampler for the number of iterations required to reach the total sample number.
    - Saves samples data to feather files after each iteration.
    - Writes the NumPyro sampler state to a pickle file after each iteration.
    Example
    -------
    >>> import discovery.samplers.numpyro as ds_numpyro
    >>> # Assume `model` is configured
    >>> npsampler = ds_numpyro.makesampler_nuts(model, num_samples =100, num_warmup=50)
    >>> ds_numpyro.run_nuts_with_checkpoints(npsampler, 10, jax.random.key(42))

    We make sure that .to_df is not overriden
    """
    # convert to pathlib object
    # make directory if it doesn't exist
    if not isinstance(outdir, Path):
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)

    # save under folder name based on pulsar name
    outdir = outdir / psr_name
    outdir.mkdir(exist_ok=True, parents=True)
    samples_file = outdir / f"{psr_name}-numpyro-samples.feather"
    checkpoint_file = outdir / f"{psr_name}-numpyro-checkpoint.pickle"


    if checkpoint_file.is_file() and samples_file.is_file() and resume:
        df = pd.read_feather(samples_file)
        num_samples_saved = df.shape[0]

        with checkpoint_file.open("rb") as f:
            checkpoint = pickle.load(f)

        total_sample_num = sampler.num_samples - num_samples_saved
        print(f"Resuming from checkpoint after {num_samples_saved} samples, {total_sample_num} samples remaining.")

        sampler.post_warmup_state = checkpoint

    else:
        df = None
        num_samples_saved = 0
        total_sample_num = sampler.num_samples

    num_checkpoints = int(jnp.ceil(total_sample_num / num_samples_per_checkpoint))
    remainder_samples = int(total_sample_num % num_samples_per_checkpoint)

    for checkpoint in range(num_checkpoints):
        if checkpoint == 0:
            sampler.num_samples = num_samples_per_checkpoint
            sampler._set_collection_params()  # Need this to update num_samples
        elif checkpoint == num_checkpoints - 1:
            # We won't need to update the collection params because we've set the post warmup state,
            # and that accomplishes the same goal.
            sampler.num_samples = remainder_samples if remainder_samples != 0 else num_samples_per_checkpoint

        sampler.run(rng_key)
        df_new = sampler.to_df()

        df = pd.concat([df, df_new]) if df is not None else df_new

        # save checkpoint, if file already exists, add a number to the end until unique
        
        while True:
            file = outdir / f"{psr_name}_outlier_analysis_arviz_data_checkpoint_{checkpoint}.nc"
            if not file.exists():
                break
            checkpoint += 1

        az_data = az.from_numpyro(sampler)
        az.to_netcdf(az_data, file)
        
        save_chain(df, samples_file)

        with checkpoint_file.open("wb") as f:
            pickle.dump(sampler.last_state, f)

        sampler.post_warmup_state = sampler.last_state

        rng_key, _ = jax.random.split(rng_key)



def get_parser():
    parser = argparse.ArgumentParser(
        description="Run spectral analysis for a single pulsar."
    )

    parser.add_argument(
        "--pulsar-name", "-p",
        required=True,
        help="Name of the pulsar (e.g. J0437-4715)"
    )

    parser.add_argument(
        "--n-frequencies", "-n",
        type=int,
        required=True,
        help="Number of frequencies to use"
    )

    parser.add_argument(
        "--spectrum-type", "-s",
        choices=["powerlaw", "freespectrum"],
        default="free",
        help="Which spectrum parameterization to use"
    )

    parser.add_argument(
        "--outdir", "-o",
        default="../results/",
        help="Directory where results will be written"
    )

    parser.add_argument(
        "--num-warmup", "-w",
        type=int,
        default=100,
        help="Number of HMC warmup / adaptation iterations (default: 100)"
    )

    parser.add_argument(
        "--num-samples", "-N",
        type=int,
        default=100,
        help="Number of posterior samples (default: 100)"
    )

    parser.add_argument(
        "--target-accept", "-a",
        type=float,
        default=0.8,
        help="Target acceptance probability for NUTS / HMC dual averaging (default: 0.8)"
    )

    parser.add_argument(
        "--max-tree-depth", "-d",
        type=int,
        default=10,
        help="Maximum NUTS tree depth (default: 10)"
    )

    parser.add_argument(
        "--outlier-threshold", "-t",
        type=float,
        default=0.1,
        help="Outlier probability threshold (default: 0.1)"
    )

    parser.add_argument(
        "--resume", "-r",
        type = bool,
        default = False,
        help="Resume from previous checkpoint if available"
    )

    parser.add_argument(
        "--checkpoint-steps", "-c",
        type=int,
        default=100,
        help="Number of samples per checkpoint (default: 100)"
    )
    
    # add parser argument for where feather files are located
    parser.add_argument(
        "--dataset-path", "-ds",
        default="../datasets/ng20_v1p1_dmx_feathers/"
    )

    return parser
