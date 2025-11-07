import sys
sys.path.append('../modules')
import utils as ou
from loguru import logger
import discovery as ds
import numpyro
from numpyro import infer
import jax

def run(args):

    logger.info(f"Attempting to run outliers for {args.pulsar_name}")
    logger.info(jax.devices())
    logger.info([jax.devices()[ii].device_kind for ii in range(len(jax.devices()))])

    ndevices = len(jax.devices())
    psrfile = [f for f in ou.ng20_v1p1_pulsar_files if args.pulsar_name in f.name]
    if len(psrfile)>1:
        raise ValueError("Give exact pulsar J or B name, please. Pulsar name matches multiple files.")
    elif len(psrfile)==0:
        raise ValueError("Could not find requested pulsar")

    logger.info(f"Loading data for {args.pulsar_name}")
    psr = ds.Pulsar.read_feather(psrfile[0])
    
    logger.info(f"Making likelihood for gibbs...")
    psrl_gibbs = ou.make_psrl_for_outlier_gibbs_step(psr, num_frequencies=args.n_frequencies)
    logger.info("Making likelihood for HMC...")
    psrl_hmc = ou.make_psrl_for_outlier_hmc_step(psr, num_frequencies=args.n_frequencies)
    
    logger.info("Making gibbs fuction for outlier analysis")
    gibbs_function = ou.make_gibbs_fn(psrl_gibbs)
    logger.info("Making numpyro model...")
    numpyro_hmc_model = ou.make_numpyro_model(psrl_gibbs)

    jgibbs = jax.jit(gibbs_function)
    
    logger.info(f"Running MCMC {args.num_warmup} warmup steps and {args.num_samples} sample steps...")
    hmc_kernel = infer.NUTS(numpyro_hmc_model, max_tree_depth=args.max_tree_depth, target_accept_prob=args.target_accept)
    kernel = infer.HMCGibbs(hmc_kernel, gibbs_fn=jgibbs, gibbs_sites=['theta','z_i', 'alpha_i', 'coeffs', 'q'])
    mcmc = infer.MCMC(kernel, num_warmup=args.num_warmup, num_samples=args.num_samples,
                      num_chains=ndevices)# , chain_method=jax.vmap)
    mcmc.run(jax.random.key(0))
    logger.info("Finished sampling")
    mcmc.print_summary()

    # TODO: SAVE THE SAMPLER STATE FOR CHECKPOINTING (SEE DAVE WRIGHT CODE)

    # TODO: SAVE THE TOTAL MCMC OUTPUT
    

if __name__=="__main__":
    # parse arguments from command line
    parser = ou.get_parser() 
    args = parser.parse_args()
    # run the analysis
    run(args)
