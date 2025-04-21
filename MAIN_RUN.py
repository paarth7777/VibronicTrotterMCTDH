"""
Basic script for generating all required MCTDH directories and input files, and
automatically submitting the jobs to a slurm queue. Example: DABNA
"""

import pickle
import numpy as np
from driver_mk_inputs import mk_input
from mode_selector import get_reduced_model
from mode_combiner import combine_by_frequency
from pennylane.labs.trotter_error import vibronic_fragments, RealspaceMatrix
from math import log2, ceil, pow
from vibronic_frags import vibronic_frags

"""
PARAMETERS DABNA
"""
# NUM_STATES = 6
# NUM_MODEs = [3, 5, 7] # list(range(1, 10 + 1))
# DELTATs = [0.01, 1.0] # [0.001, 0.01, 0.1, 1.0]
# TMAX = 500
# VIBHAMFILE = "./VCHLIB/dabna_6s10m.pkl"
# DIRNAME = "EXPS_DABNA"
# INIT_STATE = 0  # S1 state
# SUBMIT_SLURM = False
# PTHREADS = 64
# EXACT = True
# BASENAME = "dabna"

"""
PARAMETERS N4O4
"""
NUM_STATES = 5
NUM_MODEs = [3]
NUM_list =  [8] #[1, 3, 5, 7]
DELTATs = [0.01]
TMAX = 500
VIBHAMFILE = "./VCHLIB/n4o4a_sf.pkl"
DIRNAME = "EXPS_NO4A"
INIT_STATE = 3  # S0S1 state
SUBMIT_SLURM = False
PTHREADS = 64
EXACT = True
BASENAME = "no4a"

"""
Load a vibronic Hamiltonian
"""

filehandler = open(VIBHAMFILE, "rb")
omegas_total, couplings = pickle.load(filehandler)
omegas_total = np.array(omegas_total)

lambdas = couplings[0]
alphas = couplings[1]
betas = couplings[2]
m_max = len(omegas_total)

for NUM_MODE in NUM_MODEs:
    for DELTAT in DELTATs:
        for NUM in NUM_list:
            assert NUM_MODE <= m_max

            """
            Parameterize simulation instances, and generate working directories and input files.
            """

            omegas, couplings_red = get_reduced_model(
                omegas_total, couplings, m_max=NUM_MODE, order_max=2, strategy=None
            )
            # lambdas = couplings_red[0]
            # alphas = couplings_red[1]
            # betas = couplings_red[2]

            # vibronic_frags handle this internally.

            print(f">>Constructing vibronic Hamiltonian for M = {NUM_MODE}...")

            FRAG_TYPE = 'FC'       # or 'Greedy'
            SUBTYPE = 'IZ-XY_grouping'  # or 'IZ-XY_grouping', 'fit_alpha_beta'
            MOL_NAME = 'n4o4a_sf'    # assuming pickle file lives under mol/{BASENAME}.pkl

            h_list = vibronic_frags(mol=MOL_NAME, p=NUM_MODE, type=FRAG_TYPE, find=False, subtype=SUBTYPE, num_frags=NUM)
            # n_blocks = pow(2, ceil(log2(NUM_STATES))) # bug in the zeroes code

            # h_operator = sum(h_list, RealspaceMatrix.zero(states=n_blocks, modes=NUM_MODE))

            mode_labels = [f"mode{idx+1}" for idx in range(len(omegas))]
            logical_modes = combine_by_frequency(mode_labels, omegas, target_binsize=4)

            subpath = f"runs/{DIRNAME}/dt={DELTAT}"

            jobdir = f"{BASENAME}_{NUM_STATES}s_{NUM_MODE}m_t{TMAX}_{NUM}"

            if EXACT:
                jobdir += '_exact'


            parameters = {
                "N_states": NUM_STATES,
                "M_modes": NUM_MODE,
                "deltaT": DELTAT,
                "Tmax": TMAX,
                "omegas": omegas,
                "couplings": couplings_red,
                "path": f"datagen/{subpath}",  # path to mctdh folder
                "init_state": INIT_STATE,
                "exact": EXACT,
            }

            # save parameters to a pickle file in ./data

            datafile_name = f"{jobdir}_dt={DELTAT}"
            if EXACT:
                datafile_name += '_exact'
            with open(f"./data/{datafile_name}.pkl", "wb") as f:
                pickle.dump(parameters, f)

            # generate the mctdh input files

            mk_input(
                f"./{subpath}",
                jobdir,
                h_list,  # list of fragments (RealSpaceMatrices)
                DELTAT,
                n_grid=8,
                logical_modes=logical_modes,
                spfs_per_state=6,
                init_state=INIT_STATE,
                t_max=TMAX,
                pthreads=PTHREADS,
                exact=EXACT,  # overrides the spf specification with an exact calculation
                no_err_inps=False,
                make_slurm_file=True,
            )

"""
Submit the MCTDH jobs via slurm 
"""

if SUBMIT_SLURM:

    import os
    import glob
    import subprocess

    for DELTAT in DELTATs:
        DIR = f"./runs/{DIRNAME}/dt={DELTAT}"
        os.chdir(DIR)
        slurm_files = glob.glob("*.slurm")

        for file in slurm_files:
            print(f"!! Submitting {file} !!")
            subprocess.run(["sbatch", file], check=True)
        os.chdir("../../..")

print("**************\nMAIN_RUN.py successfully terminating.\n**************")
