"""
Basic script for generating all required MCTDH directories and input files, and
automatically submitting the jobs to a slurm queue. Example: (NO)4-Anthracene
"""

import pickle
import numpy as np
from driver_mk_inputs import mk_input
from mode_selector import get_reduced_model
from mode_combiner import combine_by_frequency
# from pennylane.labs.pf.vibronic import (
#     VibronicMatrix,
#     VibronicHamiltonian,
# )


"""
PARAMETERS
"""
NUM_STATES = 5
NUM_MODEs = list(range(1, 3 + 1))
DELTATs = [0.001, 0.01, 0.1, 1.0]
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

        assert NUM_MODE <= m_max

        """
        Parameterize simulation instances, and generate working directories and input files.
        """

        omegas, couplings_red = get_reduced_model(
            omegas_total, couplings, m_max=NUM_MODE, order_max=2, strategy=None
        )
        lambdas = couplings_red[0]
        alphas = couplings_red[1]
        betas = couplings_red[2]

        print(f">>Constructing vibronic Hamiltonian for M = {NUM_MODE}...")
        h_operator = VibronicHamiltonian(
            states=NUM_STATES,
            modes=NUM_MODE,
            phis=[lambdas, alphas, betas],
            omegas=omegas,
            sparse=True,
        )

        mode_labels = [f"mode{idx+1}" for idx in range(len(omegas))]
        logical_modes = combine_by_frequency(mode_labels, omegas, target_binsize=4)

        subpath = f"runs/{DIRNAME}/dt={DELTAT}"

        jobdir = f"{BASENAME}_{NUM_STATES}s_{NUM_MODE}m_t{TMAX}"

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
        # if EXACT:
        #     datafile_name += '_exact'
        with open(f"./data/{datafile_name}.pkl", "wb") as f:
            pickle.dump(parameters, f)

        # generate the mctdh input files

        mk_input(
            f"./{subpath}",
            jobdir,
            h_operator,  # VibronicHamiltonian
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
