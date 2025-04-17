from driver_rdcheck import read_populations
import pickle

"""
PARAMETERS
"""

DIRNAME = "EXPS_NO4A"
BASENAME = "no4a"
NUM_MODEs = [1,2,3]
NUM_STATE = 5
DELTATs = [0.001, 0.01, 0.1, 1.0]
TMAX = 500

for NUM_MODE in NUM_MODEs:
    for DELTAT in DELTATs:

        datafilename = f"data/{BASENAME}_{NUM_STATE}s_{NUM_MODE}m_t{TMAX}_exact_dt={DELTAT}.pkl"

        # Load the pickle object
        print(f">>Loading {datafilename}.")
        with open(datafilename, "rb") as file:
            data = pickle.load(file)

        """
        Read the population results 
        """

        popdir = f"./runs/{DIRNAME}/dt={DELTAT}/"
        popfile = f"{BASENAME}_{NUM_STATE}s_{NUM_MODE}m_t{TMAX}_exact"
        print(f">>Reading MCTDH population trajectories from {popdir}{popfile}")
        std_pops, eff_pops, errors = read_populations(
            dir=popdir, name=popfile, n_states=NUM_STATE, no_err_read=False
        )
        # Modify the object
        data["std_trajectories"] = std_pops
        data["eff_trajectories"] = eff_pops
        data["error_trajectories"] = errors

        # Save the modified object
        print(f">>Writing std/eff trajectories and the errors to {datafilename}.\n")
        with open(datafilename, "wb") as file:
            pickle.dump(data, file)

print("**************\nMAIN_INTERPRET.py successfully terminating.\n**************")
