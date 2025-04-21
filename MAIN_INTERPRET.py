from driver_rdcheck import read_populations
import pickle

"""
PARAMETERS DABNA
"""

# DIRNAME = "EXPS_DABNA"
# BASENAME = "dabna"
# NUM_MODEs = list(range(1, 10 + 1))
# NUM_STATE = 6
# DELTATs = [0.001, 0.01, 0.1, 1.0]
# TMAX = 500

"""
PARAMETERS N4O4
"""

DIRNAME = "EXPS_NO4A"
BASENAME = "no4a"
NUM_MODEs = [3]
NUM_STATE = 5
DELTATs = [0.01]
TMAX = 500
NUM_list = [1, 3, 5, 7]

for NUM_MODE in NUM_MODEs:
    for DELTAT in DELTATs:
        for NUM in NUM_list:
            datafilename = f"data/{BASENAME}_{NUM_STATE}s_{NUM_MODE}m_t{TMAX}_{NUM}_exact_dt={DELTAT}_exact.pkl"

            # Load the pickle object
            print(f">>Loading {datafilename}.")
            with open(datafilename, "rb") as file:
                data = pickle.load(file)

            """
            Read the population results 
            """

            std_pops, eff_pops, errors = read_populations(
                dir=f"./runs/{DIRNAME}/dt={DELTAT}/",
                name=f"{BASENAME}_{NUM_STATE}s_{NUM_MODE}m_t{TMAX}_{NUM}_exact",
                n_states=NUM_STATE,
                no_err_read=False,
                runrdcheck=False # 
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
