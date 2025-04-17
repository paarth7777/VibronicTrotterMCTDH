"""
Driver to generate all required MCTDH files and directories.
"""

import os
from datetime import datetime
import numpy as np
import pickle
from time import time
from py_to_op import generate_op
from py_to_inp import generate_inp, generate_spf_scheme
from mode_selector import rank_modes, get_truncated_coupling_arrays, truncate_by_states
from pennylane.labs.trotter_error import vibronic_fragments
from pennylane.labs.trotter_error import RealspaceMatrix
from pennylane.labs.trotter_error.product_formulas import trotter_error
from math import log2, ceil, pow

def mk_input(
    parent_dir,
    name,
    frag_list,  # VibronicHamiltonian
    deltaT,
    n_grid,
    logical_modes,
    spfs_per_state,
    init_state,
    t_max,
    pthreads=10,
    exact=False,  # overrides the spf specification with an exact calculation
    no_err_inps=False,
    make_slurm_file=False,
):
    """
    parent_dir: directory
    name: name
    h_operator: instance of VibronicHamiltonian
    deltaT: deltaT used in construction of effective Hamiltonian
    n_grid: gridsize used for primitive basis
    n_spfs_per_run: a list of integers. if you only want input files for 1 calculation, give a list of length 1.
        the integers in the list specify the number of SPFs per mode to use for that run. for instance,
        n_spfs_per_run = [1,4] will generate input files for 2 separate calculations: one using 1 SPF per mode,
        i.e., a TDH calculation, and one using 4 SPFs per mode, a correlated calculation.
    init_state: integer specifying which diabat should be initially populated to unity.
    t_max: propagation time of the calculation(s) in femtoseconds
    pthreads: number of threads MCTDH program will use for calculations
    exact: if True, n_spfs_per_run is overridden and only a single exact calculation is performed.
    """
    
    # h_op = h_operator.block_operator()
    frag_one = frag_list[0]
    n = frag_one.states
    m = frag_one.modes

    h_op = sum(frag_list, RealspaceMatrix.zero(states=n, modes=m))

    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir)  # Create the directory and any necessary parent directories
        print(f"\n>Parent deltaT directory created at: {parent_dir}")
    else:
        print(f"\n>Parent deltaT directory {parent_dir} already exists")

    str_op, metadata_std = generate_op(h_op, m, run_name=f"{name}")
    print(f">>Writing vibronic Hamiltonian to {parent_dir}/{name}.op")
    file_handler = open(f"{parent_dir}/{name}.op", "w")
    file_handler.write(str_op)
    file_handler.close()

    if not no_err_inps:
        print(f">>Constructing effective Hamiltonian for M = {m} and deltaT = {deltaT}...")
        t0 = time()
        err_operator = trotter_error(frag_list, deltaT)
        # err_operator = h_operator.epsilon(delta=deltaT)
        h_eff_operator = h_op + err_operator
        print(f"      Obtained effective Hamiltonian after {round(time() - t0, 3)} s.")
        t0 = time()
        str_op, metadata_eff = generate_op(h_eff_operator, m, run_name=f"{name}_err")
        print(
            f"      Obtained operator string for effective Hamiltonian after {round(time() - t0, 3)} s."
        )
        print(f">>Writing effective vibronic Hamiltonian to {parent_dir}/{name}_err.op")
        file_handler = open(f"{parent_dir}/{name}_err.op", "w")
        file_handler.write(str_op)
        file_handler.close()

    if exact:
        spf_specs = "exact"
        name_extra = "exact"

    print(f"\n>>>Preparing directory and input file for standard Hamiltonian (exact = {exact})...")
    str_input = generate_inp(
        n_states=n,
        n_modes=m,
        n_grid=n_grid,
        logical_modes=logical_modes,
        spfs_per_state=spfs_per_state,
        init_state=init_state,
        t_max=t_max,
        run_name=f"{name}",
        opname=name,
        alloc_sec=metadata_std,
        pthreads=pthreads,
        exact=exact,
    )

    # make directory if it doesnt exist
    directory = f"{parent_dir}/{name}"
    # Check if the directory exists
    if not os.path.isdir(directory):
        os.makedirs(directory)  # Create the directory and any necessary parent directories
        print(f">>>Directory created at: {directory}")
    else:
        print(">>>Directory already exists")

    file_handler = open(f"{parent_dir}/{name}.inp", "w")
    file_handler.write(str_input)
    file_handler.close()

    if not no_err_inps:
        print(
            f"\n>>>Preparing directory and input file for effective Hamiltonian (exact = {exact})..."
        )
        str_input = generate_inp(
            n_states=n,
            n_modes=m,
            n_grid=n_grid,
            logical_modes=logical_modes,
            spfs_per_state=spfs_per_state,
            init_state=init_state,
            t_max=t_max,
            run_name=f"{name}_err",
            opname=f"{name}_err",
            alloc_sec=metadata_eff,
            pthreads=pthreads,
            exact=exact,
        )

        directory = f"{parent_dir}/{name}_err"
        # Check if the directory exists
        if not os.path.isdir(directory):
            os.makedirs(directory)  # Create the directory and any necessary parent directories
            print(f">>>Directory created at: {directory}")
        else:
            print(">>>Directory already exists")

        file_handler = open(f"{parent_dir}/{name}_err.inp", "w")
        file_handler.write(str_input)
        file_handler.close()

    if make_slurm_file:
        generate_slurm_file(dir=parent_dir, job_name=f"{name}", cores=pthreads)
        if not no_err_inps:
            generate_slurm_file(dir=parent_dir, job_name=f"{name}_err", cores=pthreads)


def mk_input_from_termdict(
    term_dict, #mapped_terms
    parent_dir,
    name,
    n_grid,
    n_modes,
    n_states, 
    logical_modes,
    spfs_per_state,
    init_state,
    t_max,
    pthreads=10,
    exact=False,  # overrides the spf specification with an exact calculation
    make_slurm_file=False,
):
    """
    parent_dir: directory
    term_dict: dictionary of term labels and coefficients 
    name: name
    deltaT: deltaT used in construction of effective Hamiltonian
    n_grid: gridsize used for primitive basis
    n_spfs_per_run: a list of integers. if you only want input files for 1 calculation, give a list of length 1.
        the integers in the list specify the number of SPFs per mode to use for that run. for instance,
        n_spfs_per_run = [1,4] will generate input files for 2 separate calculations: one using 1 SPF per mode,
        i.e., a TDH calculation, and one using 4 SPFs per mode, a correlated calculation.
    init_state: integer specifying which diabat should be initially populated to unity.
    t_max: propagation time of the calculation(s) in femtoseconds
    pthreads: number of threads MCTDH program will use for calculations
    exact: if True, n_spfs_per_run is overridden and only a single exact calculation is performed.
    """

    n = n_states
    m = n_modes

    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir)  # Create the directory and any necessary parent directories
        print(f"\n>Parent deltaT directory created at: {parent_dir}")
    else:
        print(f"\n>Parent deltaT directory {parent_dir} already exists")

    t0 = time()

    str_op, metadata_eff = generate_op(param_dict=term_dict, m=m, run_name=f"{name}_err")
    print(
        f"      Obtained operator string for effective Hamiltonian after {round(time() - t0, 3)} s."
    )
    print(f">>Writing effective vibronic Hamiltonian to {parent_dir}/{name}_err.op")
    file_handler = open(f"{parent_dir}/{name}_err.op", "w")
    file_handler.write(str_op)
    file_handler.close()

    print(
        f"\n>>>Preparing directory and input file for effective Hamiltonian (exact = {exact})..."
    )
    str_input = generate_inp(
        n_states=n,
        n_modes=m,
        n_grid=n_grid,
        logical_modes=logical_modes,
        spfs_per_state=spfs_per_state,
        init_state=init_state,
        t_max=t_max,
        run_name=f"{name}_err",
        opname=f"{name}_err",
        alloc_sec=metadata_eff,
        pthreads=pthreads,
        exact=exact,
    )

    directory = f"{parent_dir}/{name}_err"
    # Check if the directory exists
    if not os.path.isdir(directory):
        os.makedirs(directory)  # Create the directory and any necessary parent directories
        print(f">>>Directory created at: {directory}")
    else:
        print(">>>Directory already exists")

    file_handler = open(f"{parent_dir}/{name}_err.inp", "w")
    file_handler.write(str_input)
    file_handler.close()

    if make_slurm_file:
        generate_slurm_file(dir=parent_dir, job_name=f"{name}_err", cores=pthreads)






def generate_slurm_file(dir, job_name, cores):

    slurm_str = f"""#!/bin/bash
# Job name:
#SBATCH --job-name=mctdh_{job_name}
#SBATCH --account=rrg-izmaylov
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paarth.jain@mail.utoronto.ca
module load gcc/13.2.0
# Command(s) to run:
mctdh86 -w {job_name}
"""
    file_handler = open(f"{dir}/{job_name}.slurm", "w")
    file_handler.write(slurm_str)
    file_handler.close()

    print(f"Created slurm submit job script {job_name}.slurm at directory: {dir}")


# if __name__ == "__main__": ############################################################## need to change h_operator

#     states = [1, 3]
#     n = len(states)
#     k = 8
#     t_max = 1000
#     init_state = 1  # python notation
#     EXACT = True

#     deltaT_series = [0.001, 0.01, 0.1, 1, 10]
#     mode_series = [2, 3, 4]

#     for m in mode_series:
#         for deltaT in deltaT_series:

#             name = f"no4a_{n}s{m}m"
#             # Create a deltaT directory if doesnt exist:
#             parent_dir = f"./{name}/deltaT={deltaT}"
#             # Check if the directory exists
#             if not os.path.isdir(parent_dir):
#                 os.makedirs(parent_dir)  # Create the directory and any necessary parent directories
#                 print(f"\n>Parent deltaT directory created at: {parent_dir}")
#             else:
#                 print(f"\n>Parent deltaT directory {parent_dir} already exists")

#             print(f"\n>>Getting Vibronic Hamiltonian for n={n}, m={m}.")

#             pkl_file = open("n4o4a_sf.pkl", "rb")
#             omegas, couplings = pickle.load(pkl_file)
#             pkl_file.close()

#             couplings_list = [couplings[c] for c in couplings]
#             couplings_list = couplings_list[:3]  # ignore cubic terms for now

#             RANKING = "C1N+F"

#             couplings_list = truncate_by_states(couplings_list, states_to_keep=states)
#             sorted_modes = rank_modes(omegas, couplings_list, ranking_type=RANKING)

#             # Obtain truncated coupling_array
#             selected_modes = list(sorted_modes.keys())[:m]
#             truncated_couplings = get_truncated_coupling_arrays(
#                 couplings_list, selected_modes, only_diagonal=False
#             )
#             truncated_omegas = [omegas[i] for i in selected_modes]
#             omegas = np.array(truncated_omegas)
#             lambdas, alphas, betas = truncated_couplings

#             print(f">>Constructing vibronic Hamiltonian for M = {m}...")
#             h_list = vibronic_fragments(n, m, omegas, [lambdas, alphas, betas])
#             n_blocks = pow(2, ceil(log2(n))) # bug in the zeroes code
#             h_operator = sum(h_list, RealspaceMatrix.zero(states=n_blocks, modes=m))

#             str_op, metadata_std = generate_op(h_operator.block_operator(), m, run_name=f"{name}")

#             # print(str_op)
#             # # exit()

#             print(f">>Writing vibronic Hamiltonian to {parent_dir}/{name}.op")

#             file_handler = open(f"{parent_dir}/{name}.op", "w")
#             file_handler.write(str_op)
#             file_handler.close()

#             print(f">>Constructing effective Hamiltonian for M = {m} and deltaT = {deltaT}...")

#             err_operator = h_operator.epsilon(delta=deltaT)
#             h_eff_operator = h_operator.block_operator() + err_operator
#             str_op, metadata_eff = generate_op(h_eff_operator, m, run_name=f"{name}_err")

#             print(f">>Writing effective vibronic Hamiltonian to {parent_dir}/{name}_err.op")
#             file_handler = open(f"{parent_dir}/{name}_err.op", "w")
#             file_handler.write(str_op)
#             file_handler.close()

#             spf_series = [8]
#             for spf_num in spf_series:

#                 print(
#                     f"\n>>>Preparing directory and input file for standard Hamiltonian (n_spf = {spf_num})..."
#                 )
#                 if not EXACT:
#                     n_spf_per_mode = [str(spf_num)] * n
#                     n_spf_per_mode = ",".join(n_spf_per_mode)
#                     n_spfs = [n_spf_per_mode] * m
#                     name_extra = f"spf{spf_num}"

#                 else:
#                     n_spfs = "exact"
#                     name_extra = "exact"

#                 str_input = generate_inp(
#                     n_states=n,
#                     n_modes=m,
#                     n_grid=k,
#                     n_spfs=n_spfs,
#                     init_state=init_state,
#                     t_max=t_max,
#                     run_name=f"{name}_{name_extra}",
#                     opname=name,
#                     alloc_sec=metadata,
#                     pthreads=pthreads,
#                     exact=EXACT,
#                 )

#                 # make directory if it doesnt exist

#                 directory = f"{parent_dir}/{name}_{name_extra}"
#                 # Check if the directory exists
#                 if not os.path.isdir(directory):
#                     os.makedirs(
#                         directory
#                     )  # Create the directory and any necessary parent directories
#                     print(f">>>Directory created at: {directory}")
#                 else:
#                     print(">>>Directory already exists")

#                 file_handler = open(f"{parent_dir}/{name}_{name_extra}.inp", "w")
#                 file_handler.write(str_input)
#                 file_handler.close()

#                 print(
#                     f"\n>>>Preparing directory and input file for effective Hamiltonian (n_spf = {spf_num})..."
#                 )
#                 str_input = generate_inp(
#                     n_states=n,
#                     n_modes=m,
#                     n_grid=k,
#                     n_spfs=n_spfs,
#                     init_state=init_state,
#                     t_max=t_max,
#                     run_name=f"{name}_{name_extra}_err",
#                     opname=f"{name}_err",
#                     alloc_sec=metadata,
#                     pthreads=pthreads,
#                 )

#                 directory = f"{parent_dir}/{name}_{name_extra}_err"
#                 # Check if the directory exists
#                 if not os.path.isdir(directory):
#                     os.makedirs(
#                         directory
#                     )  # Create the directory and any necessary parent directories
#                     print(f">>>Directory created at: {directory}")
#                 else:
#                     print(">>>Directory already exists")

#                 file_handler = open(f"{parent_dir}/{name}_{name_extra}_err.inp", "w")
#                 file_handler.write(str_input)
#                 file_handler.close()
