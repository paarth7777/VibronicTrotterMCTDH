import subprocess


def read_rdcheckfile(name, dir, n_states, runrdcheck=True):
    """
    Parses population dynamics from MCTDH output file (when rdcheck86 is unavailable).
    """

    import os

    output_path = os.path.join(dir, name, "output")
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Output file not found: {output_path}")

    with open(output_path, "r") as file:
        lines = file.readlines()

    populations = {}
    current_time = None
    pops_at_t = [0.0] * n_states

    for line in lines:
        if "Time  =" in line and "fs" in line:
            try:
                current_time = float(line.split("=")[1].split("fs")[0].strip())
            except:
                current_time = None
            pops_at_t = []

        elif "state =" in line and "pop.:" in line:
            parts = line.split()
            try:
                state = int(parts[2])  # state = X
                pop = float(parts[4])  # pop.: Y
                pops_at_t.append(pop)

                # save once we get all states
                if len(pops_at_t) == n_states:
                    populations[current_time] = pops_at_t
            except:
                continue

    print(f"[DEBUG] Parsed {len(populations)} time points from {output_path}")
    return populations

# def read_rdcheckfile(name, dir, n_states, runrdcheck=True):
#     """
#     Retreives the time series of diabatic state populations
#     """

#     if runrdcheck:
#         result = subprocess.run(
#             ["rdcheck86", "-e"], cwd=f"./{dir}/{name}", capture_output=True, text=True
#         )

#     else: #read the content of rdcheck86 -e directly 
#         with open(f"{dir}/{name}/output", "r") as file:
#         # with open(f"{dir}{name}", "r") as file: # somethings off
#             result = file.read()  # Read the entire file content
#     populations = {}

#     # Print the output
#     if runrdcheck:
#         state_pop_lines = []
#         recording = False
#         for line in result.stdout.split("\n"):
#             if recording:
#                 state_pop_lines.append(line)
#             if "  time[fs]" in line:
#                 recording = True

#     else:
#         state_pop_lines = []
#         recording = False
#         for line in result.split("\n"):
#             if recording:
#                 state_pop_lines.append(line)
#             if "  time[fs]" in line:
#                 recording = True
#     pops_at_t = []
#     for line in state_pop_lines:
#         if "time" not in line and "---" not in line and line != "":
#             print(line)
#             parsedline = " ".join(line.split()).split(" ")
#             t = float(parsedline[0])
#             state = int(parsedline[1])
#             pop = float(parsedline[2])
#             if state == 1:  # resets local counter
#                 pops_at_t = []
#             pops_at_t.append(pop)
#             if state == n_states:
#                 populations[t] = pops_at_t
#     return populations



def get_errors(dir, system, n_states, return_pop_series=True, runrdcheck=True):

    standard_pops = read_rdcheckfile(name=f"{system}", dir=dir, n_states=n_states, runrdcheck=runrdcheck)
    effective_pops = read_rdcheckfile(name=f"{system}_err", dir=dir, n_states=n_states, runrdcheck=runrdcheck)

    assert standard_pops.keys() == effective_pops.keys()

    errors = {}

    for t in standard_pops:
        pop_errs = []
        for idx in range(n_states):
            pop_errs.append(abs(standard_pops[t][idx] - effective_pops[t][idx]))
        errors[t] = pop_errs
        # print(f'Errors at {t} fs : {pop_errs}')

    if return_pop_series:
        return errors, standard_pops, effective_pops
    return errors


# m=2
# deltaT = 0.25
# n_spfs = 3
# dir = f'no4a_deltaT={deltaT}'
# name = f'no4a_{m}_spf{n_spfs}'
# error_time_series, standard_pop_series, effective_pop_series = get_errors(dir, name, n_states=4)

# # for t in error_time_series:
# #     print(f'{t}: {error_time_series[t]}')

# import matplotlib.pyplot as plt

# # max_errors = {}
# # for m in range(1, 9):
# #     name = f'no4a_{m}'
# #     error_time_series, standard_pop_series, effective_pop_series = get_errors(dir, name, n_states=4)
# #     max_errors[m] = max(error_time_series[100])

# # plt.plot(max_errors.keys(), max_errors.values())
# # plt.ylabel('Maximum population error')
# # plt.xlabel('M')
# # plt.show()

# # Create subplots
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))  # 1 row, 3 columns

# d1_color = 'blue'
# d2_color = 'red'
# d3_color = 'orange'
# d4_color = 'green'

# d1_style ='solid'
# d2_style ='dashed'
# d3_style ='dashdot'
# d4_style ='dotted'

# # Plot on each subplot
# axes[0].plot(list(standard_pop_series.keys())[:21], [e[0] for e in standard_pop_series.values()][:21], label="d1", color=d1_color, linestyle=d1_style)
# axes[0].plot(list(standard_pop_series.keys())[:21], [e[1] for e in standard_pop_series.values()][:21], label="d2", color=d2_color, linestyle=d2_style)
# axes[0].plot(list(standard_pop_series.keys())[:21], [e[2] for e in standard_pop_series.values()][:21], label="d3", color=d3_color, linestyle=d3_style)
# axes[0].plot(list(standard_pop_series.keys())[:21], [e[3] for e in standard_pop_series.values()][:21], label="d4", color=d4_color, linestyle=d4_style)
# axes[0].set_title("Populations under H")
# axes[0].set_xlabel('Time (fs)')
# axes[0].set_ylabel('Population')
# axes[0].legend()

# axes[1].plot(list(standard_pop_series.keys())[:21], [e[0] for e in effective_pop_series.values()][:21], label="d1", color=d1_color, linestyle=d1_style)
# axes[1].plot(list(standard_pop_series.keys())[:21], [e[1] for e in effective_pop_series.values()][:21], label="d2", color=d2_color, linestyle=d2_style)
# axes[1].plot(list(standard_pop_series.keys())[:21], [e[2] for e in effective_pop_series.values()][:21], label="d3", color=d3_color, linestyle=d3_style)
# axes[1].plot(list(standard_pop_series.keys())[:21], [e[3] for e in effective_pop_series.values()][:21], label="d4", color=d4_color, linestyle=d4_style)
# axes[1].set_title("Populations under effective H")
# axes[1].set_xlabel('Time (fs)')
# axes[1].set_ylabel('Population')
# axes[1].legend()

# axes[2].plot(error_time_series.keys(), [e[0] for e in error_time_series.values()], marker='x', label='d1', color=d1_color, linestyle=d1_style)
# axes[2].plot(error_time_series.keys(), [e[1] for e in error_time_series.values()], marker='o',label='d2', color=d2_color, linestyle=d2_style)
# axes[2].plot(error_time_series.keys(), [e[2] for e in error_time_series.values()], marker='s',label='d3', color=d3_color, linestyle=d3_style)
# axes[2].plot(error_time_series.keys(), [e[3] for e in error_time_series.values()], marker='v', label='d4', color=d4_color, linestyle=d4_style)
# axes[2].set_title("Population errors")
# axes[2].set_ylabel('Error')
# axes[2].set_xlabel('Time (fs)')
# axes[2].legend()

# plt.tight_layout()
# plt.savefig(f'no4a_{m}_mctdh_spf{n_spfs}')
