import os
import pickle
import matplotlib.pyplot as plt
from rdcheck_pops import get_errors

# PARAMETERS
BASENAME = "no4a"
DIRNAME = "EXPS_NO4A"
NUM_STATE = 5
NUM_MODE = 3
DELTAT = 0.01
TMAX = 500
FRAG_NUMS = [1, 3, 5, 7]

colors = ['blue', 'red', 'orange', 'green', 'purple', 'brown', 'pink', 'cyan', 'olive', 'magenta']
styles = ['solid', 'dashed', 'dashdot', 'dotted', (0, (1, 1)), (0, (3, 1, 1, 1))]

for NUM in FRAG_NUMS:
    dir_path = f"./runs/{DIRNAME}/dt={DELTAT}"
    run_name = f"{BASENAME}_{NUM_STATE}s_{NUM_MODE}m_t{TMAX}_{NUM}_exact"

    try:
        error_time_series, std_pop_series, eff_pop_series = get_errors(
            dir=dir_path,
            system=run_name,
            n_states=NUM_STATE,
            return_pop_series=True,
            runrdcheck=False,
        )
    except Exception as e:
        print(f"Skipping NUM={NUM} due to error: {e}")
        continue

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

    # Plot Standard Populations
    for i in range(NUM_STATE):
        axes[0].plot(
            list(std_pop_series.keys())[:21],
            [e[i] for e in std_pop_series.values()][:21],
            label=f"S{i+1}",
            color=colors[i % len(colors)],
            linestyle=styles[i % len(styles)],
        )

    axes[0].set_title("Standard Populations")
    axes[0].set_xlabel("Time (fs)")
    axes[0].set_ylabel("Population")
    axes[0].legend()

    # Plot Effective Populations
    for i in range(NUM_STATE):
        axes[1].plot(
            list(eff_pop_series.keys())[:21],
            [e[i] for e in eff_pop_series.values()][:21],
            label=f"S{i+1}",
            color=colors[i % len(colors)],
            linestyle=styles[i % len(styles)],
        )

    axes[1].set_title("Effective Populations")
    axes[1].set_xlabel("Time (fs)")
    axes[1].set_ylabel("Population")
    axes[1].legend()

    # Plot Errors
    for i in range(NUM_STATE):
        axes[2].plot(
            error_time_series.keys(),
            [e[i] for e in error_time_series.values()],
            label=f"S{i+1}",
            color=colors[i % len(colors)],
            linestyle=styles[i % len(styles)],
        )

    axes[2].set_title("Population Errors")
    axes[2].set_xlabel("Time (fs)")
    axes[2].set_ylabel("Abs Error")
    axes[2].legend()

    fig.suptitle(f"no4a — M={NUM_MODE}, {NUM} Fragments, Δt={DELTAT} fs")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = f"plots/no4a_m{NUM_MODE}_F{NUM}_t{TMAX}.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Saved: {output_path}")

