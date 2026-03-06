import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import copy


# CONFIGURATION
K              = 5      # Number of keyframes per individual
POP_SIZE       = 20     # Population size
N_GENERATIONS  = 30     # Number of generations
EVAL_EPISODES  = 1      # Episodes per fitness evaluation
TORQUE_MIN     = -1.0   # MuJoCo HalfCheetah action bounds
TORQUE_MAX     =  1.0
DURATION_MIN   = 10     # Min timesteps per keyframe
DURATION_MAX   = 50     # Max timesteps per keyframe
MUTATION_SIGMA = 0.15   # Std-dev for Gaussian mutation on torques
MUTATION_DUR   = 5      # Max integer perturbation on duration
CROSSOVER_PROB = 0.8    # Probability of performing crossover
TRUNCATION_K   = 10     # Top-K pool for truncation selection (50% of pop)
SEED           = 42

rng = np.random.default_rng(SEED)


def make_env(render=False):
    """Create HalfCheetah-v5. No render during training — much faster."""
    if render:
        return gym.make("HalfCheetah-v5", render_mode="human")
    return gym.make("HalfCheetah-v5")

# INDIVIDUAL REPRESENTATION
# Each individual = list of K numpy arrays, each of shape (7,):
#   [torque_0, torque_1, torque_2, torque_3, torque_4, torque_5, duration]
def random_individual():
    kf = []
    for _ in range(K):
        torques  = rng.uniform(TORQUE_MIN, TORQUE_MAX, size=6)
        duration = float(rng.integers(DURATION_MIN, DURATION_MAX + 1))
        kf.append(np.append(torques, duration))
    return kf


def starter_individual():
    """Seed the initial population with the provided starter keyframes."""
    return [
        np.array([-0.10,  0.05, -0.08,  0.12, -0.06,  0.03, 25.0]),
        np.array([ 0.20, -0.15,  0.30, -0.10,  0.25, -0.05, 40.0]),
        np.array([ 0.80, -0.40,  0.60, -0.30,  0.50, -0.20, 15.0]),
        np.array([ 0.50, -0.20,  0.40, -0.15,  0.30, -0.10, 20.0]),
        np.array([ 0.10,  0.00, -0.10,  0.05, -0.05,  0.02, 30.0]),
    ]

# FITNESS EVALUATION
def evaluate(individual, env, n_episodes=EVAL_EPISODES):
    """Run one episode with the keyframe sequence; return cumulative reward."""
    total_rewards = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        ki        = 0
        remaining = int(individual[ki][-1])
        ep_reward = 0.0
        done      = False
        while not done:
            action                         = individual[ki][:-1]
            obs, reward, term, trunc, info = env.step(action)
            ep_reward                     += reward
            remaining                     -= 1
            if remaining == 0:
                ki        = (ki + 1) % K
                remaining = int(individual[ki][-1])
            done = term or trunc
        total_rewards.append(ep_reward)
    return float(np.mean(total_rewards))


# PARENT SELECTION STRATEGIES
def binary_tournament(population, fitnesses):
    """
    Binary Tournament Selection (k=2).
    Randomly pick 2 individuals; the one with higher fitness wins.
    Low-to-moderate selection pressure; maintains diversity well.
    """
    i, j    = rng.choice(len(population), size=2, replace=False)
    winner  = i if fitnesses[i] >= fitnesses[j] else j
    return copy.deepcopy(population[winner])


def truncation_selection(population, fitnesses):
    """
    Truncation Selection.
    Only the top TRUNCATION_K individuals can be parents; sample uniformly.
    High selection pressure — fast convergence but risks losing diversity.
    """
    sorted_idx = np.argsort(fitnesses)[::-1]   # best → worst
    elite_idx  = sorted_idx[:TRUNCATION_K]     # top-K pool
    chosen     = rng.choice(elite_idx)
    return copy.deepcopy(population[chosen])


def fps_selection(population, fitnesses):
    """
    Fitness Proportionate Selection (Roulette Wheel).
    Selection probability proportional to fitness value.
    Moderate pressure; can be dominated by a few very fit individuals.
    """
    f     = np.array(fitnesses, dtype=float)
    f     = f - f.min() + 1e-6     # shift so all values are positive
    probs = f / f.sum()
    idx   = rng.choice(len(population), p=probs)
    return copy.deepcopy(population[idx])


def select_parent(population, fitnesses, method):
    if method == "tournament":
        return binary_tournament(population, fitnesses)
    elif method == "truncation":
        return truncation_selection(population, fitnesses)
    elif method == "fps":
        return fps_selection(population, fitnesses)
    else:
        raise ValueError(f"Unknown parent selection method: {method}")

# SURVIVOR SELECTION 
def survivor_truncation(population, fitnesses, offspring, off_fitnesses, n):
    """
    Truncation survivor selection (generational replacement with elitism):
    Merge parents + offspring, keep the best n individuals.
    """
    combined = population + offspring
    comb_fit = fitnesses  + off_fitnesses
    idx      = np.argsort(comb_fit)[::-1]
    return [combined[i] for i in idx[:n]], [comb_fit[i] for i in idx[:n]]


def survivor_tournament(population, fitnesses, offspring, off_fitnesses, n):
    """
    Tournament survivor selection:
    Run binary tournaments over the combined pool to fill the next generation.
    """
    combined = population + offspring
    comb_fit = fitnesses  + off_fitnesses
    new_pop, new_fit = [], []
    while len(new_pop) < n:
        i, j   = rng.choice(len(combined), size=2, replace=False)
        winner = i if comb_fit[i] >= comb_fit[j] else j
        new_pop.append(copy.deepcopy(combined[winner]))
        new_fit.append(comb_fit[winner])
    return new_pop, new_fit


def survivor_fps(population, fitnesses, offspring, off_fitnesses, n):
    """
    FPS survivor selection:
    Select next generation proportional to fitness from the combined pool.
    """
    combined = population + offspring
    comb_fit = fitnesses  + off_fitnesses
    f        = np.array(comb_fit, dtype=float)
    f        = f - f.min() + 1e-6
    probs    = f / f.sum()
    idxs     = rng.choice(len(combined), size=n, replace=False, p=probs)
    return [combined[i] for i in idxs], [comb_fit[i] for i in idxs]


def apply_survivor(population, fitnesses, offspring, off_fitnesses, n, method):
    if method == "truncation":
        return survivor_truncation(population, fitnesses, offspring, off_fitnesses, n)
    elif method == "tournament":
        return survivor_tournament(population, fitnesses, offspring, off_fitnesses, n)
    elif method == "fps":
        return survivor_fps(population, fitnesses, offspring, off_fitnesses, n)
    else:
        raise ValueError(f"Unknown survivor selection method: {method}")

# GENETIC OPERATORS
def single_point_crossover(p1, p2):
    """Single-point crossover at a random keyframe boundary."""
    if rng.random() > CROSSOVER_PROB:
        return copy.deepcopy(p1), copy.deepcopy(p2)
    pt = rng.integers(1, K)
    c1 = [copy.deepcopy(kf) for kf in p1[:pt] + p2[pt:]]
    c2 = [copy.deepcopy(kf) for kf in p2[:pt] + p1[pt:]]
    return c1, c2


def gaussian_mutation(individual, prob=0.3):
    """
    Per-gene Gaussian mutation on torques + integer mutation on duration.
    Each gene is mutated independently with probability `prob`.
    """
    ind = copy.deepcopy(individual)
    for kf in ind:
        for j in range(6):
            if rng.random() < prob:
                kf[j] = float(np.clip(
                    kf[j] + rng.normal(0, MUTATION_SIGMA),
                    TORQUE_MIN, TORQUE_MAX))
        if rng.random() < prob:
            kf[6] = float(np.clip(
                kf[6] + rng.integers(-MUTATION_DUR, MUTATION_DUR + 1),
                DURATION_MIN, DURATION_MAX))
    return ind


# FULL EA LOOP
def run_ea(parent_sel, survival_sel, label, verbose=True):
    if verbose:
        print(f"\n{'='*68}")
        print(f"  Experiment : {label}")
        print(f"  Parent sel : {parent_sel}  |  Survivor sel: {survival_sel}")
        print(f"{'='*68}")

    env = make_env(render=False)

    # Initialise population — include the provided starter keyframes
    population = [starter_individual()] + \
                 [random_individual() for _ in range(POP_SIZE - 1)]
    fitnesses  = [evaluate(ind, env) for ind in population]

    history       = []
    best_ever     = max(fitnesses)
    best_ever_ind = copy.deepcopy(population[int(np.argmax(fitnesses))])

    for gen in range(N_GENERATIONS):
        offspring, off_fitnesses = [], []

        # Generating offspring
        while len(offspring) < POP_SIZE:
            p1 = select_parent(population, fitnesses, parent_sel)
            p2 = select_parent(population, fitnesses, parent_sel)
            c1, c2 = single_point_crossover(p1, p2)
            offspring.extend([gaussian_mutation(c1), gaussian_mutation(c2)])

        # Evaluating offspring
        off_fitnesses = [evaluate(ind, env) for ind in offspring]

        # Survivor selection
        population, fitnesses = apply_survivor(
            population, fitnesses, offspring, off_fitnesses, POP_SIZE, survival_sel)

        gen_best = max(fitnesses)
        history.append(gen_best)
        if gen_best > best_ever:
            best_ever     = gen_best
            best_ever_ind = copy.deepcopy(population[int(np.argmax(fitnesses))])

        if verbose:
            print(f"  Gen {gen+1:3d}/{N_GENERATIONS}  "
                  f"best={gen_best:8.2f}  "
                  f"mean={np.mean(fitnesses):8.2f}  "
                  f"all-time best={best_ever:8.2f}")

    env.close()
    if verbose:
        print(f"\n  ✓ Done. Best reward: {best_ever:.2f}\n")
    return best_ever_ind, best_ever, history


# SAVE BEST KEYFRAMES
def save_keyframes(individual, filepath):
    with open(filepath, "w") as f:
        f.write(f"{len(individual)}\n")
        for kf in individual:
            vals  = ", ".join(f"{v:.4f}" for v in kf[:6])
            vals += f", {int(kf[6])}"
            f.write(vals + "\n")
    print(f"Best keyframes saved → {filepath}")


# CONVERGENCE PLOT
def plot_convergence(histories, labels, save_path=None):
    colors  = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    markers = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (hist, lbl) in enumerate(zip(histories, labels)):
        ax.plot(range(1, len(hist) + 1), hist,
                color=colors[i], marker=markers[i],
                markersize=4, linewidth=2.0, label=lbl, alpha=0.9)
    ax.set_xlabel("Generation", fontsize=13)
    ax.set_ylabel("Best Episode Reward", fontsize=13)
    ax.set_title(
        "Convergence Plot — EA Variants for HalfCheetah Keyframe Optimization",
        fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160)
        print(f"Convergence plot saved → {save_path}")
    plt.show()


# VISUALISE BEST SOLUTION
def visualise(individual, n_episodes=2):
    print("\nVisualising best keyframes (close window to continue)...")
    env = make_env(render=True)
    for ep in range(n_episodes):
        obs, info = env.reset()
        ki        = 0
        remaining = int(individual[ki][-1])
        ep_reward = 0.0
        done      = False
        while not done:
            action                         = individual[ki][:-1]
            obs, reward, term, trunc, info = env.step(action)
            ep_reward                     += reward
            env.render()
            remaining -= 1
            if remaining == 0:
                ki        = (ki + 1) % K
                remaining = int(individual[ki][-1])
            done = term or trunc
        print(f"  Episode {ep + 1}: reward = {ep_reward:.2f}")
    env.close()


# MAIN — 4 Experiments using the 3 selection strategies
if __name__ == "__main__":

    # 4 combinations of (parent selection, survivor selection)
    # Strategies used: Binary Tournament, Truncation, FPS 
    experiments = [
        # (parent_sel,   survival_sel,   label)
        ("truncation",  "truncation",  "Parent: Truncation     | Survivor: Truncation"),
        ("tournament",  "truncation",  "Parent: Bin.Tournament | Survivor: Truncation"),
        ("fps",         "truncation",  "Parent: FPS            | Survivor: Truncation"),
        ("tournament",  "fps",         "Parent: Bin.Tournament | Survivor: FPS"),
    ]

    all_histories       = []
    all_labels          = []
    results             = []
    global_best_fitness = -np.inf
    global_best_ind     = None
    global_best_label   = ""

    for (ps, ss, lbl) in experiments:
        best_ind, best_fit, hist = run_ea(
            parent_sel=ps, survival_sel=ss, label=lbl, verbose=True)
        all_histories.append(hist)
        all_labels.append(lbl)
        results.append((lbl, best_fit))
        if best_fit > global_best_fitness:
            global_best_fitness = best_fit
            global_best_ind     = best_ind
            global_best_label   = lbl

    #Final comparison table 
    print("\n" + "="*68)
    print("  FINAL RESULTS COMPARISON")
    print("="*68)
    print(f"  {'Configuration':<50} {'Best Reward':>10}")
    print("-"*68)
    for lbl, fit in results:
        mark = "  <- BEST" if lbl == global_best_label else ""
        print(f"  {lbl:<50} {fit:>10.2f}{mark}")
    print("="*68)
    print(f"\n  Overall best reward : {global_best_fitness:.2f}")
    print(f"  Best configuration  : {global_best_label}")

    save_keyframes(global_best_ind, "best_keyframes.txt")
    plot_convergence(all_histories, all_labels, save_path="convergence_plot.png")

    # # Uncomment to visually watch the best solution 
    # visualise(global_best_ind, n_episodes=2)

    print("\nAll done! Files saved: best_keyframes.txt, convergence_plot.png")
