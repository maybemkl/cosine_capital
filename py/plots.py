import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_word_dist(counts, p_type='bar'):
    zipf_df = (
        pd.DataFrame.from_dict(dict(counts.most_common(18)), orient='index')
        .reset_index()
        .rename(columns={'index': 'Word', 0: 'Count'})
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    plt.xticks(rotation=70)

    if p_type == 'bar':
        ax.bar(
            zipf_df["Word"],
            zipf_df["Count"],
            edgecolor="black",
            color="lightblue",
            fill=True,
            linewidth=1.5
        )
    elif p_type == 'line':
        ax.plot(
            zipf_df["Word"],
            zipf_df["Count"],
            color="blue",
            linewidth=1.5,
            marker="o",
            markersize=4,
            markerfacecolor="white"
        )

    # ticks and labels (kept same behavior)
    _, xlabels = plt.xticks()
    _, ylabels = plt.yticks()
    ax.set_xticklabels(xlabels, size=12)
    ax.set_yticklabels(ylabels, size=12)

    # axis labels
    ax.set_xlabel("Word", fontsize=15)
    ax.set_ylabel("Probability", fontsize=15)

    # grid and layout style (consistent with your other figures)
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_log(frac=False):
    if frac:
        x = 1 / np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
        y = np.log2(x)
        title = r"$\log_2$ of $x \leq 1$"
        xticklabels = [f"1/{int(1/i)}" if i != 1 else "1" for i in x]
    else:
        x = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
        y = np.log2(x)
        title = r"$\log_2$ of $x \geq 1$"
        xticklabels = [str(int(i)) for i in x]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        x, y,
        color="orange",
        linewidth=2,
        marker="o",
        markersize=5,
        markerfacecolor="white"
    )

    # --- selective tick labeling ---
    ax.set_xticks(x)
    if frac:
        keep_labels = ["1", "1/2", "1/4", "1/8", "1/16", "1/256"]
    else:
        keep_labels = ["1", "8", "16", "32", "64", "128", "256"]
    ax.set_xticklabels([lbl if lbl in keep_labels else "" for lbl in xticklabels])

    # y-axis ticks
    ax.set_yticks(np.arange(int(min(y)), int(max(y)) + 1))
    ax.set_yticklabels([str(i) for i in np.arange(int(min(y)), int(max(y)) + 1)])

    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel(r"$\log_2 x$", fontsize=16)
    ax.set_title(title, fontsize=20, pad=10)

    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.show()
    
def compare_distributions_in_plot(outcomes,
                                  first_dist, 
                                  second_dist,
                                  first_label,
                                  second_label,
                                  xlabel="Outcomes",
                                  ylabel="Probability",
                                  title="Two distributions"):
    
    x_ticks = list(range(0, len(outcomes)))
    fig, ax = plt.subplots(figsize=(8, 4))
    
    x_pos = np.arange(len(outcomes))
    width = 0.35
    
    ax.bar(
        x_pos - width/2,
        first_dist,
        width,
        label=first_label,
        edgecolor="black",
        color="lightblue",
        fill=True,
        linewidth=1.5
    )
    ax.bar(
        x_pos + width/2,
        second_dist,
        width,
        label=second_label,
        edgecolor="black",
        color="orange",
        fill=True,
        linewidth=1.5
    )
    
    # ticks and labels
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(outcomes, size=12)
    
    # y-axis ticks at 0.1 intervals
    all_values = list(first_dist) + list(second_dist)
    y_min = min(all_values) if len(all_values) > 0 else 0
    y_max = max(all_values) if len(all_values) > 0 else 1
    y_min = max(0, np.floor(y_min * 10) / 10)
    y_max = min(1, np.ceil(y_max * 10) / 10)
    y_ticks = np.arange(y_min, y_max + 0.1, 0.1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y:.1f}" for y in y_ticks], size=12)
    
    # axis labels
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    # legend and title
    ax.legend(fontsize=12)
    ax.set_title(title, fontsize=16)
    
    # grid and layout style
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_bernoulli(counts):
    fig, ax = plt.subplots(figsize=(8, 6))

    outcomes = ['heads', 'tails']
    bar_labels = ['heads', 'tails']
    bar_colors = ['tab:red', 'tab:blue']

    ax.bar(outcomes, counts, label=bar_labels, color=bar_colors)
    ax.set_xlabel('Outcome', fontsize=16)
    ax.set_ylabel('Number of tosses', fontsize=16)
    ax.set_title(f'Results from {sum(counts)} coin tosses.', fontsize=20)
    ax.legend(title='Coin toss', fontsize=12, title_fontsize=14)
   
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.show()
    
def plot_discrete_distribution(df, x_col, title, suptitle, x_lab, y_lab):
    p = sns.displot(data=df, x=x_col, kind="hist", height=6, aspect=1, discrete=True)

    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.title(title, fontsize=16)
    plt.suptitle(suptitle)
    plt.xlabel(x_lab,fontsize=14)
    plt.ylabel(y_lab,fontsize=14)
    plt.show()
    
def plot_entropy(coin_entropies, max_entr=False):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(np.linspace(0, 1, 100), coin_entropies, color="blue")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_xlabel("Probability of heads", fontsize=14)
    ax.set_ylabel("Entropy (average bits)", fontsize=14)
    ax.locator_params(axis="both", nbins=20)
    ax.grid()

    if max_entr:
        ax.plot([0.5], [1], "ro", markersize=8)
        ax.annotate("max entropy", (0.5, 1), textcoords="offset points",
                    xytext=(0, -20), ha="center")

    # left-aligned titles tied to axes coordinates
    ax.text(
        0.0, 1.08, "Entropy of a coin toss",
        ha="left", va="bottom", transform=ax.transAxes,
        fontsize=16
    )
    ax.text(
        0.0, 1.03,
        "I.e. how many bits do we need on average to encode the distribution",
        ha="left", va="bottom", transform=ax.transAxes,
        fontsize=12
    )

    plt.tight_layout()
    plt.show()

def plot_symbol_probs(uniform_probs, language):
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    
    # histogram-style bars (outline only, no fill)
    ax.bar(
        range(len(uniform_probs)), 
        uniform_probs, 
        edgecolor="black", 
        color="lightblue",
        fill=True, 
        linewidth=1.5
    )
    
    ax.set_xticks(range(len(language)))
    ax.set_xticklabels(language, fontsize=12)
    ax.set_ylabel("Probability", fontsize=14)
    ax.set_xlabel("Language", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.6)
    
    plt.tight_layout()
    plt.show()

def plot_probability_sequence(p_heads_interval):
    """
    Plot probability values over a sequence.
    
    Parameters:
    -----------
    p_heads_interval : list or array
        Probability values to plot
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    x_values = list(range(1, len(p_heads_interval) + 1))
    ax.bar(
        x_values, 
        p_heads_interval, 
        edgecolor="black",
        color="lightblue",
        fill=True,
        linewidth=1.5
    )
    
    # Set ticks for each observation
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_values, size=12)
    
    # y-axis ticks at 0.1 intervals
    y_min = min(p_heads_interval) if len(p_heads_interval) > 0 else 0
    y_max = max(p_heads_interval) if len(p_heads_interval) > 0 else 1
    y_min = max(0, np.floor(y_min * 10) / 10)
    y_max = min(1, np.ceil(y_max * 10) / 10)
    y_ticks = np.arange(y_min, y_max + 0.1, 0.1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y:.1f}" for y in y_ticks], size=12)
    
    # axis labels
    ax.set_xlabel("Coin", fontsize=15)
    ax.set_ylabel("P(H)", fontsize=15)
    
    # grid and layout style
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.show()