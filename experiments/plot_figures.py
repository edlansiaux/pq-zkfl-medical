"""
Generate publication-quality figures for the manuscript.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

# Style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'standard': '#E74C3C',   # Red
    'mlkem': '#F39C12',      # Orange
    'hybrid': '#2ECC71',     # Green
}

LABELS = {
    'standard': 'Standard FL (TLS)',
    'mlkem': 'FL + ML-KEM',
    'hybrid': 'FL + ML-KEM + ZKP + HE (Ours)',
}


def load_results(path):
    with open(path) as f:
        return json.load(f)


def fig1_accuracy_convergence(results, output_dir):
    """Figure 1: Accuracy across FL rounds for all three configurations."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    rounds = np.arange(1, 11)

    for name in ['standard', 'mlkem', 'hybrid']:
        acc = results[name]['accuracies']
        ax.plot(rounds, acc, 'o-', color=COLORS[name], label=LABELS[name],
                linewidth=2, markersize=5)

    # Mark malicious activation
    ax.axvline(x=4, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.annotate('Malicious client\nactivated',
                xy=(4, 0.55), fontsize=9, color='gray', ha='center',
                style='italic')

    ax.set_xlabel('Federated Learning Round')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Model Accuracy Under Byzantine Attack')
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlim(0.5, 10.5)
    ax.legend(loc='center right')
    ax.set_xticks(rounds)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig1_accuracy.pdf'))
    fig.savefig(os.path.join(output_dir, 'fig1_accuracy.png'))
    plt.close()
    print("  ✓ Figure 1: Accuracy convergence")


def fig2_loss_convergence(results, output_dir):
    """Figure 2: Training loss convergence."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    rounds = np.arange(1, 11)

    for name in ['standard', 'mlkem', 'hybrid']:
        loss = results[name]['losses']
        ax.plot(rounds, loss, 's-', color=COLORS[name], label=LABELS[name],
                linewidth=2, markersize=5)

    ax.axvline(x=4, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.annotate('Byzantine attack onset',
                xy=(4, max(results['standard']['losses']) * 0.7),
                fontsize=9, color='gray', ha='center', style='italic')

    ax.set_xlabel('Federated Learning Round')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Training Loss Under Byzantine Attack')
    ax.legend(loc='upper left')
    ax.set_xticks(rounds)
    ax.set_yscale('log')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig2_loss.pdf'))
    fig.savefig(os.path.join(output_dir, 'fig2_loss.png'))
    plt.close()
    print("  ✓ Figure 2: Loss convergence")


def fig3_timing_breakdown(results, output_dir):
    """Figure 3: Per-round timing breakdown."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    configs = ['standard', 'mlkem', 'hybrid']
    titles = ['Standard FL', 'FL + ML-KEM', 'Hybrid (Ours)']
    rounds = np.arange(1, 11)

    for ax, name, title in zip(axes, configs, titles):
        times = results[name]['round_times']
        ax.bar(rounds, times, color=COLORS[name], alpha=0.8, edgecolor='white')
        ax.set_xlabel('Round')
        ax.set_ylabel('Time (s)')
        ax.set_title(title)
        ax.set_xticks(rounds)
        mean_t = np.mean(times)
        ax.axhline(y=mean_t, color='black', linestyle='--', alpha=0.4)
        ax.text(10.3, mean_t, f'μ={mean_t:.2f}s', va='center', fontsize=9)

    fig.suptitle('Per-Round Computation Time', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig3_timing.pdf'))
    fig.savefig(os.path.join(output_dir, 'fig3_timing.png'))
    plt.close()
    print("  ✓ Figure 3: Timing breakdown")


def fig4_security_radar(results, output_dir):
    """Figure 4: Security posture radar chart."""
    categories = [
        'Quantum\nResistance',
        'Byzantine\nDetection',
        'Gradient\nPrivacy',
        'Channel\nConfidentiality',
        'Verifiability',
        'Long-term\nSecurity'
    ]

    # Scores (0-10) for each configuration
    scores = {
        'standard': [0, 0, 3, 6, 0, 0],
        'mlkem':    [10, 0, 3, 10, 0, 10],
        'hybrid':   [10, 10, 10, 10, 10, 10],
    }

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(polar=True))

    for name in ['standard', 'mlkem', 'hybrid']:
        values = scores[name] + scores[name][:1]
        ax.plot(angles, values, 'o-', color=COLORS[name], label=LABELS[name],
                linewidth=2, markersize=4)
        ax.fill(angles, values, alpha=0.1, color=COLORS[name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 11)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_yticklabels(['0', '2', '4', '6', '8', '10'], size=8)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
    ax.set_title('Security Posture Comparison', pad=20)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig4_security_radar.pdf'))
    fig.savefig(os.path.join(output_dir, 'fig4_security_radar.png'))
    plt.close()
    print("  ✓ Figure 4: Security radar")


def fig5_communication_overhead(results, output_dir):
    """Figure 5: Communication overhead comparison."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    configs = ['standard', 'mlkem', 'hybrid']
    means = [np.mean(results[c]['message_sizes_kb']) for c in configs]
    labels_short = ['Standard FL\n(TLS)', 'FL +\nML-KEM', 'Hybrid\n(Ours)']

    bars = ax.bar(labels_short, means,
                  color=[COLORS[c] for c in configs],
                  edgecolor='white', linewidth=1.5, width=0.5)

    # Add value labels
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                f'{val:.0f} KB', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('Mean Message Size per Round (KB)')
    ax.set_title('Communication Overhead per FL Round')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig5_communication.pdf'))
    fig.savefig(os.path.join(output_dir, 'fig5_communication.png'))
    plt.close()
    print("  ✓ Figure 5: Communication overhead")


def fig6_hybrid_breakdown(results, output_dir):
    """Figure 6: Detailed timing breakdown for hybrid protocol."""
    hybrid = results['hybrid']
    rounds = np.arange(1, 11)

    fig, ax = plt.subplots(figsize=(8, 5))

    zkp_gen = hybrid['zkp_gen_times']
    zkp_ver = hybrid['zkp_ver_times']
    he_enc = hybrid['he_encrypt_times']
    he_agg = hybrid['he_aggregate_times']
    he_dec = hybrid['he_decrypt_times']
    total = hybrid['round_times']

    # Compute "other" (local training + ML-KEM)
    other = [t - z1 - z2 - h1 - h2 - h3
             for t, z1, z2, h1, h2, h3
             in zip(total, zkp_gen, zkp_ver, he_enc, he_agg, he_dec)]

    ax.bar(rounds, other, label='Local Training + ML-KEM', color='#3498DB', alpha=0.8)
    bottom = np.array(other)

    ax.bar(rounds, he_enc, bottom=bottom, label='HE Encryption', color='#9B59B6', alpha=0.8)
    bottom += np.array(he_enc)

    ax.bar(rounds, he_agg, bottom=bottom, label='HE Aggregation', color='#1ABC9C', alpha=0.8)
    bottom += np.array(he_agg)

    ax.bar(rounds, he_dec, bottom=bottom, label='HE Decryption', color='#E67E22', alpha=0.8)
    bottom += np.array(he_dec)

    ax.bar(rounds, zkp_gen, bottom=bottom, label='ZKP Generation', color='#E74C3C', alpha=0.8)
    bottom += np.array(zkp_gen)

    ax.bar(rounds, zkp_ver, bottom=bottom, label='ZKP Verification', color='#F1C40F', alpha=0.8)

    ax.set_xlabel('Federated Learning Round')
    ax.set_ylabel('Time (s)')
    ax.set_title('Hybrid Protocol: Computation Time Breakdown')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xticks(rounds)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig6_breakdown.pdf'))
    fig.savefig(os.path.join(output_dir, 'fig6_breakdown.png'))
    plt.close()
    print("  ✓ Figure 6: Hybrid breakdown")


def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    results = load_results(os.path.join(results_dir, 'experiment_results.json'))

    print("Generating publication figures...")
    fig1_accuracy_convergence(results, figures_dir)
    fig2_loss_convergence(results, figures_dir)
    fig3_timing_breakdown(results, figures_dir)
    fig4_security_radar(results, figures_dir)
    fig5_communication_overhead(results, figures_dir)
    fig6_hybrid_breakdown(results, figures_dir)
    print("Done! Figures saved to:", figures_dir)


if __name__ == '__main__':
    main()
