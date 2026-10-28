import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# STRICT ACADEMIC AESTHETICS
# ==========================================
sns.set_theme(style="white", context="paper")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.autolayout': True,
})

PRIMARY_COLOR = '#3b5b92'
SECONDARY_COLOR = '#b23535'

INPUT_DIR = "./data/full_run_results/final_merged_results"
OUTPUT_DIR = "./data/full_run_results/graphs"

DATASET_MAP = {
    'gsm8k': 'GSM8K',
    'codefeedback': 'CodeFeedback',
    'hybrid': 'Hybrid',
    'mixed': 'Mixed',
    'wildchat': 'WildChat'
}

CONFIG_MAP = {
    'base': 'Base Model',
    'pure_math': 'Math Adapter',
    'pure_code': 'Code Adapter',
    'hybrid_adapter': 'TIES Merged',
    'routed': 'Routed System'
}

def save_academic_figure(filename_base):
    """Saves both a PNG for viewing and a high-res PDF for the paper."""
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename_base}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename_base}.pdf"), format='pdf', bbox_inches='tight')

def load_and_clean_data(input_dir: str) -> pd.DataFrame:
    all_records = []
    for f in os.listdir(input_dir):
        if f.endswith(".json"):
            with open(os.path.join(input_dir, f), 'r', encoding='utf-8') as file:
                data = json.load(file)
                items = data if isinstance(data, list) else [data]
                all_records.extend(items)
    
    df = pd.DataFrame(all_records)
    df['execution_correct'] = df['execution_correct'].fillna(False).astype(float)
    df['dataset'] = df['dataset'].map(DATASET_MAP).fillna(df['dataset'])
    df['configuration'] = df['configuration'].map(CONFIG_MAP).fillna(df['configuration'])
    return df

def plot_accuracy_heatmap(df: pd.DataFrame):
    pivot_df = df.pivot_table(
        index='dataset', 
        columns='configuration', 
        values='execution_correct', 
        aggfunc='mean'
    ) * 100

    config_order = ['Base Model', 'Math Adapter', 'Code Adapter', 'TIES Merged', 'Routed System']
    dataset_order = ['GSM8K', 'CodeFeedback', 'Hybrid', 'Mixed', 'WildChat']
    pivot_df = pivot_df.reindex(index=dataset_order, columns=config_order)

    plt.figure(figsize=(9, 5))
    ax = sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="Blues", cbar_kws={'label': 'Accuracy (%)'})
    
    plt.ylabel("Dataset", fontweight='bold')
    plt.xlabel("Model Configuration", fontweight='bold')
    
    # FIX: Rotate x-axis labels by 45 degrees so they don't overlap
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    save_academic_figure("1_accuracy_heatmap")
    plt.close()

def plot_router_efficacy(df: pd.DataFrame):
    routed_df = df[df['configuration'] == 'Routed System'].copy()
    
    task_df = routed_df[routed_df['dataset'] != 'WildChat']
    routing_acc = task_df.groupby('dataset')['routing_correct'].mean() * 100
    
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(x=routing_acc.index, y=routing_acc.values, color=PRIMARY_COLOR)
    
    plt.ylabel("Accuracy (%)", fontweight='bold')
    plt.xlabel("Dataset", fontweight='bold')
    plt.ylim(0, 100)
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=11, xytext=(0, 3), textcoords='offset points')
                    
    save_academic_figure("2a_router_accuracy")
    plt.close()

    wildchat_routed = routed_df[routed_df['dataset'] == 'WildChat']
    dist = wildchat_routed['routed_domain'].value_counts()
    clean_dist_index = [d.capitalize() for d in dist.index]
    
    plt.figure(figsize=(6, 6))
    colors = ['#d1e5f0', '#67a9cf', '#2166ac', '#fddbc7']
    plt.pie(dist.values, labels=clean_dist_index, autopct='%1.1f%%', startangle=140, colors=colors, 
            wedgeprops={'edgecolor': 'black', 'linewidth': 1})
            
    save_academic_figure("2b_wildchat_routing_distribution")
    plt.close()

def plot_latency_and_throughput(df: pd.DataFrame):
    core_df = df[df['dataset'] != 'WildChat']
    metrics = core_df.groupby('configuration')[['tps', 'adapter_switch_ms']].mean()
    
    config_order = ['Base Model', 'Math Adapter', 'Code Adapter', 'TIES Merged', 'Routed System']
    metrics = metrics.reindex(config_order)
    metrics.loc['Base Model', 'adapter_switch_ms'] = np.nan

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.set_xlabel('Model Configuration', fontweight='bold')
    ax1.set_ylabel('Throughput (Tokens per Second)', color=PRIMARY_COLOR, fontweight='bold')
    bars = ax1.bar(metrics.index, metrics['tps'], color=PRIMARY_COLOR, alpha=0.8, label="TPS")
    ax1.tick_params(axis='y', labelcolor=PRIMARY_COLOR)
    
    # FIX: Rotate x-axis labels
    ax1.set_xticklabels(metrics.index, rotation=45, ha='right')

    ax2 = ax1.twinx()  
    ax2.set_ylabel('Adapter Switch Latency (ms)', color=SECONDARY_COLOR, fontweight='bold')
    valid_ms = metrics['adapter_switch_ms'].dropna()
    ax2.plot(valid_ms.index, valid_ms.values, color=SECONDARY_COLOR, marker='s', linewidth=2.5, markersize=8, label="Switch Latency")
    ax2.tick_params(axis='y', labelcolor=SECONDARY_COLOR)
    
    max_ms = valid_ms.max() if not valid_ms.empty else 5
    ax2.set_ylim(0, max_ms * 1.5)

    save_academic_figure("3_throughput_latency")
    plt.close()

def plot_hybrid_performance(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    config_order = ['Base Model', 'Math Adapter', 'Code Adapter', 'TIES Merged', 'Routed System']

    hybrid_df = df[df['dataset'] == 'Hybrid'].groupby('configuration')['execution_correct'].mean() * 100
    hybrid_df = hybrid_df.reindex(config_order)
    
    ax = sns.barplot(x=hybrid_df.index, y=hybrid_df.values, color=PRIMARY_COLOR)
    plt.ylabel("Accuracy (%)", fontweight='bold')
    plt.xlabel("Model Configuration", fontweight='bold')
    
    # FIX: Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=11, xytext=(0, 3), textcoords='offset points')

    save_academic_figure("4a_hybrid_performance")
    plt.close()

def plot_wildchat_performance(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    config_order = ['Base Model', 'Math Adapter', 'Code Adapter', 'TIES Merged', 'Routed System']
    
    wildchat_df = df[df['dataset'] == 'WildChat'].groupby('configuration')['execution_correct'].mean() * 100
    wildchat_df = wildchat_df.reindex(config_order)
    
    ax = sns.barplot(x=wildchat_df.index, y=wildchat_df.values, color=PRIMARY_COLOR)
    plt.ylabel("Accuracy (%)", fontweight='bold')
    plt.xlabel("Model Configuration", fontweight='bold')
    
    # FIX: Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=11, xytext=(0, 3), textcoords='offset points')

    save_academic_figure("4b_wildchat_performance")
    plt.close()

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_and_clean_data(INPUT_DIR)
    
    plot_accuracy_heatmap(df)
    plot_router_efficacy(df)
    plot_latency_and_throughput(df)
    plot_hybrid_performance(df)
    plot_wildchat_performance(df)
    
    print(f"Academic graphs (PNG and PDF) saved to '{OUTPUT_DIR}'")