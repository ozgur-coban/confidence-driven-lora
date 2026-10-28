import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Professional aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})

INPUT_DIR = "./data/full_run_results/final_merged_results"
OUTPUT_DIR = "./data/full_run_results/graphs"

# Clean presentation labels
DATASET_MAP = {
    'gsm8k': 'GSM8K',
    'codefeedback': 'CodeFeedback',
    'hybrid': 'Hybrid',
    'mixed': 'Mixed',
    'wildchat': 'WildChat'
}

CONFIG_MAP = {
    'base': 'Base',
    'pure_math': 'Pure Math',
    'pure_code': 'Pure Code',
    'hybrid_adapter': 'TIES-Merged',
    'routed': 'Routed'
}

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
    
    # Apply professional labels
    df['dataset'] = df['dataset'].map(DATASET_MAP).fillna(df['dataset'])
    df['configuration'] = df['configuration'].map(CONFIG_MAP).fillna(df['configuration'])
    
    return df

def plot_catastrophic_forgetting_heatmap(df: pd.DataFrame):
    pivot_df = df.pivot_table(
        index='dataset', 
        columns='configuration', 
        values='execution_correct', 
        aggfunc='mean'
    ) * 100

    config_order = ['Base', 'Pure Math', 'Pure Code', 'TIES-Merged', 'Routed']
    dataset_order = ['GSM8K', 'CodeFeedback', 'Hybrid', 'Mixed', 'WildChat']
    pivot_df = pivot_df.reindex(index=dataset_order, columns=config_order)

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Accuracy (%)'})
    plt.title("Performance Accuracy Across Domains & Configurations", pad=20, fontsize=16, fontweight='bold')
    plt.ylabel("Dataset (Evaluation Tier)", fontweight='bold')
    plt.xlabel("Model Configuration", fontweight='bold')
    
    plt.savefig(os.path.join(OUTPUT_DIR, "1_accuracy_heatmap.png"), dpi=300)
    plt.close()

def plot_router_efficacy(df: pd.DataFrame):
    routed_df = df[df['configuration'] == 'Routed'].copy()
    
    # 1. Routing Accuracy Bar Chart
    task_df = routed_df[routed_df['dataset'] != 'WildChat']
    routing_acc = task_df.groupby('dataset')['routing_correct'].mean() * 100
    
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=routing_acc.index, y=routing_acc.values, palette="viridis")
    plt.title("MetaRouter Classification Accuracy", pad=15, fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=11, xytext=(0, 5), textcoords='offset points')
    plt.savefig(os.path.join(OUTPUT_DIR, "2a_router_accuracy.png"), dpi=300)
    plt.close()

    # 2. WildChat Routing Distribution Pie Chart
    wildchat_routed = routed_df[routed_df['dataset'] == 'WildChat']
    dist = wildchat_routed['routed_domain'].value_counts()
    # Map back domain keys to clean names for the pie chart
    clean_dist_index = [d.capitalize() for d in dist.index]
    
    plt.figure(figsize=(7, 7))
    plt.pie(dist.values, labels=clean_dist_index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    plt.title("Router Distribution on Noisy Real-World Prompts (WildChat)", pad=15, fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, "2b_wildchat_routing_distribution.png"), dpi=300)
    plt.close()

def plot_latency_and_throughput(df: pd.DataFrame):
    core_df = df[df['dataset'] != 'WildChat']
    metrics = core_df.groupby('configuration')[['tps', 'adapter_switch_ms']].mean()
    
    config_order = ['Base', 'Pure Math', 'Pure Code', 'TIES-Merged', 'Routed']
    metrics = metrics.reindex(config_order)
    
    # Nullify Base model's artificial switch latency so it doesn't plot
    metrics.loc['Base', 'adapter_switch_ms'] = np.nan

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart for TPS
    color = 'tab:blue'
    ax1.set_xlabel('Model Configuration', fontweight='bold')
    ax1.set_ylabel('Throughput (Tokens / Second)', color=color, fontweight='bold')
    bars = ax1.bar(metrics.index, metrics['tps'], color=color, alpha=0.7, label="TPS")
    ax1.tick_params(axis='y', labelcolor=color)

    # Line chart for Switch MS overlay
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Adapter Switch Latency (ms)', color=color, fontweight='bold')
    # Filter out NaNs for the line plot to avoid broken lines
    valid_ms = metrics['adapter_switch_ms'].dropna()
    ax2.plot(valid_ms.index, valid_ms.values, color=color, marker='o', linewidth=2, markersize=8, label="Switch Latency")
    ax2.tick_params(axis='y', labelcolor=color)
    
    max_ms = valid_ms.max() if not valid_ms.empty else 5
    ax2.set_ylim(0, max_ms * 1.5)

    plt.title("System Throughput vs. Dynamic Routing Overhead", pad=20, fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, "3_throughput_latency.png"), dpi=300)
    plt.close()

def plot_edge_cases(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    config_order = ['Base', 'Pure Math', 'Pure Code', 'TIES-Merged', 'Routed']

    hybrid_df = df[df['dataset'] == 'Hybrid'].groupby('configuration')['execution_correct'].mean() * 100
    hybrid_df = hybrid_df.reindex(config_order)
    sns.barplot(x=hybrid_df.index, y=hybrid_df.values, ax=axes[0], palette="magma")
    axes[0].set_title("Performance on Dual-Intent Prompts (Hybrid)", fontweight='bold')
    axes[0].set_ylabel("Execution Accuracy (%)")
    axes[0].tick_params(axis='x', rotation=45)
    
    wildchat_df = df[df['dataset'] == 'WildChat'].groupby('configuration')['execution_correct'].mean() * 100
    wildchat_df = wildchat_df.reindex(config_order)
    sns.barplot(x=wildchat_df.index, y=wildchat_df.values, ax=axes[1], palette="crest")
    axes[1].set_title("Real-World Robustness (Noisy WildChat Prompts)", fontweight='bold')
    axes[1].set_ylabel("Execution Accuracy (%)")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_edge_cases.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_and_clean_data(INPUT_DIR)
    
    plot_catastrophic_forgetting_heatmap(df)
    plot_router_efficacy(df)
    plot_latency_and_throughput(df)
    plot_edge_cases(df)
    print(f"✅ Professional graphs saved to '{OUTPUT_DIR}/'")