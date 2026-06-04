import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 9
plt.rcParams['figure.dpi'] = 150

# ============================================================
# GRAPHIC 1: METHOD EVOLUTION TIMELINE TREE
# ============================================================
fig1, ax1 = plt.subplots(figsize=(16, 10))
ax1.set_xlim(0, 14)
ax1.set_ylim(0, 11)
ax1.axis('off')
ax1.set_title("Evolution of Game AI Methods in Imperfect-Information Card Games\n(2017–2026)", 
              fontsize=14, fontweight='bold', pad=20)

# Root
root = (6.5, 9.5)
ax1.add_patch(FancyBboxPatch((root[0]-1.2, root[1]-0.4), 2.4, 0.8, 
                              boxstyle="round,pad=0.05", facecolor='#2E86AB', edgecolor='black', linewidth=2))
ax1.text(root[0], root[1], "DouZero (2022)\nDMC + Self-Play", ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Branches
branches = {
    'ResNet Variants': (1.5, 8.0, ['#6', 'Zhang 2023']),
    'Pruning/Optimization': (4.0, 8.0, ['#10, #19', 'Luo et al. 2023-24']),
    'Opponent Modeling': (6.5, 8.0, ['#4, #8', 'Zhao 2022, Sun 2023']),
    'PPO/Policy Gradient': (9.0, 8.0, ['#5, #13, #29', 'Guo 2025, Zhao 2024']),
    'LLM/Transformer': (11.5, 8.0, ['#3, #23, #26, #30, #37', 'Wang 2025, Li 2024']),
}

for label, (x, y, refs) in branches.items():
    ax1.plot([root[0], x], [root[1]-0.4, y+0.3], 'k-', linewidth=1.5, alpha=0.7)
    ax1.add_patch(FancyBboxPatch((x-1.0, y-0.35), 2.0, 0.7, 
                                  boxstyle="round,pad=0.05", facecolor='#A23B72', edgecolor='black', linewidth=1.5))
    ax1.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    ax1.text(x, y-0.3, ', '.join(refs), ha='center', va='center', fontsize=7, color='white', alpha=0.9)

# Sub-branches
subs = [
    (1.5, 6.5, "ResNet + Call Scoring (#6)"),
    (4.0, 6.5, "MSP (#10) | ODMC (#19) | Multi-DMC (#18)"),
    (6.5, 6.5, "OM Network (#4) | Role-Diff LSTM/CBAM (#8)"),
    (9.0, 6.5, "ISPPO (#5) | ACC (#29) | JRPO (#36)"),
    (11.5, 6.5, "DouMH (#37) | Tjong (#30) | SFT (#26)"),
]
for x, y, text in subs:
    ax1.plot([x, x], [7.65, 6.85], 'k--', linewidth=1, alpha=0.5)
    ax1.text(x, y, text, ha='center', va='center', fontsize=7, style='italic', 
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#F9F9F9', alpha=0.7))

# Legend
legend_elements = [mpatches.Patch(facecolor='#2E86AB', label='Root Method (DMC)'),
                   mpatches.Patch(facecolor='#A23B72', label='Major Branch'),
                   mpatches.Patch(facecolor='#F9F9F9', label='Specific Paper/Technique')]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('method_evolution_tree.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# GRAPHIC 2: BUBBLE CHART (Efficiency vs Performance)
# ============================================================
fig2, ax2 = plt.subplots(figsize=(12, 8))

# Data: (training_efficiency, win_rate_improvement, size, label, color)
papers = [
    (0.5, 0.4, 120, "#10 DouZero (MSP)", '#1f77b4'),      # baseline
    (0.85, 0.75, 250, "#19 ODMC\n(75% faster)", '#2ca02c'),
    (0.6, 0.85, 280, "#5 ISPPO\n(75% faster decision)", '#d62728'),
    (0.4, 0.9, 300, "#20 Concurrent+Multistage\n(SOTA AGS)", '#9467bd'),
    (0.55, 0.7, 200, "#13 DanZero+ (DMC→PPO)", '#ff7f0e'),
    (0.7, 0.6, 180, "#15 OADMCDou\n(Oracle Guiding)", '#17becf'),
    (0.3, 0.55, 150, "#4 DouZero+ (OM+Coach)", '#bcbd22'),
    (0.75, 0.5, 140, "#27 SQUNO (Single Q)", '#e377c2'),
    (0.5, 0.8, 220, "#29 ACC\n(Actor-Cross-Critic)", '#7f7f7f'),
]

for x, y, s, label, color in papers:
    ax2.scatter(x, y, s=s, c=color, alpha=0.7, edgecolors='black', linewidth=1)
    ax2.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', 
                 fontsize=7, ha='left', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

# Reference lines
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='DouZero Baseline (approx)')
ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

ax2.set_xlabel("Training Efficiency (relative, higher = faster/cheaper)", fontsize=11)
ax2.set_ylabel("Win Rate / Performance Improvement", fontsize=11)
ax2.set_title("Performance vs Training Efficiency\n(Size ∝ Impact/Novelty)", fontsize=13, fontweight='bold')
ax2.set_xlim(0, 1)
ax2.set_ylim(0.3, 1.0)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig('bubble_chart_efficiency_vs_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# GRAPHIC 3: METHOD CATEGORY HEATMAP (Papers vs Techniques)
# ============================================================
fig3, ax3 = plt.subplots(figsize=(14, 10))

# Define method categories
categories = [
    'DMC / DouZero\nLineage', 'PPO / Policy\nGradient', 'LLM /\nTransformer', 
    'Opponent /\nRole Modeling', 'Bidding /\nEnd-to-End', 'Reward\nShaping',
    'Action Space\nPruning', 'Fictitious Play\n/ Nash', 'Hybrid /\nOther'
]

# Papers 1-37, but mapping to key representatives
# Matrix: rows = papers (simplified), cols = categories (1 if method used)
paper_names = [
    "#1 Survey", "#4 DouZero+", "#5 ISPPO", "#6 DouRN", "#8 DouRD", "#9 AlphaDou",
    "#10 MSP", "#11 DMC Bidding", "#12 DanZero", "#13 DanZero+", "#14 2-Player",
    "#15 OADMCDou", "#16 Action Repr", "#18 Multi-DMC", "#19 ODMC", "#20 CT-MS3",
    "#21 WagerWin", "#22 DQN-IRL", "#23 LLM+Data", "#24 Two-steps+IRL", 
    "#25 BidResNet", "#26 LLM4Card", "#27 SQUNO", "#29 ACC", "#30 Tjong",
    "#32 Shanwin", "#33 TiDou", "#35 DeepBayes", "#36 TiZero", "#37 DouMH"
]

# Method presence matrix (manual coding based on summary)
# Format: [DMC, PPO, LLM, OppModel, Bidding, RewardShaping, ActionPruning, FictPlay, Hybrid]
method_matrix = [
    [1,0,0,0,0,0,0,0,0],  # #1 Survey
    [1,0,0,1,0,0,0,0,0],  # #4 DouZero+ (OM)
    [0,1,0,0,0,1,0,0,0],  # #5 ISPPO (Reward shaping)
    [1,0,0,0,1,0,0,0,0],  # #6 DouRN (ResNet + call scoring)
    [0,0,0,1,0,0,0,0,0],  # #8 DouRD (role-diff)
    [0,0,0,0,1,0,0,0,0],  # #9 AlphaDou (end-to-end)
    [1,0,0,0,0,0,1,0,0],  # #10 MSP (pruning)
    [0,0,0,0,1,0,0,0,0],  # #11 DMC bidding
    [1,0,0,0,0,0,0,0,0],  # #12 DanZero
    [1,1,0,0,0,0,0,0,0],  # #13 DanZero+ (DMC→PPO)
    [1,0,0,0,0,0,0,0,0],  # #14 2-player (DMC vs DQN)
    [1,0,0,0,0,0,1,0,0],  # #15 OADMCDou
    [0,0,0,0,0,0,1,0,0],  # #16 Action representation
    [1,0,0,0,0,0,0,0,0],  # #18 Multi-DMC
    [1,0,0,1,0,0,1,0,0],  # #19 ODMC (+opponent model, pruning)
    [0,1,0,0,1,0,0,0,0],  # #20 CT-MS3
    [0,0,0,0,0,1,0,0,0],  # #21 WagerWin
    [0,0,0,0,0,1,0,0,0],  # #22 DQN-IRL
    [0,0,1,0,0,0,0,0,0],  # #23 LLM+Data
    [0,0,0,0,0,1,1,0,0],  # #24 Two-steps+IRL
    [0,0,0,0,1,0,0,0,0],  # #25 BidResNet
    [0,0,1,0,0,0,0,0,0],  # #26 LLM4Card
    [1,0,0,0,0,0,0,0,0],  # #27 SQUNO
    [0,1,0,0,0,0,0,0,1],  # #29 ACC (hybrid critics)
    [0,0,1,0,0,1,0,0,1],  # #30 Tjong (Transformer+hierarchical)
    [0,0,0,0,0,0,1,0,0],  # #32 Shanwin
    [0,0,0,0,0,1,0,0,1],  # #33 TiDou (SL+DRL)
    [0,0,0,1,0,0,0,1,0],  # #35 DeepBayes (CFR+BIR)
    [0,1,0,0,0,1,0,0,1],  # #36 TiZero (JRPO+curriculum)
    [0,0,1,0,0,0,0,0,0],  # #37 DouMH (Transformer)
]

# Create heatmap
im = ax3.imshow(method_matrix, cmap='Blues', aspect='auto', interpolation='nearest')

# Labels
ax3.set_xticks(np.arange(len(categories)))
ax3.set_yticks(np.arange(len(paper_names)))
ax3.set_xticklabels(categories, fontsize=8, rotation=45, ha='right')
ax3.set_yticklabels(paper_names, fontsize=7)
ax3.set_xlabel("Method Category", fontsize=11)
ax3.set_ylabel("Paper", fontsize=11)
ax3.set_title("Method Category Heatmap: Which Papers Use Which Techniques?", 
              fontsize=13, fontweight='bold', pad=20)

# Colorbar
cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
cbar.set_label('Method Present (1 = yes)', fontsize=9)

# Add grid lines
ax3.set_xticks(np.arange(-0.5, len(categories), 1), minor=True)
ax3.set_yticks(np.arange(-0.5, len(paper_names), 1), minor=True)
ax3.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

# Annotate column totals
col_totals = np.sum(method_matrix, axis=0)
for i, total in enumerate(col_totals):
    ax3.text(i, len(paper_names)+0.5, f'n={total}', ha='center', va='center', 
             fontsize=8, fontweight='bold', color='darkblue')

plt.tight_layout()
plt.savefig('method_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# BONUS: NETWORK GRAPH OF METHOD RELATIONSHIPS
# ============================================================
fig4, ax4 = plt.subplots(figsize=(14, 10))
G = nx.Graph()

# Nodes: methods and key papers (simplified)
methods = ['DMC', 'PPO', 'LLM', 'OpponentModel', 'Bidding', 'RewardShaping', 'ActionPruning', 'FictPlay', 'Hybrid']
papers_small = ['#4', '#5', '#6', '#8', '#9', '#10', '#13', '#16', '#19', '#20', '#21', '#23', '#26', '#29', '#30', '#33', '#35', '#36', '#37']

# Add nodes
for m in methods:
    G.add_node(m, type='method', size=1200)
for p in papers_small:
    G.add_node(p, type='paper', size=600)

# Edges from method to paper (based on usage)
edges = [
    ('DMC', '#4'), ('DMC', '#6'), ('DMC', '#10'), ('DMC', '#13'), ('DMC', '#19'),
    ('PPO', '#5'), ('PPO', '#13'), ('PPO', '#29'), ('PPO', '#36'),
    ('LLM', '#23'), ('LLM', '#26'), ('LLM', '#30'), ('LLM', '#37'),
    ('OpponentModel', '#4'), ('OpponentModel', '#8'), ('OpponentModel', '#19'), ('OpponentModel', '#35'),
    ('Bidding', '#6'), ('Bidding', '#9'), ('Bidding', '#20'),
    ('RewardShaping', '#5'), ('RewardShaping', '#21'), ('RewardShaping', '#30'), ('RewardShaping', '#36'),
    ('ActionPruning', '#10'), ('ActionPruning', '#16'), ('ActionPruning', '#19'),
    ('FictPlay', '#35'),
    ('Hybrid', '#29'), ('Hybrid', '#30'), ('Hybrid', '#33'), ('Hybrid', '#36'),
]

G.add_edges_from(edges)

# Layout
pos = nx.spring_layout(G, k=1.2, iterations=50, seed=42)

# Node colors
node_colors = ['#1f77b4' if G.nodes[n]['type'] == 'method' else '#ff7f0e' for n in G.nodes]
node_sizes = [G.nodes[n]['size'] for n in G.nodes]

# Draw
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, edgecolors='black', linewidths=1)
nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.5, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold')

ax4.set_title("Method–Paper Network Graph\n(Clusters show dominant technique families)", 
              fontsize=13, fontweight='bold', pad=20)
ax4.axis('off')

# Legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=10, label='Method Category'),
                   plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=10, label='Paper')]
ax4.legend(handles=legend_elements, loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('method_network_graph.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ All graphics saved:\n")
print("1. method_evolution_tree.png  - Timeline showing method evolution from DouZero")
print("2. bubble_chart_efficiency_vs_performance.png - Trade-off chart")
print("3. method_heatmap.png - Which papers use which technique families")
print("4. method_network_graph.png - Relationship network")