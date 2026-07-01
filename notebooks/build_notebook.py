"""
Generates metrics_visualization.ipynb — run with: python build_notebook.py
"""
import json, textwrap

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}

def code(src):
    return {
        "cell_type": "code", "execution_count": None,
        "metadata": {}, "outputs": [],
        "source": [textwrap.dedent(src).lstrip("\n")],
    }

C = []  # cells list

# ── 0. Title ─────────────────────────────────────────────────────────────────
C.append(md("""# Financial RAG System — Metrics & Evaluation Dashboard

**10 visualisations covering every production concern:**
1. Retrieval mode comparison (Dense / Sparse / Hybrid / Reranked)
2. Chunking strategy benchmark
3. Embedding model comparison
4. Latency breakdown (component + distribution + cache)
5. Ragas evaluation scores
6. Quality & cost over time (30-day)
7. Query intent distribution
8. Token usage & cost by LLM provider
9. Re-ranking improvement (before vs after)
10. Summary scorecard
"""))

# ── 1. Installs ───────────────────────────────────────────────────────────────
C.append(code("""
import subprocess, sys
for pkg in ["matplotlib", "seaborn", "plotly", "pandas", "numpy", "kaleido"]:
    subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=False)
print("Libraries ready.")
"""))

# ── 2. Imports + style ────────────────────────────────────────────────────────
C.append(code("""
import warnings, numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go, plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore")
np.random.seed(42)

DARK_BG   = "#0f1117"
PANEL_BG  = "#1a1d27"
GRID_CLR  = "#2a2d3a"
TXT_CLR   = "#e0e0e0"
PAL = dict(dense="#4fc3f7", sparse="#ff8a65", hybrid="#81c784",
           reranked="#ce93d8", openai="#74aa9c", ollama="#ff8a65",
           gemini="#4285f4", good="#81c784", warn="#ffb74d", bad="#e57373")

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": PANEL_BG,
    "axes.edgecolor": GRID_CLR, "axes.labelcolor": TXT_CLR,
    "xtick.color": "#b0b0b0", "ytick.color": "#b0b0b0",
    "text.color": TXT_CLR,    "grid.color": GRID_CLR,
    "grid.linestyle": "--",   "grid.alpha": 0.5,
    "font.family": "monospace","font.size": 11,
    "axes.titlesize": 13,     "axes.titleweight": "bold",
})
print("Dark theme loaded.")
"""))

# ── 3. Simulate data ──────────────────────────────────────────────────────────
C.append(md("---\n## Simulated Data\n*Replace the DataFrames below with real Prometheus / PostgreSQL queries.*"))
C.append(code("""
N    = 600
DAYS = pd.date_range("2024-01-01", periods=30, freq="D")

INTENTS = ["financial_ratio","revenue_comparison","risk_analysis","cash_flow",
           "multi_company","multi_year","balance_sheet","income_statement","generic_search"]
IW      = [0.18, 0.16, 0.12, 0.11, 0.09, 0.10, 0.09, 0.08, 0.07]

df = pd.DataFrame({
    "ts":        pd.to_datetime(np.random.choice(DAYS, N)),
    "intent":    np.random.choice(INTENTS, N, p=IW),
    "provider":  np.random.choice(["openai","openai","openai","ollama","gemini"], N),
    "mode":      np.random.choice(["dense","sparse","hybrid"], N, p=[0.15,0.10,0.75]),
    "reranked":  np.random.choice([True,False], N, p=[0.85,0.15]),
    "cache_hit": np.random.choice([True,False], N, p=[0.32,0.68]),
    "ptokens":   np.random.randint(800, 3500, N),
    "ctokens":   np.random.randint(100,  800, N),
    "emb_ms":    np.random.normal(120, 30, N).clip(20),
    "ret_ms":    np.random.normal(180, 50, N).clip(40),
    "rnk_ms":    np.random.normal(200, 60, N).clip(0),
    "llm_ms":    np.random.normal(1400,400, N).clip(300),
    "faith":     np.random.beta(8, 2, N),
    "cp":        np.random.beta(7, 2, N),
    "cr":        np.random.beta(7, 3, N),
    "ar":        np.random.beta(9, 2, N),
    "chunks":    np.random.randint(1, 21, N),
})

# cache hits skip pipeline
ch = df["cache_hit"]
df.loc[ch, ["emb_ms","ret_ms","rnk_ms","llm_ms"]] = [2, 0, 0, 0]
df.loc[df["reranked"], "cp"] = (df.loc[df["reranked"],"cp"] * 1.18).clip(upper=1.0)

df["total_ms"]  = df[["emb_ms","ret_ms","rnk_ms","llm_ms"]].sum(axis=1)
df["tot_tokens"] = df["ptokens"] + df["ctokens"]
df["cost"] = df["ptokens"]*0.00015/1000 + df["ctokens"]*0.0006/1000
df.loc[df["provider"]=="ollama","cost"] = 0
df.loc[df["provider"]=="gemini","cost"] *= 0.4
df.loc[ch, "cost"] = 0

# benchmark frames
RET = pd.DataFrame({"mode":["Dense","Sparse","Hybrid","Hybrid+Rerank"],
    "cp":[0.68,0.61,0.79,0.91], "cr":[0.72,0.65,0.81,0.85],
    "latency":[160,45,210,430], "mrr":[0.61,0.54,0.73,0.84]})

CHK = pd.DataFrame({"strategy":["Recursive","Semantic","Parent-Child","Table-Aware"],
    "cp":[0.71,0.79,0.87,0.82], "faith":[0.74,0.80,0.88,0.83],
    "ar":[0.76,0.83,0.90,0.81], "tokens":[128,210,64,95], "speed":[1.2,8.4,1.8,2.1]})

EMB = pd.DataFrame({"model":["BGE-large-en","E5-large-v2","text-embed-3-small","BGE-M3"],
    "mteb":[64.2,63.1,62.3,65.0], "ms100":[310,290,180,520],
    "recall":[0.83,0.80,0.78,0.86], "cost_1m":[0.0,0.0,0.02,0.0]})

DAILY = df.groupby(df["ts"].dt.date).agg(
    n=("total_ms","count"), lat=("total_ms","mean"),
    p95=("total_ms", lambda x: x.quantile(0.95)),
    cost=("cost","sum"), tokens=("tot_tokens","sum"),
    faith=("faith","mean"), cp=("cp","mean"),
    hits=("cache_hit","sum"),
).reset_index()
DAILY["hit_rate"] = DAILY["hits"] / DAILY["n"]

print(f"N={N} | avg_lat={df.total_ms.mean():.0f}ms | "
      f"total_cost=${df.cost.sum():.2f} | cache={df.cache_hit.mean():.1%}")
df.head(3)
"""))

# ── 4. Retrieval comparison ───────────────────────────────────────────────────
C.append(md("---\n## 1. Retrieval Mode Comparison"))
C.append(code("""
fig, axes = plt.subplots(1, 3, figsize=(19, 6))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Retrieval Mode Comparison", fontsize=16, fontweight="bold", color="white", y=1.02)
clrs = [PAL["dense"], PAL["sparse"], PAL["hybrid"], PAL["reranked"]]
modes = RET["mode"].tolist()

# precision
for ax_i, (col, ylabel) in enumerate(zip(["cp","mrr"],["Context Precision@5","MRR@10"])):
    bars = axes[ax_i].bar(modes, RET[col], color=clrs, width=0.55, zorder=3)
    axes[ax_i].set_ylim(0, 1.08)
    axes[ax_i].set_title(ylabel)
    axes[ax_i].grid(axis="y", zorder=0)
    axes[ax_i].set_xticklabels(modes, rotation=15)
    for b in bars:
        axes[ax_i].text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                        f"{b.get_height():.2f}", ha="center", fontsize=10, color="white")

# scatter: latency vs precision
axes[2].scatter(RET["latency"], RET["cp"], c=clrs, s=230, zorder=5,
                edgecolors="white", linewidths=0.8)
for _, r in RET.iterrows():
    axes[2].annotate(r["mode"], (r["latency"]+10, r["cp"]), fontsize=9, color="white")
axes[2].set_xlabel("Latency (ms)"); axes[2].set_ylabel("Context Precision")
axes[2].set_title("Precision vs Latency Tradeoff"); axes[2].grid(zorder=0)

plt.tight_layout()
plt.savefig("retrieval_comparison.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.show()
print("Hybrid+Rerank: best precision at 2.7x latency cost of Dense.")
"""))

# ── 5. Chunking comparison ────────────────────────────────────────────────────
C.append(md("---\n## 2. Chunking Strategy Benchmark"))
C.append(code("""
fig, axes = plt.subplots(1, 3, figsize=(19, 6))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Chunking Strategy Benchmark", fontsize=16, fontweight="bold", color="white", y=1.02)
s_clrs = ["#4fc3f7","#81c784","#ce93d8","#ff8a65"]
strats = CHK["strategy"].tolist()
x = np.arange(len(strats))

# grouped quality bars
for i,(col,lbl) in enumerate(zip(["cp","faith","ar"],
                                  ["Context Precision","Faithfulness","Answer Relevancy"])):
    axes[0].bar(x+i*0.26, CHK[col], 0.26, label=lbl,
                color=["#4fc3f7","#81c784","#ce93d8"][i], zorder=3)
axes[0].set_xticks(x+0.26); axes[0].set_xticklabels(strats, rotation=12)
axes[0].set_ylim(0.5, 1.0); axes[0].set_title("Quality Metrics")
axes[0].legend(fontsize=8); axes[0].grid(axis="y", zorder=0)

# chunk size
bars = axes[1].bar(strats, CHK["tokens"], color=s_clrs, width=0.55, zorder=3)
axes[1].set_title("Avg Chunk Size (tokens)"); axes[1].grid(axis="y", zorder=0)
axes[1].set_xticklabels(strats, rotation=12)
for b in bars:
    axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+1,
                 str(int(b.get_height())), ha="center", fontsize=10, color="white")

# ingestion speed
bars = axes[2].bar(strats, CHK["speed"], color=s_clrs, width=0.55, zorder=3)
axes[2].set_title("Ingestion Time (s / doc)"); axes[2].grid(axis="y", zorder=0)
axes[2].set_xticklabels(strats, rotation=12)
for b in bars:
    axes[2].text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
                 f"{b.get_height():.1f}s", ha="center", fontsize=10, color="white")

plt.tight_layout()
plt.savefig("chunking_comparison.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.show()
print("Parent-Child best quality. Semantic 4.7x slower — use for narrative docs only.")
"""))

# ── 6. Embedding comparison ───────────────────────────────────────────────────
C.append(md("---\n## 3. Embedding Model Comparison"))
C.append(code("""
e_clrs = ["#81c784","#4fc3f7","#74aa9c","#ce93d8"]
fig = make_subplots(rows=1, cols=3,
    subplot_titles=["MTEB Score","Recall@5","Speed vs Quality"],
    horizontal_spacing=0.12)

for col_i, (col, fmt) in enumerate(zip(["mteb","recall"],[":.1f",":.2f"]), start=1):
    fig.add_trace(go.Bar(x=EMB["model"], y=EMB[col], marker_color=e_clrs,
        text=EMB[col], texttemplate=f"%{{text{fmt}}}", textposition="outside",
        showlegend=False), row=1, col=col_i)

fig.add_trace(go.Scatter(x=EMB["ms100"], y=EMB["mteb"], mode="markers+text",
    marker=dict(size=22, color=e_clrs, line=dict(width=1, color="white")),
    text=EMB["model"], textposition="top center", showlegend=False), row=1, col=3)
fig.update_xaxes(title_text="ms / 100 docs", row=1, col=3)
fig.update_yaxes(title_text="MTEB", row=1, col=3)

fig.update_layout(height=430, title_text="Embedding Model Benchmark",
    paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
    font=dict(color=TXT_CLR, family="monospace"))
fig.update_xaxes(tickangle=-20, gridcolor=GRID_CLR)
fig.update_yaxes(gridcolor=GRID_CLR)
fig.show()
"""))

# ── 7. Latency breakdown ──────────────────────────────────────────────────────
C.append(md("---\n## 4. Latency Breakdown"))
C.append(code("""
nc = df[~df["cache_hit"]]   # non-cached
ca = df[ df["cache_hit"]]   # cached

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Latency Breakdown", fontsize=16, fontweight="bold", color="white", y=1.02)

# stacked component bar
comp_cols   = ["emb_ms","ret_ms","rnk_ms","llm_ms"]
comp_labels = ["Embedding","Retrieval","Re-ranking","LLM"]
comp_colors = ["#4fc3f7","#81c784","#ce93d8","#ff8a65"]
bottom = 0
for col, lbl, clr in zip(comp_cols, comp_labels, comp_colors):
    avg = nc[col].mean()
    axes[0].bar(["Non-cached"], [avg], bottom=[bottom], color=clr, label=lbl, width=0.35)
    axes[0].text(0, bottom+avg/2, f"{avg:.0f}ms", ha="center", va="center",
                 fontsize=10, color="white", fontweight="bold")
    bottom += avg
axes[0].set_ylabel("ms"); axes[0].set_title("Avg Component Latency")
axes[0].legend(loc="upper right", fontsize=8); axes[0].grid(axis="y", zorder=0)

# distribution histogram
axes[1].hist(nc["total_ms"], bins=40, color="#4fc3f7", alpha=0.85, edgecolor=PANEL_BG, zorder=3)
for pct, clr, lbl in [(0.50,"#81c784","p50"),(0.95,"#ffb74d","p95"),(0.99,"#e57373","p99")]:
    v = nc["total_ms"].quantile(pct)
    axes[1].axvline(v, color=clr, linestyle="--", linewidth=1.5, label=f"{lbl}={v:.0f}ms")
axes[1].set_xlabel("ms"); axes[1].set_title("Latency Distribution (non-cached)")
axes[1].legend(fontsize=9); axes[1].grid(zorder=0)

# cached vs non-cached
axes[2].hist(nc["total_ms"], bins=35, alpha=0.7, color="#4fc3f7", label="Miss", zorder=3)
axes[2].hist(ca["total_ms"], bins=12, alpha=0.9, color="#81c784", label="Hit",  zorder=4)
axes[2].set_xlabel("ms")
axes[2].set_title(f"Cache Hit vs Miss  |  hit rate {df.cache_hit.mean():.1%}")
axes[2].legend(fontsize=9); axes[2].grid(zorder=0)

plt.tight_layout()
plt.savefig("latency_breakdown.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.show()
print(f"Cache saves ~{(1 - ca.total_ms.mean()/nc.total_ms.mean()):.0%} latency per cached query.")
"""))

# ── 8. Evaluation scores ──────────────────────────────────────────────────────
C.append(md("---\n## 5. Ragas Evaluation Score Distributions"))
C.append(code("""
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Ragas Evaluation Scores — Distribution", fontsize=16, fontweight="bold", color="white")

specs = [("faith","Faithfulness","#4fc3f7"), ("cp","Context Precision","#81c784"),
         ("cr","Context Recall","#ce93d8"),  ("ar","Answer Relevancy","#ff8a65")]

for ax, (col, title, clr) in zip(axes.flat, specs):
    data = df[col]
    ax.hist(data, bins=30, color=clr, alpha=0.85, edgecolor=PANEL_BG, zorder=3)
    mean_v = data.mean()
    ax.axvline(mean_v, color="white",   linestyle="--", linewidth=1.5, label=f"mean={mean_v:.3f}")
    ax.axvline(0.80,   color="#81c784", linestyle=":",  linewidth=1.2, alpha=0.8, label="target=0.80")
    pct = (data >= 0.8).mean()
    ax.set_title(f"{title}  |  {pct:.0%} ≥ 0.80")
    ax.set_xlabel("Score"); ax.legend(fontsize=9); ax.grid(zorder=0)

plt.tight_layout()
plt.savefig("evaluation_scores.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.show()
print(df[["faith","cp","cr","ar"]].describe().round(3).to_string())
"""))

# ── 9. Time-series ────────────────────────────────────────────────────────────
C.append(md("---\n## 6. Quality & Cost Over Time (30 Days)"))
C.append(code("""
x = DAILY["ts"].astype(str)
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
    subplot_titles=["Quality Scores (daily avg)","Query Volume & Cache Hit Rate",
                    "Token Usage & Cumulative Cost"])

fig.add_trace(go.Scatter(x=x,y=DAILY["faith"],name="Faithfulness",
    line=dict(color="#4fc3f7",width=2),mode="lines+markers"), row=1,col=1)
fig.add_trace(go.Scatter(x=x,y=DAILY["cp"],name="Context Precision",
    line=dict(color="#81c784",width=2),mode="lines+markers"), row=1,col=1)
fig.add_hrect(y0=0.8,y1=1.0,fillcolor="rgba(129,199,132,0.07)",
    line_width=0,annotation_text="Target zone",row=1,col=1)

fig.add_trace(go.Bar(x=x,y=DAILY["n"],name="Queries",
    marker_color="#ce93d8",opacity=0.7), row=2,col=1)
fig.add_trace(go.Scatter(x=x,y=DAILY["hit_rate"],name="Cache Hit Rate",
    line=dict(color="#ffb74d",width=2),mode="lines+markers"), row=2,col=1)

fig.add_trace(go.Bar(x=x,y=DAILY["tokens"],name="Tokens",
    marker_color="#4fc3f7",opacity=0.7), row=3,col=1)
fig.add_trace(go.Scatter(x=x,y=DAILY["cost"].cumsum(),name="Cumul. Cost USD",
    line=dict(color="#ff8a65",width=2.5),fill="tozeroy",
    fillcolor="rgba(255,138,101,0.12)"), row=3,col=1)

fig.update_layout(height=760, title_text="Financial RAG — 30-Day Operational Dashboard",
    paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
    font=dict(color=TXT_CLR,family="monospace"),
    legend=dict(bgcolor=PANEL_BG,bordercolor=GRID_CLR))
fig.update_yaxes(gridcolor=GRID_CLR)
fig.show()
print(f"30-day cost: ${DAILY.cost.sum():.2f}  |  total tokens: {DAILY.tokens.sum():,}")
"""))

# ── 10. Query intent distribution ────────────────────────────────────────────
C.append(md("---\n## 7. Query Intent Distribution"))
C.append(code("""
intent_counts = df["intent"].value_counts().reset_index()
intent_counts.columns = ["intent","count"]
intent_counts["pct"] = intent_counts["count"] / intent_counts["count"].sum()

i_clrs = px.colors.qualitative.Set3[:len(intent_counts)]
fig = make_subplots(rows=1, cols=2,
    subplot_titles=["Query Volume by Intent","Intent Share (%)"],
    specs=[[{"type":"bar"},{"type":"pie"}]])

fig.add_trace(go.Bar(x=intent_counts["intent"], y=intent_counts["count"],
    marker_color=i_clrs, showlegend=False,
    text=intent_counts["count"], textposition="outside"), row=1, col=1)

fig.add_trace(go.Pie(labels=intent_counts["intent"], values=intent_counts["count"],
    marker_colors=i_clrs, hole=0.38,
    textinfo="label+percent", showlegend=False), row=1, col=2)

fig.update_layout(height=450, title_text="Query Intent Analysis",
    paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
    font=dict(color=TXT_CLR, family="monospace"))
fig.update_xaxes(tickangle=-30, gridcolor=GRID_CLR)
fig.update_yaxes(gridcolor=GRID_CLR)
fig.show()
print(intent_counts.to_string(index=False))
"""))

# ── 11. Token & cost by provider ─────────────────────────────────────────────
C.append(md("---\n## 8. Token Usage & Cost by LLM Provider"))
C.append(code("""
prov = df.groupby("provider").agg(
    queries=("cost","count"), tokens=("tot_tokens","sum"),
    cost=("cost","sum"), avg_lat=("total_ms","mean")).reset_index()

prov_clrs = [PAL.get(p, "#aaaaaa") for p in prov["provider"]]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("LLM Provider Comparison", fontsize=16, fontweight="bold", color="white", y=1.02)

for ax_i, (col, title) in enumerate(zip(["tokens","cost","avg_lat"],
                                          ["Total Tokens","Total Cost (USD)","Avg Latency (ms)"])):
    bars = axes[ax_i].bar(prov["provider"], prov[col], color=prov_clrs, width=0.5, zorder=3)
    axes[ax_i].set_title(title); axes[ax_i].grid(axis="y", zorder=0)
    for b in bars:
        val = b.get_height()
        label = f"${val:.2f}" if col=="cost" else f"{val:,.0f}"
        axes[ax_i].text(b.get_x()+b.get_width()/2, val*1.02,
                        label, ha="center", fontsize=9, color="white")

plt.tight_layout()
plt.savefig("provider_comparison.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.show()
print(prov.to_string(index=False))
"""))

# ── 12. Re-ranking improvement ────────────────────────────────────────────────
C.append(md("---\n## 9. Re-ranking Impact"))
C.append(code("""
rerank_df = df.groupby("reranked")[["cp","faith","ar","total_ms"]].mean().reset_index()
rerank_df["label"] = rerank_df["reranked"].map({True:"With Reranker",False:"Without Reranker"})
r_clrs = [PAL["reranked"], PAL["dense"]]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Re-ranking Impact on Quality vs Latency", fontsize=16,
             fontweight="bold", color="white", y=1.02)

metrics = ["cp","faith","ar"]
m_labels = ["Context Precision","Faithfulness","Answer Relevancy"]
x = np.arange(len(metrics))
w = 0.35
for i, (row, clr) in enumerate(zip(rerank_df.itertuples(), r_clrs)):
    bars = axes[0].bar(x + i*w, [row.cp, row.faith, row.ar], w,
                       label=row.label, color=clr, zorder=3)
axes[0].set_xticks(x+w/2); axes[0].set_xticklabels(m_labels, rotation=12)
axes[0].set_ylim(0.5, 1.05); axes[0].set_title("Quality Metrics")
axes[0].legend(fontsize=9); axes[0].grid(axis="y", zorder=0)

# delta annotation
for i, (m_lbl, m_col) in enumerate(zip(m_labels, metrics)):
    no_r = float(rerank_df.loc[rerank_df["reranked"]==False, m_col])
    wi_r = float(rerank_df.loc[rerank_df["reranked"]==True,  m_col])
    delta = wi_r - no_r
    axes[0].annotate(f"+{delta:.2f}", xy=(i+w/2+0.05, wi_r+0.01),
                     fontsize=9, color=PAL["good"], fontweight="bold")

bars2 = axes[1].bar(rerank_df["label"], rerank_df["total_ms"],
                    color=r_clrs, width=0.45, zorder=3)
axes[1].set_title("Avg Total Latency (ms)")
axes[1].grid(axis="y", zorder=0)
for b in bars2:
    axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+10,
                 f"{b.get_height():.0f}ms", ha="center", fontsize=10, color="white")

plt.tight_layout()
plt.savefig("reranking_impact.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.show()
"""))

# ── 13. Summary scorecard ─────────────────────────────────────────────────────
C.append(md("---\n## 10. Summary Scorecard"))
C.append(code("""
nc = df[~df["cache_hit"]]

scorecard = {
    "Avg Faithfulness":       df["faith"].mean(),
    "Avg Context Precision":  df["cp"].mean(),
    "Avg Answer Relevancy":   df["ar"].mean(),
    "p50 Latency (ms)":       nc["total_ms"].quantile(0.50),
    "p95 Latency (ms)":       nc["total_ms"].quantile(0.95),
    "Cache Hit Rate":         df["cache_hit"].mean(),
    "Total Queries":          float(len(df)),
    "Total Cost USD":         df["cost"].sum(),
    "Avg Cost / Query USD":   df["cost"].mean(),
    "Hallucination Rate":     1 - df["faith"].mean(),
}

thresholds = {
    "Avg Faithfulness":      (0.80, True),
    "Avg Context Precision": (0.75, True),
    "Avg Answer Relevancy":  (0.80, True),
    "p50 Latency (ms)":      (2000, False),
    "p95 Latency (ms)":      (5000, False),
    "Cache Hit Rate":        (0.25, True),
    "Hallucination Rate":    (0.20, False),
}

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(PANEL_BG)
ax.axis("off")
ax.set_title("Production Scorecard", fontsize=16, fontweight="bold",
             color="white", pad=20)

rows, colors = [], []
for k, v in scorecard.items():
    if k in thresholds:
        thr, higher_is_better = thresholds[k]
        ok = (v >= thr) if higher_is_better else (v <= thr)
        status = "PASS" if ok else "FAIL"
        clr = PAL["good"] if ok else PAL["bad"]
    else:
        status, clr = "INFO", PAL["warn"]
    if isinstance(v, float) and v < 10:
        fmt = f"{v:.3f}" if "Rate" in k or "Faithfulness" in k or "Relevancy" in k or "Precision" in k or "Hallucination" in k else f"{v:.2f}"
    else:
        fmt = f"{v:,.1f}" if "ms" in k else (f"${v:.4f}" if "USD" in k else f"{v:,.0f}")
    rows.append([k, fmt, status])
    colors.append(["white", "white", clr])

tbl = ax.table(cellText=rows,
               colLabels=["Metric", "Value", "Status"],
               cellLoc="center", loc="center",
               cellColours=[[PANEL_BG]*3]*len(rows),
               colColours=[GRID_CLR]*3)
tbl.auto_set_font_size(False); tbl.set_fontsize(11)
tbl.scale(1, 2.0)

for (r, c), cell in tbl.get_celld().items():
    cell.set_text_props(color=colors[r-1][c] if r > 0 else "white",
                        fontweight="bold" if r == 0 else "normal")
    cell.set_edgecolor(GRID_CLR)

plt.tight_layout()
plt.savefig("scorecard.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.show()

pass_count = sum(1 for r in rows if r[2]=="PASS")
fail_count = sum(1 for r in rows if r[2]=="FAIL")
print(f"Scorecard: {pass_count} PASS / {fail_count} FAIL")
"""))

# ── Write notebook ────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": C,
}

out = "metrics_visualization.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Generated {out}  ({len(C)} cells)")
