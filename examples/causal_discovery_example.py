"""
Example script demonstrating the simplified causal discovery interface.

Usage:
    python examples/causal_discovery_example.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from RCAEval.e2e.pcmci_shapley_modules import causal_discovery, PCMCIShapleyConfig


def generate_sample_data(n_timesteps=300):
    """Generate synthetic time series with known causal structure.
    
    Causal structure: A -> B -> C, A -> C
    """
    np.random.seed(42)
    
    # Service A (root cause)
    A = np.random.randn(n_timesteps)
    
    # Service B (affected by A with lag 1)
    B = np.zeros(n_timesteps)
    for t in range(1, n_timesteps):
        B[t] = 0.7 * A[t-1] + np.random.randn() * 0.3
    
    # Service C (affected by both A and B)
    C = np.zeros(n_timesteps)
    for t in range(1, n_timesteps):
        C[t] = 0.5 * A[t-1] + 0.6 * B[t-1] + np.random.randn() * 0.3
    
    df = pd.DataFrame({
        "service_a": A,
        "service_b": B,
        "service_c": C
    })
    
    return df


def visualize_graph(graph, title="Causal Graph"):
    """Visualize the causal graph."""
    pos = nx.spring_layout(graph, seed=42)
    
    plt.figure(figsize=(10, 6))
    nx.draw_networkx_nodes(graph, pos, node_size=2000, node_color='lightblue')
    nx.draw_networkx_labels(graph, pos, font_size=12, font_weight='bold')
    
    # Draw edges with weights
    edges = graph.edges()
    weights = [graph[u][v].get('weight', 1.0) for u, v in edges]
    nx.draw_networkx_edges(graph, pos, width=[w*3 for w in weights], 
                           edge_color='gray', arrows=True, arrowsize=20,
                           connectionstyle="arc3,rad=0.1")
    
    # Add edge labels
    edge_labels = {(u, v): f"{graph[u][v].get('weight', 0):.2f}" 
                   for u, v in edges}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=10)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    return plt


def example1_basic_pcmci():
    """Example 1: Basic PCMCI usage."""
    print("=" * 60)
    print("Example 1: Basic PCMCI Causal Discovery")
    print("=" * 60)
    
    # Generate data
    df = generate_sample_data()
    print(f"Generated data with {len(df)} timesteps")
    
    # Configure PCMCI
    config = PCMCIShapleyConfig(
        causal_method="pcmci",
        pcmci_alpha=0.05,
        tau_max=3,
        pcmci_max_conds_dim=2
    )
    
    # Discover causal graph
    result = causal_discovery.discover_causal_graph(
        data=df,
        local_nodes=["service_a", "service_b", "service_c"],
        config=config,
        method="pcmci"
    )
    
    # Print results
    print(f"\nDiscovered {len(result['edges'])} causal edges:")
    for (i, j, lag) in result['edges']:
        cause = result['columns'][i]
        effect = result['columns'][j]
        strength = result['edge_strengths'].get((i, j), 0)
        print(f"  {cause} -> {effect} (lag={lag}, strength={strength:.3f})")
    
    # Visualize
    print(f"\nGraph has {result['pcmci_graph'].number_of_nodes()} nodes and "
          f"{result['pcmci_graph'].number_of_edges()} edges")
    
    return result


def example2_pc_comparison():
    """Example 2: Compare PCMCI vs PC algorithm."""
    print("\n" + "=" * 60)
    print("Example 2: PCMCI vs PC Comparison")
    print("=" * 60)
    
    df = generate_sample_data()
    
    # PCMCI
    config_pcmci = PCMCIShapleyConfig(causal_method="pcmci", tau_max=3)
    result_pcmci = causal_discovery.discover_causal_graph(
        df, ["service_a", "service_b", "service_c"], config_pcmci, "pcmci"
    )
    
    # PC
    config_pc = PCMCIShapleyConfig(causal_method="pc")
    result_pc = causal_discovery.discover_causal_graph(
        df, ["service_a", "service_b", "service_c"], config_pc, "pc"
    )
    
    print(f"\nPCMCI: {len(result_pcmci['edges'])} edges (with time lags)")
    print(f"PC:    {len(result_pc['edges'])} edges (instantaneous only)")
    
    print("\nPCMCI edges:")
    for (i, j, lag) in result_pcmci['edges']:
        print(f"  {result_pcmci['columns'][i]} -> {result_pcmci['columns'][j]} (lag={lag})")
    
    print("\nPC edges:")
    for (i, j, lag) in result_pc['edges']:
        print(f"  {result_pc['columns'][i]} -> {result_pc['columns'][j]}")


def example3_sensitivity_analysis():
    """Example 3: Test sensitivity to alpha parameter."""
    print("\n" + "=" * 60)
    print("Example 3: Sensitivity Analysis (varying alpha)")
    print("=" * 60)
    
    df = generate_sample_data()
    alphas = [0.01, 0.05, 0.10, 0.20]
    
    print(f"\n{'Alpha':<8} {'Edges':<8} {'Description'}")
    print("-" * 40)
    
    for alpha in alphas:
        config = PCMCIShapleyConfig(
            causal_method="pcmci",
            pcmci_alpha=alpha,
            tau_max=3
        )
        
        result = causal_discovery.discover_causal_graph(
            df, ["service_a", "service_b", "service_c"], config, "pcmci"
        )
        
        n_edges = len(result['edges'])
        desc = "Strict" if alpha <= 0.01 else "Moderate" if alpha <= 0.05 else "Lenient"
        print(f"{alpha:<8.2f} {n_edges:<8} {desc}")


def example4_performance_comparison():
    """Example 4: Compare performance of PCMCI vs PC."""
    import time
    
    print("\n" + "=" * 60)
    print("Example 4: Performance Comparison")
    print("=" * 60)
    
    df = generate_sample_data(n_timesteps=500)
    
    # Test PCMCI
    config_pcmci = PCMCIShapleyConfig(causal_method="pcmci", tau_max=3)
    start = time.time()
    result_pcmci = causal_discovery.discover_causal_graph(
        df, ["service_a", "service_b", "service_c"], config_pcmci, "pcmci"
    )
    time_pcmci = time.time() - start
    
    # Test PC
    config_pc = PCMCIShapleyConfig(causal_method="pc")
    start = time.time()
    result_pc = causal_discovery.discover_causal_graph(
        df, ["service_a", "service_b", "service_c"], config_pc, "pc"
    )
    time_pc = time.time() - start
    
    print(f"\nPCMCI: {time_pcmci:.3f}s ({len(result_pcmci['edges'])} edges)")
    print(f"PC:    {time_pc:.3f}s ({len(result_pc['edges'])} edges)")
    print(f"Speedup: {time_pcmci/time_pc:.1f}x faster with PC")


def example5_integration_with_pipeline():
    """Example 5: Integration with full PCMCI-Shapley pipeline."""
    print("\n" + "=" * 60)
    print("Example 5: Integration with PCMCI-Shapley Pipeline")
    print("=" * 60)
    
    from RCAEval.e2e.pcmci_shapley import pcmci_shapley
    
    # Create sample data with service metrics
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "frontend_latency": np.random.randn(n) + 100,
        "frontend_cpu": np.random.randn(n) * 10 + 50,
        "backend_latency": np.random.randn(n) + 50,
        "backend_cpu": np.random.randn(n) * 10 + 30,
        "db_latency": np.random.randn(n) + 20,
        "db_cpu": np.random.randn(n) * 10 + 60,
    })
    
    # Test with PCMCI
    print("\nRunning with PCMCI method...")
    config_pcmci = PCMCIShapleyConfig(
        causal_method="pcmci",
        tau_max=3,
        enable_pruning=False
    )
    
    try:
        result = pcmci_shapley(
            data=df,
            focus_node="frontend",
            config=config_pcmci
        )
        print(f"Service ranking (PCMCI): {result['ranks'][:5]}")
    except Exception as e:
        print(f"Error with PCMCI: {e}")
    
    # Test with PC
    print("\nRunning with PC method...")
    config_pc = PCMCIShapleyConfig(
        causal_method="pc",
        enable_pruning=False
    )
    
    try:
        result = pcmci_shapley(
            data=df,
            focus_node="frontend",
            config=config_pc
        )
        print(f"Service ranking (PC): {result['ranks'][:5]}")
    except Exception as e:
        print(f"Error with PC: {e}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("CAUSAL DISCOVERY MODULE - USAGE EXAMPLES")
    print("=" * 60)
    
    # Run examples
    example1_basic_pcmci()
    example2_pc_comparison()
    example3_sensitivity_analysis()
    example4_performance_comparison()
    example5_integration_with_pipeline()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

