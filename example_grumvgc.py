#!/usr/bin/env python3
"""
GRUMVGC使用示例
展示如何在RCAEval框架中使用GRUMVGC進行多模態根因分析
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 確保能夠導入RCAEval
sys.path.insert(0, str(Path(__file__).parent))

def create_synthetic_multimodal_data():
    """創建合成的多模態微服務監控數據"""
    
    print("🔧 Creating synthetic multimodal data...")
    
    np.random.seed(42)
    n_samples = 300
    time_points = np.arange(n_samples)
    
    # 定義服務列表
    services = ['frontend', 'auth', 'payment', 'inventory', 'notification']
    
    # 創建Metrics數據
    metric_data = {'time': time_points}
    
    for service in services:
        # CPU使用率 (0-1)
        base_cpu = np.random.uniform(0.2, 0.4)
        metric_data[f'{service}_cpu'] = base_cpu + 0.1 * np.sin(time_points * 0.1) + np.random.normal(0, 0.05, n_samples)
        
        # 內存使用率 (0-1)  
        base_memory = np.random.uniform(0.4, 0.6)
        metric_data[f'{service}_memory'] = base_memory + 0.05 * np.cos(time_points * 0.08) + np.random.normal(0, 0.03, n_samples)
        
        # 網絡流量 (MB/s)
        base_network = np.random.uniform(10, 50)
        metric_data[f'{service}_network'] = base_network + 10 * np.sin(time_points * 0.05) + np.random.normal(0, 2, n_samples)
    
    # 創建Logs時間序列數據
    logts_data = {'time': time_points}
    
    for service in services:
        # 錯誤日誌計數
        base_errors = np.random.poisson(1, n_samples)
        logts_data[f'{service}_error_count'] = base_errors
        
        # 請求日誌計數
        base_requests = np.random.poisson(20, n_samples)
        logts_data[f'{service}_request_count'] = base_requests
        
        # 警告日誌計數
        base_warnings = np.random.poisson(3, n_samples)
        logts_data[f'{service}_warning_count'] = base_warnings
    
    # 模擬異常場景：payment服務在t=200後出現問題
    inject_time = 200
    
    print(f"💥 Injecting anomaly at time {inject_time}")
    
    # Payment服務CPU和內存異常增加
    metric_data['payment_cpu'][inject_time:] += 0.4
    metric_data['payment_memory'][inject_time:] += 0.3
    
    # Payment服務錯誤日誌激增
    logts_data['payment_error_count'][inject_time:] += np.random.poisson(10, n_samples - inject_time)
    
    # 上游服務(frontend, auth)也受到影響
    metric_data['frontend_cpu'][inject_time+5:] += 0.2  # 延遲5秒影響
    metric_data['auth_cpu'][inject_time+3:] += 0.15     # 延遲3秒影響
    
    logts_data['frontend_error_count'][inject_time+5:] += np.random.poisson(3, n_samples - inject_time - 5)
    logts_data['auth_error_count'][inject_time+3:] += np.random.poisson(2, n_samples - inject_time - 3)
    
    # 下游服務(inventory, notification)也受到影響
    metric_data['inventory_cpu'][inject_time+8:] += 0.1  # 延遲8秒影響
    logts_data['notification_error_count'][inject_time+10:] += np.random.poisson(1, n_samples - inject_time - 10)
    
    # 轉換為DataFrame
    metric_df = pd.DataFrame(metric_data)
    logts_df = pd.DataFrame(logts_data)
    
    # 確保數值在合理範圍內
    for col in metric_df.columns:
        if col != 'time':
            if 'cpu' in col or 'memory' in col:
                metric_df[col] = np.clip(metric_df[col], 0, 1)
            elif 'network' in col:
                metric_df[col] = np.clip(metric_df[col], 0, None)
    
    for col in logts_df.columns:
        if col != 'time':
            logts_df[col] = np.clip(logts_df[col], 0, None)
    
    print(f"✅ Created metric data: {metric_df.shape}")
    print(f"✅ Created logts data: {logts_df.shape}")
    
    return {
        'metric': metric_df,
        'logts': logts_df
    }, inject_time


def visualize_data(multimodal_data, inject_time):
    """可視化多模態數據"""
    
    print("📊 Visualizing multimodal data...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Multimodal Microservice Monitoring Data', fontsize=16)
    
    metric_df = multimodal_data['metric']
    logts_df = multimodal_data['logts']
    
    # 繪製CPU使用率
    ax1 = axes[0, 0]
    cpu_cols = [col for col in metric_df.columns if 'cpu' in col]
    for col in cpu_cols[:3]:  # 只顯示前3個服務以避免圖表過於擁擠
        ax1.plot(metric_df['time'], metric_df[col], label=col, alpha=0.7)
    ax1.axvline(x=inject_time, color='red', linestyle='--', alpha=0.8, label='Anomaly Injection')
    ax1.set_title('CPU Usage')
    ax1.set_ylabel('CPU Usage')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 繪製內存使用率
    ax2 = axes[0, 1]
    memory_cols = [col for col in metric_df.columns if 'memory' in col]
    for col in memory_cols[:3]:
        ax2.plot(metric_df['time'], metric_df[col], label=col, alpha=0.7)
    ax2.axvline(x=inject_time, color='red', linestyle='--', alpha=0.8, label='Anomaly Injection')
    ax2.set_title('Memory Usage')
    ax2.set_ylabel('Memory Usage')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 繪製錯誤日誌計數
    ax3 = axes[1, 0]
    error_cols = [col for col in logts_df.columns if 'error_count' in col]
    for col in error_cols[:3]:
        ax3.plot(logts_df['time'], logts_df[col], label=col, alpha=0.7)
    ax3.axvline(x=inject_time, color='red', linestyle='--', alpha=0.8, label='Anomaly Injection')
    ax3.set_title('Error Log Counts')
    ax3.set_ylabel('Error Count')
    ax3.set_xlabel('Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 繪製請求日誌計數
    ax4 = axes[1, 1]
    request_cols = [col for col in logts_df.columns if 'request_count' in col]
    for col in request_cols[:3]:
        ax4.plot(logts_df['time'], logts_df[col], label=col, alpha=0.7)
    ax4.axvline(x=inject_time, color='red', linestyle='--', alpha=0.8, label='Anomaly Injection')
    ax4.set_title('Request Log Counts')
    ax4.set_ylabel('Request Count')
    ax4.set_xlabel('Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multimodal_data_visualization.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'multimodal_data_visualization.png'")
    
    return fig


def run_grumvgc_analysis(multimodal_data, inject_time):
    """運行GRUMVGC根因分析"""
    
    print("\n🚀 Running GRUMVGC analysis...")
    
    try:
        from RCAEval.e2e.grumvgc import grumvgc
        
        # 運行GRUMVGC，使用適中的參數以平衡性能和效果
        result = grumvgc(
            data=multimodal_data,
            inject_time=inject_time,
            dataset="synthetic_microservice",
            embedding_dim=64,       # 適中的嵌入維度
            seq_len=40,            # 適中的序列長度
            num_epochs=15,         # 適中的訓練輪數
            batch_size=16,         # 適中的批次大小
            dk_select_useful=True  # 啟用特徵選擇
        )
        
        print("\n🎯 GRUMVGC Analysis Results:")
        print("=" * 50)
        
        # 顯示基本信息
        print(f"📊 Total nodes analyzed: {len(result['node_names'])}")
        print(f"🔗 Causal edges found: {len(result.get('causal_edges', []))}")
        print(f"📐 Adjacency matrix shape: {result['adj'].shape}")
        
        # 顯示前10個根因
        print(f"\n🏆 Top 10 Root Causes:")
        print("-" * 30)
        for i, cause in enumerate(result['ranks'][:10], 1):
            # 簡化節點名稱顯示
            display_name = cause.replace('metric_', '').replace('logts_', '')
            print(f"{i:2d}. {display_name}")
        
        # 顯示因果邊信息
        if result.get('causal_edges'):
            print(f"\n🔗 Sample Causal Relationships:")
            print("-" * 40)
            for i, (source, target, weight) in enumerate(result['causal_edges'][:5], 1):
                source_clean = source.replace('metric_', '').replace('logts_', '')
                target_clean = target.replace('metric_', '').replace('logts_', '')
                print(f"{i}. {source_clean} → {target_clean} (strength: {weight:.3f})")
        
        # 顯示嵌入信息
        if 'embeddings_info' in result:
            info = result['embeddings_info']
            print(f"\n🧠 Embedding Information:")
            print(f"   Features processed: {info['num_features']}")
            print(f"   Embedding dimension: {info['embedding_dim']}")
        
        # 分析結果質量
        print(f"\n📈 Analysis Quality Assessment:")
        print("-" * 35)
        
        # 檢查是否正確識別了payment服務作為根因
        top_5_causes = result['ranks'][:5]
        payment_related = [cause for cause in top_5_causes if 'payment' in cause.lower()]
        
        if payment_related:
            print(f"✅ Payment service correctly identified in top 5: {payment_related}")
        else:
            print("⚠️  Payment service not in top 5 root causes")
        
        # 檢查是否識別了相關的上游服務
        upstream_services = ['frontend', 'auth']
        top_10_causes = result['ranks'][:10]
        upstream_in_top10 = [cause for cause in top_10_causes 
                           if any(service in cause.lower() for service in upstream_services)]
        
        if upstream_in_top10:
            print(f"✅ Upstream services identified: {upstream_in_top10}")
        else:
            print("⚠️  Upstream services not prominently ranked")
        
        return result
        
    except Exception as e:
        print(f"❌ GRUMVGC analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_with_baseline(multimodal_data, inject_time):
    """與基線方法比較"""
    
    print("\n📊 Comparing with baseline methods...")
    
    try:
        from RCAEval.e2e.granger_pagerank import granger_pagerank
        
        # 使用簡單的granger方法作為基線
        # 需要將多模態數據合併為單一DataFrame
        metric_df = multimodal_data['metric']
        logts_df = multimodal_data['logts']
        
        # 合併數據（簡化處理）
        combined_df = metric_df.copy()
        for col in logts_df.columns:
            if col != 'time':
                combined_df[f'log_{col}'] = logts_df[col]
        
        baseline_result = granger_pagerank(
            data=combined_df,
            inject_time=inject_time,
            dataset="synthetic_microservice"
        )
        
        print("🔄 Baseline (Granger PageRank) Results:")
        print(f"   Top 5 causes: {baseline_result['ranks'][:5]}")
        
        return baseline_result
        
    except Exception as e:
        print(f"⚠️  Baseline comparison failed: {e}")
        return None


def main():
    """主函數"""
    
    print("🎬 GRUMVGC Multimodal Root Cause Analysis Demo")
    print("=" * 55)
    
    # 步驟1: 創建合成數據
    multimodal_data, inject_time = create_synthetic_multimodal_data()
    
    # 步驟2: 可視化數據
    try:
        fig = visualize_data(multimodal_data, inject_time)
        plt.show()
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")
    
    # 步驟3: 運行GRUMVGC分析
    grumvgc_result = run_grumvgc_analysis(multimodal_data, inject_time)
    
    # 步驟4: 與基線方法比較
    baseline_result = compare_with_baseline(multimodal_data, inject_time)
    
    # 步驟5: 總結
    print("\n🎯 Analysis Summary:")
    print("=" * 25)
    
    if grumvgc_result:
        print("✅ GRUMVGC analysis completed successfully")
        print(f"   Identified {len(grumvgc_result['ranks'])} potential root causes")
        print(f"   Found {len(grumvgc_result.get('causal_edges', []))} causal relationships")
    else:
        print("❌ GRUMVGC analysis failed")
    
    if baseline_result:
        print("✅ Baseline comparison completed")
    else:
        print("⚠️  Baseline comparison unavailable")
    
    print("\n🏁 Demo completed!")
    print("\nNext steps:")
    print("1. Analyze the visualization to understand the anomaly pattern")
    print("2. Review the root cause rankings")
    print("3. Examine the causal relationships discovered")
    print("4. Compare with domain knowledge and ground truth")
    
    return grumvgc_result, baseline_result


if __name__ == "__main__":
    results = main() 