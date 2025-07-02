import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from RCAEval.io.time_series import (
    convert_mem_mb,
    drop_constant,
    drop_extra,
    drop_near_constant,
    drop_time,
    preprocess,
    select_useful_cols,
)

def svmrca(
    data, inject_time=None, dataset=None, num_loop=None, sli=None, anomalies=None, **kwargs
):
    print(f"[SVMRCA] 開始執行根因分析")
    print(f"[SVMRCA] 資料集: {dataset}")
    print(f"[SVMRCA] 故障注入時間: {inject_time}")
    print(f"[SVMRCA] 輸入數據形狀: {data.shape}")
    
    # 步驟 1: 分割數據
    print(f"\n[SVMRCA] 步驟 1/5: 分割正常與異常數據")
    if anomalies is None:
        normal_df = data[data["time"] < inject_time]
        anomal_df = data[data["time"] >= inject_time]
    else:
        normal_df = data.head(anomalies[0])
        anomal_df = data.tail(len(data) - anomalies[0])

    print(f"[SVMRCA] 正常數據: {normal_df.shape[0]} 個樣本")
    print(f"[SVMRCA] 異常數據: {anomal_df.shape[0]} 個樣本")

    # 步驟 2: 數據預處理
    print(f"\n[SVMRCA] 步驟 2/5: 數據預處理")
    print(f"[SVMRCA] 預處理前 - 正常數據維度: {normal_df.shape}")
    normal_df = preprocess(
        data=normal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    anomal_df = preprocess(
        data=anomal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )
    print(f"[SVMRCA] 預處理後 - 正常數據維度: {normal_df.shape}")
    print(f"[SVMRCA] 預處理後 - 異常數據維度: {anomal_df.shape}")

    # 步驟 3: 特徵對齊
    print(f"\n[SVMRCA] 步驟 3/5: 特徵對齊")
    intersects = [x for x in normal_df.columns if x in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]
    print(f"[SVMRCA] 共同特徵數量: {len(intersects)}")
    print(f"[SVMRCA] 特徵清單: {intersects[:10]}{'...' if len(intersects) > 10 else ''}")

    # 步驟 4: 計算異常分數
    print(f"\n[SVMRCA] 步驟 4/5: 計算各指標異常分數")
    ranks = []
    total_features = len(normal_df.columns)
    
    for idx, col in enumerate(normal_df.columns, 1):
        a = normal_df[col].to_numpy()
        b = anomal_df[col].to_numpy()

        scaler = RobustScaler().fit(a.reshape(-1, 1))
        zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
        score = max(zscores)
        ranks.append((col, score))
        
        # 每處理 10% 的特徵就輸出一次進度
        if idx % max(1, total_features // 10) == 0 or idx == total_features:
            progress = (idx / total_features) * 100
            print(f"[SVMRCA] 進度: {progress:.1f}% ({idx}/{total_features}) - 當前處理: {col}")

    # 步驟 5: 排序結果
    print(f"\n[SVMRCA] 步驟 5/5: 生成根因排序")
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    
    print(f"\n[SVMRCA] === 根因分析結果 ===")
    print(f"[SVMRCA] Top 10 根因候選:")
    for i, (feature, score) in enumerate(ranks[:10], 1):
        print(f"[SVMRCA]   {i:2d}. {feature:<30} (分數: {score:.4f})")
    
    # 統計資訊
    scores = [score for _, score in ranks]
    print(f"\n[SVMRCA] === 統計資訊 ===")
    print(f"[SVMRCA] 最高分數: {max(scores):.4f}")
    print(f"[SVMRCA] 平均分數: {sum(scores)/len(scores):.4f}")
    print(f"[SVMRCA] 分數標準差: {(sum((s - sum(scores)/len(scores))**2 for s in scores)/len(scores))**0.5:.4f}")
    
    # 檢查是否有明顯的根因
    if max(scores) > 3.0:
        print(f"[SVMRCA] 發現明顯異常! 最高分數 {max(scores):.4f} > 3.0")
    elif max(scores) > 2.0:
        print(f"[SVMRCA] 發現可疑異常，最高分數 {max(scores):.4f} > 2.0")
    else:
        print(f"[SVMRCA] 未發現明顯異常，最高分數僅 {max(scores):.4f}")
    
    print(f"[SVMRCA] 根因分析完成")
    
    ranks = [x[0] for x in ranks]

    return {
        "node_names": normal_df.columns.to_list(),
        "ranks": ranks,
    }
    
    
def mmnsigma(data, inject_time=None, dataset=None, num_loop=None, sli=None, anomalies=None, **kwargs):
    scaler_function = kwargs.get("scaler_function", StandardScaler) 

    print(f"[MMN-SIGMA] 開始執行多模態根因分析")
    print(f"[MMN-SIGMA] 資料集: {dataset}")
    print(f"[MMN-SIGMA] 故障注入時間: {inject_time}")
    print(f"[MMN-SIGMA] 使用的標準化方法: {scaler_function.__name__}")
    
    # 步驟 1: 數據解包
    print(f"\n[MMN-SIGMA] 步驟 1/8: 多模態數據解包")
    metric = data["metric"]
    logs = data["logs"]
    logts = data["logts"]
    traces = data["traces"]
    traces_err = data["tracets_err"]
    traces_lat = data["tracets_lat"]
    cluster_info = data["cluster_info"]
    
    print(f"[MMN-SIGMA] 指標數據形狀: {metric.shape}")
    print(f"[MMN-SIGMA] 日誌時間序列形狀: {logts.shape}")
    print(f"[MMN-SIGMA] 錯誤追蹤形狀: {traces_err.shape}")
    print(f"[MMN-SIGMA] 延迟追蹤形狀: {traces_lat.shape}")
    
    # ==== PREPARE DATA ====
    # 步驟 2: 指標數據重採樣
    print(f"\n[MMN-SIGMA] 步驟 2/8: 指標數據重採樣 (1s -> 15s)")
    print(f"[MMN-SIGMA] 重採樣前指標數據: {metric.shape[0]} 個時間點")
    # the metric is currently sampled for 1 seconds, resample for 15s by just take 1 point every 15 points
    metric = metric.iloc[::15, :]
    print(f"[MMN-SIGMA] 重採樣後指標數據: {metric.shape[0]} 個時間點")

    # 步驟 3: 處理指標數據
    print(f"\n[MMN-SIGMA] 步驟 3/8: 處理指標數據")
    normal_metric = metric[metric["time"] < inject_time]
    anomal_metric = metric[metric["time"] >= inject_time]
    print(f"[MMN-SIGMA] 指標 - 正常期: {normal_metric.shape[0]} 樣本")
    print(f"[MMN-SIGMA] 指標 - 異常期: {anomal_metric.shape[0]} 樣本")
    
    normal_metric = preprocess(data=normal_metric, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False))
    anomal_metric = preprocess(data=anomal_metric, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False))
    intersect = [x for x in normal_metric.columns if x in anomal_metric.columns]
    normal_metric = normal_metric[intersect]
    anomal_metric = anomal_metric[intersect]
    print(f"[MMN-SIGMA] 指標預處理後特徵數: {len(intersect)}")

    # 步驟 4: 處理日誌時間序列
    print(f"\n[MMN-SIGMA] 步驟 4/8: 處理日誌時間序列")
    logts_before = logts.shape[1] - 1  # 減去時間列
    logts = drop_constant(logts)
    logts_after = logts.shape[1] - 1
    print(f"[MMN-SIGMA] 日誌特徵: {logts_before} -> {logts_after} (移除常數特徵)")
    
    normal_logts = logts[logts["time"] < inject_time].drop(columns=["time"])
    anomal_logts = logts[logts["time"] >= inject_time].drop(columns=["time"])
    print(f"[MMN-SIGMA] 日誌 - 正常期: {normal_logts.shape[0]} 樣本")
    print(f"[MMN-SIGMA] 日誌 - 異常期: {anomal_logts.shape[0]} 樣本")

    # 步驟 5: 處理錯誤追蹤數據
    print(f"\n[MMN-SIGMA] 步驟 5/8: 處理錯誤追蹤數據")
    if dataset == "mm-tt" or dataset == "mm-ob":
        print(f"[MMN-SIGMA] 處理 {dataset} 數據集的追蹤數據")
        traces_err_before = traces_err.shape[1] - 1
        traces_err = traces_err.fillna(method='ffill')
        traces_err = traces_err.fillna(0)
        traces_err = drop_constant(traces_err)
        traces_err_after = traces_err.shape[1] - 1
        print(f"[MMN-SIGMA] 錯誤追蹤特徵: {traces_err_before} -> {traces_err_after}")

        normal_traces_err = traces_err[traces_err["time"] < inject_time].drop(columns=["time"])
        anomal_traces_err = traces_err[traces_err["time"] >= inject_time].drop(columns=["time"])
        print(f"[MMN-SIGMA] 錯誤追蹤 - 正常期: {normal_traces_err.shape[0]} 樣本")
        print(f"[MMN-SIGMA] 錯誤追蹤 - 異常期: {anomal_traces_err.shape[0]} 樣本")
    else:
        print(f"[MMN-SIGMA] 跳過錯誤追蹤數據處理 (非 mm-tt/mm-ob 數據集)")
    
    # 步驟 6: 處理延遲追蹤數據
    print(f"\n[MMN-SIGMA] 步驟 6/8: 處理延遲追蹤數據")
    if dataset == "mm-tt" or dataset == "mm-ob":
        traces_lat_before = traces_lat.shape[1] - 1
        traces_lat = traces_lat.fillna(method='ffill')
        traces_lat = traces_lat.fillna(0)
        traces_lat = drop_constant(traces_lat)
        traces_lat_after = traces_lat.shape[1] - 1
        print(f"[MMN-SIGMA] 延遲追蹤特徵: {traces_lat_before} -> {traces_lat_after}")
        
        normal_traces_lat = traces_lat[traces_lat["time"] < inject_time].drop(columns=["time"])
        anomal_traces_lat = traces_lat[traces_lat["time"] >= inject_time].drop(columns=["time"])
        print(f"[MMN-SIGMA] 延遲追蹤 - 正常期: {normal_traces_lat.shape[0]} 樣本")
        print(f"[MMN-SIGMA] 延遲追蹤 - 異常期: {anomal_traces_lat.shape[0]} 樣本")
    else:
        print(f"[MMN-SIGMA] 跳過延遲追蹤數據處理 (非 mm-tt/mm-ob 數據集)")
    
    # ==== PROCESS ====
    # 步驟 7: 計算多模態異常分數
    print(f"\n[MMN-SIGMA] 步驟 7/8: 計算多模態異常分數")
    ranks = []
    processed_features = 0
    
    # == metric ==
    print(f"[MMN-SIGMA] 處理指標特徵...")
    for col in normal_metric.columns:
        if col == "time":
            continue
        a = normal_metric[col].to_numpy()
        b = anomal_metric[col].to_numpy()

        scaler = scaler_function().fit(a.reshape(-1, 1))
        zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
        score = max(zscores)
        ranks.append((col, score))
        processed_features += 1

    print(f"[MMN-SIGMA] 完成 {processed_features} 個指標特徵")

    # == logs ==
    print(f"[MMN-SIGMA] 處理日誌特徵...")
    log_features = 0
    for col in normal_logts.columns:
        a = normal_logts[col].to_numpy()
        b = anomal_logts[col].to_numpy()

        scaler = scaler_function().fit(a.reshape(-1, 1))
        zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
        score = max(zscores)
        ranks.append((col, score))
        log_features += 1

    print(f"[MMN-SIGMA] 完成 {log_features} 個日誌特徵")

    # == traces_err ==
    if dataset == "mm-tt" or dataset == "mm-ob":
        print(f"[MMN-SIGMA] 處理錯誤追蹤特徵...")
        err_features = 0
        for col in normal_traces_err.columns:
            a = normal_traces_err[col].to_numpy()[:-2]
            b = anomal_traces_err[col].to_numpy()
                
            scaler = scaler_function().fit(a.reshape(-1, 1))
            zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
            score = max(zscores)
            ranks.append((col, score))
            err_features += 1
        print(f"[MMN-SIGMA] 完成 {err_features} 個錯誤追蹤特徵")
   
    # == traces_lat ==
    if dataset == "mm-tt" or dataset == "mm-ob":
        print(f"[MMN-SIGMA] 處理延遲追蹤特徵...")
        lat_features = 0
        for col in normal_traces_lat.columns:
            a = normal_traces_lat[col].to_numpy()
            b = anomal_traces_lat[col].to_numpy()

            scaler = scaler_function().fit(a.reshape(-1, 1))
            zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
            score = max(zscores)
            ranks.append((col, score))
            lat_features += 1
        print(f"[MMN-SIGMA] 完成 {lat_features} 個延遲追蹤特徵")

    # 步驟 8: 生成最終結果
    print(f"\n[MMN-SIGMA] 步驟 8/8: 生成最終排序結果")
    total_features = len(ranks)
    print(f"[MMN-SIGMA] 總特徵數: {total_features}")
    
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    
    print(f"\n[MMN-SIGMA] === 多模態根因分析結果 ===")
    print(f"[MMN-SIGMA] Top 15 根因候選:")
    for i, (feature, score) in enumerate(ranks[:15], 1):
        # 判斷特徵類型
        if any(metric_col in feature for metric_col in normal_metric.columns):
            feature_type = "指標"
        elif any(log_col in feature for log_col in normal_logts.columns):
            feature_type = "日誌"
        elif dataset in ["mm-tt", "mm-ob"] and "err" in feature.lower():
            feature_type = "錯誤"
        elif dataset in ["mm-tt", "mm-ob"] and "lat" in feature.lower():
            feature_type = "延遲"
        else:
            feature_type = "其他"
            
        print(f"[MMN-SIGMA]   {i:2d}. [{feature_type}] {feature:<35} (分數: {score:.4f})")
    
    # 統計各類型特徵在 Top 10 中的分布
    top10_types = {"指標": 0, "日誌": 0, "錯誤": 0, "延遲": 0, "其他": 0}
    for feature, _ in ranks[:10]:
        if any(metric_col in feature for metric_col in normal_metric.columns):
            top10_types["指標"] += 1
        elif any(log_col in feature for log_col in normal_logts.columns):
            top10_types["日誌"] += 1
        elif dataset in ["mm-tt", "mm-ob"] and "err" in feature.lower():
            top10_types["錯誤"] += 1
        elif dataset in ["mm-tt", "mm-ob"] and "lat" in feature.lower():
            top10_types["延遲"] += 1
        else:
            top10_types["其他"] += 1
    
    print(f"\n[MMN-SIGMA] === Top 10 特徵類型分布 ===")
    for ftype, count in top10_types.items():
        if count > 0:
            print(f"[MMN-SIGMA] {ftype}: {count} 個")
    
    if kwargs.get("verbose") is True:
        print(f"\n[MMN-SIGMA] === 詳細分數 (Top 20) ===")
        for r, score in ranks[:20]:
            print(f"[MMN-SIGMA] {r}: {score:.2f}")

    print(f"[MMN-SIGMA] 多模態根因分析完成")
    
    ranks = [x[0] for x in ranks]

    return {
        "ranks": ranks,
    }
    

def mmbaro(data, inject_time=None, dataset=None, num_loop=None, sli=None, anomalies=None, **kwargs):
    return mmnsigma(
        data=data,
        inject_time=inject_time,
        dataset=dataset,
        sli=sli,
        scaler_function=RobustScaler, **kwargs
    )