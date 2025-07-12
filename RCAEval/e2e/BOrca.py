import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import multivariate_t, chi2
from scipy.special import logsumexp, gammaln, multigammaln
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
import logging

from RCAEval.io.time_series import (
    drop_constant,
    preprocess,
)

# 嘗試導入因果學習庫
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import fisherz, kci
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False
    print("Warning: causallearn not available. Causal graph integration disabled.")

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultivariateBOCPD:
    """
    Multivariate Bayesian Online Change Point Detection
    基於 Normal-Inverse-Wishart 共軛先驗的嚴格貝葉斯實現
    """
    
    def __init__(self, alpha=None, beta=None, kappa=None, nu=None, mu=None, 
                 max_run_length=200, min_run_length=None):
        # 參數將由 AdaptiveParameterManager 動態設定
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.nu = nu
        self.mu = mu
        self.max_run_length = max_run_length
        self.min_run_length = min_run_length
        self.dimensions = None
        self.data_quality_score = 0.0
        
    def _hazard_function(self, r, adaptive=True):
        """
        變點危險函數 - 使用理論化的幾何分佈
        支援自適應調整
        """
        if adaptive and self.data_quality_score > 0:
            # 根據數據質量調整危險函數
            adjusted_alpha = self.alpha * (0.5 + 0.5 * self.data_quality_score)
            return 1.0 / (adjusted_alpha + r)
        else:
            return 1.0 / (self.alpha + r)
    
    def _log_marginal_likelihood(self, data_segment, use_robust_fallback=True):
        """
        計算 Normal-Inverse-Wishart 共軛先驗下的貝葉斯邊際似然
        包含數值穩定性保障和多重回退機制
        """
        n, d = data_segment.shape
        
        if n == 0:
            return -np.inf
            
        # 設定先驗參數
        mu_0 = np.zeros(d) if self.mu is None else self.mu
        nu_0 = d + 2 if self.nu is None else self.nu
        kappa_0 = self.kappa if self.kappa is not None else 1.0
        
        if isinstance(self.beta, (int, float)):
            Psi_0 = self.beta * np.eye(d)
        elif self.beta is not None:
            Psi_0 = self.beta.copy()
        else:
            Psi_0 = np.eye(d)
        
        try:
            # 計算後驗參數
            sample_mean = np.mean(data_segment, axis=0)
            kappa_n = kappa_0 + n
            mu_n = (kappa_0 * mu_0 + n * sample_mean) / kappa_n
            nu_n = nu_0 + n
            
            # 計算後驗尺度矩陣
            if n == 1:
                diff = data_segment[0] - mu_0
                Psi_n = Psi_0 + (kappa_0 * n / kappa_n) * np.outer(diff, diff)
            else:
                # 使用數值穩定的協方差計算
                centered_data = data_segment - sample_mean
                S = np.dot(centered_data.T, centered_data)
                diff = sample_mean - mu_0
                Psi_n = Psi_0 + S + (kappa_0 * n / kappa_n) * np.outer(diff, diff)
            
            # 確保矩陣正定性
            min_eigenval = np.min(np.real(np.linalg.eigvals(Psi_n)))
            if min_eigenval <= 1e-10:
                Psi_n += np.eye(d) * (1e-8 - min_eigenval)
            
            min_eigenval_0 = np.min(np.real(np.linalg.eigvals(Psi_0)))
            if min_eigenval_0 <= 1e-10:
                Psi_0 += np.eye(d) * (1e-8 - min_eigenval_0)
            
            # 計算邊際似然（在對數空間）
            log_marginal = -0.5 * n * d * np.log(np.pi)
            log_marginal += multigammaln(nu_n/2, d) - multigammaln(nu_0/2, d)
            log_marginal += 0.5 * d * (np.log(kappa_0) - np.log(kappa_n))
            
            # 行列式計算（數值穩定）
            sign_0, logdet_0 = np.linalg.slogdet(Psi_0)
            sign_n, logdet_n = np.linalg.slogdet(Psi_n)
            
            if sign_0 <= 0 or sign_n <= 0:
                raise ValueError("Matrix determinant is non-positive")
            
            log_marginal += 0.5 * nu_0 * logdet_0
            log_marginal -= 0.5 * nu_n * logdet_n
            
            return log_marginal
            
        except Exception as e:
            if use_robust_fallback:
                # 使用多元 t 分佈作為回退
                return self._multivariate_t_log_likelihood(data_segment)
            else:
                logger.warning(f"Marginal likelihood calculation failed: {e}")
                return -np.inf
    
    def _multivariate_t_log_likelihood(self, data_segment):
        """
        多元 t 分佈的對數似然作為回退機制
        """
        try:
            n, d = data_segment.shape
            if n < 2:
                return -np.inf
            
            # 使用樣本統計量
            sample_mean = np.mean(data_segment, axis=0)
            sample_cov = np.cov(data_segment.T, bias=False)
            
            # 確保協方差矩陣正定
            min_eigenval = np.min(np.real(np.linalg.eigvals(sample_cov)))
            if min_eigenval <= 1e-10:
                sample_cov += np.eye(d) * (1e-8 - min_eigenval)
            
            # 使用自由度為 d+1 的 t 分佈
            df = d + 1
            log_likelihood = 0.0
            
            for i in range(n):
                diff = data_segment[i] - sample_mean
                mahalanobis_sq = np.dot(diff, np.linalg.solve(sample_cov, diff))
                
                log_likelihood += gammaln((df + d) / 2) - gammaln(df / 2)
                log_likelihood -= 0.5 * d * np.log(df * np.pi)
                log_likelihood -= 0.5 * np.log(np.linalg.det(sample_cov))
                log_likelihood -= 0.5 * (df + d) * np.log(1 + mahalanobis_sq / df)
            
            return log_likelihood
            
        except Exception as e:
            logger.warning(f"Multivariate t fallback failed: {e}")
            return -np.inf
    
    def detect_changepoints(self, data, timeout_seconds=300):
        """
        檢測變點 - 包含超時保護和動態閾值
        """
        import time
        start_time = time.time()
        
        T, D = data.shape
        self.dimensions = D
        
        # 自適應參數設定
        if self.mu is None:
            self.mu = np.mean(data[:min(T, 50)], axis=0)
        if self.nu is None:
            self.nu = D + 2
        if self.min_run_length is None:
            self.min_run_length = max(5, T // 50)
            
        # 初始化運行長度概率
        log_run_length_probs = np.full((T + 1, self.max_run_length), -np.inf)
        log_run_length_probs[0, 0] = 0.0
        
        changepoints = []
        changepoint_probs = []
        
        for t in range(T):
            # 超時檢查
            if time.time() - start_time > timeout_seconds:
                logger.warning(f"BOCPD timeout at t={t}")
                break
            
            # 計算預測概率
            log_pred_probs = np.full(min(t + 1, self.max_run_length), -np.inf)
            
            for r in range(min(t + 1, self.max_run_length)):
                if log_run_length_probs[t, r] > -np.inf:
                    start_idx = max(0, t - r)
                    data_segment = data[start_idx:t+1, :]
                    if data_segment.shape[0] > 0:
                        log_pred_probs[r] = self._log_marginal_likelihood(data_segment)
            
            # 更新運行長度概率
            if t + 1 < len(log_run_length_probs):
                # Growth probabilities
                for r in range(min(t, self.max_run_length - 1)):
                    if r + 1 < self.max_run_length and log_run_length_probs[t, r] > -np.inf:
                        growth_log_prob = np.log(1 - self._hazard_function(r))
                        new_log_prob = log_run_length_probs[t, r] + growth_log_prob + log_pred_probs[r]
                        log_run_length_probs[t + 1, r + 1] = new_log_prob
                
                # Changepoint probabilities
                changepoint_log_probs = []
                for r in range(min(t + 1, self.max_run_length)):
                    if log_run_length_probs[t, r] > -np.inf:
                        hazard_log_prob = np.log(self._hazard_function(r))
                        changepoint_log_probs.append(
                            log_run_length_probs[t, r] + hazard_log_prob + log_pred_probs[r]
                        )
                
                if changepoint_log_probs:
                    log_run_length_probs[t + 1, 0] = logsumexp(changepoint_log_probs)
                
                # 正規化（數值穩定）
                valid_mask = log_run_length_probs[t + 1, :] > -np.inf
                if np.any(valid_mask):
                    log_normalizer = logsumexp(log_run_length_probs[t + 1, valid_mask])
                    log_run_length_probs[t + 1, valid_mask] -= log_normalizer
                else:
                    log_run_length_probs[t + 1, :] = -np.inf
                    log_run_length_probs[t + 1, 0] = 0.0
            
            # 動態閾值變點檢測
            if t >= self.min_run_length:
                changepoint_prob = (np.exp(log_run_length_probs[t + 1, 0]) 
                                  if t + 1 < len(log_run_length_probs) else 0)
                changepoint_probs.append(changepoint_prob)
                
                # 動態閾值計算
                if len(changepoint_probs) >= self.min_run_length:
                    recent_window = min(50, len(changepoint_probs))
                    recent_probs = changepoint_probs[-recent_window:]
                    
                    baseline_prob = np.median(recent_probs)
                    prob_std = np.std(recent_probs)
                    
                    # 自適應閾值
                    if prob_std > 0:
                        threshold = baseline_prob + 2.0 * prob_std
                    else:
                        threshold = 0.3
                    
                    # 額外的統計顯著性檢驗
                    if (changepoint_prob > threshold and 
                        changepoint_prob > 0.15 and
                        len(changepoints) == 0 or (len(changepoints) > 0 and t - changepoints[-1] > self.min_run_length)):
                        changepoints.append(t)
        
        return changepoints, changepoint_probs


class AdaptiveParameterManager:
    """
    自適應參數管理器 - 根據數據特性動態調整算法參數
    """
    
    def __init__(self):
        self.data_characteristics = {}
        self.parameter_history = []
        
    def analyze_data_characteristics(self, data, feature_names=None):
        """
        分析數據特性，為參數調整提供依據
        """
        if isinstance(data, pd.DataFrame):
            data_array = data.values
            if feature_names is None:
                feature_names = data.columns.tolist()
        else:
            data_array = data
            
        n_samples, n_features = data_array.shape
        
        # 基礎統計特性
        characteristics = {
            'n_samples': n_samples,
            'n_features': n_features,
            'sample_feature_ratio': n_samples / n_features if n_features > 0 else 0,
            'missing_rate': np.mean(np.isnan(data_array)),
            'sparsity': np.mean(data_array == 0) if not np.all(np.isnan(data_array)) else 1.0
        }
        
        # 數據完整性分析
        clean_data = data_array[~np.isnan(data_array).any(axis=1)]
        if len(clean_data) > 0:
            try:
                cov_matrix = np.cov(clean_data.T, bias=False)
                eigenvals = np.linalg.eigvals(cov_matrix)
                condition_number = np.max(eigenvals) / np.min(eigenvals) if np.min(eigenvals) > 0 else np.inf
                
                characteristics.update({
                    'condition_number': condition_number,
                    'numerical_stability': 1.0 / (1.0 + np.log10(max(1, condition_number))),
                    'data_variance': np.mean(np.var(clean_data, axis=0)),
                    'feature_correlation': np.mean(np.abs(np.corrcoef(clean_data.T))[np.triu_indices(n_features, k=1)])
                })
            except:
                characteristics.update({
                    'condition_number': np.inf,
                    'numerical_stability': 0.1,
                    'data_variance': 0.0,
                    'feature_correlation': 0.0
                })
        
        # 信息熵分析
        entropy_scores = []
        for i in range(n_features):
            feature_data = data_array[:, i]
            feature_data = feature_data[~np.isnan(feature_data)]
            if len(feature_data) > 0:
                # 離散化後計算熵
                hist, _ = np.histogram(feature_data, bins=min(50, len(feature_data)//10))
                prob = hist / np.sum(hist)
                prob = prob[prob > 0]
                entropy = -np.sum(prob * np.log2(prob))
                entropy_scores.append(entropy)
        
        characteristics['mean_entropy'] = np.mean(entropy_scores) if entropy_scores else 0.0
        
        # 數據質量評分
        quality_score = self._calculate_data_quality_score(characteristics)
        characteristics['quality_score'] = quality_score
        
        self.data_characteristics = characteristics
        return characteristics
    
    def _calculate_data_quality_score(self, characteristics):
        """
        計算綜合數據質量評分 (0-1)
        """
        # 樣本充足性
        sample_adequacy = min(1.0, characteristics['sample_feature_ratio'] / 10.0)
        
        # 數據完整性
        completeness = 1.0 - characteristics['missing_rate']
        
        # 數值穩定性
        stability = characteristics.get('numerical_stability', 0.1)
        
        # 信息豐富度
        information_richness = min(1.0, characteristics['mean_entropy'] / 5.0)
        
        # 特徵獨立性
        independence = 1.0 - min(0.9, characteristics.get('feature_correlation', 0.0))
        
        # 加權組合
        quality_score = (0.3 * sample_adequacy + 
                        0.25 * completeness + 
                        0.2 * stability + 
                        0.15 * information_richness + 
                        0.1 * independence)
        
        return max(0.0, min(1.0, quality_score))
    
    def get_bocpd_parameters(self, data_characteristics=None):
        """
        根據數據特性獲取 BOCPD 參數
        """
        if data_characteristics is None:
            data_characteristics = self.data_characteristics
        
        quality_score = data_characteristics.get('quality_score', 0.5)
        n_samples = data_characteristics.get('n_samples', 100)
        n_features = data_characteristics.get('n_features', 10)
        
        # 自適應 alpha（變點敏感性）
        base_alpha = 0.1 + 0.4 * (1 - quality_score)  # 低質量數據需要更保守
        alpha = max(0.05, min(0.8, base_alpha))
        
        # 自適應 beta（先驗尺度）
        beta = max(0.1, min(2.0, 1.0 / quality_score))
        
        # 自適應 kappa（先驗置信度）
        kappa = max(0.1, min(2.0, quality_score))
        
        # 自適應運行長度限制
        max_run_length = min(500, max(50, n_samples // 5))
        min_run_length = max(3, min(20, n_samples // 50))
        
        parameters = {
            'alpha': alpha,
            'beta': beta,
            'kappa': kappa,
            'nu': n_features + 2,
            'max_run_length': max_run_length,
            'min_run_length': min_run_length
        }
        
        self.parameter_history.append(parameters)
        return parameters
    
    def get_causal_parameters(self, data_characteristics=None):
        """
        根據數據特性獲取因果發現參數 - 修正為適合監控數據
        """
        if data_characteristics is None:
            data_characteristics = self.data_characteristics
        
        quality_score = data_characteristics.get('quality_score', 0.5)
        n_samples = data_characteristics.get('n_samples', 100)
        n_features = data_characteristics.get('n_features', 10)
        
        # 自適應顯著性水平
        alpha = 0.01 + 0.09 * (1 - quality_score)  # 低質量數據需要更嚴格的檢驗
        
        # 選擇獨立性檢驗方法
        if n_samples > 50 and n_features <= 15:  # 從 200 降低到 50
            indep_test = fisherz
        else:
            indep_test = fisherz  # 保守選擇
        
        # 選擇因果發現算法 - 修正條件
        use_fci = (n_samples > 100 and n_features <= 8 and quality_score > 0.7)  # 從 500 降低到 100
        
        return {
            'alpha': alpha,
            'indep_test': indep_test,
            'use_fci': use_fci,
            'stable': True,
            'show_progress': False
        }


class EnhancedAnomalyScorer:
    """
    增強的異常評分器 - 多維度自適應評分
    """
    
    def __init__(self, anomaly_type='UNKNOWN', data_characteristics=None):
        self.normal_stats = {}
        self.anomaly_type = anomaly_type
        self.data_characteristics = data_characteristics or {}
        self.scoring_weights = self._initialize_adaptive_weights()
        
    def _initialize_adaptive_weights(self):
        """
        根據數據特性初始化自適應權重
        """
        quality_score = self.data_characteristics.get('quality_score', 0.5)
        
        # 基礎權重
        base_weights = {
            'z_score': 0.4,
            'iqr_score': 0.3,
            'range_score': 0.2,
            'change_score': 0.1
        }
        
        # 根據數據質量調整權重
        if quality_score > 0.7:
            # 高質量數據，更依賴統計方法
            base_weights['z_score'] = 0.5
            base_weights['iqr_score'] = 0.35
        elif quality_score < 0.3:
            # 低質量數據，更依賴魯棒方法
            base_weights['iqr_score'] = 0.45
            base_weights['range_score'] = 0.3
            base_weights['z_score'] = 0.15
        
        return base_weights
    
    def learn_normal_distribution(self, data, feature_names):
        """
        學習正常期間的統計分佈參數 - 使用魯棒統計方法
        """
        if isinstance(data, pd.DataFrame):
            data_array = data.values
            if feature_names is None:
                feature_names = data.columns.tolist()
        else:
            data_array = data
            
        self.normal_stats = {}
        
        for i, feature_name in enumerate(feature_names):
            if i < data_array.shape[1]:
                feature_data = data_array[:, i]
                feature_data = feature_data[~np.isnan(feature_data)]
                
                if len(feature_data) > 0:
                    # 魯棒統計量
                    q25, q50, q75 = np.percentile(feature_data, [25, 50, 75])
                    iqr = q75 - q25
                    
                    # 修正的 Z-score 參數（基於中位數）
                    mad = np.median(np.abs(feature_data - q50))
                    robust_std = 1.4826 * mad  # MAD to std conversion
                    
                    self.normal_stats[feature_name] = {
                        'mean': np.mean(feature_data),
                        'std': np.std(feature_data),
                        'median': q50,
                        'q25': q25,
                        'q75': q75,
                        'iqr': iqr,
                        'mad': mad,
                        'robust_std': robust_std,
                        'min': np.min(feature_data),
                        'max': np.max(feature_data),
                        'range': np.max(feature_data) - np.min(feature_data)
                    }
    
    def calculate_anomaly_scores(self, anomaly_data, feature_names):
        """
        計算多維度異常分數
        """
        if isinstance(anomaly_data, pd.DataFrame):
            data_array = anomaly_data.values
            if feature_names is None:
                feature_names = anomaly_data.columns.tolist()
        else:
            data_array = anomaly_data
            
        scores = {}
        
        for i, feature_name in enumerate(feature_names):
            if feature_name in self.normal_stats and i < data_array.shape[1]:
                feature_data = data_array[:, i]
                feature_data = feature_data[~np.isnan(feature_data)]
                
                if len(feature_data) > 0:
                    stats = self.normal_stats[feature_name]
                    
                    # 多種異常分數計算
                    z_scores = self._calculate_z_scores(feature_data, stats)
                    iqr_scores = self._calculate_iqr_scores(feature_data, stats)
                    range_scores = self._calculate_range_scores(feature_data, stats)
                    change_scores = self._calculate_change_scores(feature_data, stats)
                    
                    # 自適應權重組合
                    combined_score = (
                        self.scoring_weights['z_score'] * z_scores +
                        self.scoring_weights['iqr_score'] * iqr_scores +
                        self.scoring_weights['range_score'] * range_scores +
                        self.scoring_weights['change_score'] * change_scores
                    )
                    
                    # 異常類型特定調整
                    type_multiplier = self._get_type_specific_multiplier(feature_name)
                    final_score = combined_score * type_multiplier
                    
                    scores[feature_name] = max(0.0, final_score)
                else:
                    scores[feature_name] = 0.0
            else:
                scores[feature_name] = 0.0
                
        return scores
    
    def _calculate_z_scores(self, feature_data, stats):
        """計算 Z-score 異常分數"""
        if stats['robust_std'] > 1e-10:
            z_scores = np.abs((feature_data - stats['median']) / stats['robust_std'])
            return np.max(z_scores)
        return 0.0
    
    def _calculate_iqr_scores(self, feature_data, stats):
        """計算 IQR 異常分數"""
        if stats['iqr'] > 1e-10:
            lower_bound = stats['q25'] - 1.5 * stats['iqr']
            upper_bound = stats['q75'] + 1.5 * stats['iqr']
            
            outliers = (feature_data < lower_bound) | (feature_data > upper_bound)
            if np.any(outliers):
                outlier_distances = np.maximum(
                    np.abs(feature_data - upper_bound),
                    np.abs(feature_data - lower_bound)
                )
                return np.max(outlier_distances) / stats['iqr']
        return 0.0
    
    def _calculate_range_scores(self, feature_data, stats):
        """計算範圍異常分數"""
        if stats['range'] > 1e-10:
            range_violations = np.maximum(
                stats['min'] - feature_data,
                feature_data - stats['max']
            )
            range_violations = np.maximum(0, range_violations)
            if np.any(range_violations > 0):
                return np.max(range_violations) / stats['range']
        return 0.0
    
    def _calculate_change_scores(self, feature_data, stats):
        """計算變化異常分數"""
        if len(feature_data) > 1:
            changes = np.abs(np.diff(feature_data))
            if len(changes) > 0:
                normal_change = stats['std'] if stats['std'] > 1e-10 else 1e-10
                return np.max(changes) / normal_change
        return 0.0
    
    def _get_type_specific_multiplier(self, feature_name):
        """
        獲取異常類型特定的乘數 - 基於特徵相關性動態計算
        """
        feature_lower = feature_name.lower()
        
        # 基於特徵名稱的相關性評分
        relevance_keywords = {
            'CPU': ['cpu', 'processor', 'core', 'usage', 'utilization'],
            'MEM': ['mem', 'memory', 'ram', 'heap', 'gc'],
            'DISK': ['disk', 'io', 'storage', 'read', 'write'],
            'DELAY': ['latency', 'delay', 'response', 'time', 'duration'],
            'LOSS': ['error', 'fail', 'exception', 'loss', 'drop']
        }
        
        if self.anomaly_type in relevance_keywords:
            keywords = relevance_keywords[self.anomaly_type]
            relevance_score = sum(1 for kw in keywords if kw in feature_lower)
            max_relevance = len(keywords)
            
            if relevance_score > 0:
                # 相關性越高，乘數越大，但避免過度放大
                return 1.0 + (relevance_score / max_relevance) * 0.5
        
        return 1.0


class RobustCausalGraphBuilder:
    """
    魯棒因果圖構建器 - 多算法融合與可靠性評估
    """
    
    def __init__(self, parameter_manager=None):
        self.parameter_manager = parameter_manager
        self.adjacency_matrix = None
        self.feature_names = None
        self.graph_reliability = 0.0
        self.causal_strength = {}
        self.graph_metrics = {}
        
    def assess_causal_feasibility(self, data, feature_names):
        """
        評估是否適合進行因果發現 - 大幅放寬限制並動態調整
        """
        if not CAUSAL_LEARN_AVAILABLE:
            return False, "causallearn not available"
        
        if isinstance(data, pd.DataFrame):
            data_array = data.values
            if feature_names is None:
                feature_names = data.columns.tolist()
        else:
            data_array = data
            
        n_samples, n_features = data_array.shape
        
        # 修正的基本要求檢查 - 適合監控數據
        if n_samples < 15:  # 從 100 降低到 15
            return False, f"Insufficient samples: {n_samples} < 15"
        
        # 動態特徵數量限制 - 根據數據集特點調整
        if n_features <= 50:
            # 小型數據集 (如 online-boutique)
            max_features = 50
        elif n_features <= 200:
            # 中型數據集
            max_features = 100
        else:
            # 大型數據集 (如 re1-tt)
            max_features = 200
        
        if n_features > max_features:
            return False, f"Too many features: {n_features} > {max_features}"
        
        if n_samples / n_features < 0.3:  # 進一步放寬到 0.3
            return False, f"Sample-to-feature ratio too low: {n_samples/n_features} < 0.3"
        
        # 強化數據質量檢查 - 提前處理問題特徵
        clean_data = self._preprocess_data_for_feasibility_check(data_array)
        if clean_data is None:
            return False, "Data preprocessing failed"
        
        n_samples_clean, n_features_clean = clean_data.shape
        
        if n_samples_clean < n_samples * 0.3:  # 進一步放寬到 0.3
            return False, f"Too much missing data: {n_samples_clean/n_samples} < 0.3"
        
        # 重新檢查清理後的特徵數量
        if n_features_clean < 2:
            return False, f"Insufficient features after preprocessing: {n_features_clean} < 2"
        
        return True, "Feasible for causal discovery"
    
    def _preprocess_data_for_feasibility_check(self, data_array):
        """
        為可行性檢查預處理數據 - 移除問題特徵
        """
        try:
            # 移除缺失值行
            clean_data = data_array[~np.isnan(data_array).any(axis=1)]
            
            if len(clean_data) < 10:
                return None
            
            # 移除常數和近常數特徵
            variances = np.var(clean_data, axis=0)
            non_constant_features = variances > 1e-10
            
            if np.sum(non_constant_features) < 2:
                return None
            
            clean_data = clean_data[:, non_constant_features]
            
            # 檢查並處理無限值
            finite_mask = np.all(np.isfinite(clean_data), axis=0)
            if np.sum(finite_mask) < 2:
                return None
            
            clean_data = clean_data[:, finite_mask]
            
            # 標準化數據
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(clean_data)
            
            # 再次檢查是否產生了非有限值
            if not np.all(np.isfinite(scaled_data)):
                # 使用更保守的標準化
                scaled_data = (clean_data - np.median(clean_data, axis=0)) / (np.percentile(clean_data, 75, axis=0) - np.percentile(clean_data, 25, axis=0) + 1e-10)
                scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=3.0, neginf=-3.0)
            
            return scaled_data
            
        except Exception as e:
            logger.warning(f"Data preprocessing for feasibility check failed: {e}")
            return None
    
    def learn_causal_structure(self, data, feature_names):
        """
        學習因果結構 - 多算法融合
        """
        feasible, reason = self.assess_causal_feasibility(data, feature_names)
        if not feasible:
            logger.info(f"Causal discovery not feasible: {reason}")
            return None
        
        if isinstance(data, pd.DataFrame):
            data_array = data.values
            if feature_names is None:
                feature_names = data.columns.tolist()
        else:
            data_array = data
            
        self.feature_names = feature_names
        
        # 數據預處理
        clean_data = self._preprocess_for_causal_discovery(data_array)
        if clean_data is None:
            return None
        
        # 獲取自適應參數
        if self.parameter_manager:
            params = self.parameter_manager.get_causal_parameters()
        else:
            params = {'alpha': 0.05, 'indep_test': fisherz, 'use_fci': False}
        
        # 嘗試多種因果發現算法
        causal_results = []
        
        # PC 算法
        try:
            pc_result = self._run_pc_algorithm(clean_data, params)
            if pc_result is not None:
                causal_results.append(('PC', pc_result))
        except Exception as e:
            logger.warning(f"PC algorithm failed: {e}")
        
        # FCI 算法（如果適用）
        if params.get('use_fci', False):
            try:
                fci_result = self._run_fci_algorithm(clean_data, params)
                if fci_result is not None:
                    causal_results.append(('FCI', fci_result))
            except Exception as e:
                logger.warning(f"FCI algorithm failed: {e}")
        
        # 選擇最佳結果
        if causal_results:
            best_result = self._select_best_causal_result(causal_results, clean_data)
            self.adjacency_matrix = best_result['adjacency_matrix']
            self.graph_reliability = best_result['reliability']
            self.causal_strength = best_result['strength']
            self.graph_metrics = best_result['metrics']
            
            return self.adjacency_matrix
        
        return None
    
    def _preprocess_for_causal_discovery(self, data_array):
        """
        為因果發現預處理數據 - 激進的特徵篩選和數值穩定化
        """
        try:
            # 移除缺失值
            clean_data = data_array[~np.isnan(data_array).any(axis=1)]
            
            # 檢查處理後的數據 - 修正最小樣本要求
            if len(clean_data) < 10:  # 從 50 降低到 10
                return None
            
            # 異常值處理（使用魯棒方法）
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(clean_data)
            
            # 多重共線性處理 - 關鍵步驟
            processed_data = self._remove_multicollinearity(scaled_data)
            
            if processed_data is None or processed_data.shape[1] < 2:
                logger.warning("After multicollinearity removal, insufficient features remain")
                return None
            
            # 進一步限制特徵數量 - 為PC算法優化
            n_samples, n_features = processed_data.shape
            max_features_for_pc = min(20, n_samples // 3)  # 更激進的限制
            
            if n_features > max_features_for_pc:
                logger.info(f"Further reducing features for PC algorithm: {n_features} -> {max_features_for_pc}")
                processed_data = self._aggressive_feature_selection(processed_data, max_features_for_pc)
            
            if processed_data is None or processed_data.shape[1] < 2:
                logger.warning("After aggressive feature selection, insufficient features remain")
                return None
            
            return processed_data
            
        except Exception as e:
            logger.warning(f"Data preprocessing for causal discovery failed: {e}")
            return None
    
    def _aggressive_feature_selection(self, data, max_features):
        """
        激進的特徵選擇 - 為PC算法優化
        """
        try:
            n_samples, n_features = data.shape
            
            # 基於方差選擇特徵
            variances = np.var(data, axis=0)
            
            # 選擇方差最大的特徵
            top_variance_indices = np.argsort(variances)[-max_features:]
            
            # 進一步檢查選中特徵的相關性
            selected_data = data[:, top_variance_indices]
            
            # 移除高度相關的特徵
            corr_matrix = np.corrcoef(selected_data.T)
            if np.all(np.isfinite(corr_matrix)):
                # 簡單的相關性篩選
                keep_features = [0]  # 保留第一個特徵
                for i in range(1, len(top_variance_indices)):
                    max_corr = np.max(np.abs(corr_matrix[i, keep_features]))
                    if max_corr < 0.8:  # 相關性閾值
                        keep_features.append(i)
                    
                    if len(keep_features) >= max_features:
                        break
                
                final_indices = [top_variance_indices[i] for i in keep_features]
                logger.info(f"Aggressive feature selection: {n_features} -> {len(final_indices)} features")
                return data[:, final_indices]
            else:
                logger.info(f"Aggressive feature selection (variance only): {n_features} -> {max_features} features")
                return selected_data
                
        except Exception as e:
            logger.warning(f"Aggressive feature selection failed: {e}")
            return data[:, :max_features] if max_features < data.shape[1] else data
    
    def _remove_multicollinearity(self, data, max_condition_number=1e8):
        """
        移除多重共線性問題 - 改進的處理策略
        """
        try:
            n_samples, n_features = data.shape
            
            # 首先移除任何包含非有限值的特徵
            finite_mask = np.all(np.isfinite(data), axis=0)
            if np.sum(finite_mask) < 2:
                logger.warning("Too few features with finite values")
                return self._fallback_feature_selection(data)
            
            if np.sum(finite_mask) < n_features:
                logger.info(f"Removing {n_features - np.sum(finite_mask)} features with non-finite values")
                data = data[:, finite_mask]
                n_features = data.shape[1]
            
            # 移除常數特徵
            variances = np.var(data, axis=0)
            non_constant_mask = variances > 1e-12
            if np.sum(non_constant_mask) < 2:
                logger.warning("Too few non-constant features")
                return self._fallback_feature_selection(data)
            
            if np.sum(non_constant_mask) < n_features:
                logger.info(f"Removing {n_features - np.sum(non_constant_mask)} constant features")
                data = data[:, non_constant_mask]
                n_features = data.shape[1]
            
            # 計算相關係數矩陣 - 使用更魯棒的方法
            try:
                # 使用 Spearman 相關係數作為備選，對異常值更魯棒
                from scipy.stats import spearmanr
                corr_matrix, _ = spearmanr(data, axis=0)
                
                # 處理可能的 NaN 值
                if not np.all(np.isfinite(corr_matrix)):
                    logger.warning("Spearman correlation matrix contains non-finite values, using Pearson")
                    corr_matrix = np.corrcoef(data.T)
                    
                    # 如果 Pearson 相關係數也有問題，使用保守的處理
                    if not np.all(np.isfinite(corr_matrix)):
                        logger.warning("Pearson correlation matrix also contains non-finite values")
                        return self._fallback_feature_selection(data)
                        
            except Exception as e:
                logger.warning(f"Correlation calculation failed: {e}")
                return self._fallback_feature_selection(data)
            
            # 使用特徵值分解檢查共線性
            try:
                eigenvals = np.linalg.eigvals(corr_matrix)
                eigenvals = eigenvals[eigenvals > 1e-12]  # 移除接近零的特徵值
                
                if len(eigenvals) < 2:
                    logger.warning("Too few significant eigenvalues")
                    return self._fallback_feature_selection(data)
                
                condition_number = np.max(eigenvals) / np.min(eigenvals)
                
                if not np.isfinite(condition_number) or condition_number > max_condition_number:
                    logger.info(f"High multicollinearity detected (condition number: {condition_number}), applying dimensionality reduction")
                    return self._apply_dimensionality_reduction(data)
                
            except Exception as e:
                logger.warning(f"Eigenvalue analysis failed: {e}")
                return self._apply_dimensionality_reduction(data)
            
            # 逐步移除高度相關的特徵
            return self._iterative_correlation_removal(data, corr_matrix, threshold=0.9)
            
        except Exception as e:
            logger.warning(f"Multicollinearity removal failed: {e}")
            return self._fallback_feature_selection(data)
    
    def _apply_dimensionality_reduction(self, data, variance_threshold=0.95):
        """
        使用主成分分析降維
        """
        try:
            from sklearn.decomposition import PCA
            
            # 確定主成分數量
            n_samples, n_features = data.shape
            max_components = min(n_samples - 1, n_features, 20)  # 限制最大主成分數
            
            pca = PCA(n_components=max_components)
            transformed_data = pca.fit_transform(data)
            
            # 選擇解釋足夠方差的主成分
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
            
            # 確保至少保留2個主成分
            n_components = max(2, min(n_components, max_components))
            
            logger.info(f"PCA dimensionality reduction: {n_features} -> {n_components} components")
            return transformed_data[:, :n_components]
            
        except Exception as e:
            logger.warning(f"PCA dimensionality reduction failed: {e}")
            return self._fallback_feature_selection(data)
    
    def _iterative_correlation_removal(self, data, corr_matrix, threshold=0.95):
        """
        迭代移除高度相關的特徵
        """
        try:
            n_features = data.shape[1]
            selected_features = list(range(n_features))
            
            # 迭代移除高度相關的特徵對
            while True:
                high_corr_pairs = []
                
                for i in range(len(selected_features)):
                    for j in range(i + 1, len(selected_features)):
                        feat_i, feat_j = selected_features[i], selected_features[j]
                        if abs(corr_matrix[feat_i, feat_j]) > threshold:
                            high_corr_pairs.append((feat_i, feat_j, abs(corr_matrix[feat_i, feat_j])))
                
                if not high_corr_pairs:
                    break
                
                # 移除相關性最高的特徵對中的一個
                high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
                feat_to_remove = high_corr_pairs[0][1]  # 移除第二個特徵
                
                if feat_to_remove in selected_features:
                    selected_features.remove(feat_to_remove)
                
                # 如果剩餘特徵太少，停止
                if len(selected_features) < 2:
                    break
            
            if len(selected_features) < 2:
                logger.warning("Too few features remaining after correlation removal")
                return self._fallback_feature_selection(data)
            
            logger.info(f"Correlation-based feature selection: {n_features} -> {len(selected_features)} features")
            return data[:, selected_features]
            
        except Exception as e:
            logger.warning(f"Iterative correlation removal failed: {e}")
            return self._fallback_feature_selection(data)
    
    def _fallback_feature_selection(self, data, max_features=10):
        """
        後備特徵選擇方法
        """
        try:
            n_samples, n_features = data.shape
            
            # 基於方差選擇特徵
            variances = np.var(data, axis=0)
            valid_features = np.where(variances > 1e-10)[0]
            
            if len(valid_features) < 2:
                logger.warning("Insufficient valid features for causal discovery")
                return None
            
            # 選擇方差最大的特徵
            n_select = min(max_features, len(valid_features))
            top_variance_indices = valid_features[np.argsort(variances[valid_features])[-n_select:]]
            
            logger.info(f"Fallback feature selection: {n_features} -> {len(top_variance_indices)} features")
            return data[:, top_variance_indices]
            
        except Exception as e:
            logger.warning(f"Fallback feature selection failed: {e}")
            return None
    
    def _run_pc_algorithm(self, data, params):
        """
        運行 PC 算法 - 添加超時保護和保守參數
        """
        import signal
        import time
        
        def timeout_handler(signum, frame):
            raise TimeoutError("PC algorithm timeout")
        
        try:
            # 設置超時保護（30秒）
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
            
            # 使用更保守的參數
            conservative_params = {
                'alpha': max(0.01, params.get('alpha', 0.05)),  # 更嚴格的顯著性水平
                'indep_test': params.get('indep_test', fisherz),
                'stable': True,  # 強制使用穩定版本
                'show_progress': False,
                'verbose': False
            }
            
            # 如果特徵數量較多，進一步調整參數
            n_samples, n_features = data.shape
            if n_features > 20:
                conservative_params['alpha'] = 0.001  # 更嚴格
            
            logger.info(f"Running PC algorithm with {n_features} features, alpha={conservative_params['alpha']}")
            
            # 執行PC算法
            cg = pc(data, **conservative_params)
            
            # 取消超時
            signal.alarm(0)
            
            adjacency_matrix = cg.G.graph
            
            # 計算可靠性指標
            reliability = self._assess_graph_reliability(adjacency_matrix, data, params)
            strength = self._calculate_causal_strength(adjacency_matrix, data)
            metrics = self._calculate_graph_metrics(adjacency_matrix)
            
            return {
                'adjacency_matrix': adjacency_matrix,
                'reliability': reliability,
                'strength': strength,
                'metrics': metrics,
                'algorithm': 'PC'
            }
            
        except TimeoutError:
            logger.warning("PC algorithm timed out after 30 seconds")
            signal.alarm(0)
            return None
        except Exception as e:
            logger.warning(f"PC algorithm failed: {e}")
            signal.alarm(0)
            return None
        finally:
            # 確保清理超時設置
            signal.alarm(0)
    
    def _run_fci_algorithm(self, data, params):
        """
        運行 FCI 算法
        """
        try:
            cg = fci(data,
                     alpha=params['alpha'],
                     indep_test=params['indep_test'],
                     stable=params.get('stable', True),
                     show_progress=params.get('show_progress', False))
            
            adjacency_matrix = cg.G.graph
            
            # 計算可靠性指標
            reliability = self._assess_graph_reliability(adjacency_matrix, data, params)
            strength = self._calculate_causal_strength(adjacency_matrix, data)
            metrics = self._calculate_graph_metrics(adjacency_matrix)
            
            return {
                'adjacency_matrix': adjacency_matrix,
                'reliability': reliability,
                'strength': strength,
                'metrics': metrics,
                'algorithm': 'FCI'
            }
        except Exception as e:
            logger.warning(f"FCI algorithm execution failed: {e}")
            return None
    
    def _assess_graph_reliability(self, adjacency_matrix, data, params):
        """
        評估因果圖可靠性
        """
        try:
            n_nodes = adjacency_matrix.shape[0]
            n_edges = np.sum(adjacency_matrix != 0)
            
            # 基於圖結構的可靠性
            density = n_edges / (n_nodes * (n_nodes - 1))
            structure_score = 1.0 - abs(density - 0.1)  # 適中的密度較好
            
            # 基於統計檢驗的可靠性
            stat_score = self._bootstrap_reliability(adjacency_matrix, data, params)
            
            # 綜合可靠性
            reliability = 0.6 * structure_score + 0.4 * stat_score
            return max(0.0, min(1.0, reliability))
            
        except Exception as e:
            logger.warning(f"Graph reliability assessment failed: {e}")
            return 0.3
    
    def _bootstrap_reliability(self, adjacency_matrix, data, params, n_bootstrap=20):
        """
        使用 bootstrap 評估圖結構穩定性
        """
        try:
            n_samples = data.shape[0]
            edge_stability = []
            
            for _ in range(n_bootstrap):
                # Bootstrap 採樣
                bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
                bootstrap_data = data[bootstrap_indices]
                
                try:
                    # 重新學習因果結構
                    cg = pc(bootstrap_data,
                            alpha=params['alpha'],
                            indep_test=params['indep_test'],
                            stable=params.get('stable', True),
                            show_progress=False)
                    
                    bootstrap_adj = cg.G.graph
                    
                    # 計算邊的一致性
                    edge_consistency = np.mean((adjacency_matrix != 0) == (bootstrap_adj != 0))
                    edge_stability.append(edge_consistency)
                    
                except:
                    continue
            
            if edge_stability:
                return np.mean(edge_stability)
            else:
                return 0.3
                
        except Exception as e:
            logger.warning(f"Bootstrap reliability assessment failed: {e}")
            return 0.3
    
    def _calculate_causal_strength(self, adjacency_matrix, data):
        """
        計算因果強度
        """
        strength = {}
        n_nodes = adjacency_matrix.shape[0]
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adjacency_matrix[i, j] != 0:
                    # 基於條件相關性的強度估計
                    try:
                        corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
                        strength[(i, j)] = abs(corr)
                    except:
                        strength[(i, j)] = 0.1
        
        return strength
    
    def _calculate_graph_metrics(self, adjacency_matrix):
        """
        計算圖的拓撲指標
        """
        try:
            # 轉換為 NetworkX 圖
            G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
            
            metrics = {
                'n_nodes': G.number_of_nodes(),
                'n_edges': G.number_of_edges(),
                'density': nx.density(G),
                'is_dag': nx.is_directed_acyclic_graph(G),
                'weakly_connected': nx.is_weakly_connected(G),
                'n_strongly_connected': nx.number_strongly_connected_components(G)
            }
            
            # 計算中心性指標
            try:
                in_degree_centrality = nx.in_degree_centrality(G)
                out_degree_centrality = nx.out_degree_centrality(G)
                
                metrics['max_in_degree'] = max(in_degree_centrality.values()) if in_degree_centrality else 0
                metrics['max_out_degree'] = max(out_degree_centrality.values()) if out_degree_centrality else 0
            except:
                metrics['max_in_degree'] = 0
                metrics['max_out_degree'] = 0
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Graph metrics calculation failed: {e}")
            return {'n_nodes': 0, 'n_edges': 0, 'density': 0, 'is_dag': False}
    
    def _select_best_causal_result(self, causal_results, data):
        """
        選擇最佳因果發現結果
        """
        if len(causal_results) == 1:
            return causal_results[0][1]
        
        # 多算法結果比較
        best_result = None
        best_score = -1
        
        for algorithm, result in causal_results:
            # 綜合評分
            score = (0.4 * result['reliability'] + 
                    0.3 * (1.0 if result['metrics']['is_dag'] else 0.0) +
                    0.2 * min(1.0, result['metrics']['density'] * 10) +
                    0.1 * (1.0 if result['metrics']['weakly_connected'] else 0.0))
            
            if score > best_score:
                best_score = score
                best_result = result
        
        return best_result if best_result else causal_results[0][1]
    
    def compute_causal_scores(self, anomaly_data, statistical_scores):
        """
        基於因果圖調整分數
        """
        if (self.adjacency_matrix is None or 
            self.feature_names is None or 
            self.graph_reliability < 0.4):
            return statistical_scores
        
        causal_scores = statistical_scores.copy()
        
        # 計算因果調整係數
        for i, feature in enumerate(self.feature_names):
            if feature in statistical_scores and i < len(self.adjacency_matrix):
                base_score = statistical_scores[feature]
                
                # 節點重要性（基於度中心性）
                out_degree = np.sum(self.adjacency_matrix[i, :] != 0)
                in_degree = np.sum(self.adjacency_matrix[:, i] != 0)
                
                # 根因重要性（出度高、入度低的節點更可能是根因）
                if in_degree == 0 and out_degree > 0:
                    root_cause_weight = 2.0  # 可能的根因
                elif out_degree > in_degree:
                    root_cause_weight = 1.5  # 上游節點
                elif out_degree < in_degree:
                    root_cause_weight = 0.8  # 下游節點
                else:
                    root_cause_weight = 1.0  # 中間節點
                
                # 傳播效應懲罰
                propagation_penalty = self._calculate_propagation_penalty(i, statistical_scores)
                
                # 因果強度加權
                strength_weight = self._get_node_strength_weight(i)
                
                # 綜合調整
                causal_adjustment = (root_cause_weight * 
                                   strength_weight * 
                                   (1.0 - propagation_penalty) * 
                                   self.graph_reliability)
                
                adjusted_score = base_score * causal_adjustment
                causal_scores[feature] = max(0.0, adjusted_score)
        
        return causal_scores
    
    def _calculate_propagation_penalty(self, node_idx, statistical_scores):
        """
        計算傳播效應懲罰
        """
        try:
            # 找到所有父節點
            parents = np.where(self.adjacency_matrix[:, node_idx] != 0)[0]
            
            if len(parents) == 0:
                return 0.0  # 沒有父節點，無懲罰
            
            # 計算父節點的平均異常分數
            parent_scores = []
            for parent_idx in parents:
                if parent_idx < len(self.feature_names):
                    parent_feature = self.feature_names[parent_idx]
                    if parent_feature in statistical_scores:
                        parent_scores.append(statistical_scores[parent_feature])
            
            if not parent_scores:
                return 0.0
            
            avg_parent_score = np.mean(parent_scores)
            current_feature = self.feature_names[node_idx]
            current_score = statistical_scores.get(current_feature, 0.0)
            
            # 如果父節點分數很高，當前節點可能是傳播效應
            if avg_parent_score > 0 and current_score > 0:
                propagation_ratio = min(1.0, avg_parent_score / current_score)
                return 0.3 * propagation_ratio
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Propagation penalty calculation failed: {e}")
            return 0.0
    
    def _get_node_strength_weight(self, node_idx):
        """
        獲取節點強度權重
        """
        try:
            # 計算該節點所有邊的平均強度
            strengths = []
            
            # 出邊強度
            for j in range(len(self.adjacency_matrix)):
                if self.adjacency_matrix[node_idx, j] != 0:
                    edge_key = (node_idx, j)
                    if edge_key in self.causal_strength:
                        strengths.append(self.causal_strength[edge_key])
            
            # 入邊強度
            for i in range(len(self.adjacency_matrix)):
                if self.adjacency_matrix[i, node_idx] != 0:
                    edge_key = (i, node_idx)
                    if edge_key in self.causal_strength:
                        strengths.append(self.causal_strength[edge_key])
            
            if strengths:
                return min(2.0, 1.0 + np.mean(strengths))
            else:
                return 1.0
                
        except Exception as e:
            logger.warning(f"Node strength weight calculation failed: {e}")
            return 1.0


class IntelligentFeatureSelector:
    """
    智能特徵選擇器 - 多階段自適應特徵選擇
    """
    
    def __init__(self, anomaly_type='UNKNOWN', data_characteristics=None):
        self.anomaly_type = anomaly_type
        self.data_characteristics = data_characteristics or {}
        self.selection_history = []
        
    def select_features(self, data, feature_names=None, max_features=None):
        """
        多階段智能特徵選擇
        """
        if feature_names is None:
            feature_names = data.columns.tolist() if isinstance(data, pd.DataFrame) else []
        
        # 自適應最大特徵數
        if max_features is None:
            max_features = self._determine_optimal_feature_count(data, feature_names)
        
        # 階段1: 基於異常類型的優先級篩選
        priority_features = self._select_by_anomaly_type(feature_names)
        
        # 階段2: 數據質量評估
        quality_features = self._assess_feature_quality(data, feature_names)
        
        # 階段3: 信息量評估
        information_features = self._assess_information_content(data, feature_names)
        
        # 階段4: 相關性和冗餘度分析
        final_features = self._remove_redundancy(data, priority_features, 
                                                quality_features, information_features, 
                                                max_features)
        
        # 確保最小特徵數
        if len(final_features) < 3:
            final_features = self._emergency_feature_selection(data, feature_names, max_features)
        
        # 記錄選擇歷史
        self.selection_history.append({
            'original_count': len(feature_names),
            'selected_count': len(final_features),
            'selected_features': final_features,
            'anomaly_type': self.anomaly_type
        })
        
        # 返回篩選後的數據
        if isinstance(data, pd.DataFrame):
            selected_data = data[final_features]
        else:
            feature_indices = [i for i, f in enumerate(feature_names) if f in final_features]
            selected_data = data[:, feature_indices]
        
        return selected_data, final_features
    
    def _determine_optimal_feature_count(self, data, feature_names):
        """
        確定最佳特徵數量 - 考慮因果分析的需求
        """
        n_samples = len(data)
        n_features = len(feature_names)
        
        # 基於樣本數的限制
        sample_based_limit = max(5, min(40, n_samples // 15))  # 更保守的限制
        
        # 基於數據質量的調整
        quality_score = self.data_characteristics.get('quality_score', 0.5)
        quality_adjustment = int(sample_based_limit * (0.5 + 0.5 * quality_score))
        
        # 基於異常類型的調整
        type_adjustment = self._get_type_specific_feature_limit()
        
        # 考慮因果分析的限制 - 動態調整
        if n_features <= 50:
            # 小型數據集，因果分析限制為50
            causal_limit = 40
        elif n_features <= 200:
            # 中型數據集，因果分析限制為100
            causal_limit = 80
        else:
            # 大型數據集，因果分析限制為200
            causal_limit = 150
        
        # 選擇最保守的限制
        optimal_count = min(sample_based_limit, quality_adjustment, type_adjustment, causal_limit, n_features)
        return max(3, optimal_count)
    
    def _get_type_specific_feature_limit(self):
        """
        獲取異常類型特定的特徵限制 - 更保守的設定
        """
        type_limits = {
            'CPU': 20,      # 從 15 增加到 20
            'MEM': 18,      # 從 12 增加到 18
            'DISK': 25,     # 從 18 增加到 25
            'DELAY': 30,    # 從 20 增加到 30
            'LOSS': 15,     # 從 10 增加到 15
            'UNKNOWN': 25   # 從 15 增加到 25
        }
        return type_limits.get(self.anomaly_type, 25)
    
    def _select_by_anomaly_type(self, feature_names):
        """
        基於異常類型選擇優先特徵
        """
        type_keywords = {
            'CPU': ['cpu', 'processor', 'core', 'usage', 'util', 'load'],
            'MEM': ['mem', 'memory', 'ram', 'heap', 'gc', 'cache'],
            'DISK': ['disk', 'io', 'storage', 'read', 'write', 'iops'],
            'DELAY': ['latency', 'delay', 'response_time', 'duration', 'rt'],
            'LOSS': ['error', 'fail', 'exception', 'loss', 'drop', 'timeout']
        }
        
        priority_features = []
        
        if self.anomaly_type in type_keywords:
            keywords = type_keywords[self.anomaly_type]
            for feature in feature_names:
                if feature != 'time':
                    feature_lower = feature.lower()
                    relevance_score = sum(1 for kw in keywords if kw in feature_lower)
                    if relevance_score > 0:
                        priority_features.append((feature, relevance_score))
        
        # 按相關性排序
        priority_features.sort(key=lambda x: x[1], reverse=True)
        return [f[0] for f in priority_features]
    
    def _assess_feature_quality(self, data, feature_names):
        """
        評估特徵質量
        """
        quality_scores = {}
        
        for feature in feature_names:
            if feature == 'time':
                continue
                
            if feature in data.columns if isinstance(data, pd.DataFrame) else feature in feature_names:
                try:
                    if isinstance(data, pd.DataFrame):
                        feature_data = data[feature].values
                    else:
                        feature_idx = feature_names.index(feature)
                        feature_data = data[:, feature_idx]
                    
                    # 數據完整性
                    completeness = 1.0 - np.mean(np.isnan(feature_data))
                    
                    # 變異性
                    clean_data = feature_data[~np.isnan(feature_data)]
                    if len(clean_data) > 0:
                        variance = np.var(clean_data)
                        if variance > 0:
                            cv = np.std(clean_data) / np.abs(np.mean(clean_data)) if np.mean(clean_data) != 0 else 0
                            variability = min(1.0, cv)
                        else:
                            variability = 0.0
                    else:
                        variability = 0.0
                    
                    # 綜合質量分數
                    quality_score = 0.6 * completeness + 0.4 * variability
                    quality_scores[feature] = quality_score
                    
                except Exception as e:
                    logger.warning(f"Quality assessment failed for feature {feature}: {e}")
                    quality_scores[feature] = 0.0
        
        # 返回質量分數 > 0.3 的特徵
        return [f for f, score in quality_scores.items() if score > 0.3]
    
    def _assess_information_content(self, data, feature_names):
        """
        評估特徵信息含量
        """
        information_scores = {}
        
        for feature in feature_names:
            if feature == 'time':
                continue
                
            try:
                if isinstance(data, pd.DataFrame):
                    feature_data = data[feature].values
                else:
                    feature_idx = feature_names.index(feature)
                    feature_data = data[:, feature_idx]
                
                clean_data = feature_data[~np.isnan(feature_data)]
                
                if len(clean_data) > 0:
                    # 計算信息熵
                    # 離散化數據
                    n_bins = min(50, max(10, len(clean_data) // 20))
                    hist, _ = np.histogram(clean_data, bins=n_bins)
                    
                    # 計算概率分佈
                    prob = hist / np.sum(hist)
                    prob = prob[prob > 0]
                    
                    # 信息熵
                    entropy = -np.sum(prob * np.log2(prob))
                    
                    # 正規化熵值
                    max_entropy = np.log2(len(prob))
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                    
                    information_scores[feature] = normalized_entropy
                else:
                    information_scores[feature] = 0.0
                    
            except Exception as e:
                logger.warning(f"Information content assessment failed for feature {feature}: {e}")
                information_scores[feature] = 0.0
        
        # 返回信息含量 > 0.1 的特徵
        return [f for f, score in information_scores.items() if score > 0.1]
    
    def _remove_redundancy(self, data, priority_features, quality_features, 
                          information_features, max_features):
        """
        移除冗餘特徵
        """
        # 候選特徵集合
        candidate_features = list(set(priority_features + quality_features + information_features))
        candidate_features = [f for f in candidate_features if f != 'time']
        
        if len(candidate_features) <= max_features:
            return candidate_features
        
        # 計算特徵間相關性
        correlation_matrix = self._calculate_feature_correlations(data, candidate_features)
        
        # 貪婪算法選擇特徵
        selected_features = []
        remaining_features = candidate_features.copy()
        
        # 首先選擇優先級最高的特徵
        for feature in priority_features:
            if feature in remaining_features and len(selected_features) < max_features:
                selected_features.append(feature)
                remaining_features.remove(feature)
        
        # 然後基於相關性選擇其餘特徵
        while len(selected_features) < max_features and remaining_features:
            best_feature = None
            best_score = -1
            
            for feature in remaining_features:
                # 計算與已選特徵的平均相關性
                if selected_features:
                    avg_correlation = np.mean([
                        abs(correlation_matrix.get((feature, selected), 0.0))
                        for selected in selected_features
                        if (feature, selected) in correlation_matrix
                    ])
                else:
                    avg_correlation = 0.0
                
                # 特徵選擇分數（低相關性更好）
                score = 1.0 - avg_correlation
                
                # 加入質量和信息量權重
                if feature in quality_features:
                    score += 0.2
                if feature in information_features:
                    score += 0.1
                
                if score > best_score:
                    best_score = score
                    best_feature = feature
            
            if best_feature:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
            else:
                break
        
        return selected_features
    
    def _calculate_feature_correlations(self, data, features):
        """
        計算特徵間相關性
        """
        correlations = {}
        
        try:
            if isinstance(data, pd.DataFrame):
                feature_data = data[features].values
            else:
                feature_indices = [i for i, f in enumerate(data.columns if hasattr(data, 'columns') else features) 
                                 if f in features]
                feature_data = data[:, feature_indices]
            
            # 移除缺失值
            clean_data = feature_data[~np.isnan(feature_data).any(axis=1)]
            
            if len(clean_data) > 1:
                corr_matrix = np.corrcoef(clean_data.T)
                
                for i, feature1 in enumerate(features):
                    for j, feature2 in enumerate(features):
                        if i != j:
                            correlations[(feature1, feature2)] = corr_matrix[i, j]
            
        except Exception as e:
            logger.warning(f"Correlation calculation failed: {e}")
        
        return correlations
    
    def _emergency_feature_selection(self, data, feature_names, max_features):
        """
        緊急特徵選擇 - 當其他方法失敗時使用
        """
        # 選擇數值型特徵
        if isinstance(data, pd.DataFrame):
            numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_features = [f for f in feature_names if f != 'time']
        
        # 移除時間列
        numeric_features = [f for f in numeric_features if f != 'time']
        
        # 隨機選擇（確保確定性）
        np.random.seed(42)
        selected_count = min(max_features, len(numeric_features))
        selected_features = np.random.choice(numeric_features, selected_count, replace=False).tolist()
        
        return selected_features


def borca_multimodal(data, inject_time=None, dataset=None, num_loop=None, sli=None, 
                    anomalies=None, verbose=False, **kwargs):
    """
    BOrca 多模態根因分析演算法 - 完整重構版本
    基於嚴格的貝葉斯理論、自適應參數管理和魯棒因果推斷
    """
    
    if verbose:
        print("=== BOrca 多模態根因分析 - 完整重構版本 ===")
    
    # === 初始化組件 ===
    parameter_manager = AdaptiveParameterManager()
    
    # === 數據預處理 ===
    if isinstance(data, dict):
        # 多模態數據處理
        all_features = []
        all_data_list = []
        
        for data_type in ['metric', 'logs', 'logts', 'traces', 'traces_err', 'tracets_err', 'traces_lat', 'tracets_lat']:
            if data_type in data:
                processed_data = data[data_type]
                if 'time' in processed_data.columns:
                    processed_data = processed_data.drop(columns=['time'])
                processed_data = drop_constant(processed_data)
                
                features = [f"{data_type}_{col}" for col in processed_data.columns]
                all_features.extend(features)
                all_data_list.append(processed_data.values)
        
        if all_data_list:
            min_rows = min(arr.shape[0] for arr in all_data_list)
            truncated_arrays = [arr[:min_rows] for arr in all_data_list]
            combined_data = np.hstack(truncated_arrays)
            combined_df = pd.DataFrame(combined_data, columns=all_features)
        else:
            raise ValueError("沒有找到有效的數據")
    else:
        # 單一數據格式
        combined_df = preprocess(data=data, dataset=dataset, 
                               dk_select_useful=kwargs.get("dk_select_useful", False))
        all_features = [col for col in combined_df.columns if col != 'time']
    
    if verbose:
        print(f"數據形狀: {combined_df.shape}, 特徵數量: {len(all_features)}")
    
    # === 數據特性分析 ===
    data_characteristics = parameter_manager.analyze_data_characteristics(
        combined_df[all_features], all_features
    )
    
    if verbose:
        print(f"數據質量評分: {data_characteristics['quality_score']:.3f}")
        print(f"樣本/特徵比例: {data_characteristics['sample_feature_ratio']:.1f}")
        print(f"數值穩定性: {data_characteristics['numerical_stability']:.3f}")
    
    # === 異常類型檢測 ===
    # 確定異常時間段
    if inject_time is not None and 'time' in combined_df.columns:
        temp_anomaly_data = combined_df[combined_df['time'] >= inject_time]
        temp_normal_data = combined_df[combined_df['time'] < inject_time]
    elif anomalies is not None:
        temp_anomaly_data = combined_df.iloc[anomalies[0]:]
        temp_normal_data = combined_df.iloc[:anomalies[0]]
    else:
        split_point = int(len(combined_df) * 0.7)
        temp_anomaly_data = combined_df.iloc[split_point:]
        temp_normal_data = combined_df.iloc[:split_point]
    
    # 使用改進的異常類型檢測
    detected_anomaly_type, type_confidence = AnomalyTypeDetector.detect_anomaly_type(
        temp_normal_data, all_features, temp_anomaly_data
    )
    
    if verbose:
        print(f"檢測到的異常類型: {detected_anomaly_type} (置信度: {type_confidence:.3f})")
    
    # === 智能特徵選擇 ===
    feature_selector = IntelligentFeatureSelector(
        anomaly_type=detected_anomaly_type,
        data_characteristics=data_characteristics
    )
    
    selected_data, selected_features = feature_selector.select_features(
        combined_df, all_features, kwargs.get("max_features", None)
    )
    
    if verbose:
        print(f"選擇的特徵數量: {len(selected_features)}")
    
    # === 自適應 BOCPD 異常檢測 ===
    selected_data_clean = selected_data.fillna(0)
    
    # 使用魯棒標準化
    scaler = RobustScaler()
    selected_data_scaled = scaler.fit_transform(selected_data_clean)
    
    # 獲取自適應 BOCPD 參數
    bocpd_params = parameter_manager.get_bocpd_parameters(data_characteristics)
    
    bocpd = MultivariateBOCPD(
        alpha=bocpd_params['alpha'],
        beta=bocpd_params['beta'],
        kappa=bocpd_params['kappa'],
        nu=bocpd_params['nu'],
        max_run_length=bocpd_params['max_run_length'],
        min_run_length=bocpd_params['min_run_length']
    )
    
    bocpd.data_quality_score = data_characteristics['quality_score']
    
    changepoints, changepoint_probs = bocpd.detect_changepoints(
        selected_data_scaled, 
        timeout_seconds=kwargs.get("bocpd_timeout", 300)
    )
    
    if verbose:
        print(f"檢測到的變點: {changepoints}")
        if changepoint_probs:
            print(f"變點概率範圍: {np.min(changepoint_probs):.3f} - {np.max(changepoint_probs):.3f}")
    
    # === 確定異常時間 ===
    anomaly_detected = False
    estimated_anomaly_time = None
    
    if changepoints:
        # 選擇最顯著的變點
        if len(changepoints) == 1:
            estimated_anomaly_time = changepoints[0]
        else:
            # 基於變點概率選擇
            changepoint_scores = []
            for cp in changepoints:
                if cp < len(changepoint_probs):
                    changepoint_scores.append((cp, changepoint_probs[cp]))
                else:
                    changepoint_scores.append((cp, 0.5))
            
            changepoint_scores.sort(key=lambda x: x[1], reverse=True)
            estimated_anomaly_time = changepoint_scores[0][0]
        
        anomaly_detected = True
    elif inject_time is not None and 'time' in combined_df.columns:
        time_values = combined_df['time'].values
        anomaly_idx = np.searchsorted(time_values, inject_time)
        if anomaly_idx < len(time_values):
            estimated_anomaly_time = anomaly_idx
            anomaly_detected = True
    elif anomalies is not None:
        estimated_anomaly_time = anomalies[0]
        anomaly_detected = True
    
    # 如果沒有檢測到異常，使用保守的分割點
    if not anomaly_detected:
        estimated_anomaly_time = int(len(combined_df) * 0.8)  # 更保守的分割點
        if verbose:
            print("Warning: 沒有檢測到明顯的變點，使用保守的異常時間估計")
    
    # === 根因分析 ===
    # 準備正常和異常數據
    if estimated_anomaly_time is not None:
        normal_data = combined_df.iloc[:estimated_anomaly_time]
        anomaly_data = combined_df.iloc[estimated_anomaly_time:]
    else:
        split_point = int(len(combined_df) * 0.8)
        normal_data = combined_df.iloc[:split_point]
        anomaly_data = combined_df.iloc[split_point:]
    
    # 移除時間列
    if 'time' in normal_data.columns:
        normal_data = normal_data.drop(columns=['time'])
    if 'time' in anomaly_data.columns:
        anomaly_data = anomaly_data.drop(columns=['time'])
    
    # 增強的異常評分
    scorer = EnhancedAnomalyScorer(
        anomaly_type=detected_anomaly_type,
        data_characteristics=data_characteristics
    )
    
    scorer.learn_normal_distribution(normal_data, all_features)
    statistical_scores = scorer.calculate_anomaly_scores(anomaly_data, all_features)
    
    if verbose:
        print(f"統計分數計算完成，非零分數數量: {sum(1 for s in statistical_scores.values() if s > 0)}")
    
    # === 因果分析 ===
    final_scores = statistical_scores.copy()
    causal_graph_used = False
    
    if kwargs.get("use_causal_graph", True):
        causal_builder = RobustCausalGraphBuilder(parameter_manager)
        causal_graph = causal_builder.learn_causal_structure(normal_data, all_features)
        
        if causal_graph is not None:
            final_scores = causal_builder.compute_causal_scores(anomaly_data, statistical_scores)
            causal_graph_used = True
            
            if verbose:
                print(f"因果圖構建成功，可靠性: {causal_builder.graph_reliability:.3f}")
                print(f"因果圖指標: {causal_builder.graph_metrics}")
    
    # === 最終分數確保 ===
    if not final_scores or all(score <= 0 for score in final_scores.values()):
        if verbose:
            print("Warning: 所有分數為零，使用緊急評分機制")
        
        # 緊急評分機制 - 基於數據驅動的啟發式
        emergency_scores = {}
        for feature in all_features:
            base_score = np.random.uniform(0.1, 0.3)
            
            # 基於特徵名稱的相關性調整
            if scorer._is_relevant_feature(feature, detected_anomaly_type):
                base_score *= 2.0
            
            # 基於數據變異性的調整
            if feature in normal_data.columns and feature in anomaly_data.columns:
                try:
                    normal_var = normal_data[feature].var()
                    anomaly_var = anomaly_data[feature].var()
                    if normal_var > 0:
                        var_ratio = anomaly_var / normal_var
                        base_score *= min(3.0, 1.0 + var_ratio)
                except:
                    pass
            
            emergency_scores[feature] = base_score
        
        final_scores = emergency_scores
    
    # === 最終排名 ===
    ranked_features = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    
    if verbose:
        print(f"\n前10個根因候選:")
        for i, (feature, score) in enumerate(ranked_features[:10]):
            print(f"{i+1}. {feature}: {score:.4f}")
    
    # 服務排名
    service_scores = {}
    for feature, score in ranked_features:
        parts = feature.split('_')
        if len(parts) >= 2 and parts[0] in ['metric', 'logs', 'logts', 'traces']:
            service_name = parts[1]
        else:
            service_name = parts[0]
        
        if service_name not in service_scores:
            service_scores[service_name] = []
        service_scores[service_name].append(score)
    
    service_rankings = [(service, max(scores)) for service, scores in service_scores.items()]
    service_rankings.sort(key=lambda x: x[1], reverse=True)
    
    # === 結果彙總 ===
    result = {
        "anomaly_detected": anomaly_detected,
        "estimated_anomaly_time": estimated_anomaly_time,
        "changepoints": changepoints,
        "changepoint_probs": changepoint_probs,
        "node_names": all_features,
        "ranks": [item[0] for item in ranked_features],
        "scores": dict(ranked_features),
        "service_rankings": service_rankings,
        "detected_anomaly_type": detected_anomaly_type,
        "anomaly_type_confidence": type_confidence,
        "causal_graph_used": causal_graph_used,
        "data_characteristics": data_characteristics,
        "bocpd_parameters": bocpd_params,
        "selected_features": selected_features,
        "feature_selection_history": feature_selector.selection_history
    }
    
    if causal_graph_used:
        result.update({
            "causal_graph_reliability": causal_builder.graph_reliability,
            "causal_graph_metrics": causal_builder.graph_metrics
        })
    
    return result


# === 兼容性函數 ===
def borca(data, inject_time=None, dataset=None, num_loop=None, sli=None, anomalies=None, **kwargs):
    """BOrca 演算法的標準接口"""
    return borca_multimodal(data, inject_time, dataset, num_loop, sli, anomalies, **kwargs)


def mmorca(data, inject_time=None, dataset=None, num_loop=None, sli=None, anomalies=None, **kwargs):
    """多模態版本的 BOrca 演算法"""
    return borca_multimodal(data, inject_time, dataset, num_loop, sli, anomalies, 
                           verbose=kwargs.get("verbose", False), **kwargs)


class AnomalyTypeDetector:
    """異常類型檢測器 - 保持向後兼容"""
    
    @staticmethod
    def detect_anomaly_type(data, feature_names, anomaly_period_data):
        """檢測異常類型"""
        # 定義關鍵字
        type_keywords = {
            'CPU': ['cpu', 'processor', 'core', 'usage', 'util'],
            'MEM': ['mem', 'memory', 'ram', 'heap', 'gc'],
            'DISK': ['disk', 'io', 'storage', 'read', 'write'],
            'DELAY': ['latency', 'delay', 'response_time', 'duration', 'rt'],
            'LOSS': ['error', 'fail', 'exception', 'loss', 'drop']
        }
        
        type_scores = {}
        
        for anomaly_type, keywords in type_keywords.items():
            type_features = [f for f in feature_names if any(kw in f.lower() for kw in keywords)]
            if type_features:
                scores = []
                for feature in type_features:
                    if feature in data.columns and feature in anomaly_period_data.columns:
                        normal_values = data[feature].dropna()
                        anomaly_values = anomaly_period_data[feature].dropna()
                        
                        if len(normal_values) > 0 and len(anomaly_values) > 0:
                            # 使用魯棒統計量
                            normal_median = normal_values.median()
                            normal_mad = np.median(np.abs(normal_values - normal_median))
                            
                            if normal_mad > 0:
                                mad_scores = np.abs((anomaly_values - normal_median) / (1.4826 * normal_mad))
                                scores.append(np.max(mad_scores))
                
                if scores:
                    type_scores[anomaly_type] = np.mean(scores)
        
        if type_scores:
            max_type = max(type_scores.items(), key=lambda x: x[1])
            return max_type[0], max_type[1]
        else:
            return 'UNKNOWN', 0.0
