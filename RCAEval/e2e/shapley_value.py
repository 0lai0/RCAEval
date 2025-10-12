"""
Shapley Value 計算模組

提供基於合作賽局理論的公平貢獻分配算法，用於根因分析中的服務貢獻度量化。

主要功能：
1. ShapleyValueCalculator: 精確和近似 Shapley Value 計算
2. CausalShapleyValueCalculator: 因果約束的 Shapley Value 計算（NEW）
3. 系統異常函數創建
4. 服務優先級管理
5. 緩存機制優化

"""

import itertools
import math
import time
import random
from collections import defaultdict, OrderedDict, deque
from typing import Dict, List, Callable, Any, Optional, Set, Tuple


def extract_base_service_name(service_name: str) -> str:
    """
    從帶窗口標識的服務名提取基礎服務名
    
    例如:
        'serviceA_w0' -> 'serviceA'
        'serviceA' -> 'serviceA'
        'gateway_w2' -> 'gateway'
    
    Args:
        service_name: 帶窗口標識的服務名
        
    Returns:
        str: 基礎服務名
    """
    if '_w' in service_name and service_name.split('_')[-1].startswith('w'):
        return '_'.join(service_name.split('_')[:-1])
    return service_name


def build_dependency_graph(services: List[str], edges: set) -> Dict[str, Set[str]]:
    """
    從 CPG 邊建構依賴圖
    
    Args:
        services: 服務名稱列表
        edges: CPG 邊集合，格式為 (source, target, weight)
    
    Returns:
        Dict[str, Set[str]]: 依賴圖 {service: set of dependencies}
    """
    # 初始化依賴圖
    dep_graph = {service: set() for service in services}
    
    # 從邊提取依賴關係
    for edge in edges:
        if len(edge) >= 2:
            source = extract_base_service_name(edge[0])
            target = extract_base_service_name(edge[1])
            
            # 邊的方向：source 依賴於 target (source -> target 表示 target 影響 source)
            # 在因果圖中，如果 A -> B，表示 A 的異常可能導致 B 的異常
            # 因此在 Shapley 排列中，A 應該在 B 之前
            if source in dep_graph and target in services:
                # target 必須在 source 之前（target 是 source 的前驅）
                dep_graph[source].add(target)
    
    return dep_graph


def topological_sort_all_orders(services: List[str], dep_graph: Dict[str, Set[str]]) -> List[List[str]]:
    """
    生成所有符合拓撲順序的排列
    
    使用深度優先搜索生成所有有效的拓撲排序
    
    Args:
        services: 服務名稱列表
        dep_graph: 依賴圖 {service: set of dependencies}
    
    Returns:
        List[List[str]]: 所有符合拓撲順序的排列
    """
    # 計算入度
    in_degree = {s: 0 for s in services}
    for s in services:
        for dep in dep_graph.get(s, set()):
            if dep in in_degree:
                in_degree[s] += 1
    
    all_orders = []
    
    def dfs(current_order: List[str], remaining: Set[str], current_in_degree: Dict[str, int]):
        if not remaining:
            all_orders.append(current_order[:])
            return
        
        # 找出所有入度為 0 的節點（可以加入的節點）
        available = [s for s in remaining if current_in_degree[s] == 0]
        
        if not available:
            # 有環，無法繼續（理論上不應該發生）
            return
        
        # 嘗試每個可用節點
        for node in available:
            # 選擇這個節點
            current_order.append(node)
            new_remaining = remaining - {node}
            
            # 更新入度
            new_in_degree = current_in_degree.copy()
            for dependent in services:
                if dependent in new_remaining and node in dep_graph.get(dependent, set()):
                    new_in_degree[dependent] -= 1
            
            # 遞歸
            dfs(current_order, new_remaining, new_in_degree)
            
            # 回溯
            current_order.pop()
    
    dfs([], set(services), in_degree)
    return all_orders


def sample_causal_permutation(services: List[str], dep_graph: Dict[str, Set[str]]) -> List[str]:
    """
    隨機採樣一個符合因果約束的排列
    
    使用 Kahn's 算法的隨機化版本
    
    Args:
        services: 服務名稱列表
        dep_graph: 依賴圖 {service: set of dependencies}
    
    Returns:
        List[str]: 符合拓撲順序的隨機排列
    """
    # 計算入度
    in_degree = {s: 0 for s in services}
    for s in services:
        for dep in dep_graph.get(s, set()):
            if dep in in_degree:
                in_degree[s] += 1
    
    # 初始化隊列（入度為 0 的節點）
    available = [s for s in services if in_degree[s] == 0]
    result = []
    
    while available:
        # 隨機選擇一個可用節點
        node = random.choice(available)
        available.remove(node)
        result.append(node)
        
        # 更新鄰居的入度
        for neighbor in services:
            if neighbor not in result and node in dep_graph.get(neighbor, set()):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    available.append(neighbor)
    
    # 處理可能的環（添加剩餘節點）
    if len(result) < len(services):
        remaining = [s for s in services if s not in result]
        random.shuffle(remaining)
        result.extend(remaining)
    
    return result


class ShapleyValueCalculator:
    """
    Shapley Value 計算器
    
    實現精確和蒙特卡羅近似兩種計算方法，支持大規模服務的貢獻度量化。
    
    特性：
    - 精確計算：適用於小規模服務集合（≤10個服務）
    - 蒙特卡羅近似：適用於大規模服務集合
    - LRU 緩存：提升計算效率
    - 統計信息：提供計算性能指標
    """
    
    def __init__(self, cache_size: int = 10000, max_services_for_exact: int = 10, 
                 monte_carlo_samples: int = 1000, enable_parallel: bool = False):
        """
        初始化 Shapley Value 計算器
        
        Args:
            cache_size: 緩存大小，用於存儲子集價值計算結果
            max_services_for_exact: 使用精確計算的最大服務數量
            monte_carlo_samples: 蒙特卡羅採樣次數
            enable_parallel: 是否啟用並行計算（未來功能）
        """
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.max_services_for_exact = max_services_for_exact
        self.monte_carlo_samples = monte_carlo_samples
        self.enable_parallel = enable_parallel
        
        self.stats = {
            'exact_calculations': 0,
            'approx_calculations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0,
            'total_combinations': 0,
            'avg_calculation_time': 0.0
        }
    
    def calculate_shapley_values(
        self, 
        services: List[str], 
        system_anomaly_function: Callable[[List[str]], float],
        max_services_for_exact: Optional[int] = None,
        monte_carlo_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        計算所有服務的 Shapley Value
        
        Args:
            services: 服務名稱列表
            system_anomaly_function: 系統異常評估函數 v(S)
            max_services_for_exact: 使用精確計算的最大服務數量（None 使用默認值）
            monte_carlo_samples: 蒙特卡羅採樣次數（None 使用默認值）
            
        Returns:
            Dict[str, float]: 每個服務的 Shapley Value
        """
        n = len(services)
        
        if n == 0:
            return {}
        
        # 使用實例配置或參數
        max_exact = max_services_for_exact if max_services_for_exact is not None else self.max_services_for_exact
        num_samples = monte_carlo_samples if monte_carlo_samples is not None else self.monte_carlo_samples
        
        start_time = time.time()
        
        # 選擇計算方法
        if n <= max_exact:
            total_combinations = 2 ** n
            print(f"  Using exact Shapley calculation ({n} services, {total_combinations} combinations)")
            result = self._exact_shapley(services, system_anomaly_function)
            self.stats['exact_calculations'] += 1
            self.stats['total_combinations'] += total_combinations
        else:
            print(f"  Using Monte Carlo approximation ({n} services, {num_samples} samples)")
            result = self._approximate_shapley(services, system_anomaly_function, num_samples=num_samples)
            self.stats['approx_calculations'] += 1
            self.stats['total_combinations'] += num_samples
        
        elapsed_time = time.time() - start_time
        self.stats['total_time'] += elapsed_time
        
        # 更新平均計算時間
        total_calcs = self.stats['exact_calculations'] + self.stats['approx_calculations']
        if total_calcs > 0:
            self.stats['avg_calculation_time'] = self.stats['total_time'] / total_calcs
        
        print(f"  Shapley calculation completed in {elapsed_time:.2f}s")
        
        return result
    
    def _exact_shapley(
        self, 
        services: List[str], 
        system_anomaly_function: Callable[[List[str]], float]
    ) -> Dict[str, float]:
        """
        精確計算 Shapley Value
        
        時間複雜度: O(n * 2^n)
        適用於小規模服務集合
        
        Args:
            services: 服務名稱列表
            system_anomaly_function: 系統異常評估函數
            
        Returns:
            Dict[str, float]: 每個服務的 Shapley Value
        """
        n = len(services)
        shapley_values = {service: 0.0 for service in services}
        
        # 遍歷所有子集大小 (0 到 n)
        for subset_size in range(n + 1):
            # 生成該大小的所有子集
            for subset_tuple in itertools.combinations(services, subset_size):
                subset = list(subset_tuple)
                
                # 獲取子集的系統異常分數（帶緩存）
                v_S = self._get_subset_value(subset, system_anomaly_function)
                
                # 對每個不在子集中的服務
                for service in services:
                    if service not in subset:
                        # 計算加入該服務後的分數
                        extended_subset = subset + [service]
                        v_S_union_i = self._get_subset_value(
                            extended_subset, 
                            system_anomaly_function
                        )
                        
                        # 邊際貢獻
                        marginal_contribution = v_S_union_i - v_S
                        
                        # Shapley Value 公式的權重
                        weight = (
                            math.factorial(subset_size) * 
                            math.factorial(n - subset_size - 1) / 
                            math.factorial(n)
                        )
                        
                        # 累加
                        shapley_values[service] += weight * marginal_contribution
        
        return shapley_values
    
    def _approximate_shapley(
        self, 
        services: List[str], 
        system_anomaly_function: Callable[[List[str]], float],
        num_samples: int = 1000
    ) -> Dict[str, float]:
        """
        蒙特卡羅近似 Shapley Value
        
        時間複雜度: O(n * num_samples)
        適用於大規模服務集合
        
        Args:
            services: 服務名稱列表
            system_anomaly_function: 系統異常評估函數
            num_samples: 蒙特卡羅採樣次數
            
        Returns:
            Dict[str, float]: 每個服務的近似 Shapley Value
        """
        import numpy as np
        
        n = len(services)
        shapley_values = {service: 0.0 for service in services}
        
        # 使用固定種子確保可重現性
        np.random.seed(42)
        
        # 批量生成排列以提高效率
        batch_size = min(100, num_samples)
        
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            current_batch_size = batch_end - batch_start
            
            # 批量生成隨機排列
            permutations = []
            for _ in range(current_batch_size):
                permutations.append(np.random.permutation(services).tolist())
            
            # 處理當前批次
            for permuted_services in permutations:
                # 對每個位置的服務計算邊際貢獻
                for i, service in enumerate(permuted_services):
                    # 前綴（已加入的服務）
                    prefix = permuted_services[:i]
                    
                    # 包含當前服務
                    with_service = permuted_services[:i+1]
                    
                    # 計算邊際貢獻
                    v_prefix = self._get_subset_value(prefix, system_anomaly_function)
                    v_with = self._get_subset_value(with_service, system_anomaly_function)
                    
                    marginal_contribution = v_with - v_prefix
                    
                    # 累加
                    shapley_values[service] += marginal_contribution
        
        # 平均化
        for service in shapley_values:
            shapley_values[service] /= num_samples
        
        return shapley_values
    
    def _get_subset_value(
        self, 
        subset: List[str], 
        system_anomaly_function: Callable[[List[str]], float]
    ) -> float:
        """
        帶緩存的子集價值計算
        
        Args:
            subset: 服務子集
            system_anomaly_function: 系統異常評估函數
            
        Returns:
            float: 子集價值
        """
        # 生成緩存鍵（排序後的元組）
        cache_key = tuple(sorted(subset))
        
        # 檢查緩存
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            # 移動到末尾（LRU）
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]
        
        # 緩存未命中，計算
        self.stats['cache_misses'] += 1
        value = system_anomaly_function(subset)
        
        # 存入緩存
        self.cache[cache_key] = value
        
        # 檢查緩存大小
        if len(self.cache) > self.cache_size:
            # 移除最舊的項目
            self.cache.popitem(last=False)
        
        return value
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        返回計算統計信息
        
        Returns:
            Dict[str, Any]: 包含緩存命中率、計算時間等統計信息
        """
        cache_total = self.stats['cache_hits'] + self.stats['cache_misses']
        cache_hit_rate = (
            self.stats['cache_hits'] / cache_total 
            if cache_total > 0 else 0.0
        )
        
        total_calcs = self.stats['exact_calculations'] + self.stats['approx_calculations']
        
        return {
            **self.stats,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.cache),
            'total_calculations': total_calcs,
            'efficiency_ratio': cache_hit_rate * 100,  # 效率比率
            'avg_time_per_calculation': self.stats['avg_calculation_time']
        }
    
    def clear_cache(self):
        """清空緩存"""
        self.cache.clear()
        self.stats['cache_hits'] = 0
        self.stats['cache_misses'] = 0


class CausalShapleyValueCalculator(ShapleyValueCalculator):
    """
    因果約束的 Shapley Value 計算器
    
    擴展標準 ShapleyValueCalculator，利用 CPG 的因果依賴關係
    只考慮符合拓撲順序的排列，避免不符合物理現實的情況。
    
    主要優勢：
    1. 利用時間序列和因果依賴信息
    2. 避免下游服務被誤判為根因
    3. 提高根因識別精準度
    4. 結果更符合實際服務調用順序
    
    理論基礎：
    - 標準 Shapley Value 考慮所有 n! 種排列
    - Causal Shapley Value 只考慮符合拓撲順序的排列
    - 這些排列更符合實際的因果傳播路徑
    """
    
    def __init__(self, cache_size: int = 10000, max_services_for_exact: int = 10, 
                 monte_carlo_samples: int = 1000, enable_parallel: bool = False):
        """
        初始化 Causal Shapley Value 計算器
        
        Args:
            cache_size: 緩存大小
            max_services_for_exact: 使用精確計算的最大服務數量
            monte_carlo_samples: 蒙特卡羅採樣次數
            enable_parallel: 是否啟用並行計算
        """
        super().__init__(cache_size, max_services_for_exact, monte_carlo_samples, enable_parallel)
        
        # 新增統計信息
        self.stats['causal_calculations'] = 0
        self.stats['valid_permutations'] = 0
        self.stats['constraint_efficiency'] = 0.0
    
    def calculate_causal_shapley_values(
        self, 
        services: List[str], 
        edges: set,
        system_anomaly_function: Callable[[List[str]], float],
        max_services_for_exact: Optional[int] = None,
        monte_carlo_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        計算因果約束的 Shapley Value
        
        Args:
            services: 服務名稱列表
            edges: CPG 邊集合
            system_anomaly_function: 系統異常評估函數
            max_services_for_exact: 使用精確計算的最大服務數量
            monte_carlo_samples: 蒙特卡羅採樣次數
            
        Returns:
            Dict[str, float]: 每個服務的 Causal Shapley Value
        """
        n = len(services)
        
        if n == 0:
            return {}
        
        # 構建依賴圖
        dep_graph = build_dependency_graph(services, edges)
        
        # 使用實例配置或參數
        max_exact = max_services_for_exact if max_services_for_exact is not None else self.max_services_for_exact
        num_samples = monte_carlo_samples if monte_carlo_samples is not None else self.monte_carlo_samples
        
        start_time = time.time()
        
        # 選擇計算方法
        if n <= max_exact:
            # 精確計算：生成所有符合拓撲順序的排列
            print(f"  Using exact Causal Shapley calculation ({n} services)")
            result = self._exact_causal_shapley(services, dep_graph, system_anomaly_function)
            self.stats['exact_calculations'] += 1
        else:
            # 近似計算：採樣符合拓撲順序的排列
            print(f"  Using Causal Monte Carlo approximation ({n} services, {num_samples} samples)")
            result = self._approximate_causal_shapley(services, dep_graph, system_anomaly_function, num_samples)
            self.stats['approx_calculations'] += 1
        
        elapsed_time = time.time() - start_time
        self.stats['total_time'] += elapsed_time
        self.stats['causal_calculations'] += 1
        
        # 更新平均計算時間
        total_calcs = self.stats['exact_calculations'] + self.stats['approx_calculations']
        if total_calcs > 0:
            self.stats['avg_calculation_time'] = self.stats['total_time'] / total_calcs
        
        print(f"  Causal Shapley calculation completed in {elapsed_time:.2f}s")
        
        return result
    
    def _exact_causal_shapley(
        self, 
        services: List[str], 
        dep_graph: Dict[str, Set[str]],
        system_anomaly_function: Callable[[List[str]], float]
    ) -> Dict[str, float]:
        """
        精確計算 Causal Shapley Value
        
        生成所有符合拓撲順序的排列並計算
        
        Args:
            services: 服務名稱列表
            dep_graph: 依賴圖
            system_anomaly_function: 系統異常評估函數
            
        Returns:
            Dict[str, float]: Causal Shapley Values
        """
        n = len(services)
        shapley_values = {service: 0.0 for service in services}
        
        # 生成所有符合拓撲順序的排列
        valid_orders = topological_sort_all_orders(services, dep_graph)
        
        if not valid_orders:
            # 如果無法生成有效排列（可能有環），降級到標準方法
            print("  Warning: No valid topological orders found, falling back to standard Shapley")
            return self._exact_shapley(services, system_anomaly_function)
        
        num_valid_orders = len(valid_orders)
        self.stats['valid_permutations'] = num_valid_orders
        
        # 理論上的總排列數
        total_permutations = math.factorial(n)
        self.stats['constraint_efficiency'] = num_valid_orders / total_permutations
        
        print(f"    Valid topological orders: {num_valid_orders} / {total_permutations} ({self.stats['constraint_efficiency']:.2%})")
        
        # 對每個有效排列計算邊際貢獻
        for order in valid_orders:
            for i, service in enumerate(order):
                # 前綴（已加入的服務）
                prefix = order[:i]
                with_service = order[:i+1]
                
                # 計算邊際貢獻
                v_prefix = self._get_subset_value(prefix, system_anomaly_function)
                v_with = self._get_subset_value(with_service, system_anomaly_function)
                
                marginal_contribution = v_with - v_prefix
                shapley_values[service] += marginal_contribution
        
        # 平均化
        for service in shapley_values:
            shapley_values[service] /= num_valid_orders
        
        return shapley_values
    
    def _approximate_causal_shapley(
        self, 
        services: List[str], 
        dep_graph: Dict[str, Set[str]],
        system_anomaly_function: Callable[[List[str]], float],
        num_samples: int = 1000
    ) -> Dict[str, float]:
        """
        蒙特卡羅近似 Causal Shapley Value
        
        隨機採樣符合拓撲順序的排列
        
        Args:
            services: 服務名稱列表
            dep_graph: 依賴圖
            system_anomaly_function: 系統異常評估函數
            num_samples: 採樣次數
            
        Returns:
            Dict[str, float]: 近似 Causal Shapley Values
        """
        n = len(services)
        shapley_values = {service: 0.0 for service in services}
        
        # 設置隨機種子以確保可重現性
        random.seed(42)
        
        # 採樣符合拓撲順序的排列
        for sample_idx in range(num_samples):
            # 生成一個隨機的符合拓撲順序的排列
            order = sample_causal_permutation(services, dep_graph)
            
            # 對每個服務計算邊際貢獻
            for i, service in enumerate(order):
                prefix = order[:i]
                with_service = order[:i+1]
                
                # 計算邊際貢獻
                v_prefix = self._get_subset_value(prefix, system_anomaly_function)
                v_with = self._get_subset_value(with_service, system_anomaly_function)
                
                marginal_contribution = v_with - v_prefix
                shapley_values[service] += marginal_contribution
        
        # 平均化
        for service in shapley_values:
            shapley_values[service] /= num_samples
        
        self.stats['valid_permutations'] = num_samples
        
        return shapley_values
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        返回計算統計信息（包含因果約束信息）
        
        Returns:
            Dict[str, Any]: 包含緩存命中率、計算時間、約束效率等統計信息
        """
        base_stats = super().get_statistics()
        
        # 添加因果約束相關統計
        base_stats['causal_calculations'] = self.stats.get('causal_calculations', 0)
        base_stats['valid_permutations'] = self.stats.get('valid_permutations', 0)
        base_stats['constraint_efficiency'] = self.stats.get('constraint_efficiency', 0.0)
        
        return base_stats


def get_service_priority(service_name: str) -> float:
    """
    根據服務名稱和類型給予優先級分數
    
    調整後的優先級分層：
    - Level 1 (0.6): 上游入口或核心服務（frontend, gateway, ui, auth, order等）
    - Level 2 (0.5): 中間業務邏輯服務（product, cart, user, shipping等）
    - Level 3 (0.4): 下游基礎設施或輔助服務（db, mongo, redis, email等）
    
    主要改進：縮小權重差異範圍，減少對 Shapley Value 計算的影響
    
    Args:
        service_name: 服務名稱
        
    Returns:
        float: 優先級分數，範圍[0.4, 0.6]
    """
    service_name_lower = service_name.lower()
    
    # Level 1: 上游入口或核心服務 (分數最高，但降低)
    if any(p in service_name_lower for p in ['frontend', 'front-end', 'ui', 'gateway', 'auth', 'order']):
        return 0.6  # 從 1.0 降低到 0.6
    
    # Level 2: 中間業務邏輯
    elif any(p in service_name_lower for p in ['product', 'cart', 'user', 'shipping', 'payment', 'checkout', 'catalogue', 'catalog']):
        return 0.5  # 從 0.7 降低到 0.5
    
    # Level 3: 下游基礎設施或輔助服務 (分數最低，保持)
    elif any(p in service_name_lower for p in ['db', 'mongo', 'redis', 'email', 'cache', 'queue']):
        return 0.4  # 保持 0.4
    
    # 默認中等優先級
    return 0.5


def create_system_anomaly_function(all_events: List) -> Callable[[List[str]], float]:
    """
    創建改進的系統異常評估函數
    
    該函數計算給定服務子集對整體系統異常的貢獻度
    
    主要改進：
    1. 減少優先級權重影響
    2. 使用最大值和平均值的混合
    3. 突出異常服務的貢獻
    4. 預計算和緩存優化
    
    Args:
        all_events: 所有聚合事件列表
    
    Returns:
        callable: 系統異常函數 v(S)
    """
    # 預先計算每個服務的異常分數和優先級（避免重複計算）
    service_anomaly_scores = {}
    service_priority_cache = {}
    
    for event in all_events:
        base_service = extract_base_service_name(event.service_name)
        
        if hasattr(event, 'anomaly_score'):
            if base_service not in service_anomaly_scores:
                service_anomaly_scores[base_service] = 0.0
            
            # 取該服務所有窗口中的最大異常分數
            service_anomaly_scores[base_service] = max(
                service_anomaly_scores[base_service],
                event.anomaly_score
            )
    
    # 預計算所有服務的優先級權重
    for service in service_anomaly_scores.keys():
        priority_weight = get_service_priority(service)
        # 將 [0.4, 0.6] 映射到 [0.9, 1.1]，減少權重影響
        service_priority_cache[service] = 0.9 + (priority_weight - 0.4) * 0.2 / 0.2
    
    def improved_system_anomaly_function(service_subset: List[str]) -> float:
        """
        改進的系統異常分數計算
        
        策略：
        1. 減少優先級權重影響（從 0.4-1.0 縮小到 0.9-1.1）
        2. 使用最大值和加權平均的混合
        3. 突出異常服務的貢獻
        4. 使用預計算的緩存提高效率
        """
        if not service_subset:
            return 0.0
        
        total_anomaly = 0.0
        max_anomaly = 0.0
        anomaly_scores = []
        
        for service in service_subset:
            # 獲取服務的異常分數
            service_anomaly = service_anomaly_scores.get(service, 0.0)
            anomaly_scores.append(service_anomaly)
            
            # 使用預計算的優先級權重
            adjusted_priority = service_priority_cache.get(service, 1.0)
            
            # 加權累加
            weighted_anomaly = service_anomaly * adjusted_priority
            total_anomaly += weighted_anomaly
            max_anomaly = max(max_anomaly, service_anomaly)
        
        # 計算平均值
        avg_anomaly = total_anomaly / len(service_subset)
        
        # 使用混合策略：70% 加權平均 + 30% 最大值
        # 這樣既能反映整體異常，又能突出最異常的服務
        mixed_score = 0.7 * avg_anomaly + 0.3 * max_anomaly
        
        # 額外獎勵：如果有多個高異常分數的服務，給予額外分數
        high_anomaly_count = sum(1 for score in anomaly_scores if score > 0.5)
        if high_anomaly_count > 1:
            mixed_score *= (1.0 + 0.1 * (high_anomaly_count - 1))
        
        return mixed_score
    
    return improved_system_anomaly_function


def calculate_shapley_contributions(
    services: List[str],
    all_events: List,
    max_services_for_exact: int = 10,
    monte_carlo_samples: int = 1000,
    cache_size: int = 10000
) -> Dict[str, float]:
    """
    便捷函數：計算服務的 Shapley Value 貢獻度
    
    Args:
        services: 服務名稱列表
        all_events: 所有聚合事件列表
        max_services_for_exact: 使用精確計算的最大服務數量
        monte_carlo_samples: 蒙特卡羅採樣次數
        cache_size: 緩存大小
        
    Returns:
        Dict[str, float]: 每個服務的 Shapley Value 貢獻度
    """
    # 創建系統異常函數
    system_anomaly_func = create_system_anomaly_function(all_events)
    
    # 計算 Shapley Values
    calculator = ShapleyValueCalculator(
        cache_size=cache_size,
        max_services_for_exact=max_services_for_exact,
        monte_carlo_samples=monte_carlo_samples
    )
    shapley_values = calculator.calculate_shapley_values(
        services,
        system_anomaly_func
    )
    
    # 歸一化到 [0, 1]
    if shapley_values:
        max_shapley = max(shapley_values.values())
        min_shapley = min(shapley_values.values())
        if max_shapley > min_shapley:
            # 使用 min-max 歸一化
            shapley_values = {
                k: (v - min_shapley) / (max_shapley - min_shapley)
                for k, v in shapley_values.items()
            }
        else:
            # 如果所有值相同，平均分配
            shapley_values = {k: 1.0/len(shapley_values) for k in shapley_values.keys()}
    
    return shapley_values


# 使用示例
if __name__ == "__main__":
    print("Shapley Value Calculator Module")
    print("=" * 50)
    
    # 創建測試數據
    class MockEvent:
        def __init__(self, service_name, anomaly_score):
            self.service_name = service_name
            self.anomaly_score = anomaly_score
    
    # 模擬事件數據
    test_events = [
        MockEvent('serviceA_w0', 0.8),
        MockEvent('serviceA_w1', 0.9),
        MockEvent('serviceB_w0', 0.3),
        MockEvent('serviceC_w0', 0.2)
    ]
    
    # 測試服務列表
    test_services = ['serviceA', 'serviceB', 'serviceC']
    
    print(f"Testing with services: {test_services}")
    print(f"Mock events: {len(test_events)}")
    
    # 計算 Shapley Values
    shapley_values = calculate_shapley_contributions(test_services, test_events)
    
    print(f"\nShapley Values:")
    for service, value in shapley_values.items():
        print(f"  {service}: {value:.4f}")
    
    # 測試計算器統計信息
    calculator = ShapleyValueCalculator()
    system_func = create_system_anomaly_function(test_events)
    calculator.calculate_shapley_values(test_services, system_func)
    
    stats = calculator.get_statistics()
    print(f"\nStatistics:")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"  Total time: {stats['total_time']:.2f}s")
    print(f"  Cache size: {stats['cache_size']}")
    
    print("\nModule test completed!")
