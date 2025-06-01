# RCAEval Docker 環境設置與操作指南

## 📋 概述

RCAEval 是一個用於微服務系統根因分析的基準測試平台。本專案提供了完整的 Docker 環境，支援兩種不同的 Python 環境配置，以滿足不同 RCA 方法的需求。

## 🏗️ 架構說明

### 環境配置
- **Base OS**: Ubuntu 22.04 LTS
- **Python 版本**: 
  - Python 3.10 (預設環境) - 用於大部分 RCA 方法
  - Python 3.8 (RCD 環境) - 專門用於 RCD 相關功能
- **Java**: OpenJDK 11 (用於部分 Java 相關工具)

### 主要組件
1. **系統依賴**: graphviz, Cairo, XML 處理庫等
2. **Python 套件**: 根據 requirements.txt 和 requirements_rcd.lock 安裝
3. **客製化庫**: 透過 script/link.sh 連結修改過的庫文件
4. **PyRCA**: 自動安裝用於 HT 和 E-Diagnosis 方法

### 服務架構
- **rcaeval**: 預設環境容器 (Python 3.10)
- **rcaeval-rcd**: RCD 專用環境容器 (Python 3.8)
- **rcaeval-jupyter**: Jupyter Notebook 服務 (Port 8888)
- **rcaeval-api**: HTTP API 服務 (Port 8080)

## 🚀 快速開始

### 1. 建構 Docker 映像

```bash
# 使用 docker-compose (推薦)
docker compose build

# 或直接使用 Docker
docker build -t rcaeval .
```

### 2. 啟動環境

#### 方式一：背景執行所有服務 (推薦)

```bash
# 啟動所有服務並在背景執行
docker compose up -d

# 查看服務狀態
docker compose ps

# 查看服務日誌
docker compose logs -f
```

#### 方式二：互動式執行特定服務

```bash
# 啟動預設環境 (Python 3.10) - 互動模式
docker compose run --rm rcaeval /bin/bash

# 啟動 RCD 環境 (Python 3.8) - 互動模式
docker compose run --rm rcaeval-rcd /bin/bash

# 僅啟動 Jupyter Notebook 服務
docker compose up -d rcaeval-jupyter
# 瀏覽器開啟: http://localhost:8888
```

#### 方式三：直接使用 Docker

```bash
# 預設環境
docker run -it --rm -v $(pwd):/app rcaeval

# RCD 環境
docker run -it --rm -v $(pwd):/app rcaeval \
  /bin/bash -c "source env-rcd/bin/activate && /bin/bash"
```

### 3. 進入運行中的容器

```bash
# 進入預設環境容器
docker compose exec rcaeval /bin/bash

# 進入 RCD 環境容器
docker compose exec rcaeval-rcd /bin/bash

# 檢查已啟動的環境
source env/bin/activate  # 或 source env-rcd/bin/activate
python --version
```

### 4. 停止服務

```bash
# 停止所有服務
docker compose down

# 停止特定服務
docker compose stop rcaeval-jupyter

# 停止並移除所有容器和網路
docker compose down --volumes
```

## 📁 目錄結構與掛載

### 容器內路徑
- `/app` - 專案根目錄
- `/app/data` - 資料目錄 (持久化儲存)
- `/app/results` - 結果目錄 (持久化儲存)
- `/app/env` - Python 3.10 虛擬環境
- `/app/env-rcd` - Python 3.8 虛擬環境

### 持久化儲存
Docker volumes 用於資料持久化：
- `rcaeval-data` - 掛載到 `/app/data`
- `rcaeval-results` - 掛載到 `/app/results`

### 服務端口
- **8888**: Jupyter Notebook 介面
- **8080**: HTTP API 服務

## 🔧 環境使用

### 切換 Python 環境

```bash
# 在容器內切換到預設環境 (Python 3.10)
source env/bin/activate

# 切換到 RCD 環境 (Python 3.8)
source env-rcd/bin/activate

# 檢查當前 Python 版本
python --version
```

### 驗證安裝

```bash
# 執行基本測試
python -m pytest tests/test.py::test_basic

# 檢查套件安裝
pip list | grep -E "(causal|pyagrum|torch)"
```

## 📊 執行實驗

### 背景服務模式執行實驗

```bash
# 1. 啟動所有服務
docker compose up -d

# 2. 進入預設環境執行實驗
docker compose exec rcaeval bash -c "source env/bin/activate && python main.py --method baro --dataset online-boutique"

# 3. 進入 RCD 環境執行實驗
docker compose exec rcaeval-rcd bash -c "source env-rcd/bin/activate && python main.py --method rcd --dataset your-dataset"

# 4. 批次執行多個實驗
docker compose exec rcaeval bash -c "
source env/bin/activate
for method in baro pc_pagerank microcause; do
    echo 'Running method: $method'
    python main.py --method $method --dataset online-boutique
done
"
```

### 互動模式執行實驗

```bash
# 進入容器進行互動式操作
docker compose exec rcaeval /bin/bash

# 在容器內執行
source env/bin/activate
python main.py --method baro --dataset online-boutique
```

### 使用 Jupyter Notebook

```bash
# 啟動 Jupyter 服務
docker compose up -d rcaeval-jupyter

# 瀏覽器開啟 http://localhost:8888
# 無需密碼即可直接使用
```

## 🛠️ 開發與除錯

### 服務管理

```bash
# 查看所有服務狀態
docker compose ps

# 查看特定服務日誌
docker compose logs rcaeval
docker compose logs rcaeval-jupyter

# 重啟特定服務
docker compose restart rcaeval

# 查看資源使用情況
docker stats
```

### 除錯模式

```bash
# 啟動除錯容器
docker compose run --rm rcaeval /bin/bash

# 或進入運行中的容器
docker compose exec rcaeval /bin/bash

# 使用 VS Code 附加到容器
# 1. 啟動背景服務: docker compose up -d
# 2. 在 VS Code 中使用 "Dev Containers: Attach to Running Container"
# 3. 選擇 rcaeval-default 或 rcaeval-rcd
```

### 效能監控

```bash
# 監控所有容器資源使用
docker stats

# 查看特定容器的詳細資訊
docker compose exec rcaeval htop

# 查看磁碟使用情況
docker compose exec rcaeval df -h
```

## 📦 資料管理

### 資料準備

```bash
# 將資料複製到容器 (服務運行中)
docker compose cp ./your-data.csv rcaeval:/app/data/

# 或直接編輯本地檔案 (自動同步到容器)
echo "your data" > data/your-file.txt
```

### 結果導出

```bash
# 從容器複製結果
docker compose cp rcaeval:/app/results ./local-results

# 或直接存取本地掛載的目錄
ls -la results/
```

## 🔍 故障排除

### 常見問題

#### 1. 服務啟動失敗
```bash
# 查看服務日誌
docker compose logs rcaeval

# 重新建構並啟動
docker compose down
docker compose build --no-cache
docker compose up -d
```

#### 2. 端口衝突
```bash
# 修改 docker-compose.yml 中的端口映射
ports:
  - "9999:8888"  # 將 8888 改為 9999
```

#### 3. 容器無法存取
```bash
# 檢查容器狀態
docker compose ps

# 重啟服務
docker compose restart rcaeval
```

#### 4. 記憶體不足
```bash
# 在 docker-compose.yml 中添加資源限制
deploy:
  resources:
    limits:
      memory: 8G
```

### 除錯指令

```bash
# 檢查環境變數
docker compose exec rcaeval env

# 檢查 Python 路徑
docker compose exec rcaeval python -c "import sys; print(sys.path)"

# 檢查已安裝套件
docker compose exec rcaeval pip list

# 檢查網路連通性
docker compose exec rcaeval ping rcaeval-jupyter
```

## 🔧 自訂配置

### 修改服務配置

```yaml
# 在 docker-compose.yml 中自訂環境變數
environment:
  - CUSTOM_VAR=value
  - DEBUG=1
  - PYTHONPATH=/app:/custom/path
```

### 添加新服務

```yaml
# 在 docker-compose.yml 中添加新服務
rcaeval-worker:
  build: .
  volumes:
    - .:/app
  environment:
    - PYTHONPATH=/app
  command: /bin/bash -c "source env/bin/activate && python worker.py"
```

## 📋 維護與更新

### 服務更新

```bash
# 停止服務
docker compose down

# 更新映像
docker compose build --pull

# 重新啟動
docker compose up -d

# 清理舊映像
docker image prune -f
```

### 資料備份與還原

```bash
# 備份 volumes
docker run --rm -v rcaeval-data:/data -v $(pwd):/backup ubuntu \
  tar czf /backup/rcaeval-data-backup.tar.gz -C /data .

# 還原 volumes
docker run --rm -v rcaeval-data:/data -v $(pwd):/backup ubuntu \
  tar xzf /backup/rcaeval-data-backup.tar.gz -C /data
```

## 🚦 效能最佳化

### 服務資源分配

```yaml
# 在 docker-compose.yml 中調整資源限制
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
    reservations:
      cpus: '1.0'
      memory: 2G
```

### 批次處理最佳化

```bash
# 使用背景服務執行大量實驗
docker compose exec -d rcaeval bash -c "
source env/bin/activate
nohup python batch_experiments.py > /app/results/batch.log 2>&1 &
"

# 監控執行進度
docker compose exec rcaeval tail -f /app/results/batch.log
```

## 📞 支援與協助

### 快速診斷

```bash
# 一鍵健康檢查
docker compose exec rcaeval bash -c "
echo '=== System Info ==='
uname -a
echo '=== Python Version ==='
python --version
echo '=== Package Status ==='
pip list | head -10
echo '=== Disk Space ==='
df -h /app
echo '=== Memory Usage ==='
free -h
"
```

如遇到問題，請：

1. 使用上述健康檢查指令收集資訊
2. 查看服務日誌：`docker compose logs`
3. 檢查專案的 GitHub Issues
4. 確保 Docker 和 docker-compose 版本相容
5. 確保系統有足夠的記憶體和儲存空間

## 📄 授權聲明

本專案包含多個開源組件，各自擁有不同的授權條款。詳細資訊請參閱 `LICENSES/` 目錄中的相關檔案。