FROM ubuntu:22.04

# 設置環境變數
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Taipei

# 安裝系統依賴
RUN apt-get update -y && \
    apt-get install -y \
        build-essential \
        libxml2 \
        libxml2-dev \
        zlib1g-dev \
        python3-tk \
        graphviz \
        software-properties-common \
        git \
        wget \
        curl \
        openjdk-11-jre-headless \
        pkg-config \
        libcairo2-dev \
        libgirepository1.0-dev \
        && rm -rf /var/lib/apt/lists/*

# 安裝 Python 3.10 和 3.8
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3.10-distutils \
        python3.8 \
        python3.8-dev \
        python3.8-venv \
        python3.8-distutils \
        && rm -rf /var/lib/apt/lists/*

# 分別安裝 pip - 修復 Python 3.8 的 pip 安裝問題
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

RUN wget https://bootstrap.pypa.io/pip/3.8/get-pip.py && \
    python3.8 get-pip.py && \
    rm get-pip.py

# 設置 Python 3.10 為預設
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# 設置工作目錄
WORKDIR /app

# 複製專案文件
COPY . .

# 創建 default 環境 (Python 3.10) - 修復依賴順序問題
RUN python3.10 -m venv env && \
    . env/bin/activate && \
    pip install --upgrade pip==20.0.2 && \
    pip install wheel && \
    pip install numpy && \
    pip install -e .[default]

# 創建 RCD 環境 (Python 3.8)
RUN python3.8 -m venv env-rcd && \
    . env-rcd/bin/activate && \
    pip install --upgrade pip==20.0.2 && \
    pip install wheel && \
    pip install numpy && \
    pip install -e .[rcd]

# 執行連結腳本來連結自定義的庫文件
RUN chmod +x script/link.sh && \
    bash script/link.sh || true

# 安裝 PyRCA (用於 HT 和 E-Diagnosis 方法)
RUN . env/bin/activate && \
    git clone https://github.com/salesforce/PyRCA.git /tmp/PyRCA && \
    cd /tmp/PyRCA && \
    pip install -e . && \
    cd /app && \
    rm -rf /tmp/PyRCA

# 創建數據目錄
RUN mkdir -p data results

# 設置權限
RUN chmod -R 755 /app

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python --version || exit 1

# 暴露常用端口（如需要 web 介面）
EXPOSE 8080 8888

# 設置標籤
LABEL maintainer="RCAEval Project"
LABEL description="Docker environment for RCAEval benchmark - Root Cause Analysis for Microservice Systems"
LABEL version="1.1.2"

# 預設命令：啟動 default 環境的 bash
CMD ["/bin/bash", "-c", "source env/bin/activate && echo 'RCAEval environment ready. Use env-rcd for RCD-specific tasks.' && /bin/bash"]