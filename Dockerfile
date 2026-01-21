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

# Install Python 3.10 and 3.8
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

# Install pip for each Python version - fix pip installation issue for Python 3.8
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

RUN wget https://bootstrap.pypa.io/pip/3.8/get-pip.py && \
    python3.8 get-pip.py && \
    rm get-pip.py

# Set Python 3.10 as the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# 設置工作目錄
WORKDIR /app

# 複製專案文件
COPY . .

# Create default environment (Python 3.10) - fix dependency ordering issues
RUN python3.10 -m venv env && \
    . env/bin/activate && \
    pip install --upgrade pip==20.0.2 && \
    pip install wheel && \
    pip install numpy && \
    pip install -e .[default]

# Create RCD environment (Python 3.8)
RUN python3.8 -m venv env-rcd && \
    . env-rcd/bin/activate && \
    pip install --upgrade pip==20.0.2 && \
    pip install wheel && \
    pip install numpy && \
    pip install -e .[rcd]

# 執行連結腳本來連結自定義的庫文件
RUN chmod +x script/link.sh && \
    bash script/link.sh || true

# Install PyRCA (used by HT and E-Diagnosis methods)
RUN . env/bin/activate && \
    git clone https://github.com/salesforce/PyRCA.git /tmp/PyRCA && \
    cd /tmp/PyRCA && \
    pip install -e . && \
    cd /app && \
    rm -rf /tmp/PyRCA

# Create data and results directories
RUN mkdir -p data results

# Set permissions
RUN chmod -R 755 /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python --version || exit 1

# Expose commonly used ports (for web interfaces etc.)
EXPOSE 8080 8888

# Set image labels
LABEL maintainer="RCAEval Project"
LABEL description="Docker environment for RCAEval benchmark - Root Cause Analysis for Microservice Systems"
LABEL version="1.1.2"

# Default command: start bash in the default environment
CMD ["/bin/bash", "-c", "source env/bin/activate && echo 'RCAEval environment ready. Use env-rcd for RCD-specific tasks.' && /bin/bash"]