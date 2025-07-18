FROM python:3.13-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    swig \
    build-essential \
    libopenblas-dev \
    libatlas-base-dev \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# faiss 소스 코드 다운로드 및 빌드
RUN git clone https://github.com/facebookresearch/faiss.git /tmp/faiss \
    && cd /tmp/faiss \
    && mkdir build && cd build \
    && cmake -DFAISS_ENABLE_GPU=OFF -DBUILD_SHARED_LIBS=ON -DPython_EXECUTABLE=$(which python3) .. \
    && make -j$(nproc) \
    && make install \
    && cd /app \
    && rm -rf /tmp/faiss

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 파일 복사
COPY . .

# Streamlit 실행
CMD ["streamlit", "run", "rag_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
