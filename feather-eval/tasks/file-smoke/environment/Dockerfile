FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    file poppler-utils libreoffice-calc libreoffice-writer libreoffice-impress \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip uninstall -y openpyxl python-docx python-pptx \
    && python -m pip install --no-cache-dir --force-reinstall \
       paper-docx==0.2.0 paper-pptx==0.2.0 paper-xlsx==0.2.1 \
       'reportlab>=4,<5' 'pypdf>=6,<7' 'pdfplumber>=0.11,<0.12' \
       'pdf2image>=1.17,<2' 'Pillow>=10.3,<13' \
    && paper-docx-doctor && paper-pptx-doctor && paper-xlsx-doctor
RUN useradd --create-home --uid 1001 agent && mkdir /workspace && chown agent:agent /workspace
WORKDIR /workspace
COPY --chown=agent:agent inventory.csv /workspace/inventory.csv
