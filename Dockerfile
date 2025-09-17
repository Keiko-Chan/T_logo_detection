FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

COPY . .

#RUN pip install numpy
#RUN pip install uvicorn
#RUN pip install fastapi
#RUN pip install pillow
#RUN pip install python-multipart
#RUN pip install opencv-python
#RUN pip install pyyaml
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p models

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]