docker run -p 8000:8000 \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/configs:/app/configs \
  tbank-detector