#!/bin/sh

# Render가 주입해주는 PORT 환경 변수를 사용하여 Gunicorn을 실행합니다.
gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 --no-preload --graceful-timeout 60 app:app