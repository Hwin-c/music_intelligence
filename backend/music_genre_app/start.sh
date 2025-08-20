#!/bin/sh
# Gunicorn 워커 타임아웃을 프론트엔드와 동일하게 300초(5분)로 설정합니다.
# 0으로 설정하여 비활성화하는 것보다 안정적입니다.
gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 300 app:app