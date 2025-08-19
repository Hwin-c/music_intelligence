#!/bin/sh
# --no-preload와 --graceful-timeout 옵션을 제거합니다.
# --timeout 0 은 Render에서 긴 부팅 시간을 허용하는 데 유용하므로 유지합니다.
gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 app:app