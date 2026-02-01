#!/bin/bash

# 1. Bật Worker chạy ngầm (Background)
# Dùng & để nó không chặn tiến trình tiếp theo
python -m app.workers.process_worker &

# 2. Bật Web API (Tiến trình chính)
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}