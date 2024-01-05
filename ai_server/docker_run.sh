#!/bin/bash
cd /root/workspace/OCR_Center/ai_server && CUDA_VISIBLE_DEVICES=0 nohup python manage_ocr_center.py >/dev/null 2>&1 &

while true; do
  sleep 6
done
