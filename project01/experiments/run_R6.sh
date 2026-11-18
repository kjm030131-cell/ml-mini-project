#!/bin/bash
# R6 실행 순서 (2코어·7GB: SVC 는 동시에 1개만)
cd "$(dirname "$0")"
python3 R6_meta.py svc15 & python3 R6_meta.py lgbm & wait
python3 R6_meta.py svcR4 & python3 R6_meta.py lr & wait
python3 R6_meta.py raw
python3 R6_meta.py meta
echo ALL_DONE
