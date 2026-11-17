#!/bin/bash
# R1+R2 한 세션 실행 (메모리 7GB: 동시 2개까지). experiments 폴더에서 실행.
set -x
python3 R1_stack_cv2.py prep && python3 R2_stack_diverse.py prep || exit 1
python3 R1_stack_cv2.py svc15 & python3 R1_stack_cv2.py lgbm & wait
python3 R1_stack_cv2.py raw & python3 R2_stack_diverse.py poly15 & wait
python3 R2_stack_diverse.py svcH & python3 R2_stack_diverse.py et & wait
python3 R1_stack_cv2.py lr
python3 R1_stack_cv2.py meta
python3 R2_stack_diverse.py meta
echo ALL_DONE
