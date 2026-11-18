#!/bin/bash
# R7: 스모크 후 3개 설정을 차례로 실행 (메모리 7GB 안, 동시 1개)
cd "$(dirname "$0")"
python3 R7_quadrant.py smoke || exit 1
for m in q15 q3 q15sk; do python3 R7_quadrant.py $m; done
echo ALLDONE
