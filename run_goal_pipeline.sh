#!/usr/bin/env bash
set -euo pipefail

cd /workspace/attention-is-all-you-need-for-vlms
mkdir -p runs/goal-pipeline
exec >> runs/goal-pipeline/pipeline.log 2>&1
echo "$(date -Is) pipeline-start"

# Serialize the GPU queue while the network transfer runs unattended.
while pgrep -f '[w]get .*gqa-images.zip' >/dev/null || pgrep -f '[s]tream_extract_gqa.py' >/dev/null || pgrep -f '[b]enchmark_efficiency.py' >/dev/null || pgrep -f '[r]un.py evaluate' >/dev/null; do sleep 60; done

mkdir -p data/finecops/images
if [ "$(find data/finecops/images -type f -name '*.jpg' | wc -l)" -lt 4313 ]; then
  if [ -f /workspace/gqa-images.zip ] && [ "$(stat -c%s /workspace/gqa-images.zip)" -ge 21817965542 ]; then
    mapfile -t entries < <(sed 's#^#images/#' data/finecops/gqa_needed.txt)
    unzip -q -j /workspace/gqa-images.zip "${entries[@]}" -d data/finecops/images || true
    if [ "$(find data/finecops/images -type f -name '*.jpg' | wc -l)" -lt 4313 ]; then
      rm -f data/finecops/images/*.jpg
      python extract_finecops_images.py --archive /workspace/gqa-images.zip --list data/finecops/gqa_needed.txt --output data/finecops/images
    fi
  fi
  test "$(find data/finecops/images -type f -name '*.jpg' | wc -l)" -ge 4313
fi
if [ ! -f data/finecops-test.jsonl ]; then
  python prepare_finecops.py --annotations data/finecops/test_expression_pos_coco_format.json --expressions data/finecops/test_expression_pos.json --images data/finecops/images --output data/finecops-test.jsonl
fi

checkpoint() {
  case "$1:$2" in
    A4:0) echo runs/day2-local/refcocog-siglip2-A4-s0/best.pt;;
    A4:1) echo runs/day2-new/refcocog-siglip2-A4-s1/best.pt;;
    A4:2) echo runs/day2-new/refcocog-siglip2-A4-s2/best.pt;;
    S4:0) echo runs/publishable-v2/refcocog-siglip2-S4-s0-recovered/best.pt;;
    S4:1) echo runs/day2-new/refcocog-siglip2-S4-s1/best.pt;;
    S4:2) echo runs/day2-new/refcocog-siglip2-S4-s2/best.pt;;
    A8:0) echo runs/publishable-v2/refcocog-siglip2-A8-s0/best.pt;;
    A8:1) echo runs/publishable-v2/refcocog-siglip2-A8-s1/best.pt;;
    A8:2) echo runs/publishable-v2/refcocog-siglip2-A8-s2/best.pt;;
  esac
}
for variant in A4 S4 A8; do
  for seed in 0 1 2; do
    output="runs/finecops-eval/finecops-siglip2-${variant}-s${seed}.pt"
    if [ ! -f "$output" ]; then
      python run.py evaluate --checkpoint "$(checkpoint "$variant" "$seed")" --data data/finecops-test.jsonl --output "$output" --batch-size 32 --mass 0.8
    fi
  done
done
if [ ! -f runs/finecops-analysis/finecops_summary.md ]; then
  python analyze_finecops.py --data data/finecops-test.jsonl --predictions runs/finecops-eval --output runs/finecops-analysis
fi

if [ ! -f runs/efficiency/measurements.json ]; then
  python benchmark_efficiency.py --data data/refcocog-test.jsonl --checkpoint runs/day2-local/refcocog-siglip2-A4-s0/best.pt --output runs/efficiency/measurements.json --batch-size 16 --repeats 40 --warmups 8
fi
python summarize_efficiency.py --input runs/efficiency/measurements.json --output runs/efficiency/efficiency_table

# A single-seed CLIP control is deliberately the smallest transfer test.
for variant in A4 S4; do
  out="runs/clip-control/refcocog-clip-${variant}-s0"
  if [ ! -f "$out/best.pt" ]; then
    # Keep the same effective batch (64) while reducing frozen-backbone microbatch overhead.
    python run.py train --train data/refcocog-train.jsonl --val data/refcocog-val.jsonl --output "$out" --backbone clip --variant "$variant" --seed 0 --learning-rate 0.0003 --batch-size 64 --accumulate 1 --steps 5000 --warmup-steps 250 --eval-every 500
  fi
  test_output="runs/clip-control/refcocog-clip-${variant}-s0-test.pt"
  if [ ! -f "$test_output" ]; then
    python run.py evaluate --checkpoint "$out/best.pt" --data data/refcocog-test.jsonl --output "$test_output" --batch-size 16 --mass 0.8
  fi
done
if [ -f runs/clip-control/refcocog-clip-A4-s0-test.pt ] && [ -f runs/clip-control/refcocog-clip-S4-s0-test.pt ]; then
  python analyze_clip_control.py --a4 runs/clip-control/refcocog-clip-A4-s0-test.pt --s4 runs/clip-control/refcocog-clip-S4-s0-test.pt --output runs/clip-control/clip_control_summary.json
fi

if [ -f runs/refcoco-eval/refcoco-A4-s0-testA.pt ] && [ -f runs/refcoco-eval/refcoco-A4-s0-testB.pt ] && [ -f runs/refcoco-eval/refcoco-S4-s0-testA.pt ] && [ -f runs/refcoco-eval/refcoco-S4-s0-testB.pt ]; then
  python analyze_refcoco_eval.py --predictions runs/refcoco-eval --output runs/refcoco-eval/refcoco_seed0
fi

mkdir -p docs/results/finecops docs/results/efficiency docs/results/clip-control docs/results/refcoco
cp runs/finecops-analysis/finecops_*.md runs/finecops-analysis/finecops_*.csv runs/finecops-analysis/finecops_*.json docs/results/finecops/ 2>/dev/null || true
cp runs/finecops-analysis/*.png docs/results/finecops/ 2>/dev/null || true
cp runs/efficiency/efficiency_table.csv runs/efficiency/efficiency_table.md runs/efficiency/measurements.json docs/results/efficiency/ 2>/dev/null || true
cp runs/clip-control/clip_control_summary.* docs/results/clip-control/ 2>/dev/null || true
cp runs/refcoco-eval/refcoco_seed0.* docs/results/refcoco/ 2>/dev/null || true
python synthesize_paper_findings.py --root .
cat >> docs/progress-log.md <<'EOF'

## 2026-08-14 — Publishable expansion pipeline completed

- FineCops-Ref positive-test evaluation, decoder efficiency, and the minimal one-seed CLIP-family control were run from immutable existing checkpoints and frozen `tau = 0.8`; raw run artifacts remain under `runs/`.
- FineCops slices use only official level and tuple-type metadata; no LLM-derived labels or test-time tuning were added. Efficiency reports cached-decoder and full-pipeline measurements separately.
EOF
cat >> docs/decision-log.md <<'EOF'

## 2026-08-14 — Keep the publishable expansion bounded

- **Decision:** Stop after FineCops-Ref, decoder efficiency, and one matched CLIP-family control; do not launch RefCOCO+, extra classic seeds, or another backbone automatically.
- **Reason:** These three measurements directly test the remaining failure-boundary, efficiency, and backbone-transfer questions while preserving the completed evidence.
EOF
git add docs/results docs/progress-log.md docs/decision-log.md
git commit -m "Record publishable expansion results"
git push origin HEAD:runpod-publishable-results
echo "$(date -Is) pipeline-complete"
