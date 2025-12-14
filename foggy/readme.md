# Single-Image Dehazing (Foggy)

This project runs Dark Channel Prior (DCP) and AOD-Net on foggy images and evaluates them with NIQE.

## How to Run

### 1. SOTS-Outdoor hazy images

```bash
python src/preprocess.py --in data/sots_o/hazy --out outputs/sots_o/original --long_side 1024

python src/dcp.py --in outputs/sots_o/original --out outputs/sots_o/dcp \
  --window 15 --omega 0.95 --radius 60 --eps 1e-3 --t0 0.1

python src/aodnet_infer.py --in outputs/sots_o/original --out outputs/sots_o/aodnet \
  --weights ckpt/aodnet.pth

python src/eval_niqe.py \
  --orig outputs/sots_o/original \
  --dcp outputs/sots_o/dcp \
  --aod outputs/sots_o/aodnet \
  --csv niqe_results_sots_o.csv \
  --summary niqe_summary_sots_o.txt

python src/viz.py \
  --orig outputs/sots_o/original \
  --dcp outputs/sots_o/dcp \
  --aod outputs/sots_o/aodnet \
  --out figs/sots_o \
  --max_examples 8 \
  --niqe_csv niqe_results_sots_o.csv
```

### 2. Real foggy images

```bash
python src/preprocess.py --in data/real --out outputs/real/original --long_side 1024

python src/dcp.py --in outputs/real/original --out outputs/real/dcp \
  --window 15 --omega 0.95 --radius 60 --eps 1e-3 --t0 0.1

python src/aodnet_infer.py --in outputs/real/original --out outputs/real/aodnet \
  --weights ckpt/aodnet.pth

python src/eval_niqe.py \
  --orig outputs/real/original \
  --dcp outputs/real/dcp \
  --aod outputs/real/aodnet \
  --csv niqe_results_real.csv \
  --summary niqe_summary_real.txt

python src/viz.py \
  --orig outputs/real/original \
  --dcp outputs/real/dcp \
  --aod outputs/real/aodnet \
  --out figs/real \
  --max_examples 8 \
  --niqe_csv niqe_results_real.csv