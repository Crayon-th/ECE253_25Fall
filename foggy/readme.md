python src/preprocess.py --in data/real --out outputs/real/original --long_side 1024
python src/dcp.py --in outputs/real/original --out outputs/real/dcp --window 15 --omega 0.95 --radius 60 --eps 1e-3 --t0 0.1
python src/aodnet_infer.py --in outputs/real/original --out outputs/real/aodnet --weights ckpt/aodnet.pth
python src/eval_niqe.py --orig outputs/real/original --dcp outputs/real/dcp --aod outputs/real/aodnet --csv niqe_results_real.csv --summary niqe_summary_real.txt
python src/viz.py --orig outputs/real/original --dcp outputs/real/dcp --aod outputs/real/aodnet --out figs/real --max_examples 8 --niqe_csv niqe_results_real.csv

python src/combine_reports.py --csvs niqe_results_sots_o.csv niqe_results_real.csv --tags sots_o real --out_csv niqe_results_all.csv --out_summary niqe_summary_all.txt --fig figs/niqe_bar_all.png