# Digit5, irc-msda
nohup python main.py --config DigitFive.yaml --target-domain mnist -bp ../../../ --temperature 0.8 --s_intra 0.3 --s_inter 0.0 --t_intra 0.0 --t_inter 0.0 --pl 3 --pj 0 --gpu 3  > ./log/baseline_mnist_wgcc_sg00_sl03_tg00_tl00.txt 2>&1 &

nohup python main.py --config DigitFive.yaml --target-domain mnistm -bp ../../../ --temperature 0.8 --s_intra 0.3 --s_inter 0.0 --t_intra 0.0 --t_inter 0.0 --pl 3 --pj 0 --gpu 4  > ./log/baseline_mnistm_wgcc_sg00_sl03_tg00_tl00.txt 2>&1 &

nohup python main.py --config DigitFive.yaml --target-domain svhn -bp ../../../ --temperature 0.8 --s_intra 0.3 --s_inter 0.0 --t_intra 0.0 --t_inter 0.0 --pl 3 --pj 0 --gpu 5  > ./log/baseline_svhn_wgcc_sg00_sl03_tg00_tl00.txt 2>&1 &

nohup python main.py --config DigitFive.yaml --target-domain syn -bp ../../../ --temperature 0.8 --s_intra 0.3 --s_inter 0.0 --t_intra 0.0 --t_inter 0.0 --pl 3 --pj 0 --gpu 3  > ./log/baseline_syn_wgcc_sg00_sl03_tg00_tl00.txt 2>&1 &

nohup python main.py --config DigitFive.yaml --target-domain usps -bp ../../../ --temperature 0.8 --s_intra 0.3 --s_inter 0.0 --t_intra 0.0 --t_inter 0.0 --pl 3 --pj 0 --gpu 4  > ./log/baseline_usps_wgcc_sg00_sl03_tg00_tl00.txt 2>&1 &
