torchrun --nproc_per_node=3 train.py -C forest_wos2_60d_r8_all -f -1
torchrun --nproc_per_node=3 train.py -C forest_ws2_60d_expert -f -1
torchrun --nproc_per_node=3 train.py -C forest_ws2_60d_r8_all -f -1
torchrun --nproc_per_node=3 train.py -C forest_ws2_30d_r8_all -f -1
torchrun --nproc_per_node=3 train.py -C forest_ws2_60d_r10 -f -1
torchrun --nproc_per_node=3 train.py -C forest_ws2_72d_r16 -f -1
torchrun --nproc_per_node=3 train.py -C forest_ws2_60d_r8_fix_bands -f -1
torchrun --nproc_per_node=3 train.py -C forest_ws2_final_v2 -f -1
