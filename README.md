## Code for CS329D project
This repository contains my code for CS329D project. It is forked from WILDS github repository and I have added my implementation for CDAN, BSP, NWD and BNM methods.
- `examples/algorithms/DANN.py` and `examples/models/domain_adversarial_network.py` is modified to allow BSP penalty, NWD penalty, CDAN and CDANE.
- `examples/algorithms/ERM.py` is modified to allow BNM method. 
- `examples/models/cp_impl.py` is an additional module implemented by me. It contains useful classes and functions for CDAN, BSP, NWD and BNM.
- `examples/train.py` is modified to save features to carry out analysis later.
- `cdan_analysis.ipynb` notebook is implemented to do post-training analysis of learned features from CDAN and CDANE.
- `bsp_analysis.ipynb` notebook is implemented to do post-training analysis of learned features from BSP.
- `ztest.md` has commands used to run most of the experiments.
- `compile_results.py` aggregates results and shows it in one table.