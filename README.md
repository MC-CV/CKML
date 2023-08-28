# CKML
This is our Tensorflow implementation for CKML.

The code has been tested running under Python 3.6.15. The required packages are as follows:
- nvidia-tensorflow == 1.15.4+nv20.10
- tensorflow-determinism == 0.3.0
- numpy == 1.19.5
- scipy == 1.7.3


For Yelp data, use the following command to train and test
```
python labcode_yelp.py --data yelp
```

For Online Retail data, 
```
python labcode_retail.py --data retail
```

For Tmall data, 
```
python labcode_retail.py --data tmall
```

## Citation
If you want to use our codes and datasets in your research, please cite:
```
@article{ckml,
title = {Coarse-to-Fine Knowledge-Enhanced Multi-Interest Learning Framework for Multi-Behavior Recommendation},
author = {Meng, Chang and Zhao, Ziqi and Guo, Wei and Zhang, Yingxue and Wu, Haolun and Gao, Chen and Li, Dong and Li, Xiu and Tang, Ruiming},
year = {2023},
volume = {42},
number = {1},
journal = {ACM Trans. Inf. Syst.},
articleno = {30},
numpages = {27}
}
```
