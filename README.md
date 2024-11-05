# SiCL: Silhouette-Driven Contrastive Learning for Unsupervised Person Re-Identification with Clothes Change

 
:rocket: SiCL is dedicated to an innovative task: **unsupervised clothes changing person re-identification**. 

:star: Within the realm of clothing changing person re-identification, SiCL proudly stands as the inaugural unsupervised methodology to attain commendable outcomes across a multitude of datasets!!!


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maskcl-semantic-mask-driven-contrastive/unsupervised-person-re-identification-on-ltcc)](https://paperswithcode.com/sota/unsupervised-person-re-identification-on-ltcc?p=maskcl-semantic-mask-driven-contrastive)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maskcl-semantic-mask-driven-contrastive/unsupervised-person-re-identification-on-prcc)](https://paperswithcode.com/sota/unsupervised-person-re-identification-on-prcc?p=maskcl-semantic-mask-driven-contrastive)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maskcl-semantic-mask-driven-contrastive/unsupervised-person-re-identification-on-vc)](https://paperswithcode.com/sota/unsupervised-person-re-identification-on-vc?p=maskcl-semantic-mask-driven-contrastive)

:heart: We warmly welcome and encourage fellow researchers to engage in enlightening discussions and exchanges on this topic!!!

📚**Paper Link** 

https://arxiv.org/abs/2305.13600

📚**Dataset**：

We evaluate SiCL on Six datasets:
 
| Dataset | Link |
| ------- | ------- 
| PRCC | https://www.isee-ai.cn/~yangqize/clothing.html|
| LTCC | https://naiq.github.io/LTCC_Perosn_ReID.html | 
| Celeb-ReID| https://github.com/Huang-3/Celeb-reID | 
| Celeb-ReID-Light| https://github.com/Huang-3/Celeb-reID | 
| VC-Clothes| https://wanfb.github.io/dataset.html | 
| DeepChange| https://github.com/PengBoXiangShang/deepchange | 

:speech_balloon: **Remark**:

In SiCL, during the training process, it needs to prepare the original dataset along with the corresponding dataset of person masks.

The mask images used in the **SiCL** are generated based on human parsing networks. 

In this study, we employed [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) for pedestrian silhouette extraction. 

After generating the corresponding mask image dataset, you should change the dataset path in **CMC.py** and **/utils/dataset/data/preprocessor.py**.







:heart: **We extend a warm invitation to researchers to venture into exploring diverse approaches in generating pedestrian semantic information and conducting experiments on SiCL. 

:heart:We are greatly anticipating the researchers' invaluable contributions in terms of sharing and providing feedback on their experimental outcomes!**





:bulb:**Train**:
```
sh run_code.sh
```


