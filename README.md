# MaskCL: Semantic Mask-Driven Contrastive Learning for Unsupervised Person Re-Identification with Clothes Change

 
MaskCL focuses on a novel task: **unsupervised clothes changing person re-identification**. 

:star: We have achieved remarkably outstanding results!!!! 

:star: And to the best of our knowledge, **this is the first work of its kind in the domain of clothes changing person re-identification**!!!

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maskcl-semantic-mask-driven-contrastive/unsupervised-person-re-identification-on-ltcc)](https://paperswithcode.com/sota/unsupervised-person-re-identification-on-ltcc?p=maskcl-semantic-mask-driven-contrastive)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maskcl-semantic-mask-driven-contrastive/unsupervised-person-re-identification-on-vc)](https://paperswithcode.com/sota/unsupervised-person-re-identification-on-vc?p=maskcl-semantic-mask-driven-contrastive)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maskcl-semantic-mask-driven-contrastive/unsupervised-person-re-identification-on-prcc)](https://paperswithcode.com/sota/unsupervised-person-re-identification-on-prcc?p=maskcl-semantic-mask-driven-contrastive)

:heart: We warmly welcome and encourage fellow researchers to engage in enlightening discussions and exchanges on this topic!!!

📚**Paper Link** 

https://arxiv.org/abs/2305.13600

📚**Dataset**：

We evaluate MaskCL on Six datasets:
 
| Dataset | Link |
| ------- | ------- 
| PRCC | https://www.isee-ai.cn/~yangqize/clothing.html|
| LTCC | https://naiq.github.io/LTCC_Perosn_ReID.html | 
| Celeb-ReID| https://github.com/Huang-3/Celeb-reID | 
| Celeb-ReID-Light| https://github.com/Huang-3/Celeb-reID | 
| VC-Clothes| https://wanfb.github.io/dataset.html | 
| DeepChange| https://github.com/PengBoXiangShang/deepchange | 

📚**Remark**:

:speech_balloon: In MaskCL, during the training process, it needs to prepare the original dataset along with the corresponding dataset of person masks.

The mask images used in the **MaskCL** are generated based on human parsing networks. 

In this study, we employed [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) for pedestrian silhouette extraction. 

We extend a warm invitation to researchers to venture into exploring diverse approaches in generating pedestrian semantic information and conducting experiments on MaskCL. 

We are greatly anticipating the researchers' invaluable contributions in terms of sharing and providing feedback on their experimental outcomes!


After generating the corresponding mask image dataset, you should change the dataset path in **CMC.py** and **/utils/dataset/data/preprocessor.py**.


:bulb:**Train**:
```
sh run_code.sh
```

:loudspeaker: We will release the model weight soon!


