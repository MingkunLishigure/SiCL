# MaskCL: Semantic Mask-Driven Contrastive Learning for Unsupervised Person Re-Identification with Clothes Change

The code of :**MaskCL: Semantic Mask-Driven Contrastive Learning for Unsupervised Person Re-Identification with Clothes Change**

 

MaskCL focuses on a novel task: unsupervised clothes changing person re-identification. 

We have achieved remarkably outstanding results, and to the best of our knowledge, this is the first work of its kind in the domain of clothes changing person re-identification. 

We warmly welcome and encourage fellow researchers to engage in enlightening discussions and exchanges on this topic!


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maskcl-semantic-mask-driven-contrastive/unsupervised-person-re-identification-on-ltcc)](https://paperswithcode.com/sota/unsupervised-person-re-identification-on-ltcc?p=maskcl-semantic-mask-driven-contrastive)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maskcl-semantic-mask-driven-contrastive/unsupervised-person-re-identification-on-vc)](https://paperswithcode.com/sota/unsupervised-person-re-identification-on-vc?p=maskcl-semantic-mask-driven-contrastive)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maskcl-semantic-mask-driven-contrastive/unsupervised-person-re-identification-on-prcc)](https://paperswithcode.com/sota/unsupervised-person-re-identification-on-prcc?p=maskcl-semantic-mask-driven-contrastive)



**Paper Link** 

https://arxiv.org/abs/2305.13600


**Train**:
```
sh run_code.sh
```

**Remark**:
It should be noted that the mask images used in the **MaskCL** are generated based on human parsing networks. In this study, we employed [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) for pedestrian silhouette extraction. We extend a warm invitation to researchers to venture into exploring diverse approaches in generating pedestrian semantic information and conducting experiments on MaskCL. We are greatly anticipating the researchers' invaluable contributions in terms of sharing and providing feedback on their experimental outcomes!
