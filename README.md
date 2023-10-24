# RADE
code for paper 'Reference-Assisted Dialogue Evaluation for Open-Domain Dialogue (ACL2023)'.

## Introduction

Evaluating open-domain dialogue systems is challenging for reasons such as the one-to-many problem, i.e., many appropriate responses other than just the golden response.   
As of now, automatic evaluation methods need better consistency with humans, while reliable human evaluation can be time- and cost-intensive. 
To this end, we propose the Reference-Assisted  Dialogue Evaluation (RADE) approach under the multi-task learning framework, which leverages the pre-created utterance as reference other than the gold response to relief the one-to-many problem. 
Specifically, RADE explicitly compares reference and the candidate response to predict their overall scores.
Moreover, an auxiliary response generation task enhances prediction via a shared encoder.



## Dataset

We first collect a pre-train dataset based on existing works. To support RADE, we also extend three datasets with additional rated responses other than just a golden response by human annotation.

All the datasets can be found at `./dataset` folder.



## Todo

-  Add the code to reproduce our result.



## Contact us

Mail to shizhl@mail.sdu.edu.cn



## Citation

```txt
@inproceedings{shi2023rade,
  title={RADE: Reference-Assisted Dialogue Evaluation for Open-Domain Dialogue},
  author={Shi, Zhengliang and Sun, Weiwei and Zhang, Shuo and Zhang, Zhen and Ren, Pengjie and Ren, Zhaochun},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={12856--12875},
  year={2023}
}
```





 
