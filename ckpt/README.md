---
license: cc-by-nc-sa-4.0
language:
  - en
tags:
  - vision-language pretraining
  - medical
  - ct diagnosis
---

# RADAR: An Expert-Level Generalist AI for Abdominal CT Diagnosis
[![Paper](https://img.shields.io/badge/Science-Paper-2B6CB0?logo=google-scholar&logoColor=white)](https://www.science.org/doi/10.1126/science.aec6129)
[![GitHub](https://img.shields.io/badge/GitHub-Code-B85C38?logo=github&logoColor=white)](https://github.com/alibaba-damo-academy/damo-radar)
[![Zenodo](https://img.shields.io/badge/Zenodo-Code-0F766E?logo=zenodo&logoColor=white)](https://zenodo.org/records/21271172)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-7C3F58?logo=creativecommons&logoColor=white)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

RADAR is a generalist vision-language model trained on over 400,000 contrast-enhanced abdominal CT examinations with 15 million anatomy-aware image–text pairs, learning directly from clinical reports without manual annotation. RADAR provides a scalable and versatile framework for radiology AI, demonstrating expert-level performance across both routine and complex clinical tasks.

<p align="center">
  <img src="radar_fig0.png" alt="RADAR Overview" width="90%">
</p>

## Model Preparation

Pre-trained checkpoints are available on [HuggingFace](https://huggingface.co/datasets/radar-generalist/RADAR).

| File | Description | Destination |
| --- | --- | --- |
| `checkpoint_radar_pretrain.pth` | RADAR pre-trained on RAD-CT | `radar/ckpt/checkpoint_radar_pretrain.pth` |
| `bert-base-chinese` | BERT tokenizer and model (Chinese) | `radar/ckpt/bert-base-chinese/` |
| `bert-base-uncased` | BERT tokenizer and model (English) | `radar/ckpt/bert-base-uncased/` |
| `checkpoint_unet.pth` | Pretrained VisionBranch (UNet) checkpoint | `radar/ckpt/checkpoint_unet.pth` |
| `checkpoint_radar_plus.pth` | RADAR+ checkpoint trained from scratch on Merlin-CT-Train | `radar/ckpt/checkpoint_radar_plus.pth` |
| `checkpoint_radar_plus_finetuned_on_merlin.pth` | RADAR+ checkpoint pretrained on RAD-CT and finetuned on Merlin-CT-Train | `radar/ckpt/checkpoint_radar_plus_finetuned_on_merlin.pth` |

## License

This project is released under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

> The Radar model is currently intended for research purposes only. Further improvements and prospective clinical studies are still required before it can be used directly for clinical deployment.


## Citation

If you use these models in your research, please cite:

```bibtex
@article{damo-radar-2026,
    author = {Qi Zhang and Jianpeng Zhang and Weiwei Cao and Zilin Lu and Wanxing Chang and Haonan Ding and Cao Chen and Zhi Li and Xing Xue and Sinuo Wang and Shaoteng Zhang and Yutong Xie and Yong Xia and Qi Wu and Zhongyi Shui and Xi Li and Zhilin Zheng and Yanjie Zhou and Tony C.W. Mok and Yingda Xia and Hongkan Wang and Xianghua Ye and Tao Ma and Jie Peng and Xiaoguang Wang and Jian Ding and Yuming Gao and Huazhen Ye and Yiping Liu and Dongjie Chen and Zhaomin Ni and Jianwen Ning and Wei Zhang and Jian Liu and Chaohui Yu and Shenghong Ju and Jianfeng Zhang and Wenbo Xiao and Ling Zhang and Tingbo Liang },
    title = {An expert-level generalist AI for abdominal CT diagnosis},
    journal = {Science},
    volume = {393},
    number = {6817},
    pages = {eaec6129},
    year = {2026},
    doi = {10.1126/science.aec6129},
    URL = {https://www.science.org/doi/abs/10.1126/science.aec6129}
}
```