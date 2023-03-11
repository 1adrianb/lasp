## LASP: Text-to-Text Optimization for Language-Aware Soft Prompting of Vision & Language Models 

PDF with the paper: [[here]](https://www.adrianbulat.com/downloads/CVPR2023/LASP.pdf)

Soft prompt learning has recently emerged as one of the
methods of choice for adapting V&L models to a down-
stream task using a few training examples. However, cur-
rent methods significantly overfit the training data, suffering from large accuracy degradation when tested on unseen classes from the same domain. To this end, in this paper, we make the following 4 contributions: (1) To alleviate base class overfitting, we propose a novel Language-Aware Soft Prompting (LASP) learning method by means of a text-to-text cross-entropy loss that maximizes the probability of the learned prompts to be correctly classified with respect to pre-defined hand-crafted textual prompts. (2) To increase the representation capacity of the prompts, we propose grouped LASP where each group of prompts is optimized with respect to a separate subset of textual prompts. (3) We identify a visual-language misalignment introduced by prompt learning and LASP, and more importantly, propose a re-calibration mechanism to address it. (4) We show that LASP is inherently amenable to including, during training, virtual classes, i.e. class names for which no visual samples are available, further increasing the robustness of the learned prompts. Through evaluations on 11 datasets, we show that our approach (a) significantly outperforms all prior works on soft prompting, and (b) matches and surpasses, for the first time, the accuracy on novel classes obtained by hand-crafted prompts and CLIP for 8 out of 11 test datasets.

## Setup

Install the required dependencies: 
```bash
pip install -r requirements.txt
```
Download and prepare the datasets using the instructions listed [here](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md). Alternatively, you can use the download helper utility, simply runing: ``python datasets/utils/download_datasets.py --root path_to_datasets``

## Usage

*Note:* The schedulers where tweaked compared with the original paper in the interest of training speed, switching from the batch_size=1 setup of CoCoOp to batch_size=N. As a result the numbers will be different. Overall the new numbers are higher than in the original paper, even for G=1:

Evaluation results:

|      | ImageNet | Caltech101 | Oxford Pets | Stanford Cars | Flowers 102 | Food 101 | FGVC  | SUN397 | DTD   | EuroSAT | UCF101 | Avg.  |
| ---- | -------- | ---------- | ----------- | ------------- | ----------- | -------- | ----- | ------ | ----- | ------- | ------ | ----- |
| Base | 76.25    | 98.45      | 95.64       | 76.25         | 97.62       | 90.84    | 35.51 | 81.34  | 80.63 | 94.57   | 85.47  | 82.96 |
| New  | 71.17    | 94.43      | 97.39       | 71.99         | 74.16       | 91.62    | 38.15 | 78.45  | 63.36 | 85.05   | 77.33  | 76.64 |
| H    | 73.62    | 96.39      | 96.50       | 74.06         | 84.29       | 91.23    | 36.78 | 79.87  | 70.96 | 89.55   | 81.20  | 79.67 |

Logs can be found in ```logs/``` folder.


## Citation

```bibtex
@inproceedings{bulat2023lasp,
  title={LASP: Text-to-Text Optimization for Language-Aware Soft Prompting of Vision \& Language Models},
  author={Bulat, Adrian and Tzimiropoulos, Georgios},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23232--23241},
  year={2023}
}
```

## Acknowledgment 

Code based on the [CoOp repository](https://github.com/KaiyangZhou/CoOp).