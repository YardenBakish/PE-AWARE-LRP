# PyTorch Implementation of Revisiting LRP: Positional Attribution as the Missing Ingredient for Transformer Explainability
<div class="grid" markdown>
<img src="/PropogationOverATTNPE.jpg" alt="Alt text" width="400" style="display: inline-block;" height="400"  >
<img src="/PE_LRP_examples.png" alt="Alt text" width="400" height="400" style="display: inline-block" >
</div>

## Usage

### [DEMO](https://colab.research.google.com/drive/1NyRSQG2IlCJ362FAFdgh3EZcpNFnjsqF?usp=sharing),


## Vision
- Segmentation Tests:
  * Download the data [gtsegs_ijcv.mat](http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat), and place it in the main folder.
  * Run the following:  <pre> ``` python analyze_seg_results.py  --check-all ``` </pre> 
    
- Perturbation Tests: 
  * Download the weights (.pth file) for [DeiT](https://github.com/facebookresearch/deit/blob/main/README_deit.md)  (Tiny, Small, or Base)
  * Download the [Imagenet](https://www.image-net.org/) dataset. In config.py, line 25, put the download path
  * Run the following:  <pre> ```python analyze_pert_results.py --check-all --fract <> --model <*.pth> --batch-size <> ``` </pre> 

## NLP
- Finetune:
  To finetune Llama-Tiny and Llama-2-7b on the IMDB dataset, do as follows:
  * make NLP/ your current working directory
  * Download [IMDB Moview Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), and place it in /attDatasets
  * Run the following:  <pre> ```python main_imdb.py --num-warmup-steps 5 --lr 5e-4 --model-size {llama_tiny | llama_2_7b} ``` </pre>
  * Both models should converge to 92% accuracy within ~10 epochs.
- Perturbation Tests:
  We provide emperical results in the article for [IMDB Moview Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), [ARC-Easy](https://huggingface.co/datasets/allenai/ai2_arc/viewer/ARC-Easy), and [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia). To reproduce the results, run:  <pre> ```python eval_pert_arc.py --mode pert --model-size {llama_3_8b} --pe --single-norm``` </pre> <pre> ```python eval_pert_wiki.py --mode pert --model-size {llama_3_8b} --pe --single-norm``` </pre> <pre> ```python eval_pert_imdb.py --mode pert --model-size {llama_tiny | llama_2_7b} --pe --single-norm``` </pre>
  Omit --pe to achieve results for the standard method.

## Citation
If you use this codebase, or otherwise found our work valuable, please cite:
 <pre> TBD </pre> 

## Acknowledgments
The code is heavily inspired by [Transformer-Explainability](https://github.com/hila-chefer/Transformer-Explainability) for Vision models, and [AttnLRP](https://github.com/rachtibat/LRP-eXplains-Transformers/tree/main) for NLP. Thanks for their wonderful works.
