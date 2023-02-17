# NEW! TFDS API for Meta-Dataset

To accompany the presentation of the [VTAB+MD paper](https://openreview.net/pdf?id=Q0hm0_G1mpH)
at NeurIPS 2021's Datasets and Benchmarks track, we are releasing a TensorFlow
Datasets-based implementation of Meta-Dataset's input pipeline which is
compatible with both the original Meta-Dataset protocol (MD-v1) and the updated
protocol designed for VTAB+MD (MD-v2). See [the documentation page](meta_dataset/data/tfds/README.md)
for more information and example code snippets.

# Meta-Dataset

This repository contains accompanying code for the article introducing
Meta-Dataset, [arxiv.org/abs/1903.03096](https://arxiv.org/abs/1903.03096) and the follow-up paper that proposes the VTAB+MD merged benchmark [arxiv.org/abs/2104.02638](http://arxiv.org/abs/2104.02638). It also contains accompanying code and checkpoints for the CrossTransformers
[https://arxiv.org/abs/2007.11498](https://arxiv.org/abs/2007.11498) and FLUTE [https://arxiv.org/abs/2105.07029](https://arxiv.org/abs/2105.07029) follow-up works, which improve performance.

This code is provided here in order to give more details on the implementation
of the data-providing pipeline, our back-bones and models, as well as the
experimental setting.

See below for [user instructions](#user-instructions), including how to:

1.  [install](#installation) the software,
2.  [download and convert](#downloading-and-converting-datasets) the data, and
3.  [train](#training) implemented models.

See this
[introduction notebook](https://github.com/google-research/meta-dataset/blob/main/Intro_to_Metadataset.ipynb)
for a demonstration of how to sample data from the pipeline (episodes or
batches).

In order to run the experiments described in the first version of the arXiv
article, [arxiv.org/abs/1903.03096v1](https://arxiv.org/abs/1903.03096v1),
please use the instructions, code, and configuration files at version
[arxiv_v1](https://github.com/google-research/meta-dataset/tree/arxiv_v1) of
this repository.

We are currently working on updating the instructions, code, and configuration
files to reproduce the results in the second version of the article,
[arxiv.org/abs/1903.03096v2](https://arxiv.org/abs/1903.03096v2). You can follow
the progess in branch
[arxiv_v2_dev](https://github.com/google-research/meta-dataset/tree/arxiv_v2_dev)
of this repository.

This is not an officially supported Google product.

## Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples

_Eleni Triantafillou, Tyler Zhu, Vincent Dumoulin, Pascal Lamblin, Utku Evci,
Kelvin Xu, Ross Goroshin, Carles Gelada, Kevin Swersky, Pierre-Antoine Manzagol,
Hugo Larochelle_

Few-shot classification refers to learning a classifier for new classes given
only a few examples. While a plethora of models have emerged to tackle it, we
find the procedure and datasets that are used to assess their progress lacking.
To address this limitation, we propose Meta-Dataset: a new benchmark for
training and evaluating models that is large-scale, consists of diverse
datasets, and presents more realistic tasks. We experiment with popular
baselines and meta-learners on Meta-Dataset, along with a competitive method
that we propose. We analyze performance as a function of various characteristics
of test tasks and examine the models' ability to leverage diverse training
sources for improving their generalization. We also propose a new set of
baselines for quantifying the benefit of meta-learning in Meta-Dataset. Our
extensive experimentation has uncovered important research challenges and we
hope to inspire work in these directions.

## CrossTransformers: spatially-aware few-shot transfer

_Carl Doersch, Ankush Gupta, Andrew Zisserman_

This is a Transformer-based neural network architecture which can find coarse
spatial correspondence between the query and the support images, and then infer
class membership by computing distances between spatially-corresponding
features.  The paper also introduces SimCLR episodes, which are episodes that
require SimCLR-style instance recognition, and therefore encourage features
which capture more than just the training-set categories.  This algorithm is
SOTA on Meta-Dataset (train-on-ILSVRC) as of NeurIPS 2020.

Configuration files for CrossTransformers with and without SimCLR episodes (CTX
and CTX+SimCLR Eps from the paper) can be found in
`learn/gin/default/crosstransformer*`.  We also have pretrained checkpoints for
these two configurations:
[CTX](https://storage.googleapis.com/dm_crosstransformer/ctx.zip),
and
[CTX+SimCLR Eps](https://storage.googleapis.com/dm_crosstransformer/ctx_simclreps.zip),
as well as
[CTX+SimCLR Eps+BOHB Aug](https://storage.googleapis.com/dm_crosstransformer/ctx_simclreps_bohbaug.zip).
Note that these were retrained from the versions reported in the paper, but
their performance should be on-par.  The network structure is the same for all
three models, and so they can be loaded using either of the CrossTransformer
config files.

## Learning a Universal Template for Few-shot Dataset Generalization (FLUTE)
_Eleni Triantafillou, Hugo Larochelle, Richard Zemel, Vincent Dumoulin

Few-shot Learning with a Universal TEmplate (FLUTE) is a model designed for the 
strong generalization challenge of few-shot learning classes from unseen
datasets. At the time of publication (ICML 2021), it achieved SOTA on
Meta-Dataset (train-on-all). It works by leveraging the training datasets to 
learn a 'universal template' that can be repurposed to solve diverse test tasks, 
by appropriately 'filling in the blanks' of the template each time, with an 
appropriate set of FiLM parameters that are learned with gradient descent in 
each test task.

Configuration files for training FLUTE, as well as the dataset classifier used
in FLUTE's Blender network can be found in `learn/gin/default/flute.gin` and
`learn/gin/default/flute_dataset_classifier.gin`, respectively. Configuration
files for testing different variants of FLUTE can be found in
`learn/gin/best/flute*` The results reported in the paper were obtained with
`learn/gin/best/flute.gin`.

The training script for FLUTE is `train_flute.py`. We also have pre-trained
checkpoints for FLUTE and its Blender network: https://console.cloud.google.com/storage/gresearch/flute

# Leaderboard (in progress)

The tables below were generated by
[this notebook](https://github.com/google-research/meta-dataset/blob/main/Leaderboard.ipynb).

## Adding a new model to the leaderboard

1.  Gather accuracy results and 95% confidence intervals, as well as the number
    of episodes used for the CI (minimum 600).
2.  If you were affected by
    [#54](https://github.com/google-research/meta-dataset/issues/54),
    make sure the evaluation on Traffic Sign is done on shuffled samples. We
    encourage you to re-train your best model (or at least perform validation
    again) as well.
3.  Create an
    [issue](https://github.com/google-research/meta-dataset/issues/new),
    with the name of the model, results, as well as the article to cite or any
    other relevant information to include, and label it "leaderboard".
    Alternatively, submit a PR with an update to the notebook.

<!-- Beginning of content generated by `Leaderboard.ipynb` -->

## Training on ImageNet only

Method                     |Avg rank                   |ILSVRC (test)              |Omniglot                   |Aircraft                   |Birds                      |Textures                   |QuickDraw                  |Fungi                      |VGG Flower                 |Traffic signs              |MSCOCO                     
---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------
k-NN [[1]]                 |14.6                       |41.03±1.01&nbsp;(15)       |37.07±1.15&nbsp;(16)       |46.81±0.89&nbsp;(15)       |50.13±1.00&nbsp;(15.5)     |66.36±0.75&nbsp;(13)       |32.06±1.08&nbsp;(16)       |36.16±1.02&nbsp;(13)       |83.10±0.68&nbsp;(12)       |44.59±1.19&nbsp;(15)       |30.38±0.99&nbsp;(15.5)     
Finetune [[1]]             |10.45                      |45.78±1.10&nbsp;(13)       |60.85±1.58&nbsp;(11.5)     |68.69±1.26&nbsp;(5)        |57.31±1.26&nbsp;(14)       |69.05±0.90&nbsp;(9.5)      |42.60±1.17&nbsp;(13.5)     |38.20±1.02&nbsp;(11)       |85.51±0.68&nbsp;(9)        |66.79±1.31&nbsp;(5)        |34.86±0.97&nbsp;(13)       
MatchingNet [[1]]          |13.55                      |45.00±1.10&nbsp;(13)       |52.27±1.28&nbsp;(14)       |48.97±0.93&nbsp;(13)       |62.21±0.95&nbsp;(12.5)     |64.15±0.85&nbsp;(15)       |42.87±1.09&nbsp;(13.5)     |33.97±1.00&nbsp;(14)       |80.13±0.71&nbsp;(15)       |47.80±1.14&nbsp;(12.5)     |34.99±1.00&nbsp;(13)       
ProtoNet [[1]]             |10.75                      |50.50±1.08&nbsp;(10.5)     |59.98±1.35&nbsp;(11.5)     |53.10±1.00&nbsp;(10.5)     |68.79±1.01&nbsp;(8.5)      |66.56±0.83&nbsp;(13)       |48.96±1.08&nbsp;(11)       |39.71±1.11&nbsp;(9)        |85.27±0.77&nbsp;(9)        |47.12±1.10&nbsp;(14)       |41.00±1.10&nbsp;(10.5)     
fo-MAML [[1]]              |12.25                      |45.51±1.11&nbsp;(13)       |55.55±1.54&nbsp;(13)       |56.24±1.11&nbsp;(8.5)      |63.61±1.06&nbsp;(12.5)     |68.04±0.81&nbsp;(9.5)      |43.96±1.29&nbsp;(13.5)     |32.10±1.10&nbsp;(15)       |81.74±0.83&nbsp;(14)       |50.93±1.51&nbsp;(10.5)     |35.30±1.23&nbsp;(13)       
RelationNet [[1]]          |15.55                      |34.69±1.01&nbsp;(16)       |45.35±1.36&nbsp;(15)       |40.73±0.83&nbsp;(16)       |49.51±1.05&nbsp;(15.5)     |52.97±0.69&nbsp;(16)       |43.30±1.08&nbsp;(13.5)     |30.55±1.04&nbsp;(16)       |68.76±0.83&nbsp;(16)       |33.67±1.05&nbsp;(16)       |29.15±1.01&nbsp;(15.5)     
fo-Proto-MAML [[1]]        |9.25                       |49.53±1.05&nbsp;(10.5)     |63.37±1.33&nbsp;(8.5)      |55.95±0.99&nbsp;(8.5)      |68.66±0.96&nbsp;(8.5)      |66.49±0.83&nbsp;(13)       |51.52±1.00&nbsp;(9.5)      |39.96±1.14&nbsp;(6.5)      |87.15±0.69&nbsp;(6)        |48.83±1.09&nbsp;(12.5)     |43.74±1.12&nbsp;(9)        
ALFA+fo-Proto-MAML [[3]]   |7.1                        |52.80±1.11&nbsp;(8.5)      |61.87±1.51&nbsp;(8.5)      |63.43±1.10&nbsp;(6)        |69.75±1.05&nbsp;(6.5)      |70.78±0.88&nbsp;(7)        |59.17±1.16&nbsp;(5.5)      |41.49±1.17&nbsp;(6.5)      |85.96±0.77&nbsp;(9)        |60.78±1.29&nbsp;(8)        |48.11±1.14&nbsp;(5.5)      
ProtoNet (large) [[4]]     |7.25                       |53.69±1.07&nbsp;(6)        |68.50±1.27&nbsp;(5.5)      |58.04±0.96&nbsp;(7)        |74.07±0.92&nbsp;(4.5)      |68.76±0.77&nbsp;(9.5)      |53.30±1.06&nbsp;(8)        |40.73±1.15&nbsp;(6.5)      |86.96±0.73&nbsp;(6)        |58.11±1.05&nbsp;(9)        |41.70±1.08&nbsp;(10.5)     
CTX [[4]]                  |2.5                        |62.76±0.99&nbsp;(2.5)      |**82.21**±1.00&nbsp;(2)    |**79.49**±0.89&nbsp;(1.5)  |80.63±0.88&nbsp;(3)        |75.57±0.64&nbsp;(4)        |72.68±0.82&nbsp;(2)        |51.58±1.11&nbsp;(2.5)      |**95.34**±0.37&nbsp;(1)    |82.65±0.76&nbsp;(3)        |59.90±1.02&nbsp;(3.5)      
BOHB [[5]]                 |7.85                       |51.92±1.05&nbsp;(8.5)      |67.57±1.21&nbsp;(5.5)      |54.12±0.90&nbsp;(10.5)     |70.69±0.90&nbsp;(6.5)      |68.34±0.76&nbsp;(9.5)      |50.33±1.04&nbsp;(9.5)      |41.38±1.12&nbsp;(6.5)      |87.34±0.59&nbsp;(6)        |51.80±1.04&nbsp;(10.5)     |48.03±0.99&nbsp;(5.5)      
SimpleCNAPS [[14],[7]]     |8.75                       |54.80±1.20&nbsp;(6)        |62.00±1.30&nbsp;(8.5)      |49.20±0.90&nbsp;(13)       |66.50±1.00&nbsp;(10.5)     |71.60±0.70&nbsp;(5.5)      |56.60±1.00&nbsp;(7)        |37.50±1.20&nbsp;(11)       |82.10±0.90&nbsp;(12)       |63.10±1.10&nbsp;(6.5)      |45.80±1.00&nbsp;(7.5)      
TransductiveCNAPS [[14],[8]]|8.6                        |54.10±1.10&nbsp;(6)        |62.90±1.30&nbsp;(8.5)      |48.40±0.90&nbsp;(13)       |67.30±0.90&nbsp;(10.5)     |72.50±0.70&nbsp;(5.5)      |58.00±1.00&nbsp;(5.5)      |37.70±1.10&nbsp;(11)       |82.80±0.80&nbsp;(12)       |61.80±1.10&nbsp;(6.5)      |45.80±1.00&nbsp;(7.5)      
TSA_resnet18 [[12]]        |3.8                        |59.50±1.10&nbsp;(4)        |78.20±1.20&nbsp;(4)        |72.20±1.00&nbsp;(4)        |74.90±0.90&nbsp;(4.5)      |77.30±0.70&nbsp;(3)        |67.60±0.90&nbsp;(4)        |44.70±1.00&nbsp;(4)        |90.90±0.60&nbsp;(4)        |82.50±0.80&nbsp;(3)        |59.00±1.00&nbsp;(3.5)      
TSA_resnet34 [[12]]        |2.25                       |63.73±0.99&nbsp;(2.5)      |**82.58**±1.11&nbsp;(2)    |**80.13**±1.01&nbsp;(1.5)  |83.39±0.80&nbsp;(2)        |79.61±0.68&nbsp;(2)        |71.03±0.84&nbsp;(3)        |51.38±1.17&nbsp;(2.5)      |94.05±0.45&nbsp;(2.5)      |81.71±0.95&nbsp;(3)        |**61.67**±0.95&nbsp;(1.5)  
PMF-DINOSmall [[15]]       |**1.5**                    |**75.51**±0.72&nbsp;(1)    |**82.81**±1.10&nbsp;(2)    |78.38±1.09&nbsp;(3)        |**85.18**±0.77&nbsp;(1)    |**86.95**±0.60&nbsp;(1)    |**74.47**±0.83&nbsp;(1)    |**55.16**±1.09&nbsp;(1)    |94.66±0.48&nbsp;(2.5)      |**90.04**±0.81&nbsp;(1)    |**62.60**±0.96&nbsp;(1.5)  

## Training on all datasets

Method                     |Avg rank                   |ILSVRC (test)              |Omniglot                   |Aircraft                   |Birds                      |Textures                   |QuickDraw                  |Fungi                      |VGG Flower                 |Traffic signs              |MSCOCO                     
---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------
k-NN [[1]]                 |16.85                      |38.55±0.94&nbsp;(16.5)     |74.60±1.08&nbsp;(18)       |64.98±0.82&nbsp;(19)       |66.35±0.92&nbsp;(14.5)     |63.58±0.79&nbsp;(15.5)     |44.88±1.05&nbsp;(19)       |37.12±1.06&nbsp;(15.5)     |83.47±0.61&nbsp;(15.5)     |40.11±1.10&nbsp;(18)       |29.55±0.96&nbsp;(17)       
Finetune [[1]]             |14.1                       |43.08±1.08&nbsp;(14.5)     |71.11±1.37&nbsp;(19)       |72.03±1.07&nbsp;(15.5)     |59.82±1.15&nbsp;(17)       |69.14±0.85&nbsp;(9.5)      |47.05±1.16&nbsp;(18)       |38.16±1.04&nbsp;(15.5)     |85.28±0.69&nbsp;(14)       |66.74±1.23&nbsp;(3)        |35.17±1.08&nbsp;(15)       
MatchingNet [[1]]          |16.4                       |36.08±1.00&nbsp;(18)       |78.25±1.01&nbsp;(16.5)     |69.17±0.96&nbsp;(17.5)     |56.40±1.00&nbsp;(18)       |61.80±0.74&nbsp;(17)       |60.81±1.03&nbsp;(15.5)     |33.70±1.04&nbsp;(18)       |81.90±0.72&nbsp;(17)       |55.57±1.08&nbsp;(9.5)      |28.79±0.96&nbsp;(17)       
ProtoNet [[1]]             |14.5                       |44.50±1.05&nbsp;(14.5)     |79.56±1.12&nbsp;(16.5)     |71.14±0.86&nbsp;(15.5)     |67.01±1.02&nbsp;(14.5)     |65.18±0.84&nbsp;(13.5)     |64.88±0.89&nbsp;(14)       |40.26±1.13&nbsp;(14)       |86.85±0.71&nbsp;(13)       |46.48±1.00&nbsp;(16)       |39.87±1.06&nbsp;(13.5)     
fo-MAML [[1]]              |16.25                      |37.83±1.01&nbsp;(16.5)     |83.92±0.95&nbsp;(14.5)     |76.41±0.69&nbsp;(13)       |62.43±1.08&nbsp;(16)       |64.16±0.83&nbsp;(15.5)     |59.73±1.10&nbsp;(17)       |33.54±1.11&nbsp;(18)       |79.94±0.84&nbsp;(18)       |42.91±1.31&nbsp;(17)       |29.37±1.08&nbsp;(17)       
RelationNet [[1]]          |17.8                       |30.89±0.93&nbsp;(19)       |86.57±0.79&nbsp;(13)       |69.71±0.83&nbsp;(17.5)     |54.14±0.99&nbsp;(19)       |56.56±0.73&nbsp;(19)       |61.75±0.97&nbsp;(15.5)     |32.56±1.08&nbsp;(18)       |76.08±0.76&nbsp;(19)       |37.48±0.93&nbsp;(19)       |27.41±0.89&nbsp;(19)       
fo-Proto-MAML [[1]]        |12.6                       |46.52±1.05&nbsp;(13)       |82.69±0.97&nbsp;(14.5)     |75.23±0.76&nbsp;(14)       |69.88±1.02&nbsp;(12.5)     |68.25±0.81&nbsp;(11.5)     |66.84±0.94&nbsp;(13)       |41.99±1.17&nbsp;(13)       |88.72±0.67&nbsp;(11)       |52.42±1.08&nbsp;(12.5)     |41.74±1.13&nbsp;(11)       
CNAPs [[2]]                |11.2                       |50.80±1.10&nbsp;(11.5)     |91.70±0.50&nbsp;(8.5)      |83.70±0.60&nbsp;(8.5)      |73.60±0.90&nbsp;(11)       |59.50±0.70&nbsp;(18)       |74.70±0.80&nbsp;(12)       |50.20±1.10&nbsp;(8.5)      |88.90±0.50&nbsp;(11)       |56.50±1.10&nbsp;(9.5)      |39.40±1.00&nbsp;(13.5)     
SUR [[6]]                  |8.45                       |56.10±1.10&nbsp;(8)        |93.10±0.50&nbsp;(5.5)      |84.60±0.70&nbsp;(6.5)      |70.60±1.00&nbsp;(12.5)     |71.00±0.80&nbsp;(7.5)      |81.30±0.60&nbsp;(4)        |64.20±1.10&nbsp;(4.5)      |82.80±0.80&nbsp;(15.5)     |53.40±1.00&nbsp;(12.5)     |50.10±1.00&nbsp;(8)        
SUR-pnf [[6]]              |9.2                        |56.00±1.10&nbsp;(8)        |90.00±0.60&nbsp;(11.5)     |79.70±0.80&nbsp;(11.5)     |75.90±0.90&nbsp;(8.5)      |72.50±0.70&nbsp;(5.5)      |76.70±0.70&nbsp;(9.5)      |49.80±1.10&nbsp;(8.5)      |90.00±0.60&nbsp;(8.5)      |52.20±0.80&nbsp;(12.5)     |50.20±1.10&nbsp;(8)        
SimpleCNAPS [[14],[7]]     |8.4                        |56.50±1.10&nbsp;(8)        |91.90±0.60&nbsp;(8.5)      |83.80±0.60&nbsp;(8.5)      |76.10±0.90&nbsp;(8.5)      |70.00±0.80&nbsp;(9.5)      |78.30±0.70&nbsp;(7.5)      |49.10±1.20&nbsp;(8.5)      |91.30±0.60&nbsp;(7)        |59.20±1.00&nbsp;(7)        |42.40±1.10&nbsp;(11)       
TransductiveCNAPS [[14],[8]]|6.95                       |57.90±1.10&nbsp;(3.5)      |94.30±0.40&nbsp;(3.5)      |84.70±0.50&nbsp;(6.5)      |78.80±0.70&nbsp;(4.5)      |66.20±0.80&nbsp;(13.5)     |77.90±0.60&nbsp;(7.5)      |48.90±1.20&nbsp;(8.5)      |92.30±0.40&nbsp;(4)        |59.70±1.10&nbsp;(7)        |42.50±1.10&nbsp;(11)       
URT [[9]]                  |6.85                       |55.70±1.00&nbsp;(8)        |94.40±0.40&nbsp;(3.5)      |85.80±0.60&nbsp;(5)        |76.30±0.80&nbsp;(8.5)      |71.80±0.70&nbsp;(5.5)      |**82.50**±0.60&nbsp;(2)    |63.50±1.00&nbsp;(4.5)      |88.20±0.60&nbsp;(11)       |51.10±1.10&nbsp;(15)       |52.20±1.10&nbsp;(5.5)      
URT-pf [[9]]               |8.55                       |55.50±1.10&nbsp;(8)        |90.20±0.60&nbsp;(11.5)     |79.80±0.70&nbsp;(11.5)     |77.50±0.80&nbsp;(6)        |73.50±0.70&nbsp;(4)        |75.80±0.70&nbsp;(11)       |48.10±0.90&nbsp;(11.5)     |91.90±0.50&nbsp;(4)        |52.00±1.40&nbsp;(12.5)     |52.10±1.00&nbsp;(5.5)      
FLUTE [[10]]               |6.75                       |51.80±1.10&nbsp;(11.5)     |93.20±0.50&nbsp;(5.5)      |87.20±0.50&nbsp;(4)        |79.20±0.80&nbsp;(4.5)      |68.80±0.80&nbsp;(11.5)     |79.50±0.70&nbsp;(5.5)      |58.10±1.10&nbsp;(6)        |91.60±0.60&nbsp;(4)        |58.40±1.10&nbsp;(7)        |50.00±1.00&nbsp;(8)        
URL [[11]]                 |2.95                       |57.51±1.08&nbsp;(3.5)      |**94.51**±0.41&nbsp;(1.5)  |88.59±0.46&nbsp;(3)        |80.54±0.69&nbsp;(2.5)      |76.17±0.67&nbsp;(2.5)      |**81.94**±0.56&nbsp;(2)    |68.75±0.95&nbsp;(2.5)      |92.11±0.48&nbsp;(4)        |63.34±1.19&nbsp;(4.5)      |54.03±0.96&nbsp;(3.5)      
TSA [[12]]                 |2.4                        |57.35±1.05&nbsp;(3.5)      |**94.96**±0.38&nbsp;(1.5)  |**89.33**±0.44&nbsp;(1.5)  |81.42±0.74&nbsp;(2.5)      |76.74±0.72&nbsp;(2.5)      |**82.01**±0.57&nbsp;(2)    |67.40±0.99&nbsp;(2.5)      |92.18±0.52&nbsp;(4)        |83.55±0.90&nbsp;(2)        |55.75±1.06&nbsp;(2)        
TriM [[13]]                |7.55                       |58.60±1.00&nbsp;(3.5)      |92.00±0.60&nbsp;(8.5)      |82.80±0.70&nbsp;(10)       |75.30±0.80&nbsp;(8.5)      |71.20±0.80&nbsp;(7.5)      |77.30±0.70&nbsp;(9.5)      |48.50±1.00&nbsp;(11.5)     |90.50±0.50&nbsp;(8.5)      |63.00±1.00&nbsp;(4.5)      |52.80±1.10&nbsp;(3.5)      
PMF-DINOSmall [[15]]       |**2.25**                   |**73.52**±0.80&nbsp;(1)    |92.17±0.57&nbsp;(8.5)      |**89.49**±0.52&nbsp;(1.5)  |**91.04**±0.37&nbsp;(1)    |**85.73**±0.62&nbsp;(1)    |79.43±0.67&nbsp;(5.5)      |**74.99**±0.94&nbsp;(1)    |**95.30**±0.44&nbsp;(1)    |**89.85**±0.76&nbsp;(1)    |**59.69**±1.02&nbsp;(1)    

## References

[1]: #1-triantafillou-et-al-2020
[2]: #2-requeima-et-al-2019
[3]: #3-baik-et-al-2020
[4]: #4-doersch-et-al-2020
[5]: #5-saikia-et-al-2020
[6]: #6-dvornik-et-al-2020
[7]: #7-bateni-et-al-2020
[8]: #8-bateni-et-al-2022a
[9]: #9-liu-et-al-2021a
[10]: #10-triantafillou-et-al-2021
[11]: #11-li-et-al-2021a
[12]: #12-li-et-al-2021b
[13]: #13-liu-et-al-2021b
[14]: #14-bateni-et-al-2022b
[15]: #15-hu-et-al-2022

###### \[1\] Triantafillou et al. (2020)

Eleni Triantafillou, Tyler Zhu, Vincent Dumoulin, Pascal Lamblin, Utku Evci, Kelvin Xu, Ross Goroshin, Carles Gelada, Kevin Swersky, Pierre-Antoine Manzagol, Hugo Larochelle; [_Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples_](https://arxiv.org/abs/1903.03096); ICLR 2020.


###### \[2\] Requeima et al. (2019)

James Requeima, Jonathan Gordon, John Bronskill, Sebastian Nowozin, Richard E. Turner; [_Fast and Flexible Multi-Task Classification Using Conditional Neural Adaptive Processes_](https://arxiv.org/abs/1906.07697); NeurIPS 2019.


###### \[3\] Baik et al. (2020)

Sungyong Baik, Myungsub Choi, Janghoon Choi, Heewon Kim, Kyoung Mu Lee; [_Meta-Learning with Adaptive Hyperparameters_](https://papers.nips.cc/paper/2020/hash/ee89223a2b625b5152132ed77abbcc79-Abstract.html); NeurIPS 2020.


###### \[4\] Doersch et al. (2020)

Carl Doersch, Ankush Gupta, Andrew Zisserman; [_CrossTransformers: spatially-aware few-shot transfer_](https://arxiv.org/abs/2007.11498); NeurIPS 2020.


###### \[5\] Saikia et al. (2020)

Tonmoy Saikia, Thomas Brox, Cordelia Schmid; [_Optimized Generic Feature Learning for Few-shot Classification across Domains_](https://arxiv.org/abs/2001.07926); arXiv 2020.


###### \[6\] Dvornik et al. (2020)

Nikita Dvornik, Cordelia Schmid, Julien Mairal; [_Selecting Relevant Features from a Multi-domain Representation for Few-shot Classification_](https://arxiv.org/abs/2003.09338); ECCV 2020.


###### \[7\] Bateni et al. (2020)

Peyman Bateni, Raghav Goyal, Vaden Masrani, Frank Wood, Leonid Sigal; [_Improved Few-Shot Visual Classification_](https://openaccess.thecvf.com/content_CVPR_2020/html/Bateni_Improved_Few-Shot_Visual_Classification_CVPR_2020_paper.html); CVPR 2020.


###### \[8\] Bateni et al. (2022a)

Peyman Bateni, Jarred Barber, Jan-Willem van de Meent, Frank Wood; [_Enhancing Few-Shot Image Classification with Unlabelled Examples_](https://openaccess.thecvf.com/content/WACV2022/html/Bateni_Enhancing_Few-Shot_Image_Classification_With_Unlabelled_Examples_WACV_2022_paper.html); WACV 2022.


###### \[9\] Liu et al. (2021a)

Lu Liu, William Hamilton, Guodong Long, Jing Jiang, Hugo Larochelle; [_Universal Representation Transformer Layer for Few-Shot Image Classification_](https://arxiv.org/abs/2006.11702); ICLR 2021.


###### \[10\] Triantafillou et al. (2021)

Eleni Triantafillou, Hugo Larochelle, Richard Zemel, Vincent Dumoulin; [_Learning a Universal Template for Few-shot Dataset Generalization_](https://arxiv.org/abs/2105.07029); ICML 2021.


###### \[11\] Li et al. (2021a)

Wei-Hong Li, Xialei Liu, Hakan Bilen; [_Universal Representation Learning from Multiple Domains for Few-shot Classification_](https://arxiv.org/pdf/2103.13841.pdf); ICCV 2021.


###### \[12\] Li et al. (2021b)

Wei-Hong Li, Xialei Liu, Hakan Bilen; [_Cross-domain Few-shot Learning with Task-specific Adapters_](https://arxiv.org/pdf/2107.00358.pdf); arXiv 2021.


###### \[13\] Liu et al. (2021b)

Yanbin Liu, Juho Lee, Linchao Zhu, Ling Chen, Humphrey Shi, Yi Yang; [_A Multi-Mode Modulator for Multi-Domain Few-Shot Classification_](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_A_Multi-Mode_Modulator_for_Multi-Domain_Few-Shot_Classification_ICCV_2021_paper.pdf); ICCV 2021.


###### \[14\] Bateni et al. (2022b)

Bateni Peyman, Jarred Barber, Raghav Goyal, Vaden Masrani, Jan-Willem van de Meent, Leonid Sigal, and Frank Wood.; [_Beyond Simple Meta-Learning: Multi-Purpose Models for Multi-Domain, Active and Continual Few-Shot Learning._](https://arxiv.org/abs/2201.05151); arXiv 2022.


###### \[15\] Hu et al. (2022)

Shell Xu Hu, Da Li, Jan Stühmer, Minyoung Kim and Timothy Hospedales.; [_Pushing the Limits of Simple Pipelines for Few-Shot Learning: External Data and Fine-Tuning Make a Difference._](https://arxiv.org/abs/2204.07305); CVPR 2022.


<!-- End of content generated by `Leaderboard.ipynb` -->

# User instructions

## Installation

Meta-Dataset is generally compatible with Python 2 and Python 3, but some parts
of the code may require Python 3. The code works with TensorFlow 2, although it
makes extensive use of `tf.compat.v1` internally.

-   We recommend you follow
    [these instructions](https://www.tensorflow.org/install/pip) to install
    TensorFlow.
-   A list of packages to install is available in `requirements.txt`, you can
    install them using `pip`.
-   Clone the `meta-dataset` repository. Most command lines start with `python
    -m meta_dataset.<something>`, and should be typed from within that clone
    (where a `meta_dataset` Python module should be visible).
-   To reproduce the CrossTransformers training, you will need data augmentation
    code from [simclr](https://github.com/google-research/simclr), which is
    autimatically downloaded by `setup.py`.

## Downloading and converting datasets

Meta-Dataset uses several established datasets, that are available from
different sources. You can find below a summary of these datasets, as well as
instructions to download them and convert them into a common format.

In addition to the datasets below, the FLUTE paper reported results on 3 extra
datasets, following recent work. You can find instructions for downloading and 
converting those 3 additional datasets (MNIST, CIFAR-10 and CIFAR-100) in the
CNAPs [repo](https://github.com/cambridge-mlg/cnaps).

For brevity of the command line examples, we assume the following environment
variables are defined:

-   `$DATASRC`: root of where the original data is downloaded and potentially
    extracted from compressed files. This directory does not need to be
    available after the data conversion is done.
-   `$SPLITS`: directory where `*_splits.json` files will be created, one per
    dataset. For instance, `$SPLITS/fungi_splits.json` contains information
    about which classes are part of the meta-training, meta-validation, and
    meta-test set. These files are only used during the dataset conversion
    phase, but can help troubleshooting later. To re-use the
    [canonical splits](https://github.com/google-research/meta-dataset/tree/main/meta_dataset/dataset_conversion/splits)
    instead of re-generating them, you can make it point to
    `meta_dataset/dataset_conversion` in your checkout.
-   `$RECORDS`: root directory that will contain the converted datasets (one per
    sub-directory). This directory needs to be available during training and
    evaluation.

### Dataset summary

Dataset (other names)                                                                                                                        | Number of classes (train/valid/test)    | Size on disk                 | Conversion time
-------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- | ---------------------------- | ---------------
ilsvrc\_2012 (ImageNet, ILSVRC) \[[instructions](doc/dataset_conversion.md#ilsvrc_2012)\]                                                  | 1000 (712/158/130, hierarchical)        | \~140 GiB                    | 5 to 13 hours
omniglot \[[instructions](doc/dataset_conversion.md#omniglot)\]                                                                            | 1623 (883/81/659, by alphabet: 25/5/20) | \~60 MiB                     | few seconds
aircraft (FGVC-Aircraft) \[[instructions](doc/dataset_conversion.md#aircraft)\]                                                            | 100 (70/15/15)                          | \~470 MiB (2.6 GiB download) | 5 to 10 minutes
cu\_birds (Birds, CUB-200-2011) \[[instructions](doc/dataset_conversion.md#cu_birds)\]                                                     | 200 (140/30/30)                         | \~1.1 GiB                    | \~1 minute
dtd (Describable Textures, DTD) \[[instructions](doc/dataset_conversion.md#dtd)\]                                                          | 47 (33/7/7)                             | \~600 MiB                    | few seconds
quickdraw (Quick, Draw!) \[[instructions](doc/dataset_conversion.md#quickdraw)\]                                                           | 345 (241/52/52)                         | \~50 GiB                     | 3 to 4 hours
fungi (FGVCx Fungi) \[[instructions](doc/dataset_conversion.md#fungi)\]                                                                    | 1394 (994/200/200)                      | \~13 GiB                     | 5 to 15 minutes
vgg\_flower (VGG Flower) \[[instructions](doc/dataset_conversion.md#vgg_flower)\]                                                          | 102 (71/15/16)                          | \~330 MiB                    | \~1 minute
traffic\_sign (Traffic Signs, German Traffic Sign Recognition Benchmark, GTSRB) \[[instructions](doc/dataset_conversion.md#traffic_sign)\] | 43 (0/0/43, test only)                  | \~50 MiB (263 MiB download)  | \~1 minute
mscoco (Common Objects in Context, COCO) \[[instructions](doc/dataset_conversion.md#mscoco)\]                                              | 80 (0/40/40, validation and test only)  | \~5.3 GiB (18 GiB download)  | 4 hours
*Total (All datasets)*                                                                                                                       | *4934 (3144/598/1192)*                  | *\~210 GiB*                  | *12 to 24 hours*

### Meta-Dataset-v2
In order to make the combined benchmark (VTAB+MD) compatible with each other, Meta-Dataset-v2 makes some changes on the existing pipelines. When converting the ImageNet dataset please use `ilsvrc\_2012\_v2` \([instructions](doc/dataset_conversion.md#ilsvrc_2012)\) in order to make it a training only dataset. Also,`VGG Flowers` is reserved as a VTAB task in VTAB+MD, so there is no need to convert it. For more details check the [paper](http://arxiv.org/abs/2104.02638).

In order to run existing meta-learners with the updated training, validation and test classes you can refer to the `learn/gin/setups/imagenet_v2.gin` `learn/gin/setups/all_v2.gin`. These files are meant to be drop in replacements for `learn/gin/setups/imagenet.gin` and `learn/gin/setups/all.gin` files respectively.
## Training

Experiments are defined via [gin](google/gin-config) configuration files, that
are under `meta_dataset/learn/gin/`:

-   `setups/` contain generic setups for classes of experiment, for instance
    which datasets to use (`imagenet` or `all`), parameters for sampling the
    number of ways and shots of episodes.
-   `models/` define settings for different meta-learning algorithms (baselines,
    prototypical networks, MAML...)
-   `default/` contains files that each correspond to one experiment, mostly
    defining a setup and a model, with default values for training
    hyperparameters.
-   `best/` contains files with values for training hyperparameters that
    achieved the best performance during hyperparameter search.

There are three main architectures, also called "backbones" (or "embedding
networks"): `four_layer_convnet` (sometimes `convnet` for short), `resnet`, and
`wide_resnet`. These architectures can be used by all baselines and episodic
models. Another backbone, `relationnet_convnet` (similar to `four_layer_convnet`
but without pooling on the last layer), is only used by RelationNet (and
baseline, for pre-training purposes).  CrossTransformers use a larger backbone
`resnet34`, which is similar to `resnet` but with more layers.

### Reproducing results

See [Reproducing best results](doc/reproducing_best_results.md) for
instructions to launch training experiments with the values of hyperparameters
that were selected in the paper. The hyperparameters (including the backbone,
whether to train from scratch or from pre-trained weights, and the number of
training updates) were selected using only the validation classes of the ILSVRC
2012 dataset for all experiments. Even when training on "all" datasets, the
validation classes of the other datasets were not used.

### Adding `task_adaptation` code to the path
In order to use `data.read_episodes` module you need to get task_adaptation
code. You can do that by running following code.
```bash
git clone https://github.com/google-research/task_adaptation.git
export PYTHONPATH=$PYTHONPATH:$PWD
```
