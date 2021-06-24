# ADCC

##### Original Implementation of the paper Revisiting The Evaluation of Class Activation Mapping for Explainability: A Novel Metric and Experimental Analysis, appeared at CVPRW'2021 Responsible Computer Vision as oral and poster presentation.

ADCC is a more solid and unbiased benchmark for evaluating CAMs for explainability purposes.
It considers more than one aspect of how is the resulting saliency map (read the [paper](https://openaccess.thecvf.com/content/CVPR2021W/RCV/html/Poppi_Revisiting_the_Evaluation_of_Class_Activation_Mapping_for_Explainability_A_CVPRW_2021_paper.html) for a more detailed overview)

### How it works
Given:
1. An input image
2. A saliency map
3. An explanation map
4. A CNN
5. A saliency map extractor (a callable returning an upsampled saliency map)

it computes the Average Drop, the Coherency and the Complexity, to return the final ADCC score.

### Steps

##### Run the example
Using default parameters
```shell
main.py
```

Using custom parameters
```shell
main.py --image [path-to-input-image,str] --model [name-of-the-CNN,str]
```

##### Use it as a module
The ADCC module provides all the computation needed to return the final score, given the 5 inputs previously mentioned.

##### What it returns
The ADCC module simply returns the ADCC score in [0,1] range

##### Other infos
This repo is implemented in PyTorch

##### If you use this repo in your publication please cite

```
@inproceedings{poppi2021revisiting,
  title={Revisiting The Evaluation of Class Activation Mapping for Explainability: A Novel Metric and Experimental Analysis},
  author={Poppi, Samuele and Cornia, Marcella and Baraldi, Lorenzo and Cucchiara, Rita},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year={2021}
}
```

For any info contact [samuele.poppi@unimore.it](mailto:samuele.poppi@unimore.it?subject=ADCC%20GitHub%20Repo)

