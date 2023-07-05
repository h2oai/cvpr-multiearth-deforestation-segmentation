# CVPR 2023 Workshop on Multimodal Learning for Earth and Environment
https://sites.google.com/view/rainforest-challenge/multiearth-2023

# Deforestation segmentation

![Pipeline](figures_for_paper/earth-unetV3.png)

### Data preparation
Download all provided datasets

```
./azcopy cp "https://rainforestchallenge.blob.core.windows.net/multiearth2023-dataset-final/?sv=2021-10-04&st=2023-04-11T13%3A09%3A30Z&se=2023-06-12T13%3A09%3A00Z&sr=c&sp=racwdxltf&sig=33jltVx0OUwTPNS4PK%2F%2FZBntwjJaqAHdocj%2BpHRDp88%3D" data/multiearth2023-dataset-final --recursive
```
Install the required dependencies
`pip install -r requirements.txt`

Extract each relevant indvidual bands from the *.nc files for the flexible data loader.

```
python src/extract_sat_images.py
```

### Modeling
Train the final deforestation segmentation models

```
cd src
./forest.sh
```

Create the binary deforestation masks with ForestSubmission.ipynb for the test submission then upload the zipped images.

Each experiment takes 2-3 hours using 3 NVIDIA Quadro RTX 6000 cards. Our best single model achieves 90.4 Pixel Accuracy, 0.871 F1 and 0.792 IoU on the unseen test area. Training all the models with small configuration changes and blending raw predictions before binarization gives small performance boost up to 90.8 Pixel Accuracy, 0.876 F1 and 0.799 IoU.


# Acknowledgements
We would like to thank for Philipp, Pascal, Christof for sharing their solution https://github.com/pascal-pfeiffer/kaggle-rsna-2022-5th-place
Their easy to configure framework was useful for training the binary segmentation models.

