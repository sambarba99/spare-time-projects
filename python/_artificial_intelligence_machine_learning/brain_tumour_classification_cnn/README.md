## Classification of brain tumour MRIs with a PyTorch Convolutional Neural Network (CNN)

<p align="center">
	<img src="images/data_samples_raw.png"/>
	<br/>
	<img src="images/data_samples_augmented.png"/>
</p>

Test set performance:

<p align="center">
	<img src="images/test_confusion_matrix.png"/>
</p>

Saliency maps for some sample images:

<p align="center">
	<img src="images/saliency_map1.png"/>
	<br/>
	<img src="images/saliency_map2.png"/>
	<br/>
	<img src="images/saliency_map3.png"/>
</p>

Learned filters of each conv layer, and corresponding feature maps of a sample image:

<p align="center">
	<img src="images/conv1_filters.png"/>
	<br/>
	<img src="images/conv2_filters.png"/>
	<br/>
	<img src="images/conv3_filters.png"/>
	<br/>
	<img src="images/conv1_feature_map.png"/>
	<br/>
	<img src="images/conv2_feature_map.png"/>
	<br/>
	<img src="images/conv3_feature_map.png"/>
</p>

Model architecture:

<p align="center">
	<img src="images/model_architecture.png"/>
</p>

Source:
- [Brain Tumour MRIs](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle dataset)
