# Caltech101
Just my take with Transfer Learning on [Caltech101](https://en.wikipedia.org/wiki/Caltech_101) dataset using ResNet-50 Convolutional Neural Network.

Simple web application:
- [Recognition-App](https://recognition-caltech.herokuapp.com/)

Saved model can be used locally for predicting as:  
- UI using [Gradio](https://www.gradio.app/)  
  
`$ python predict.py --usage gradio`  
  <img src="https://github.com/TomislavZupanovic/Caltech101/blob/master/notebooks/Gradio.jpg" width="700" height="350">  
- Random image from test dataset  
  
`$ python predict.py --usage random`  
<img src="https://github.com/TomislavZupanovic/Caltech101/blob/master/notebooks/Random_image.jpeg" width="680" height="280">  
