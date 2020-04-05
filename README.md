# ProgressiveInGAN
## Paper: InGAN: Capturing and Remapping the "DNA" of a Natural Image
http://www.wisdom.weizmann.ac.il/~vision/ingan/

### Description
In this project I used InGan idea and used Progressive GAN in order to do great inhancment in the performance.

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pytorch](https://pytorch.org/)
- [websocket](https://websockets.readthedocs.io/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

We recommend to install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 


### Run

In order to use the code you need to run the train.py file , marked line in there can be updated as the epochs in each stage. 
for example : 
trainer.train([3, 8, 12, 16], [0.5, 0.5, 0.5], [16, 16, 16, 16])
means we have 4 stage with 3 epochs at first stage 8 epochs at stage 2 and 16 at stage 4, second argument is the percent of stablization
at each stage(first stage is not included so its number of stages -1) , last argument should be as the number of stages.
when changing the number of stages you should change in config "stage_size" paramter, to the number of stages you are looking for -1(5 stages should be 4).


image name and output folder is determined at the config file also at the of it.
in case of using a slackbot add true as a last argument to trainer.train and you will get update, make sure you change at slack_weizmann.py
the arguments of SLACK_BOT_TOKEN  and SLACK_OAUTH_TOKEN  to the relevant token which can be achieved from the slack api.

### Data

Any image(Such as those which are inside the git) can be added, you need one image to do all the learning.


**Target**
- `Generation`: Generate a new image using the given image.


