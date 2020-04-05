# ProgressiveInGAN
ProgressiveInGAN

In order to use the code you need to run the train.py file , marked line in there can be updated as the epochs in each stage. 
for example : 
trainer.train([3, 8, 12, 16], [0.5, 0.5, 0.5], [16, 16, 16, 16])
means we have 4 stage with 3 epochs at first stage 8 epochs at stage 2 and 16 at stage 4, second argument is the percent of stablization
at each stage(first stage is not included so its number of stages -1) , last argument should be as the number of stages.
when changing the number of stages you should change in config "stage_size" paramter, to the number of stages you are looking for -1(5 stages should be 4).


image name and output folder is determined at the config file also at the of it.
in case of using a slackbot add true as a last argument to trainer.train and you will get update, make sure you change at slack_weizmann.py
the arguments of SLACK_BOT_TOKEN  and SLACK_OAUTH_TOKEN  to the relevant token which can be achieved from the slack api.

Enjoy!
