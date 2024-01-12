The whole project includes one main.py file, a taskA.py file for the task A, and  a taskB.py file for task B.  In the folder of task B 3 pre-trained model file were put inside. The random forest model is trained with n_estimators=5 due to the github file limit, and the better accuracy is get when n_estimators= 100.

Directly running the main.py file can get the accuracy result of each method by using the pre-trained model. In the main file, there are some commented(#) code are used to train and save the model, and also plot the learning curve of each model.

The code needs packages:
medmnist
numpy
matplotlib
sklearn
tqdm
torch
