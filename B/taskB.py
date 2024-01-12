import medmnist
import numpy as np
from medmnist import Evaluator,PathMNIST
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader

accus=[]

def import_B():
    data = np.load('./dataset/pathmnist.npz') #load data

    x_train_3d = data['train_images']
    y_train = data['train_labels']
    x_test_3d = data['test_images']
    y_test = data['test_labels']

    x_train = x_train_3d.reshape((x_train_3d.shape[0], -1)) #reduce dimention from 3d to 2d
    x_test = x_test_3d.reshape((x_test_3d.shape[0], -1)) #reduce dimention from 3d to 2d

    return x_train,y_train,x_test,y_test

def newimport_B():

    #set pre-transform pipeline
    transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize(mean=[.5], std=[.5])
    ])

    #get dataset
    train_data = PathMNIST(split='train', transform=transform, download=True)
    test_data = PathMNIST(split='test', transform=transform, download=True)

    #load dataset
    train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)
    train_eval = DataLoader(dataset=train_data, batch_size=8, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=8, shuffle=False)

    return train_loader,train_eval,test_loader

def decision_tree_model(x_train,y_train):
    #train model
    dt_model = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=10, min_samples_split=50)
    dt_model.fit(x_train,y_train)
    #save model
    torch.save(dt_model,'dt.pt')
    print("save success")

def dt_learning_curve(x_train,y_train,x_test,y_test):
    test = []
    for i in range(20):
       dt_model = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=i+1,min_samples_split=50)
       dt_model.fit(x_train,y_train)
       y_pred =  dt_model.predict(x_test)
       score = accuracy_score(y_test, y_pred)
       test.append(score)
    plt.plot(range(1, 21), test)
    plt.xlabel('max depth')
    plt.ylabel('accuracy')
    plt.show()

def decision_tree(path,x_test,y_test):
    #test model using the pre-saved model
    dtmodel = torch.load(path)
    y_pred =  dtmodel.predict(x_test)
    
    print('Decision Tree test Accuracy: ', accuracy_score(y_test, y_pred))


def random_forest_model(x_train,y_train):
    #train model 
    rf_model=RandomForestClassifier(n_estimators=300)
    rf_model.fit(x_train,y_train)
    #save model
    torch.save(rf_model,'rf.pt')
    print("save success")

def random_forest(path,x_test,y_test):
    #test model using the pre-saved model
    rfmodel = torch.load(path)
    y_pred=rfmodel.predict(x_test)
    
    print("Random Forest test Accuracy: ", accuracy_score(y_test, y_pred))

class ConvNet(nn.Module):
    #define CNN model
       def __init__(self):
           super(ConvNet, self).__init__()

           self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

           self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

           self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
           self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

           self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 9))

       def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def test(model,test_loader):
        #test model
        model.eval()
        y_score = torch.tensor([])

        with torch.no_grad():
            for x, y in test_loader:
           
                pred_cnn = model(x)
                pred_cnn = pred_cnn.softmax(dim=-1)

                y_score = torch.cat((y_score, pred_cnn), 0)

            y_score = y_score.detach().numpy()
        
            evaluator = Evaluator('pathmnist', 'test')
            metrics = evaluator.evaluate(y_score)
            accu = metrics[1]
            print('CNN test Accuracy:%.3f' % (accu))
        
        return accu

def CNN_B_model(train_loader,test_loader):

    model = ConvNet()
    epoches = 20
    optimizer = optim.SGD(model.parameters(), lr= 0.001, momentum=0.9)
    loss_func = nn.CrossEntropyLoss() 
    accus=[]
    
    for epoch in range(epoches):
        #train model
        model.train()
        for x, y in tqdm(train_loader):
        
            optimizer.zero_grad()
            pred_cnn = model(x)
       
            y = y.squeeze().long()
            loss = loss_func(pred_cnn, y)
        
            loss.backward()
            optimizer.step()
        #test model
        accu = test(model,test_loader)
        accus.append(accu)

    #save model
    torch.save(model.state_dict(),'cnn.pt')
    print("save success")
    #plot learning curve
    plt.plot(range(epoches), accus)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()
    
    return accus
  

def CNN_B(path,test_loader):
    #test model using the pre-saved model
    model = ConvNet()
    model.load_state_dict(torch.load(path))
    
    test(model,test_loader)
