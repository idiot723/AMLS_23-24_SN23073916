from A.taskA import import_A, KNN
from B.taskB import import_B, newimport_B, decision_tree, random_forest,CNN_B

def main():
    x_train,y_train,x_test,y_test = import_A()
    x2_train,y2_train,x2_test,y2_test = import_B()
    train_loader2,train_eval2,test_loader2 = newimport_B()
    print("\nTASK A\nFor KNN mathod:\n")
    KNN(x_train,y_train,x_test,y_test)
    print("\nTASK B\nFor decision tree mathod:\n")
    #train and save the models
    #decision_tree_model(x_train,y_train)
    #random_forest_model(x_train,y_train)
    #CNN_B_model(train_loader)
    #test with pre-trained models
    decision_tree('dt.pt',x2_test,y2_test)
    print("\nFor random forest mathod:\n")
    random_forest('rf.pt',x2_test,y2_test)
    print("\nFor CNN mathod:\n")
    CNN_B('cnn.pt',test_loader2)
        

main()
