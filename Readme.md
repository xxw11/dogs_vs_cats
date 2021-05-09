# Readme



```python
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,10,kernel_size = 5)
        self.conv2 = torch.nn.Conv2d(10,20,kernel_size = 5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(56180,1000)
        self.fc2 = torch.nn.Linear(1000,2)
        
    def forward(self,x):
        batch_size = x.size(0)
        x = x.view(batch_size,3,224,224)
        x = self.pooling(F.relu(self.conv1(x)))
        x = self.pooling(F.relu(self.conv2(x)))
        x = x.view(batch_size,-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

卷积-池化-卷积-池化-全连接-全连接

```
[1,  100] loss: 0.687
[1,  200] loss: 0.685
Accuracy on test set: 56 %
[2,  100] loss: 0.681
[2,  200] loss: 0.678
Accuracy on test set: 57 %
[3,  100] loss: 0.675
[3,  200] loss: 0.668
Accuracy on test set: 58 %
[4,  100] loss: 0.667
[4,  200] loss: 0.660
Accuracy on test set: 60 %
[5,  100] loss: 0.658
[5,  200] loss: 0.655
Accuracy on test set: 61 %
[6,  100] loss: 0.647
[6,  200] loss: 0.645
Accuracy on test set: 63 %
[7,  100] loss: 0.637
[7,  200] loss: 0.633
Accuracy on test set: 63 %
[8,  100] loss: 0.631
[8,  200] loss: 0.623
Accuracy on test set: 64 %
[9,  100] loss: 0.623
[9,  200] loss: 0.610
Accuracy on test set: 65 %
[10,  100] loss: 0.609
[10,  200] loss: 0.610
Accuracy on test set: 66 %
[11,  100] loss: 0.610
[11,  200] loss: 0.612
Accuracy on test set: 66 %
[12,  100] loss: 0.601
[12,  200] loss: 0.594
Accuracy on test set: 67 %
[13,  100] loss: 0.600
[13,  200] loss: 0.595
Accuracy on test set: 67 %
[14,  100] loss: 0.595
[14,  200] loss: 0.593
Accuracy on test set: 68 %
[15,  100] loss: 0.587
[15,  200] loss: 0.591
Accuracy on test set: 69 %
[16,  100] loss: 0.586
[16,  200] loss: 0.585
Accuracy on test set: 69 %
[17,  100] loss: 0.590
[17,  200] loss: 0.587
Accuracy on test set: 68 %
[18,  100] loss: 0.581
[18,  200] loss: 0.579
Accuracy on test set: 70 %
[19,  100] loss: 0.578
[19,  200] loss: 0.570
Accuracy on test set: 70 %
```

