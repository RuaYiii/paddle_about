import paddle
import paddle.vision as vision
import paddle.text as text
import numpy as np
from paddle.io import Dataset ,DataLoader
from paddle.vision.transforms import ToTensor
import paddle.nn.functional as F
import cv2 as cv
#关于如何构建自己的数据集
paddle.set_device('gpu') #使用GPU

#train_dataset = vision.datasets.MNIST(mode='train', transform=ToTensor()) ##会下载数据集
#val_dataset = vision.datasets.MNIST(mode='test', transform=ToTensor()) #会下载数据集


date_path="E:/py_project/paddle_about/img_npy/"
model_path="E:\py_project\paddle_about\checkpoint\model_final.pdparams"

print("@@@@load data")
train_xy=np.load(date_path+"train.npy")


#需要抽象一个dataset类
class MyDataset(Dataset):
    def __init__(self, data):
        super(MyDataset,self).__init__()
        self.data=data
    def __getitem__(self, index):
        image=self.data[0][index].astype('float32')
        label=self.data[1][index].astype('float32')
        return image, label

    def __len__(self):
        return len(self.data[0])
train_set=MyDataset(train_xy)

#关于模型封装
class MyNet(paddle.nn.Layer):
    def __init__(self, sz):
        super(MyNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(3,64,5,stride=1, padding=2)
        #self.conv2 = paddle.nn.Conv2D(64,64,3,stride=1, padding=1)
        self.conv3 = paddle.nn.Conv2D(64,32,3,stride=1, padding=1)
        self.conv4 = paddle.nn.Conv2D(32,3,3,stride=1, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        #x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        #x = paddle.nn.functional.pixel_shuffle(x,3)
        return x
sz_y=train_xy[1][0].shape
epoch_num = 100
batch_size = 20
learning_rate = 0.00001
model = MyNet(sz_y[0])
opt = paddle.optimizer.Adam(learning_rate=learning_rate,parameters=model.parameters())
def train(model,opt):
    model.train()
    train_loader = paddle.io.DataLoader(train_set,
                                        shuffle=True,
                                        batch_size=batch_size)
    for epoch in range(epoch_num):
        print(f"@@@@epoch:{epoch}")
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            #print(f"batch_id:{batch_id}")
            y_data = paddle.to_tensor(data[1])
            #print(f"batch_id:{batch_id}")
            #y_data = paddle.unsqueeze(y_data, 1)
            logits = model(x_data)
            #print(f"@@@{logits.shape}")
            #print(f"@@@{y_data.shape}")
            loss = F.mse_loss(logits, y_data)
            #loss =paddle.nn.MSELoss()
            if batch_id % 1000 == 0:
                print(f"loss:{loss}")    
                #print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))
            loss.backward()
            opt.step()
            opt.clear_grad()
        model.eval()
        accuracies = []
        losses = []
        model.train()
#导入之前的训练数据
'''
layer_state_dict = paddle.load(model_path)
model.set_dict(layer_state_dict)
model=paddle.Model(model)
model.prepare(optimizer=paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()), 
              loss=paddle.nn.CrossEntropyLoss(), 
              metrics=paddle.metric.Accuracy())
'''              
#进行推理
#img_path="E:/py_project/paddle_about/asset/test_o2.png"
#img = cv.imread(img_path) #(485, 478, 3)

#一些测试-推理代码
def img_cov(img_np): 
    res=[]
    for i in range(3):
        res.append(img_np[:,:,i])
    return np.array(res)
def cov_img(img_np):
    r=img_np[0]
    g=img_np[1]
    b=img_np[2]
    res=np.append(r[:,:,np.newaxis],g[:,:,np.newaxis],axis=2)
    res=np.append(res,b[:,:,np.newaxis],axis=2)
    return res
#cov=img_cov(img) 
#full_t=cov[np.newaxis,np.newaxis,:,:,:].astype(dtype='float32')
#res_=model.predict(full_t)
#res=np.array(res_)
#print(res[0][0][0].shape)
#img_res=cov_img(res[0][0][0])
#cv.imwrite("E:/py_project/paddle_about/asset/test2.png",img_res)

#模型结构
#params_info = paddle.summary(model,(1,3,256,256))
#print(params_info)


print("start")
train(model,opt)
print("end")
#paddle.save(model.state_dict(), 'E:/py_project/paddle_about/checkpoint/test_net.pdparams')
#paddle.save(adam.state_dict(), "E:/py_project/paddle_about/checkpoint/adam.pdopt")
#paddle.save(prog, "temp/model.pdmodel")
paddle.save(model.state_dict(),"inference_model2.pdparams")
#model.save('checkpoint/model_final',training=True)
#paddle.save(prog, "temp/model.pdmodel")
#读取权重然后导出部属用模型
model_for_save = MyNet(sz_y[0])
layer_state_dict = paddle.load("inference_model2.pdparams")
model_for_save.set_dict(layer_state_dict)
model_for_save=paddle.Model(model_for_save)
model_for_save.prepare(optimizer=paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters()), 
              loss=paddle.nn.MSELoss(), 
              metrics=paddle.metric.Accuracy())
img_path="E:/py_project/paddle_about/asset/test_o2.png"
img = cv.imread(img_path) #(485, 478, 3)
cov=img_cov(img) 
full_t=cov[np.newaxis,np.newaxis,:,:,:].astype(dtype='float32')
res_=model_for_save.predict(full_t)
res=np.array(res_)
img_res=cov_img(res[0][0][0])
cv.imwrite("for_out.png",img_res)

model_for_save.save('inference_model2', False)  # save for inference


