import numpy as np
import colorsys
from PIL import Image
import cv2 as cv
import paddle


#为了处理之前的卷积层的输出问题

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

img_path="E:/py_project/paddle_about/asset/test_o.png"
img = cv.imread(img_path) #(485, 478, 3) 注意，中文路径有问题

loaded_layer = paddle.jit.load("./inference_model2")
loaded_layer.eval()

x = paddle.randn([1, 784], 'float32')
img = img.astype(dtype='float32')
img=img_cov(img)
pred = loaded_layer(img[np.newaxis,:,:,:])
#print(img.shape)
print(pred.shape)
img=cov_img(pred[0])
cv.imwrite("E:/py_project/paddle_about/asset/test.png",img)
#test= paddle.load(model_path)
#help(test)
#test.predict()
