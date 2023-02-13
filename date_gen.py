import numpy as np
from PIL import Image
def test_err_fiffusion(x,rank):  #可以到达一个诡异的效果，或者可以用于怪核
    #转换的级别
    #转换的角度 - 给指定方向加上去 sin:1:cos
    angle=30
    #rank=50 #上限是255
    r,c=x.shape[0],x.shape[1]
    for i in range(r):
        for j in range(c):
            temp=x[i,j] / rank
            err= temp-int(temp)
            err_base=err*rank #扩散的基础值
            x[i,j]=int(temp)
            if(j+1<c):
                x[i,j+1]+=err_base/3
                if(i+1<r):
                    x[i+1,j+1]+=err_base/3
            if(i+1<r):
                x[i+1,j]+=err_base/3
    return x*rank
    
test_img_path="E:/py_project/halftone/asset/人像.png"
image = Image.open(test_img_path).convert('RGB')
image.load()
r_o, g_o, b_o= image.split() #拆分
image_arr=np.array(image)
r=image_arr[:,:,0]
g=image_arr[:,:,1]
b=image_arr[:,:,2]
res_x_r=[]
res_x_g=[]
res_x_b=[]
res_y_r=[]
res_y_g=[]
res_y_b=[]
for i in range(255):
    last_r= test_err_fiffusion(r,i+1)
    last_g= test_err_fiffusion(g,i+1)
    last_b= test_err_fiffusion(b,i+1)

    res_x_r.append(last_r)
    res_x_g.append(last_g)
    res_x_b.append(last_b)
    if(i==0):
        res_y_r.append(r)
        res_y_g.append(g)
        res_y_b.append(b)
    else:
        res_y_r.append(last_r)
        res_y_g.append(last_g)
        res_y_b.append(last_b)
    print(i)
np.save("E:/py_project/paddle_about/img_npy/x_g.npy",np.array(res_x_r))
np.save("E:/py_project/paddle_about/img_npy/x_g.npy",np.array(res_x_g))
np.save("E:/py_project/paddle_about/img_npy/x_b.npy",np.array(res_x_b))
np.save("E:/py_project/paddle_about/img_npy/y_r.npy",np.array(res_y_r))
np.save("E:/py_project/paddle_about/img_npy/y_g.npy",np.array(res_y_g))
np.save("E:/py_project/paddle_about/img_npy/y_b.npy",np.array(res_y_b))