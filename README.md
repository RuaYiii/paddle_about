# 这是一个测试场，说白了是学习复建过程中的产物

主要是paddlepaddle的代码

## 目前测试场有什么呢？

由上个学习项目（ https://github.com/RuaYiii/Halftones-for-img-and-video ）用垃圾方法实现半调效果但是过程比较诡异然后产出了二百多张序列帧————视频太大了，我不打算上传，其实传到他处了————于是就有了一个小小小小小小型的垃圾测试集，用来出一点垃圾图
- 有一个一次性的训练代码，并导出权重参数，然后载入权重参数并导出一个直接用于推理的模型
- 有一个单独直接用这个b模型并且输出序列帧的代码
- 对了，一部分测试代码没删除，所以边输出边把每帧给你显示个一下下
- 由于生成序列帧视频是那输出单张图改的，所以也可以输出单张图，有示例，可以看看asset的这个文件夹
- 而且由于色彩模式是rgb，所以理应是0-255 也就是uint8,但是模型的输出是有些超常的，这里笔者并没有限制，在输出单张图的时候会有奇妙的诡异效果，但是视频的话其实就是，一坨不可视的答辩了。

以上，祝你生活愉快

**对了，训练文件真的很大，所以没上传，但是模型是上传的**