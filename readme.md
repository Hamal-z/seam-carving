# seam carving算法 python实现

数字图像处理课上了解到了seam carving算法，对这个自适应非等比缩放方法比较感兴趣，于是尝试用python来实现。

图像处理方面用了opencv以及numpy，能量函数方面，尝试先对原图像进行高斯模糊，再使用Sobel或者Canny算法计算能量，进一步得到能量累加图后用动态规划找到能量最小的seam line。

最终效果支持双向缩小，自定义缩小程度，缩小过程中显示每次删除的seam line。


-----
Python 版本：`3.6.4`

numpy 版本：`1.14.2`

opencv-contrib-python 版本：`3.4.0.12`

------


| 原图 | 能量图 |
| :------: | :------: |
| ![原图](https://dzwonsemrish7.cloudfront.net/items/2T240W260u0U021z0M1u/t.jpg?v=290fe016) | ![energy](https://dzwonsemrish7.cloudfront.net/items/1i1K180a3w113m1v0F39/ec.jpg?v=57f7aa35) | 

| 横向、纵向缩小20% | 横向缩小33% |
| :------: | :------: |
| ![20](https://dzwonsemrish7.cloudfront.net/items/0O251B1o0n16043t0f21/20%2020.jpg?v=a31b675a) | ![33](https://dzwonsemrish7.cloudfront.net/items/441f2x3q3c1v180e3u1T/33.jpg?v=f1347eb6) |



------

动态演示

![gif](https://dzwonsemrish7.cloudfront.net/items/1j3C2n0Q1B3R1G182u3B/Screen%20Recording%202018-07-22%20at%2005.13%20下午.gif?v=5f93daf2)









