import numpy as np
import matplotlib.pyplot as plt

# 1.像素点的理解
def test01():

    # 构建200*200,像素值全为0的图像
    img = np.zeros([200,200])
    print(img)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()

    # 构建200*200,像素值全为255的图像
    img = np.full([200, 200], 255)
    print(img)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()


# 2.图像通道的理解
def test02():

    # 从磁盘中读取彩色图像
    img = plt.imread('data/lena.png')
    # 显示图像H*W*C
    print(img.shape)  # 512*510*3

    img = np.transpose(img, [2, 0, 1])
    print(img.shape)  # 512*510*3

    for chanel in img:
        print(chanel)
        plt.imshow(chanel)
        plt.show()

    # 显示图像H*W*C
    img = np.transpose(img, [1, 2, 0])
    plt.imshow(img)
    plt.show()



if __name__ == '__main__':
    test02()
