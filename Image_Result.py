import cv2 as cv
import matplotlib.pyplot as plt

def Image_Results():
    noOfDataset = 4

    for n in range(noOfDataset):
        metFile1 = './Image_Results/Unet-%d.jpg' % (n + 1)
        metFile2 = './Image_Results/ResUnet-%d.jpg' % (n + 1)
        metFile3 = './Image_Results/TransResUnet-%d.jpg' % (n + 1)
        metFile4 = './Image_Results/AHCNN-%d.jpg' % (n + 1)
        metFile5 = './Image_Results/IFHBA-AHCNN-%d.jpg' % (n + 1)

        GT = './Image_Results/Dataset-%d-GT.png' % (n + 1)
        image1 = cv.imread(metFile1)
        image2 = cv.imread(metFile2)
        image3 = cv.imread(metFile3)
        image4 = cv.imread(metFile4)
        image5 = cv.imread(metFile5)
        gt = cv.imread(GT)
        plt.subplot(1, 6, 1)
        plt.title('Groundtruth')
        plt.imshow(gt)
        plt.subplot(1, 6, 2)
        plt.title('Unet')
        plt.imshow(image1)
        plt.subplot(1, 6, 3)
        plt.title('ResUnet')
        plt.imshow(image2)
        plt.subplot(2, 6, 4)
        plt.title('TransResUnet')
        plt.imshow(image3)
        plt.subplot(2, 6, 5)
        plt.title('AHCNN')
        plt.imshow(image4)
        plt.subplot(2, 6, 6)
        plt.title('IFHBA-AHCNN')
        plt.imshow(image5)
        plt.show()


if __name__ == '__main__':
    Image_Results()
