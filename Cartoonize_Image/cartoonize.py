import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading image using file path and return as numpy array
def read_img(file_path):
    img = cv2.imread(file_path)
    return img

def edge_detection(img, line_width, blur_amount):
    gray_scale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Smooting gray scale image(this will helps in better edges detection)

    # In median blur, the central element of the image 
    # .... is replaced by the median of all the pixels
    gray_scale_img_blur = cv2.medianBlur(gray_scale_img, blur_amount)


    # Now detecting edges
    # Syntax: 
    #   cv2.adaptiveThreshold(source, maxVal, adaptiveMethod, thresholdType, blocksize, constant)
    img_edges = cv2.adaptiveThreshold(gray_scale_img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_width, blur_amount)

    return img_edges

def color_quantization(img, k_value, epochs, accuracy):  # k_value is number of clusters    
    data = np.float32(img)
    data = data.reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, epochs, accuracy)

    # Now applying kmeans algorithm
    # Syntax: 
    #   cv2.kmeans(samples, nclusters(K), criteria, attempts, flags)    
    ret, centers_position, centroid = cv2.kmeans(data, k_value, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centroid = np.uint8(centroid)
    result = centroid[centers_position.flatten()]

    # reshaping image to its actual shape
    result = result.reshape(img.shape)

    return result



def generate_cartoonize_img(img, LINE_WIDTH, BLUR_VALUE, TOTAL_COLORS, EPOCHS, ACCURACY):
    # img = read_img(img_path)

    edgeImg = edge_detection(img, LINE_WIDTH, BLUR_VALUE)
    quantized_img = color_quantization(img, TOTAL_COLORS, EPOCHS, ACCURACY)

    # bilateralFilter is used for removing noice and smothing
    blurred_img = cv2.bilateralFilter(quantized_img, d=7, sigmaColor=200, sigmaSpace=200)
    cartoonized_img = cv2.bitwise_and(blurred_img, blurred_img, mask = edgeImg)
    
    return cartoonized_img
    


if __name__ == "__main__":    
    img_path = "./face.jpg"
    LINE_WIDTH  = 7
    BLUR_VALUE = 5
    TOTAL_COLORS = 4
    EPOCHS = 50
    ACCURACY = 0.02

    cartoon_img = generate_cartoonize_img(img_path, LINE_WIDTH, BLUR_VALUE, TOTAL_COLORS, EPOCHS, ACCURACY)

    # img = read_img(img_path)
    # cartoon_img = color_quantization(img, TOTAL_COLORS, EPOCHS, ACCURACY)

    plt.imshow(cartoon_img)
    plt.show()
    




    
