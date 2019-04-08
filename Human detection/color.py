import cv2
import numpy as np
from sklearn.cluster import KMeans


def histogram(clusters):
    n = np.arange(0, len(np.unique(clusters.labels_)) + 1)
    hist, _ = np.histogram(clusters.labels_, bins=n)
    hist = hist.astype('float32')
    hist /= hist.sum()
    return hist


img = cv2.imread('batman.png')
height, width, _ = np.shape(img)
image = img.reshape((height * width, 3))
k=8
clusters = KMeans(n_clusters=k)
clusters.fit(image)

histogram = histogram(clusters)
combined = zip(histogram, clusters.cluster_centers_)
combined = sorted(combined, key=lambda x: x[0], reverse=True)

threshold=19
for _, rgbs in enumerate(combined):
    x=rgbs[1]
    red, green, blue = int(x[2]), int(x[1]), int(x[0])
    rgb=(red, green, blue)
    #if abs(red-green)>threshold and abs(green-blue)>threshold and abs(red-blue)>threshold:
    print('#%02x%02x%02x' % rgb)
