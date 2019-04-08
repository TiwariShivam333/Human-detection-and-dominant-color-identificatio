# Human-detection-and-dominant-color-identification

## Dominant color identification
### Approach
Used KMeans algorithm to partitions `n` observations(pixels) into `k` clusters, then i count the number of pixels in each cluster to identify the dominant colors, now once i've got the `k` dominant colours, I sort them by most common first.

For removal of (black, green, grey) colours, i took help of RGB values of the identified colours, i figured out that these three colours have difference between their r,g and b value very less, for that i defined a threshold such that, if the difference between r,g,b values is greater than the threshold, then that colour will not be in the range of either of the three colours(black, grey, white).

### Run
``` python3 color.py```

Change the image file, to see the result for any other image, and can change the value of `k` and `threshold` according to requirement.





## Human detection

### Approach
Have used HOG feature descriptor, to generalize the object in such a way that the same object produces output very close to the same feature descriptor when viewed under different conditions. It uses a `global` feature to describe a person rather than a collection of `local` features.HOG uses histogram of size 64x128 pixels. So after image preprocessing(Feature description, histogram normalization, block normalization) have used a `MLP` for classification with single hidden layer of size `64` and trained the images, then compared the observed value for a test image, if it's greater than 0.5 then there is a human present and if not then there is not.

### Run 
```python3 human.py```
