# Human-detection-and-dominant-color-identification

## Dominant color identification
### Approach
Used KMeans algorithm to partitions n observations(pixels) into k clusters, then i count the number of pixels in each cluster to identify the dominant colors, now once i've got the `k` dominant colours, I sort them by most common first, now for removal of (black, green, grey) colours, i took help of RGB values of the identified colours, i figured out that these three colours have difference between their r,g and b value very less, for that i defined a threshold such that, if the difference between r,g,b values is greater than the threshold, then that colour will not be in the range of either of the three colours(black, grey, white).

### Run
``` python3 color.py```

Change the image file, to see the result for any other image, and can change the value of `k` and `threshold` according to requirement.
