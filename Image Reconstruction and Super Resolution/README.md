4. Begin by exploring the [instructor's notebook](https://github.com/nipunbatra/ml-teaching/blob/master/notebooks/siren.ipynb) that introduces the application of Random Fourier Features (RFF) for image reconstruction. Demonstrate the following applications using the cropped image from the notebook:
    - Superresolution: perform superresolution on the image shown in notebook to enhance its resolution by factor 2. Show a qualitative comparison of original and reconstructed image.
    - The above only helps us with a qualitative comparison. Let us now do a quantitative comparison. First, skim read this article: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution
        - Start with a 400x400 image (ground truth high resolution).
        - Resize it to a 200x200 image (input image)
        - Use RFF + Linear regression to increase the resolution to 400x400 (predicted high resolution image)
        - Compute the following metrics:
            - RMSE on predicted v/s ground truth high resolution image
            - Peak SNR
    - Completing Image with Random Missing Data: Apply RFF to complete the image with 10%, 20%, and so on up to 90% of its data missing randomly. Randomly remove portions of the data, train the model on the remaining data, and predict on the entire image. Display the reconstructed images for each missing data percentage and show the metrics calculated above. What do you conclude?.

5. Use the [instructor's notebook](https://github.com/nipunbatra/ml-teaching/blob/master/notebooks/movie-recommendation-knn-mf.ipynb) on matrix factorisation, and solve the following questions. 
    - Use the above image from Q4 and complete the rectangular missing patch for three cases, i.e. a rectangular block of 30X30 is assumed missing from the image. Choose rank `r` yourself. Vary the patch location as follows.
        1. an area with mainly a single color.
        2. an area with 2-3 different colors.
        3. an area with at least 5 different colors.
    
        Perform Gradient Descent till convergence, plot the selected patches, original and reconstructed images, compute the metrics mentioned in Q4 and write your observations. Obtain the reconstruction using RFF + Linear regression and compare the two.

    - Vary patch size (NxN) for ```N = [20, 40, 60, 80, 100]``` and peform Gradient Descent till convergence. Demonstrate the variation in reconstruction quality by making appropriate plots and metrics.
    
        Reconstruct the same patches using RFF. Compare the results and write your observations.
            
    - Write a function using this [reference](https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html) and use alternating least squares instead of gradient descent to repeat Part 1, 2 of Q5, using your written function.
    
    - Consider a patch of size (100x100) with at least 5 colors. Vary the low-rank value as ```r = [5, 10, 50, 100]``` . Use Gradient Descent and plot the reconstructed images to demonstrate difference in reconstruction quality. Write your observations.
