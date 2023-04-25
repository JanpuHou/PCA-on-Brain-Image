import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA

my_image = imread("brain.jpg")
print(my_image.shape)

plt.figure(figsize=[12,8])
plt.imshow(my_image)
plt.show()

# Here first, we will be grayscaling our image, and then We’ll perform PCA on the matrix with all the components. 

#greyscaling the image
image_sum = my_image.sum(axis=2)
print(image_sum.shape)
 
new_image = image_sum/image_sum.max()
print(new_image.max())
 
plt.figure(figsize=[12,8])
plt.imshow(new_image, cmap=plt.cm.gray)
plt.show()


pca = PCA()
pca.fit(new_image)

# Getting the cumulative variance
 
var_cumu = np.cumsum(pca.explained_variance_ratio_)*100
 

# How many PCs explain 95% of the variance?
k = np.argmax(var_cumu>95)
print("Number of components explaining 95% variance: "+ str(k))
#print("\n")

plt.figure(figsize=[10,5])
plt.title('Cumulative Explained Variance explained by the components')
plt.ylabel('Cumulative Explained variance')
plt.xlabel('Principal components')
plt.axvline(x=k, color="k", linestyle="--")
plt.axhline(y=95, color="r", linestyle="--")
ax = plt.plot(var_cumu)
plt.show()

#Reconstructing using Inverse Transform
ipca = IncrementalPCA(n_components=k)
image_recon = ipca.inverse_transform(ipca.fit_transform(new_image))
 
# Plotting the reconstructed image
plt.figure(figsize=[12,8])
plt.imshow(image_recon,cmap = plt.cm.gray)
plt.show()


# Function to reconstruct and plot image for a given number of components
 
def plot_at_k(k):
    ipca = IncrementalPCA(n_components=k)
    image_recon = ipca.inverse_transform(ipca.fit_transform(new_image))
    plt.imshow(image_recon,cmap = plt.cm.gray)
    
#setting different amounts of K
ks = [10, 20, 40, 60, 80, 100]
 
plt.figure(figsize=[15,9])
 
for i in range(6):
    plt.subplot(2,3,i+1)
    plot_at_k(ks[i])
    plt.title("Components: "+str(ks[i]))
 
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.show()