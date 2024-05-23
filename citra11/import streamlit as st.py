import streamlit as st
from skimage import io, color, measure, morphology, draw, img_as_float
from skimage.filters import sobel
from skimage.segmentation import active_contour
import numpy as np
import matplotlib.pyplot as plt

def convex_hull(image):
    grayscale = color.rgb2gray(image)
    binary = grayscale > 0.5
    chull = morphology.convex_hull_image(binary)
    return chull

def skeletonization(image):
    grayscale = color.rgb2gray(image)
    binary = grayscale > 0.5
    skeleton = morphology.skeletonize(binary)
    return skeleton

def active_contour_image(image):
    grayscale = color.rgb2gray(image)
    s = np.linspace(0, 2 * np.pi, 400)
    r = 100 + 100 * np.sin(s)
    c = 100 + 100 * np.cos(s)
    init = np.array([r, c]).T

    snake = active_contour(sobel(grayscale), init, alpha=0.015, beta=10, gamma=0.001)
    fig, ax = plt.subplots()
    ax.imshow(grayscale, cmap=plt.cm.gray)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    return fig

st.title("Image Manipulation App")
st.write("Upload an image to perform operations on it")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = io.imread(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    operation = st.selectbox(
        "Choose an operation",
        ("Convex Hull", "Skeletonization", "Active Contour")
    )

    if operation == "Convex Hull":
        chull_image = convex_hull(image)
        st.image(chull_image, caption='Convex Hull Image', use_column_width=True, clamp=True)
    elif operation == "Skeletonization":
        skeleton_image = skeletonization(image)
        st.image(skeleton_image, caption='Skeletonized Image', use_column_width=True, clamp=True)
    elif operation == "Active Contour":
        fig = active_contour_image(image)
        st.pyplot(fig)
