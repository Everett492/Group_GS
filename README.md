# Working Log in Group_GS

## Webs Used

1. [nerfstudio official document](https://docs.nerf.studio/)

## Nerfstudio Start

### Installation

[Installation Instruction](https://docs.nerf.studio/quickstart/installation.html)

1. Create environment:

    ```text
    conda create --name nerfstudio -y python=3.8
    conda activate nerfstudio
    pyhton -m pip install --upgrade pip
    ```

2. Dependencies:
    - PyTorch
        - PyTorch
        - Build necessary CUDA extensions.

            ```text
            conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
            ```

    - tiny-cuda-nn

        ```text
        pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
        ```

3. Installing nerfstudio

    ```text
    pip install nerfstudio
    ```

### Run First Model

[Training Your First Model](https://docs.nerf.studio/quickstart/first_nerf.html)

1. Train and run viewer:
    - Test dataset downloading: original order `ns-download-data nerfstudio --capture-name=poster` is not effective. I've uploaded the dataset to github.

## NeRF: Neural Radiance Field

### Theoretical Part

1. Original paper: [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/pdf/2003.08934.pdf)
    - Inrtroduction:
        - A static scene is represented by a continuous `5D funtion` that outputs the `radiance` emitted in each direction (θ, φ) at each point (x, y, z) in space, and a `density` at each point which acts like a differential opacity controlling how much radiance is accumulated bt a ray passing through (x, y, z).
        - The 5D funtion is represented by an MLP.
        - Rendering step:
            - March camera rays through the scene to generate a sampled set of 3D points.
            - Use those points and their corresponding 2D viewing directions as input to the NN to produce an output set of colors and densities.
            - Use classical volume rendering techniques to accumulate those colors and densities into a 2D image.

            ![NeRF Rendering](./image_note/NeRF_Rendering.jpg)

        - Disadvantages of basic implementation:
            - low resolution, inefficient
            - solution 1: positional encoding
            - solution 2: hierarchical sampling
    - Neural Radiance Field Scence Representation
        - The 5D vector-valued function:
            - Input: (x, y, z, θ, φ)
            - Output: (r, g, b, σ)
            - Approximate this function with an MPL network: `FΘ : (x, d) → (c, σ)`.
        - Density σ is predicted as a function of only the location x. Color c is predicted as a function of both location and viewing direction.
        - Network Architecture:
        ![MLP Architecture](./image_note/NeRF_MLP.png)
    - Volume Rendering with Radiance Fields
        - The 5D neural rediance field repersents a scence as the volume density and directional emitted radiance at any point in sapce.
        - The volume density σ(x) can be interpreted as the differential probability of a ray termination at an infinitesimal particle at location x.
        - The expected color C(r) of camera ray r(t) = o + td with near and far bounds tn and tf is:
        ![Color C(r) Calculation](./image_note/Cr.png)
        - The function T(t) denotes accumulated transmittance along the ray from tn to t, the probability that the ray travels from tn to t without hitting any other particle.
        - Integral to Quadrature:
            - Stratified Sampling Approach: pertition [tn, tf] into N evenly-spaced bins and then dran one sample uniformly at random from within each bin
                ![Stratified Sampling](./image_note/Stratified_Sampling.png)
            - Use the samples above to estimate C(r):
                ![C(r) Estimating](./image_note/Cr_Estimate.png)
    - Optimizing a Neural Radiance Field
        - Positional encoding
            - The encoding function is applied separately to each of the three coordinate values in x and to the three components of the Cartesian viewing direction unit vector d.
            - For x, L = 10. For d, L = 4.
            ![Encoding Function](./image_note/NeRF_Encoding_Function.png)
        - Hierarchical volume sampling
        - Implementation details
        ![NeRF Loss Function](./image_note/NeRF_Loss_Function.png)

### NeRFactor
