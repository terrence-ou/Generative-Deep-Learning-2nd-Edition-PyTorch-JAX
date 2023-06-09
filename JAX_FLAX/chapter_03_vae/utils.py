''''
#############################################################
IMPORTANT:

This python file adapts from David Foster's code provided 
in his book Generative Deep Learning 2nd Edition:
https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/03_vae/03_vae_faces/vae_utils.py
The original code is using Apache-2.0 license; this file adapts 
the original implementation to the JAX context

#############################################################
'''

import numpy as np
import matplotlib.pyplot as plt


# Get the vector towards the given feature
def get_vector_from_label(dataset, embedding_dim, label, encode_fn, state, rng):
    # Initialize parameters
    current_sum_POS = np.zeros(shape=embedding_dim, dtype=np.float32)
    current_n_POS = 0
    current_mean_POS = np.zeros(shape=embedding_dim, dtype=np.float32)
    
    current_sum_NEG = np.zeros(shape=embedding_dim, dtype=np.float32)
    current_n_NEG = 0
    current_mean_NEG = np.zeros(shape=embedding_dim, dtype=np.float32)

    current_vector = np.zeros(shape=embedding_dim, dtype=np.float32)
    current_dist = 0

    print('label: ' + label)
    print('images | POS move | NEG move | distance | 𝛥 distance:')

    total_POS_samples = 5000
    curr_iter = 0
    while current_n_POS < total_POS_samples:
        # Sampling new POS and NEG images
        batch = next(iter(dataset))
        imgs = batch[0].numpy()
        attributes = batch[1].numpy()
        z = encode_fn(state, imgs, rng)
        z_POS = z[attributes==1]
        z_NEG = z[attributes==-1]
        
        # Updated both mean vector for both POS and NEG samples
        if len(z_POS) > 0:
            current_sum_POS = current_sum_POS + np.sum(z_POS, axis=0)
            current_n_POS += len(z_POS)
            new_mean_POS = current_sum_POS / current_n_POS
            movement_POS = np.linalg.norm(new_mean_POS - current_mean_POS)
        
        if len(z_NEG) > 0:
            current_sum_NEG = current_sum_NEG + np.sum(z_NEG, axis=0)
            current_n_NEG += len(z_NEG)
            new_mean_NEG = current_sum_NEG / current_n_POS
            movement_NEG = np.linalg.norm(new_mean_NEG - current_mean_NEG)
        
        # Updated the feature vector
        current_vector = new_mean_POS - new_mean_NEG
        new_dist = np.linalg.norm(current_vector)
        dist_change = new_dist - current_dist

        # Print the vector-finding process
        placeholder = '|  '
        if curr_iter % 5 == 0:
            print(f'{current_n_POS:6d}', placeholder, 
                  f'{movement_POS:6.3f}', placeholder,
                  f'{movement_NEG:6.3f}', placeholder,
                  f'{new_dist:6.3f}', placeholder,
                  f'{dist_change:6.3f}')
        
        current_mean_POS = np.copy(new_mean_POS)
        current_mean_NEG = np.copy(new_mean_NEG)
        current_dist = np.copy(new_dist)

        # When the changing distance is very small, terminate the while loop
        stop_thresh = 8e-2
        if np.sum([movement_POS, movement_NEG]) < stop_thresh:
            current_vector = current_vector / current_dist
            print('Found the ' + label + ' vector')
            break
    
        curr_iter += 1
    return current_vector
    

# Plot the feature transitions
def add_vector_to_images(dataset, feature_vec, encode_fn, decode_fn, state, rng):
    n_plots = 5
    factors = np.arange(-4, 5)
    batch = next(iter(dataset))
    imgs = batch[0].numpy()

    # Get image embeddings
    z = encode_fn(state, imgs, rng)
    
    fig = plt.figure(figsize=(18, 10))
    counter = 1

    for i in range(n_plots):
        img = imgs[i]
        ax = fig.add_subplot(n_plots, len(factors) + 1, counter)
        ax.axis('off')
        ax.imshow(img)
        counter += 1
        # Add transition images
        for factor in factors:
            new_z_sample = z[i] + feature_vec * factor
            generated_img = decode_fn(state, np.expand_dims(new_z_sample, axis=0))[0]
            ax = fig.add_subplot(n_plots, len(factors) + 1, counter)
            ax.axis('off')
            ax.imshow(generated_img)
            counter += 1
    plt.show()


# Morph between two faces
def morph_faces(dataset, encode_fn, decode_fn, state, rng):
    factors = np.arange(0.0, 1.0, 0.1)

    sample_faces = next(iter(dataset))[0][:2] # sample two faces
    imgs = sample_faces.numpy()
    z = encode_fn(state, imgs, rng)
    fig = plt.figure(figsize=(18, 8))
    counter = 1

    face_a = imgs[0]
    face_b = imgs[1]

    # show original face
    ax = fig.add_subplot(1, len(factors) + 2, counter)
    ax.axis('off')
    ax.imshow(face_a)

    counter += 1

    # plot transitions
    for factor in factors:
        factored_z = z[0] * (1 - factor) + z[1] * factor
        generated_img = decode_fn(state, np.expand_dims(factored_z, axis=0))[0]
        ax = fig.add_subplot(1, len(factors) + 2, counter)
        ax.axis('off')
        ax.imshow(generated_img)
        counter += 1
    
    # show target face
    ax = fig.add_subplot(1, len(factors) + 2, counter)
    ax.axis('off')
    ax.imshow(face_b)

    plt.show()