import numpy as np
import matplotlib.pyplot as plt


def get_vector_from_label(dataset, embedding_dim, label, encode_fn, state, rng):
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
        batch = next(iter(dataset))
        imgs = batch[0].numpy()
        attributes = batch[1].numpy()
        z = encode_fn(state, imgs, rng)
        z_POS = z[attributes==1]
        z_NEG = z[attributes==-1]
        
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

        stop_thresh = 8e-2
        if np.sum([movement_POS, movement_NEG]) < stop_thresh:
            current_vector = current_vector / current_dist
            print('Found the ' + label + ' vector')
            break
    
        curr_iter += 1
    return current_vector
    
