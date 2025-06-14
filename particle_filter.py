import json
import os
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# change IDs to your IDs.
ID1 = "318452364"
ID2 = "207767021"

ID = "HW3_{0}_{1}".format(ID1, ID2)
RESULTS = 'results'
os.makedirs(RESULTS, exist_ok=True)
IMAGE_DIR_PATH = "Images"

# SET NUMBER OF PARTICLES
N = 100

# Initial Settings
s_initial = [297,    # x center
             139,    # y center
              16,    # half width
              43,    # half height
               0,    # velocity x
               0]    # velocity y


def predict_particles(s_prior: np.ndarray) -> np.ndarray:
    """Progress the prior state with time and add noise.

    Note that we explicitly did not tell you how to add the noise.
    We allow additional manipulations to the state if you think these are necessary.

    Args:
        s_prior: np.ndarray. The prior state.
    Return:
        state_drifted: np.ndarray. The prior state after drift (applying the motion model) and adding the noise.
    """
    s_prior = s_prior.astype(float)
    state_drifted = s_prior.copy()

    # Check if this is our first prediction call
    first_prediction = not getattr(predict_particles, "initialized", False)

    if first_prediction:
        # Initial frame: spread particles around starting position
        startup_noise = np.random.normal(0, 6, s_prior.shape)
        state_drifted = state_drifted + startup_noise
        # Set flag to indicate we've been initialized
        predict_particles.initialized = True
    else:
        # Subsequent frames: apply motion model then add uncertainty

        # Move particles based on their velocities
        state_drifted[0, :] += state_drifted[4, :]  # x position update
        state_drifted[1, :] += state_drifted[5, :]  # y position update

        # Add random noise to account for motion uncertainty
        location_noise = np.random.normal(0, 4, (2, s_prior.shape[1]))
        speed_noise = np.random.normal(0, 2, (2, s_prior.shape[1]))

        state_drifted[0:2, :] += location_noise  # Add noise to x,y positions
        state_drifted[4:6, :] += speed_noise  # Add noise to velocities

    state_drifted = state_drifted.astype(int)
    return state_drifted


def compute_normalized_histogram(image: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Compute the normalized histogram using the state parameters.

    Args:
        image: np.ndarray. The image we want to crop the rectangle from.
        state: np.ndarray. State candidate.

    Return:
        hist: np.ndarray. histogram of quantized colors.
    """
    state = np.floor(state)
    state = state.astype(int)
    hist = np.zeros((16, 16, 16))

    # Extract rectangle parameters from state vector
    center_x, center_y = state[0], state[1]
    half_width, half_height = state[2], state[3]

    # Calculate patch boundaries with image bounds checking
    left_edge = max(0, center_x - half_width)
    right_edge = min(image.shape[1], center_x + half_width)
    top_edge = max(0, center_y - half_height)
    bottom_edge = min(image.shape[0], center_y + half_height)

    # Extract the region of interest
    image_patch = image[top_edge:bottom_edge, left_edge:right_edge]

    # Handle empty patch case
    if image_patch.size == 0:
        hist[:] = 1.0 / (16 ** 3)  # Set uniform distribution
    else:
        # Quantize colors from 8-bit to 4-bit and compute histogram
        quantized_colors = (image_patch // 16).reshape(-1, 3)
        color_histogram, _ = np.histogramdd(
            quantized_colors,
            bins=(16, 16, 16),
            range=((0, 16), (0, 16), (0, 16))
        )
        hist = color_histogram

    hist = np.reshape(hist, 16 * 16 * 16)

    # normalize safely
    total_pixels = hist.sum()
    if total_pixels > 0:
        hist = hist / total_pixels
    else:
        hist = np.ones_like(hist) / (16 ** 3)

    return hist


def sample_particles(previous_state: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Sample particles from the previous state according to the cdf.

    If additional processing to the returned state is needed - feel free to do it.

    Args:
        previous_state: np.ndarray. previous state, shape: (6, N)
        cdf: np.ndarray. cummulative distribution function: (N, )

    Return:
        s_next: np.ndarray. Sampled particles. shape: (6, N)
    """

    # Initialize output array with same shape as input
    s_next = np.zeros(previous_state.shape)

    # Resample each particle position
    for particle_idx in range(previous_state.shape[1]):
        # Generate random value between 0 and 1
        random_value = np.random.uniform(0, 1)

        # Find first CDF entry that exceeds our random value
        selected_idx = np.searchsorted(cdf, random_value)

        # Ensure we don't exceed array bounds
        selected_idx = min(selected_idx, previous_state.shape[1] - 1)

        # Copy the selected particle's state
        s_next[:, particle_idx] = previous_state[:, selected_idx]

    return s_next


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate Bhattacharyya Distance between two histograms p and q.

    Args:
        p: np.ndarray. first histogram.
        q: np.ndarray. second histogram.

    Return:
        distance: float. The Bhattacharyya Distance.
    """
    # Calculate Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p * q))
    
    # Calculate Bhattacharyya distance
    distance = -np.log(bc)
    return distance


def show_particles(image: np.ndarray, state: np.ndarray, W: np.ndarray, frame_index: int, ID: str,
                  frame_index_to_mean_state: dict, frame_index_to_max_state: dict,
                  ) -> tuple:
    fig, ax = plt.subplots(1)
    image = image[:,:,::-1]  # Convert BGR to RGB for matplotlib
    plt.imshow(image)
    plt.title(ID + " - Frame mumber = " + str(frame_index))

    # Avg particle box-plot in green
    # Calculate weighted average particle position
    x_avg = np.sum(state[0, :] * W)
    y_avg = np.sum(state[1, :] * W)
    w_avg = 2 * np.sum(state[2, :] * W)  # Convert half-width to full width
    h_avg = 2 * np.sum(state[3, :] * W)  # Convert half-height to full height

    # Convert center coordinates to top-left corner for rectangle
    x_avg = x_avg - w_avg / 2
    y_avg = y_avg - h_avg / 2

    rect = patches.Rectangle((x_avg, y_avg), w_avg, h_avg, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # calculate Max particle box-plot in red
    # Find best particle (highest weight)
    max_particle_idx = np.argmax(W)
    x_max = state[0, max_particle_idx]
    y_max = state[1, max_particle_idx]
    w_max = 2 * state[2, max_particle_idx]
    h_max = 2 * state[3, max_particle_idx]

    # Convert center coordinates to top-left corner
    x_max = x_max - w_max / 2
    y_max = y_max - h_max / 2

    rect = patches.Rectangle((x_max, y_max), w_max, h_max, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show(block=False) # Plot while allowing the program keep running

    fig.savefig(os.path.join(RESULTS, ID + "-" + str(frame_index) + ".png"))
    frame_index_to_mean_state[frame_index] = [float(x) for x in [x_avg, y_avg, w_avg, h_avg]]
    frame_index_to_max_state[frame_index] = [float(x) for x in [x_max, y_max, w_max, h_max]]
    return frame_index_to_mean_state, frame_index_to_max_state


def main():
    state_at_first_frame = np.matlib.repmat(s_initial, N, 1).T
    S = predict_particles(state_at_first_frame)

    # LOAD FIRST IMAGE
    image = cv2.imread(os.path.join(IMAGE_DIR_PATH, "001.png"))

    # COMPUTE NORMALIZED HISTOGRAM
    q = compute_normalized_histogram(image, s_initial)

    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    W = np.zeros(N)
    for i in range(N):
        p = compute_normalized_histogram(image, S[:, i])
        W[i] = np.exp(-9 * bhattacharyya_distance(p, q))
    W = W / np.sum(W)
    C = np.cumsum(W)

    images_processed = 1

    # MAIN TRACKING LOOP
    image_name_list = os.listdir(IMAGE_DIR_PATH)
    image_name_list.sort()
    frame_index_to_avg_state = {}
    frame_index_to_max_state = {}
    for image_name in image_name_list[1:]:

        S_prev = S

        # LOAD NEW IMAGE FRAME
        image_path = os.path.join(IMAGE_DIR_PATH, image_name)
        current_image = cv2.imread(image_path)

        # SAMPLE THE CURRENT PARTICLE FILTERS
        S_next_tag = sample_particles(S_prev, C)

        # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE
        S = predict_particles(S_next_tag)

        # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
        W = np.zeros(N)
        for i in range(N):
            p = compute_normalized_histogram(current_image, S[:, i])
            W[i] = np.exp(-9 * bhattacharyya_distance(p, q))
        W = W / np.sum(W)
        C = np.cumsum(W)

        # CREATE DETECTOR PLOTS
        images_processed += 1
        if 0 == images_processed%10:
            frame_index_to_avg_state, frame_index_to_max_state = show_particles(
                current_image, S, W, images_processed, ID, frame_index_to_avg_state, frame_index_to_max_state)

    with open(os.path.join(RESULTS, 'frame_index_to_avg_state.json'), 'w') as f:
        json.dump(frame_index_to_avg_state, f, indent=4)
    with open(os.path.join(RESULTS, 'frame_index_to_max_state.json'), 'w') as f:
        json.dump(frame_index_to_max_state, f, indent=4)


if __name__ == "__main__":
    main()
