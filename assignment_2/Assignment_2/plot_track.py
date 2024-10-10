import numpy as np
import matplotlib.pyplot as plt

STARTING = 0.8  # the value of the starting line
FINISHING = 0.4  # the value of the finishing line


def build_track_a(save_map=False):
    track = np.ones(shape=(32, 17))
    track[14:, 0] = 0
    track[22:, 1] = 0
    track[-3:, 2] = 0
    track[:4, 0] = 0
    track[:3, 1] = 0
    track[0, 2] = 0
    track[6:, -8:] = 0

    track[6, 9] = 1

    track[:6, -1] = FINISHING
    track[-1, 3:9] = STARTING
    if save_map:
        with open('data/track_a.npy', 'wb') as f:
            np.save(f, track)
    return track


def build_track_b(save_map=False):
    track = np.ones(shape=(30, 32))

    for i in range(14):
        track[:(-3 - i), i] = 0

    track[3:7, 11] = 1
    track[2:8, 12] = 1
    track[1:9, 13] = 1

    track[0, 14:16] = 0
    track[-17:, -9:] = 0
    track[12, -8:] = 0
    track[11, -6:] = 0
    track[10, -5:] = 0
    track[9, -2:] = 0

    track[-1] = np.where(track[-1] == 0, 0, STARTING)
    track[:, -1] = np.where(track[:, -1] == 0, 0, FINISHING)
    if save_map:
        with open('data/track_b.npy', 'wb') as f:
            np.save(f, track)
    return track


def build_track_c(save_map=False):
    track = np.ones(shape=(12, 10))
    # track[10:, 0] = 0
    track[12:, 1] = 0
    # track[-3:, 2] = 0
    track[:3, 0] = 0
    # track[:3, 1] = 0
    track[0, 1] = 0
    track[6:, 7:] = 0
    # track[:,0] = 0
    # track[:, 1] = 0

    track[6, 7] = 1

    track[:6, -1] = FINISHING
    track[-1, :7] = STARTING

    if save_map:
        with open('data/track_c.npy', 'wb') as f:
            np.save(f, track)
    return track

if __name__ == "__main__":
    # create tracks and save them as .npy files
    # track_a = build_track_a(save_map=True)
    # track_b = build_track_b(save_map=True)
    track_c = build_track_c(save_map=True)

    # check if the map properly built
    plt.figure(figsize=(10, 5), dpi=150)
    for i, map_type in enumerate(['c']):
        with open(f'./data/track_{map_type}.npy', 'rb') as f:
            track = np.load(f)
        ax = plt.subplot(1, 2, i + 1)
        ax.imshow(track, cmap='GnBu')
        ax.set_title(f'track {map_type}', fontdict={'fontsize': 13, 'fontweight': 'bold'})

    plt.tight_layout()
    plt.savefig('./data/maps.png')
    plt.show()