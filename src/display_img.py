"""Display tools"""

import matplotlib.pyplot as plt


def compare_img(img_1, img_arr_1, name_1, img_2, img_arr_2, name_2, cmap=None) -> None:
    """Display 2 images and their histogram for comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(img_1)
    ax1.axis("off")
    ax1.set_title(name_1)

    ax2.imshow(img_2, cmap=cmap)
    ax2.axis("off")
    ax2.set_title(name_2)

    plt.tight_layout()
    plt.show()
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.hist(img_arr_1.flatten(), bins=range(256))
    ax1.set_title(name_1)

    ax2.hist(img_arr_2.flatten(), bins=range(256))
    ax2.set_title(name_2)

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    help()
