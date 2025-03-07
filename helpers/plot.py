import matplotlib.pyplot as plt
from PIL import Image


def plot_image_row(
    images: list[Image.Image],
    figsize: tuple[int, int] | None = None,
    texts: list[str] | None = None,
) -> None:
    """
    Display a row of images with optional titles.

    Args:
        images: List of PIL images
        figsize: Tuple of width and height of the figure
        texts: List of strings for the titles of the images
    """
    if len(images) == 0:
        return
    elif len(images) == 1:
        # Plot a single image
        images[0].show()
        return

    num_images = len(images)
    if texts is None:
        texts = [""] * num_images  # Create empty titles if none provided

    # Create a figure and a set of subplots
    fig, axes = plt.subplots(1, num_images, figsize=figsize)

    # If there's only one image, axes is not a list, so we make it a list
    if num_images == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, texts):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")  # Hide axes

    plt.tight_layout()
    plt.show()


def plot_image_grid(
    images: list[Image.Image],
    n_columns: int,
    figsize: tuple[int, int] | None = None,
    texts: list[str] | None = None,
) -> None:
    """
    Display images in a grid layout with optional titles.

    Args:
        images: List of PIL images
        n_columns: Number of columns in the grid
        figsize: Tuple of width and height of the figure
        texts: List of strings for the titles of the images
    """
    num_images = len(images)
    n_rows = (num_images + n_columns - 1) // n_columns  # Calculate number of rows
    n_rows += 1 if num_images % n_columns != 0 else 0

    if texts is None:
        texts = [""] * num_images  # Create empty titles if none provided

    fig, axes = plt.subplots(n_rows, n_columns, figsize=figsize)

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    for ax, img, title in zip(axes, images, texts):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")  # Hide axes

    # Hide any remaining empty subplots
    for ax in axes[num_images:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_images(
    images: list[Image.Image],
    figsize: tuple[int, int] | None = None,
    texts: list[str] | None = None,
    n_columns: int = 3,
) -> None:
    """
    Display images in a grid layout with optional titles.

    Args:
        images: List of PIL images
        figsize: Tuple of width and height of the figure
        texts: List of strings for the titles of the images
        n_columns: Number of columns in the grid
    """
    num_images = len(images)
    n_rows = (num_images + n_columns - 1) // n_columns
    n_rows += 1 if num_images % n_columns != 0 else 0

    # If there's only one row, use plot_image_row
    if n_rows == 1:
        plot_image_row(images, figsize, texts)
    else:
        plot_image_grid(images, n_columns, figsize, texts)
