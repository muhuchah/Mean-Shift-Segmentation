import matplotlib.pyplot as plt


def load_image(image, title='Just an image'):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.savefig('output.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()

