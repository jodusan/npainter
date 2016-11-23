import json
import glob
from libs import utils, gif, painter
import matplotlib.pyplot as plt

from libs.painter import train

"""
    NeuralPainter - npainter

    Simple neural network for painting images. It works by using x,y coordinates of
    each pixel as input and tries to predict their rgb values. This somewhat gives
    the resulting generated picture a look of a painted one, that is based on the
    original input image.

    gh repo: https://github.com/dulex123/npainter
"""

config = {}
with open('config.json', 'r') as f:
    config = json.load(f)

    input_folder = config['input_folder']
    output_folder = config['output_folder']
    learning_rate = float(config['learning_rate'])
    num_iters = int(config['num_iterations'])
    batch_size = int(config['batch_size'])
    n_neurons = int(config['n_neurons'])
    n_layers = int(config['n_layers'])
    activ_fn = config['activation_fn']
    final_activ_fn = config['final_activation_fn']
    make_gifs = config['make_gifs']
    gif_step = int(config['gif_step'])

    imgs_path = glob.glob(input_folder + "/*.jpg")
    imgs_path.sort()
    imgs = []
    for i, img_path in enumerate(imgs_path):
        print(img_path)
        img = plt.imread(img_path)

        imgs = train(img, gif_step, learning_rate, batch_size, num_iters, n_neurons, n_layers, activ_fn, final_activ_fn)
        plt.imsave(fname=output_folder + "/" + str(i) + ".jpg", arr=imgs[-1])
