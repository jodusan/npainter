# NeuralPainter - npainter

Simple neural network written in tensorflow for painting images. It works by using [x,y] coordinates of each pixel from the input image and tries to predict their respective [r,g,b] values. The generated pictures have a look of a hand-painted picture.


## Dependencies

* [`tensorflow`](https://www.tensorflow.org/) Tensorflow library.
* `python 3.5` 
* `matplotlib`
* `numpy`

## Usage

Usage is configured with config.json file that has to be in the same directory as the npaint.py script. Script will take all .jpg images from the input folder and run for num_iterations on each one before outputing the result. It can be used on videos as well, since you can extract all frames from video with [`ffmpeg`](http://stackoverflow.com/questions/8287759/extracting-frames-from-mp4-flv)

### Config options:

* `input_folder` path to folder where input images are located
* `output_folder` folder where to write the output images to
* `n_neurons` number of neurons per layer
* `activation_fn` activation function after each hidden layer, "relu" or "tanh" are valid options
* `final_activation_fn` activation function of the last layer
* `n_layers` number of layers
* `batch_size`
* `num_iterations`
* `learning_rate` 

### Running the script

```
   python3 npaint.py 
```

### Notes

It took me around 18 hours on quadcore i5-4690 to paint 500 frames (20sec) of 720p video with 50 iterations and 13 layers with 24 neurons/layer. It can be seen on this [painted video](https://www.youtube.com/watch?v=4Jc6vNpQo58) & [original video](https://www.youtube.com/watch?v=hkSzmPnzF-o).
# Credits

Released under the [MIT License].<br>
Authored and maintained by Dušan Josipović.

> Blog [dulex123.github.io](http://dulex123.github.io) &nbsp;&middot;&nbsp;
> GitHub [@dulex123](https://github.com/dulex123) &nbsp;&middot;&nbsp;
> Twitter [@josipovicd](https://twitter.com/josipovicd)

[MIT License]: http://mit-license.org/

