# **OpenGL/CUDA Mandelbrot**

## Author

- **[Flávio Santos](https://github.com/fs000) - a22100771**

## Summary

This project was built on **CUDA C** with the main goal of generating an interactive Mandelbrot Fractal.
It uses the [`GLEW`](https://glew.sourceforge.net/) and [`FreeGLUT`](https://freeglut.sourceforge.net/) 
libraries to access **OpenGL** functionality, including user input.

**Disclaimer:** Due to this being the last project of the subject and also the teaching component of the 
cycle of studies, the conclusion to this report also carries out a brief mention to the previous projects 
(which can be found [here](https://github.com/fs000/MatrixMultiplication) and [here](https://github.com/fs000/Mandelbrot-Fractal)).

## Instructions

- **Click** anywhere on the fractal with the `Left Mouse Button` and **drag** the `Mouse` to **move** the fractal;
- Use the `Mouse Wheel` to **zoom in / out** on the `Mouse`'s current position;
- **Hold** `Shift` and use the `Mouse Wheel` to **increase/decrease** the number of **iterations**;
- **Click** anywhere on the fractal with the `Right Mouse Button` to reset the fractal to its starting conditions.

## Discussion

As this project's name suggests, it takes advantage of the GPU's power to parallelize tasks in order to 
create an interactive Fractal display.

This project was technically already included in the [previous one](https://github.com/fs000/Mandelbrot-Fractal) 
as an extra. As such, in order to have this be a standalone version, it was as simple as adding a `main()` function 
at the bottom of the only script present (`kernel.cu`). Some very light changes were also implemented.

The `kernel.cu` file is divided into the following regions:

1. **Global Variables**
    This section contains all variables that are required in different parts of the code (i.e. multiple methods).
    Note that `WIDTH` and `HEIGHT` can be manually changed before building the project. These variables control 
	both the OpenGL window size and the fractal texture size (which will surely have an impact on performance).

2. **Mandelbrot Fractal Kernel**
    This section is the most important for the project at hand. It contains the *kernel* with which the fractal 
	is generated (`generateFractal`), all other `__device__` type helper methods it requires and some `__constant__` 
	variables which are responsible for coloring.
    One method that might catch some attention is `toFractalCoords`. This method transforms any coordinate into 
	their correspondant in fractal space. As this is used outside of the *kernel* it not only sports the `__device__` 
	tag but also the `__host__` one, informing the compiler that this function can be run by both the GPU and CPU.

3. **OpenGL/CUDA Interop**
    This section is responsible for setting up any and all OpenGL functionalities through the use of several `glew` 
	and `glut` methods. Mainly, it starts the OpenGL context, and sets up all input handling and main loop work 
	(such as the drawing of a quad facing the *camera* and its texture).
    The `render()` method is arguably the most important. It's here where the program executes the *kernel*, 
	registers events (in order to calculate the frame time) and maps and unmaps resources from Cuda to OpenGL 
	(using `cudaGraphics` functions). The main resource being passed around is the *bitmap* that is "painted" by 
	the *kernel* (`fractal_d`). It is a `uchar3` (in other words, a vector of 3 `unsigned char`), and as such can 
	hold up to 256 values per coordinate (meaning a total of **16 777 216** colors can be processed).

4. **Input Handling**
    This section of the code controls the program's response based on user input. It detects such inputs based on event 
	callbacks passed from `glut`.
    Some fractal generation global variables (`center`, `scale` and `iterations`) are all manipulated by the user here.

## Results

Most (if not all) interactive display applications are rendered in real-time, meaning the content being generated is 
processed just as it is being displayed. In order to save GPU workload, this project does not follow that.

It instead updates the display (through calling `glutPostRedisplay()`), only when the user interacts with the fractal 
image, manipulating it (e.g: increasing scale, moving up, etc.).

Why is this important? Because generally applications display the amount of frames they are able to produce per second, 
something that simply cannot be done here. However, we can still measure *frame time*, which in this case encapsulates 
the time necessary to generate a fractal display.

Additionally, different sections of the fractal will require more or less time to generate. With this in mind, and in 
order to have equally valid *frame time* results analysis, the view of the fractal being evaluated is that of the 
*start* or *reset* (basically, the entire overview of the fractal).

The results below were obtained on an `NVIDIA GeForce GTX 1080`:

#### Avg Frame Time for Fractal Generation
|          |  800 x 600  |  1280 x 720  |   1920 x 1080  |
| :------- | :---------: | :----------: | :------------: |
| **100**  |   6.068 ms  |    8.068 ms  |    16.014 ms   |
| **1000** |  12.062 ms  |   17.062 ms  |    34.872 ms   |
| **5000** |  41.410 ms  |   58.466 ms  |   118.686 ms   |

![Avg Frame Time Graph](/images/avg_time.svg)

As we can see from either the table or graph, it comes as no surprise that the increase in *frame time* is related 
to the image dimensions and iteration count. 

Either way, these are all really good results! ~6 ms to produce such an impressive image is a feat in itself.
Also, always keep in mind, that these results will shift back and forth a lot as we dive into any of the fractal's 
regions (outside of the set). Some, even with a low resolution and iteration count, can take as high as ~60 ms to 
produce an image! 

## Conclusions

As I've previously stated on my [previous project](https://github.com/fs000/Mandelbrot-Fractal), these are the type 
of projects I love to dive into, because the final results are always so gratifying to contemplate!

It was a tough challenge, coming back to `C`, but now, not only do I have a better grasp of this powerful language, 
but I also learned how to use it to access the raw power of GPUs and use a parallelization built device to try and 
get the most out of it in order to accelerate any algorithm.

It was very amusing to learn all this in class, and then apply it to a *real-world* application and see the results for myself.

Not only that, but this project also allowed me to learn more about `OpenGL`. I knew about it, but just has never 
used it. And now that I have, even though my contact was limited, I feel ready to do something else with the amassed knowledge.

The interoperability between `CUDA` and `OpenGL` was yet another test to my abilities. I didn't know if it could be done, 
but the more I searched on the topic, the more I understood that it could, in fact, be accomplished and it wasn't too 
different from what I had done thus far (on the previous project).

Overall, I enjoyed all of the semester's projects. I learned a lot from all of them, and can't wait to expand and take 
this knowledge elsewhere. It will certainly come in handy in my future!

## Thanks

Finally, I'd like to thank our teacher [José Rogado](https://github.com/jrogado) for all his invaluable input and teachings 
throughout the making of this project and the entirety of the semester.

His grasp on all the subjects we learned and his very good teaching capabilities were definitely a factor on the quality of my work.

Even when some of us were building beyond the requirements, and exploring these topics by ourselves, he always had the availability, 
patience and knowledge to, if not aid us *then and there*, push us in the right direction.
