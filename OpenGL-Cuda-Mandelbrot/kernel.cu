#include <stdio.h>
#include <math.h>
#include <Windows.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

#pragma region Global Variables

// The amount of bytes per pixel
const int BYTES_PER_PIXEL = 3;   /// Red, Green, Blue

// The main window title
const char *main_title = "CUDA/OpenGL MandelBrot Fractal";

// The window/texture dimensions
const int WIDTH = 1280,
		  HEIGHT = 720;

// The window's aspect ratio
const double ASPECT_RATIO = WIDTH / (double)HEIGHT;

// Fractal display variables
double2 center{ -0.75, 0.0 };
double scale = 1.0;
int iterations = 100;

// The time necessary to generate an image
float frame_time_ms = 0;

// Cuda related variables
GLuint pbo = 0, tex = 0;
struct cudaGraphicsResource *cuda_pbo_resource;

#pragma endregion

#pragma region Mandelbrot Fractal Kernel

// Ultra fractal colors
__constant__ uchar3 colorMap[5] = {
	{0, 7, 100},
	{32, 107, 203},
	{237, 255, 255},
	{255, 170, 0},
	{0, 2, 0}
};

// Ultra fractal colors' points
__constant__ double colorPoints[5] = {
	0.0,
	0.16,
	0.42,
	0.6425,
	0.8575
};

// Transform window coordinates into fractal coordinates
__host__ __device__ double toFractalCoords(int n, int dimension, double scale, double extra = 1.0) {
	return (n - (dimension / 2.0)) * (scale / (dimension / 2.5) * extra);
}

// Returns a 0-1 double representing the distance in percentage between 2 colors
__device__ double calcParam(double t, double lp, double rp) {
	return (t - lp) / (rp - lp);
}

// Apply a hermite interpolation between 2 colors
__device__ uchar3 hermColor(int index, double param) {

	double t = param * param * (3.0f - 2.0f * param);

	return { (unsigned char)(((1.0 - t) * (double)colorMap[index].x) + (t * (double)colorMap[index + 1].x)),
			 (unsigned char)(((1.0 - t) * (double)colorMap[index].y) + (t * (double)colorMap[index + 1].y)),
			 (unsigned char)(((1.0 - t) * (double)colorMap[index].z) + (t * (double)colorMap[index + 1].z)) };
}

// Returns a color
__device__ uchar3 getColor(double t) {

	// Loop through color length - 1
	for (int i = 0; i < 4; i++) {

		// Verify if the given t is between 2 colors
		if (t >= colorPoints[i] && t < colorPoints[i + 1]) {

			// If so, calculate the percentage
			double param = calcParam(t, colorPoints[i], colorPoints[i + 1]);

			// And returned an interpolated color
			return hermColor(i, param);
		}
	}

	// If t is outside any 2 colors, return the last color
	return colorMap[4];
}

// Calculation of Z^2 + C (returns a color directly)
__device__ uchar3 iterateMandel(double c_real, double c_imag, int max_iters) {

	int n = 0;
	double real = c_real;
	double imag = c_imag;

	while (n < max_iters) {

		double real2 = real * real;
		double imag2 = imag * imag;
		imag = 2.0 * real * imag + c_imag;
		real = real2 - imag2 + c_real;

		if (real2 + imag2 > 20.0) {
			// Apply color smoothing and return a color
			double sl = (double)(n + 4.0) - log2(log2(real2 + imag2)) / log2(2.0);
			return getColor(sl / (double)max_iters);
		}

		n++;
	}

	// Return black
	return {};
}

// Main Fractal generation method
__global__ void generateFractal(int width, int height,
									  double aspect_ratio,
									  double2 center,
									  double scale,
									  int max_iters,
									  uchar3 *fractal) {

	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;

	if ((px < width) && (py < height)) {

		// Invert Y because OpenGL's tex coordinates start at bottom left!
		double cX = toFractalCoords(px, width, scale, aspect_ratio) + center.x;
		double cY = -toFractalCoords(py, height, scale) + center.y;

		uchar3 color = iterateMandel(cX, cY, max_iters);

		int index = py * width + px;

		fractal[index] = color;
	}
}

#pragma endregion

#pragma region OpenGL/CUDA Interop

// Create method references
void initialize(int, char *[]);
void initWindow(int, char *[]);
void initPixelBuffer(void);
void render(void);
void drawTexture(void);
void resize(int, int);
void display(void);
void exitCudaInterop(void);
void mouseDrag(int, int);
void mousePress(int, int, int, int);
void mouseWheel(int, int, int, int);
void mouseDrag(int, int);

// Init OpenGL / Cuda
void initialize(int argc, char *argv[]) {

	initWindow(argc, argv);

	// Print OpenGL version used to the console
	fprintf(
		stdout,
		"INFO: OpenGL Version: %s\n",
		glGetString(GL_VERSION)
	);

	// Set up 2D orthographic region
	gluOrtho2D(0, WIDTH, HEIGHT, 0);

	// Set up mouse functions
	glutMouseWheelFunc(mouseWheel);
	glutMouseFunc(mousePress);
	glutMotionFunc(mouseDrag);

	// Set up display functions
	glutReshapeFunc(resize);
	glutDisplayFunc(display);

	// Set up the cuda/opengl interop (pixel buffer / texture)
	initPixelBuffer();
}

// Don't forget to free the returned char* !!!
char *createWindowTitle() {

	// Create string buffer
	size_t buffer_size = 1024; // arbitrary buffer size
	char *buffer = (char *)malloc(buffer_size * sizeof(char));

	// Concatenate 'strings'
	snprintf(buffer, buffer_size, 
			 "%s | Resolution: %d x %d | Center: %.2f; %.2f | Iterations: %d | Scale: %.2f | Frame Time: %.2f ms",
			 main_title, WIDTH, HEIGHT, center.x, center.y, iterations, 1.0 / scale, frame_time_ms);

	// Return the buffer (pointer)
	return buffer;
}

// Updates OpenGL's window title
void updateWindowTitle() {

	// Set new window title
	char *window_title = createWindowTitle();
	glutSetWindowTitle(window_title);
	free(window_title);
}

// Set up glut and glew
void initWindow(int argc, char *argv[]) {

	glutInit(&argc, argv);

	//glutInitContextVersion(3, 3);
	glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
	glutInitContextProfile(GLUT_CORE_PROFILE);

	glutSetOption(
		GLUT_ACTION_ON_WINDOW_CLOSE,
		GLUT_ACTION_GLUTMAINLOOP_RETURNS
	);

	glutInitWindowSize(WIDTH, HEIGHT);

	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);

	int window_handle = glutCreateWindow("TITLE");

	// Print error if the program couldn't create a window
	if (window_handle < 1) {
		fprintf(
			stderr,
			"ERROR: Could not create a new rendering window.\n"
		);
		getchar();
		exit(EXIT_FAILURE);
	}

	glewInit();
}

// Sets up the pixel buffer to be modified by CUDA and consumed by OpenGL
void initPixelBuffer() {

	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, BYTES_PER_PIXEL * WIDTH * HEIGHT * sizeof(GLubyte), 0,
				 GL_STREAM_DRAW);
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,
								 cudaGraphicsMapFlagsWriteDiscard);
}

// Handle window reshaping
void resize(int width, int height) {
	// Lock the display
	glutReshapeWindow(WIDTH, HEIGHT);
}

// Handle OpenGL rendering
void render() {

	// The "bitmap" to be used by the device
	uchar3 *fractal_d = 0;

	// Cuda events to record and calculate frame time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&fractal_d, NULL, cuda_pbo_resource);

	// Define blocks and threads
	dim3 grid_size((WIDTH + 31) / 32, (HEIGHT + 31) / 32);
	dim3 block_size(32, 32);

	// Record the first event
	cudaEventRecord(start);

	// Execute the kernel
	generateFractal<<<grid_size, block_size>>>(WIDTH, HEIGHT, ASPECT_RATIO, center, scale, iterations, fractal_d);

	// Record the second event
	cudaEventRecord(stop);

	// Block CPU execution until the "stop" event is recorded
	cudaEventSynchronize(stop);

	// Calculate frame time in milliseconds
	cudaEventElapsedTime(&frame_time_ms, start, stop);

	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

// Draw a quad and texture with the same size as the display window
void drawTexture() {

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, HEIGHT);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(WIDTH, HEIGHT);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(WIDTH, 0.0f);
	glEnd();
	glDisable(GL_TEXTURE_2D);
}

// The main display loop
void display() {

	glClear(GL_COLOR_BUFFER_BIT);
	render();
	drawTexture();
	glutSwapBuffers();
	updateWindowTitle();
}

// Release CUDA resources/buffers
void exitCudaInterop() {

	if (pbo) {
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffers(1, &pbo);
		glDeleteTextures(1, &tex);
	}
}

#pragma endregion

#pragma region Input Handling

int pressed;
int2 startCoords;

// Handle mouse button pressing
void mousePress(int button, int state, int x, int y) {

	// Register the pressed button
	pressed = button;

	// Check if the mouse was pressed down
	if (state == GLUT_DOWN) {

		// Check left button press
		if (button == GLUT_LEFT_BUTTON) {

			// Save (window) coordinates where user clicked
			startCoords = { x, y };

			// Check right button press
		} else if (button == GLUT_RIGHT_BUTTON) {

			// Reset main variables
			center = { -0.75, 0.0 };
			scale = 1.0;
			iterations = 100;
		}
	}
}

// Handle mouse dragging
void mouseDrag(int x, int y) {

	// Verify that user pressed left mouse button
	if (pressed == GLUT_LEFT_BUTTON) {

		// Move fractal center based on mouse movement
		int2 delta = { (x - startCoords.x) * ASPECT_RATIO, (y - startCoords.y) * ASPECT_RATIO };
		center.x -= delta.x * 0.002 * scale;
		center.y += delta.y * 0.002 * scale;
		startCoords = { x, y };

		// Call a redraw
		glutPostRedisplay();
	}
}

// Zoom in/out based on mouse position
void zoom(int dir, int x, int y) {

	// Save the old position (after converting window coordinates into fractal coordinates)
	double oldX = toFractalCoords(x, WIDTH, scale, ASPECT_RATIO);
	double oldY = toFractalCoords(y, HEIGHT, scale);

	// Apply scale change
	scale *= (1.0f - dir * 0.04);

	// Move center based on amount of zoom applied
	center.x -= toFractalCoords(x, WIDTH, scale, ASPECT_RATIO) - oldX;
	center.y += toFractalCoords(y, HEIGHT, scale) - oldY;
}

// Increase/Decrease ther number of iterations
void modifyIters(int dir) {
	iterations += dir * 10;
}

// Handle mouse wheel movement
void mouseWheel(int button, int dir, int x, int y) {

	// Get modifier key state for keyboard events
	int mod = glutGetModifiers();

	// Check if the user is pressing the SHIFT key
	if (mod == GLUT_ACTIVE_SHIFT) {
		// Change number of iterations if true
		modifyIters(dir);
	} else {
		// Apply zoom on fractal if false
		zoom(dir, x, y);
	}

	// Call a redraw
	glutPostRedisplay();
}

#pragma endregion

// The main program function
int main(int argc, char *argv[]) {

	// Initialize OpenGL and CUDA interoperability
	initialize(argc, argv);

	// Let glut handle the main rendering loop
	glutMainLoop();

	// Release CUDA resources when closing the application
	atexit(exitCudaInterop);

	// Equivalent of "return 0"
	exit(EXIT_SUCCESS);
}
