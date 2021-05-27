// BasicOpenCLApplication.cpp : Defines the entry point for the console application.

#include "stdafx.h"
#include "OpenCLContext.h"
#include "Chrono.h"
#include <math.h>
#include <iostream>
#include <string>
#include <fstream>
using namespace std;

void SaveBMP(char *fname, unsigned char *image, int width, int height, int componentPerPixel = 1, int reverseColor = 0)
{
	FILE *destination;
	int i, j;
	int *pt;
	char name[512], hdr[0x36];
	unsigned char *imsource = new unsigned char[width*height * 3];
	//int al=(ImageSize*3)%4;

	if (componentPerPixel == 1)
		for (i = 0;i<width*height * 3;i++)
			imsource[i] = image[i / 3];
	else
		for (i = 0;i<width*height * 3;i++)
			imsource[i] = image[i];
	if (reverseColor)
		for (j = 0;j<height;j++)
			for (i = 0;i<width;i++)
			{
				unsigned char aux;
				aux = imsource[3 * (i + width*j)];
				imsource[3 * (i + width*j)] = imsource[3 * (i + width*j) + 2];
				imsource[3 * (i + width*j) + 2] = aux;
			}
	strcpy(name, fname);
	i = (int)strlen(name);
	if (!((i>4) && (name[i - 4] == '.') && (name[i - 3] == 'b') && (name[i - 2] == 'm') && (name[i - 1] == 'p')))
	{
		name[i] = '.';
		name[i + 1] = 'b';
		name[i + 2] = 'm';
		name[i + 3] = 'p';
		name[i + 4] = 0;
	}
	if ((destination = fopen(name, "wb")) == NULL)
		perror("erreur de creation de fichier\n");
	hdr[0] = 'B';
	hdr[1] = 'M';
	pt = (int *)(hdr + 2);// file size
	*pt = 0x36 + width*height * 3;
	pt = (int *)(hdr + 6);//reserved
	*pt = 0x0;
	pt = (int *)(hdr + 10);// image address
	*pt = 0x36;
	pt = (int *)(hdr + 14);// size of [0E-35]
	*pt = 0x28;
	pt = (int *)(hdr + 0x12);// Image width
	*pt = width;
	pt = (int *)(hdr + 0x16);// Image heigth
	*pt = height;
	pt = (int *)(hdr + 0x1a);// color planes
	*pt = 1;
	pt = (int *)(hdr + 0x1c);// bit per pixel
	*pt = 24;
	for (i = 0x1E;i<0x36;i++)
		hdr[i] = 0;
	fwrite(hdr, 0x36, 1, destination);
	fwrite(imsource, width*height * 3, 1, destination);
	fclose(destination);
	delete[] imsource;
}
/// Arrays and enum to be completed with the relevant kernel names!
enum KERNEL_CODE
{
	GRAPH_DRAWING = 0,
	LAST_KERNEL = 1
};
KernelDescriptor myKernelDecriptors[LAST_KERNEL] = {
	{ GRAPH_DRAWING,	"GRAPH_DRAWING", "" }
};
/// End of kernel specific info


void SimpleGraphDrawing(unsigned char *image, int dim[2], double range[2][2])
{
	Chrono c;
	for (int j = 0;j<dim[1];j++)
		for (int i = 0;i<dim[0];i++)
		{
			float x = range[0][0] + (i + 0.5)*(range[0][1] - range[0][0]) / dim[0]; //Create x coordinates within the range [range[0][0] .. range[0][1]] 
			float y = range[1][0] + (j + 0.5)*(range[1][1] - range[1][0]) / dim[1]; //Create x coordinates within the range [range[1][0] .. range[1][1]] 
			float val = (x*x + y*y - 1);
			val = val*val*val - x*x*y*y*y;
			image[j*dim[0] + i] = (val>0) * 255; //setting up the (i,j) pixel value to either 0 or 255
		}
	c.PrintElapsedTime("time CPU (s): ");
}

void SimpleGraphDrawingGPU(OpenCLContext &context, unsigned char *image, int dim[2], double range[2][2])
{
	Chrono c;
	int n = dim[0] * dim[1];
	int blocking = true;
	double range_1d[4] = { range[0][0],range[0][1],range[1][0],range[1][1]};
	cl_int graphKernel = GRAPH_DRAWING;
	cl_int error;

	size_t globalWorkSize[2] = { 64,64 };

	//reserving space in GPU memory for arguments
	cl_mem buffer = clCreateBuffer(context.GetContext(), CL_MEM_READ_WRITE, n * sizeof(char), NULL, &error);
	cl_mem buffer_dim = clCreateBuffer(context.GetContext(), CL_MEM_READ_WRITE, 2 * sizeof(int), NULL, &error);
	cl_mem buffer_range = clCreateBuffer(context.GetContext(), CL_MEM_READ_WRITE, 4 * sizeof(double), NULL, &error);

	//writing arguments to GPU memory
	error = clEnqueueWriteBuffer(context.GetCommandQueue(0), buffer, blocking, 0, n * sizeof(char), image, 0, NULL, NULL);
	error = clEnqueueWriteBuffer(context.GetCommandQueue(0), buffer_dim, blocking, 0, 2 * sizeof(int), dim, 0, NULL, NULL);
	error = clEnqueueWriteBuffer(context.GetCommandQueue(0), buffer_range, blocking, 0, 4 * sizeof(double), range_1d, 0, NULL, NULL);

	//setting kernel arguments
	error = clSetKernelArg(context.GetKernel(graphKernel), 0, sizeof(cl_mem), &buffer);
	error = clSetKernelArg(context.GetKernel(graphKernel), 1, sizeof(cl_mem), &buffer_dim);
	error = clSetKernelArg(context.GetKernel(graphKernel), 2, sizeof(cl_mem), &buffer_range);

	//running kernel and reading back results
	error = clEnqueueNDRangeKernel(context.GetCommandQueue(0), context.GetKernel(graphKernel), 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	clFinish(context.GetCommandQueue(0));
	error = clEnqueueReadBuffer(context.GetCommandQueue(0), buffer, blocking, 0, n*sizeof(char), image, 0, NULL, NULL);
	error = clReleaseMemObject(buffer);
	error = clReleaseMemObject(buffer_dim);
	error = clReleaseMemObject(buffer_range);

	c.PrintElapsedTime("time GPU (s): ");
	return;
}

void handle_user_input_formula(OpenCLContext context) {

	//Creates input file streams
	ifstream cl_template("KernelTemplate.cl");
	ifstream cl_formula("formula.cfg");

	//creates output stream for temporary file
	ofstream temp_cl_code("TempKernel.cl");

	string markerstring = "//m";
	string tempstring, tempstring2;

	//copy everything from MyKernels.cl to a temporary file
	//insert the formula from formula.cfg at the marker
	while (cl_template >> tempstring)
	{
		printf("while loop run");
		if (tempstring != markerstring) {
			temp_cl_code << tempstring;
			temp_cl_code << " ";
		} else {
			while (cl_formula >> tempstring2) {
				temp_cl_code << tempstring2;
				temp_cl_code << " ";
			}
		}
	}

	//close file streams
	cl_template.close();
	cl_formula.close();
	temp_cl_code.close();

	context.ReadKernelFile("TempKernel.cl");
}

int main(int argc, char* argv[])
{
	OpenCLContext context;

	context.ReadKernelFile("MyKernels.cl");
	//handle_user_input_formula(context);

	context.BuildKernels(myKernelDecriptors, LAST_KERNEL);

	int dims[2] = { 512,512 };
	double range[2][2] = { { -1.4,1.4 },{ -1.1,1.3 } };

	//read dimension and range value from file
	std::fstream valuesfile("values.cfg", std::ios_base::in);
	valuesfile >> dims[0] >> dims[1] >> range[0][0] >> range[0][1] >> range[1][0] >> range[1][1];
	valuesfile.close();

	//print new dimension and range values
	printf("dimensions: %ix%i\n", dims[0], dims[1]);
	printf("range: %.6f to %.6f, %.6f to %.6f\n", range[0][0], range[0][1], range[1][0], range[1][1]);

	unsigned char *image = new unsigned char[dims[0] * dims[1]];
	SetConsoleColor(14);
	SimpleGraphDrawing(image, dims, range); //largest 64bit prime
	SetConsoleColor(13);
	SimpleGraphDrawingGPU(context, image, dims, range);
	//printf("graph 0: %i\n", image[0]); //debugging
	//printf("graph 5: %i\n", image[5]); //debugging
	//printf("graph 10: %i\n", image[10]); //debugging
	SaveBMP("graph.bmp", image, dims[0], dims[1]);
	delete[] image;
	return 0;
}
/*
•30 % : OpenCL code is correct, simple and working.
•25% : Kernel  is  properly  called  and  the  right  parameters  are  passed.Parameters  should  include  the(x, y) sampling range as well as the image dimensions.
•15% : Data transfer between CPU and GPU is correct.
•10% : A 2D grid of computation is used and enough parallelism is generated.
•15% : The  final  formula  is  taken  from  a  user  input(e.g.a  file)  instead  of  being  hardcoded, as  well  as  the coordinate range.For marking purpose make sure you have a default equation working.
•5% : Submission follows guidelines and does not involve extra work(e.g.correcting code).

results: Mention running times here. May be similar

1024 x 1024 results:

time CPU (s): 0.040 s
time GPU (s): 0.003 s

16384 x 16384 results:

time CPU (s): 10.005 s
time GPU (s): 0.337 s

*/