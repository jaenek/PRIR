// Copyright (C) 2017 Basile Fraboni
// Copyright (C) 2014 Ivan Kutskir
// All Rights Reserved
// You may use, distribute and modify this code under the
// terms of the MIT license. For further details please refer
// to : https://mit-license.org/
//

//!
//! \file blur.cpp
//! \author Basile Fraboni
//! \date 2017
//!
//! \brief The software is a C++ implementation of a fast
//! Gaussian blur algorithm by Ivan Kutskir. For further details
//! please refer to :
//! http://blog.ivank.net/fastest-gaussian-blur.html
//!
//! Floating point single precision version
//!

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <cmath>
#include <chrono>
#include <mpi.h>
#include <omp.h>

void die(const char * msg) {
	puts(msg);
	exit(1);
}

//!
//! \fn void std_to_box(int boxes[], float sigma, int n)
//!
//! \brief this function converts the standard deviation of
//! Gaussian blur into dimensions of boxes for box blur. For
//! further details please refer to :
//! https://www.peterkovesi.com/matlabfns/#integral
//! https://www.peterkovesi.com/papers/FastGaussianSmoothing.pdf
//!
//! \param[out] boxes   boxes dimensions
//! \param[in] sigma    Gaussian standard deviation
//! \param[in] n        number of boxes
//!
void std_to_box(int boxes[], float sigma, int n)
{
    // ideal filter width
    float wi = std::sqrt((12*sigma*sigma/n)+1);
    int wl = std::floor(wi);
    if(wl%2==0) wl--;
    int wu = wl+2;

    float mi = (12*sigma*sigma - n*wl*wl - 4*n*wl - 3*n)/(-4*wl - 4);
    int m = std::round(mi);

    for(int i=0; i<n; i++)
        boxes[i] = ((i < m ? wl : wu) - 1) / 2;
}

//!
//! \fn void horizontal_blur_rgb(float * in, float * out, int w, int h, int c, int r)
//!
//! \brief this function performs the horizontal blur pass for box blur.
//!
//! \param[in,out] in       source channel
//! \param[in,out] out      target channel
//! \param[in] w            image width
//! \param[in] h            image height
//! \param[in] c            image channels
//! \param[in] r            box dimension
//!
void horizontal_blur_rgb(float * in, float * out, int w, int h, int c, int r)
{
    float iarr = 1.f / (r+r+1);
    for(int i=0; i<h; i++)
    {
        int ti = i*w;
        int li = ti;
        int ri = ti+r;

        float fv[3] = { in[ti*c+0], in[ti*c+1], in[ti*c+2] };
        float lv[3] = { in[(ti+w-1)*c+0], in[(ti+w-1)*c+1], in[(ti+w-1)*c+2] };
        float val[3] = { (r+1)*fv[0], (r+1)*fv[1], (r+1)*fv[2] };

        for(int j=0; j<r; j++)
        {
            val[0] += in[(ti+j)*c+0];
            val[1] += in[(ti+j)*c+1];
            val[2] += in[(ti+j)*c+2];
        }

        for(int j=0; j<=r; j++, ri++, ti++)
        {
            val[0] += in[ri*c+0] - fv[0];
            val[1] += in[ri*c+1] - fv[1];
            val[2] += in[ri*c+2] - fv[2];
            out[ti*c+0] = val[0]*iarr;
            out[ti*c+1] = val[1]*iarr;
            out[ti*c+2] = val[2]*iarr;
        }

        for(int j=r+1; j<w-r; j++, ri++, ti++, li++)
        {
            val[0] += in[ri*c+0] - in[li*c+0];
            val[1] += in[ri*c+1] - in[li*c+1];
            val[2] += in[ri*c+2] - in[li*c+2];
            out[ti*c+0] = val[0]*iarr;
            out[ti*c+1] = val[1]*iarr;
            out[ti*c+2] = val[2]*iarr;
        }

        for(int j=w-r; j<w; j++, ti++, li++)
        {
            val[0] += lv[0] - in[li*c+0];
            val[1] += lv[1] - in[li*c+1];
            val[2] += lv[2] - in[li*c+2];
            out[ti*c+0] = val[0]*iarr;
            out[ti*c+1] = val[1]*iarr;
            out[ti*c+2] = val[2]*iarr;
        }
    }
}
void horizontal_blur_rgb_omp(float * in, float * out, int w, int h, int c, int r)
{
    float iarr = 1.f / (r+r+1);
    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i<h; i++)
    {
        int ti = i*w;
        int li = ti;
        int ri = ti+r;

        float fv[3] = { in[ti*c+0], in[ti*c+1], in[ti*c+2] };
        float lv[3] = { in[(ti+w-1)*c+0], in[(ti+w-1)*c+1], in[(ti+w-1)*c+2] };
        float val[3] = { (r+1)*fv[0], (r+1)*fv[1], (r+1)*fv[2] };

        for(int j=0; j<r; j++)
        {
            val[0] += in[(ti+j)*c+0];
            val[1] += in[(ti+j)*c+1];
            val[2] += in[(ti+j)*c+2];
        }

        for(int j=0; j<=r; j++, ri++, ti++)
        {
            val[0] += in[ri*c+0] - fv[0];
            val[1] += in[ri*c+1] - fv[1];
            val[2] += in[ri*c+2] - fv[2];
            out[ti*c+0] = val[0]*iarr;
            out[ti*c+1] = val[1]*iarr;
            out[ti*c+2] = val[2]*iarr;
        }

        for(int j=r+1; j<w-r; j++, ri++, ti++, li++)
        {
            val[0] += in[ri*c+0] - in[li*c+0];
            val[1] += in[ri*c+1] - in[li*c+1];
            val[2] += in[ri*c+2] - in[li*c+2];
            out[ti*c+0] = val[0]*iarr;
            out[ti*c+1] = val[1]*iarr;
            out[ti*c+2] = val[2]*iarr;
        }

        for(int j=w-r; j<w; j++, ti++, li++)
        {
            val[0] += lv[0] - in[li*c+0];
            val[1] += lv[1] - in[li*c+1];
            val[2] += lv[2] - in[li*c+2];
            out[ti*c+0] = val[0]*iarr;
            out[ti*c+1] = val[1]*iarr;
            out[ti*c+2] = val[2]*iarr;
        }
    }
}

//!
//! \fn void total_blur_rgb(float * in, float * out, int w, int h, int c, int r)
//!
//! \brief this function performs the total blur pass for box blur.
//!
//! \param[in,out] in       source channel
//! \param[in,out] out      target channel
//! \param[in] w            image width
//! \param[in] h            image height
//! \param[in] c            image channels
//! \param[in] r            box dimension
//!
void total_blur_rgb(float * in, float * out, int w, int h, int c, int r)
{
     // radius range on either side of a pixel + the pixel itself
    float iarr = 1.f / (r+r+1);
    for(int i=0; i<w; i++)
    {
        int ti = i;
        int li = ti;
        int ri = ti+r*w;

        float fv[3] = {in[ti*c+0], in[ti*c+1], in[ti*c+2] };
        float lv[3] = {in[(ti+w*(h-1))*c+0], in[(ti+w*(h-1))*c+1], in[(ti+w*(h-1))*c+2] };
        float val[3] = {(r+1)*fv[0], (r+1)*fv[1], (r+1)*fv[2] };

        for(int j=0; j<r; j++)
        {
            val[0] += in[(ti+j*w)*c+0];
            val[1] += in[(ti+j*w)*c+1];
            val[2] += in[(ti+j*w)*c+2];
        }

        for(int j=0; j<=r; j++, ri+=w, ti+=w)
        {
            val[0] += in[ri*c+0] - fv[0];
            val[1] += in[ri*c+1] - fv[1];
            val[2] += in[ri*c+2] - fv[2];
            out[ti*c+0] = val[0]*iarr;
            out[ti*c+1] = val[1]*iarr;
            out[ti*c+2] = val[2]*iarr;
        }

        for(int j=r+1; j<h-r; j++, ri+=w, ti+=w, li+=w)
        {
            val[0] += in[ri*c+0] - in[li*c+0];
            val[1] += in[ri*c+1] - in[li*c+1];
            val[2] += in[ri*c+2] - in[li*c+2];
            out[ti*c+0] = val[0]*iarr;
            out[ti*c+1] = val[1]*iarr;
            out[ti*c+2] = val[2]*iarr;
        }

        for(int j=h-r; j<h; j++, ti+=w, li+=w)
        {
            val[0] += lv[0] - in[li*c+0];
            val[1] += lv[1] - in[li*c+1];
            val[2] += lv[2] - in[li*c+2];
            out[ti*c+0] = val[0]*iarr;
            out[ti*c+1] = val[1]*iarr;
            out[ti*c+2] = val[2]*iarr;
        }
    }
}

void total_blur_rgb_omp(float * in, float * out, int w, int h, int c, int r)
{
     // radius range on either side of a pixel + the pixel itself
    float iarr = 1.f / (r+r+1);
    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i<w; i++)
    {
        int ti = i;
        int li = ti;
        int ri = ti+r*w;

        float fv[3] = {in[ti*c+0], in[ti*c+1], in[ti*c+2] };
        float lv[3] = {in[(ti+w*(h-1))*c+0], in[(ti+w*(h-1))*c+1], in[(ti+w*(h-1))*c+2] };
        float val[3] = {(r+1)*fv[0], (r+1)*fv[1], (r+1)*fv[2] };

        for(int j=0; j<r; j++)
        {
            val[0] += in[(ti+j*w)*c+0];
            val[1] += in[(ti+j*w)*c+1];
            val[2] += in[(ti+j*w)*c+2];
        }

        for(int j=0; j<=r; j++, ri+=w, ti+=w)
        {
            val[0] += in[ri*c+0] - fv[0];
            val[1] += in[ri*c+1] - fv[1];
            val[2] += in[ri*c+2] - fv[2];
            out[ti*c+0] = val[0]*iarr;
            out[ti*c+1] = val[1]*iarr;
            out[ti*c+2] = val[2]*iarr;
        }

        for(int j=r+1; j<h-r; j++, ri+=w, ti+=w, li+=w)
        {
            val[0] += in[ri*c+0] - in[li*c+0];
            val[1] += in[ri*c+1] - in[li*c+1];
            val[2] += in[ri*c+2] - in[li*c+2];
            out[ti*c+0] = val[0]*iarr;
            out[ti*c+1] = val[1]*iarr;
            out[ti*c+2] = val[2]*iarr;
        }

        for(int j=h-r; j<h; j++, ti+=w, li+=w)
        {
            val[0] += lv[0] - in[li*c+0];
            val[1] += lv[1] - in[li*c+1];
            val[2] += lv[2] - in[li*c+2];
            out[ti*c+0] = val[0]*iarr;
            out[ti*c+1] = val[1]*iarr;
            out[ti*c+2] = val[2]*iarr;
        }
    }
}

//!
//! \fn void box_blur_rgb(float * in, float * out, int w, int h, int c, int r)
//!
//! \brief this function performs a box blur pass.
//!
//! \param[in,out] in       source channel
//! \param[in,out] out      target channel
//! \param[in] w            image width
//! \param[in] h            image height
//! \param[in] c            image channels
//! \param[in] r            box dimension
//!
void box_blur_rgb(float *& in, float *& out, int w, int h, int c, int r)
{
    std::swap(in, out);
    horizontal_blur_rgb(out, in, w, h, c, r);
    total_blur_rgb(in, out, w, h, c, r);
}

void box_blur_rgb_omp(float *& in, float *& out, int w, int h, int c, int r)
{
    std::swap(in, out);
    horizontal_blur_rgb_omp(out, in, w, h, c, r);
    total_blur_rgb_omp(in, out, w, h, c, r);
}

//!
//! \fn void fast_gaussian_blur_rgb(float * in, float * out, int w, int h, int c, float sigma)
//!
//! \brief this function performs a fast Gaussian blur. Applying several
//! times box blur tends towards a true Gaussian blur. Three passes are sufficient
//! for good results. For further details please refer to :
//! http://blog.ivank.net/fastest-gaussian-blur.html
//!
//! \param[in,out] in       source channel
//! \param[in,out] out      target channel
//! \param[in] w            image width
//! \param[in] h            image height
//! \param[in] c            image channels
//! \param[in] sigma        gaussian std dev
//!
void fast_gaussian_blur_rgb(float *& in, float *& out, int w, int h, int c, float sigma, int rank, int size, bool omp)
{
    // sigma conversion to box dimensions
    int boxes[3];
    std_to_box(boxes, sigma, 3);
	if (!omp) {
		box_blur_rgb(in, out, w, h, c, boxes[0]);
		box_blur_rgb(out, in, w, h, c, boxes[1]);
		box_blur_rgb(in, out, w, h, c, boxes[2]);
	} else if (omp) {
		box_blur_rgb_omp(in, out, w, h, c, boxes[0]);
		box_blur_rgb_omp(out, in, w, h, c, boxes[1]);
		box_blur_rgb_omp(in, out, w, h, c, boxes[2]);
	}
}

int main(int argc, char * argv[])
{
	int rank = 0, size = 1, width = 6240, height = 4160, channels = 3, isize = 0;
	unsigned char *image_data;
	float sigma, *old_image, *new_image;
	std::chrono::system_clock::time_point start;
    float tab[10];

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    omp_set_num_threads(2);

	sigma = argc > 1 ? std::atof(argv[1]) : 3.;
    for(int k = 0; k<10; k++)
    {
        if (rank == 0) {
            if( argc < 2 ) die("Error: Provide blur sigma as an argument.");
            const char * image_file = "./sander-crombach-6b3r1WAjPBI-unsplash.jpg";

            // image loading
            image_data = stbi_load(image_file, &width, &height, &channels, 0);
            std::cout << "Source image: " << width<<"x" << height << " ("<<channels<<")" << " sigma: " << sigma << std::endl;
            if(channels < 3)
            {
                std::cout<< "Input images must be RGB images."<<std::endl;
                exit(1);
            }

            // copy data
            isize = width * height * channels;

            // output channels r,g,b
            printf("isize: %d\n", isize);
            new_image = new float[isize];
            old_image = new float[isize];

            // channels copy r,g,b
            for(int i = 0; i < isize; ++i)
                old_image[i] = image_data[i] / 255.f;

            // per channel filter
            
        }

        if (width % size != 0 && rank == 0) {
            MPI_Finalize();
            die("Error: Image cannot be divided by the number of processes.\n");
        }
        int local_width = width / size;
        int local_size = local_width * height * channels;

        printf("width: %d, local: %d, rank: %d, size: %d\n", width, local_width, rank, size);

        float *local_in = new float[local_size];
        float *local_out = new float[local_size];
        
        if(rank == 0) {
            MPI_Scatter(old_image, local_size, MPI_FLOAT, local_in, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
        } else {
            MPI_Scatter(NULL, local_size, MPI_FLOAT, local_in, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }

        printf("local_in: %f, rank: %d\n", local_in[0], rank);
        start = std::chrono::system_clock::now();
        fast_gaussian_blur_rgb(local_in, local_out, local_width, height, channels, sigma, rank, size, false);
        
        if(rank==0){
            MPI_Gather(local_out, local_size, MPI_FLOAT, new_image, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gather(local_out, local_size, MPI_FLOAT, NULL, 0, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
        auto end = std::chrono::system_clock::now();

        delete[] local_in;
        delete[] local_out;

        if (rank == 0) {
            

            // stats
            float elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
            //std::cout << "time " << elapsed << "ms" << std::endl;
            tab[k] = elapsed;

            // channels copy r,g,b
            for(int i = 0; i < isize; ++i)
                image_data[i] = (unsigned char) std::min(255.f, std::max(0.f, 255.f * new_image[i]));

            // save
            const char * output_file = "blur_1.jpg";
            stbi_write_jpg(output_file, width, height, channels, image_data, 100);
            stbi_image_free(image_data);

            // clean memory
            delete[] new_image;
            delete[] old_image;
        }
    }
    if (rank == 0){
        float sum = 0;
        for(int i=0; i < 10; i++)
        {
            sum += tab[i];
        }
        float avg = sum/10;
        std::cout << "Source image: " << width<<"x" << height << " ("<<channels<<")" << " sigma: " << sigma << std::endl;
        std::cout << "time " << avg << "ms" << std::endl;
    }
	MPI_Finalize();

	return 0;
}
