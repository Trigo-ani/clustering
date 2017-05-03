/*
	PROJECT : Clustering Images Using OpenMP
	This Version (Simplified) : Clustering text files - Sequential
	
	DESCRIPTION: This program takes input the filenames and clusters them together based on 100% similarity matches.
	
	ENVIRONMENT: uBuntu/UNIX-Based, Windows
	NOTE: If you're using Windows, please make sure you're running the executable with Adminstrative Privileges, and all the permissions
		to the files to be accessed are granted to the program.
		
	-- Bala Kumar & Anirudh Trigunayat (149105348 & 149105168)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <list>
#include <vector>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>
#include <dirent.h>
#include <omp.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <opencv2/imgproc/imgproc.hpp>

#define LIMIT 0.3

using namespace cv;
using namespace std;


/* Other Utils */
int isEqual(string filea, string fileb) {
	
	Mat img1 = imread(filea, CV_LOAD_IMAGE_COLOR);
	Mat img2 = imread(fileb, CV_LOAD_IMAGE_COLOR);
	
	if (img1.total() != img2.total()) {
		return 0;
	}

	Mat result;

	Mat gray1, gray2;
	cvtColor(img1, gray1, CV_BGR2GRAY);
	cvtColor(img2, gray2, CV_BGR2GRAY);

	int threshold = (double)(gray1.rows * gray1.cols) * (1 - LIMIT); 

	compare(gray1 , gray2  , result , CMP_EQ );
//	cout << "Comparing " << filea << " and " << fileb << endl;
	int similarPixels  = countNonZero(result);
	if ( similarPixels  > threshold ) 
		return 1;
	else
		return 0;
}



int main(void) {
	int n = 0, clusters = -1, numthreads;
	short *flag;
	DIR *d;
	struct dirent *dir;
		
	d = opendir(".");
	while ((dir = readdir(d)) != NULL) {
		int len = strlen(dir->d_name);
		if (dir->d_type == DT_REG && len > 4 &&
        dir->d_name[len - 4] == '.' &&
        dir->d_name[len - 3] == 'j' &&
        dir->d_name[len - 2] == 'p' &&
        dir->d_name[len - 1] == 'g')
		{ /* If the entry is a regular file */
			n++;
		}
	}
	
	if(n == 0) {
		printf("No text files found. Please make sure there are some files in \'.txt\' format the working directory\n");
		exit(EXIT_FAILURE);
	}
	
	//Allocate the memory for storing the filenames, flags, and the clusters
	vector<string> files;
	list<string> *clusterEles = new list<string>[n];

	flag = (short *)calloc(sizeof(short), n);
	
	d = opendir(".");
	if (d)
	{
		for (int i = 0; (dir = readdir(d)) != NULL;) {
			int len = strlen(dir->d_name);
			if (dir->d_type == DT_REG && len > 4 &&
			dir->d_name[len - 4] == '.' &&
			dir->d_name[len - 3] == 'j' &&
			dir->d_name[len - 2] == 'p' &&
			dir->d_name[len - 1] == 'g') {
				string temp;
				for(i = 0; dir->d_name[i] != '\0'; i++)
					temp.push_back(dir->d_name[i]);

				files.push_back(temp);
				i++;
			}
		}
	}

	double start = omp_get_wtime();
	omp_set_nested(1);
	//Clustering
	for (int i = 0; i < n; i++) {
		//If the file exists in any other cluster, skip scanning.
		
		if (flag[i])
			continue;

		//If it doesnt, then create a new cluster and add file to the cluster.	
		clusters++;
		clusterEles[clusters].push_back(files[i]);
		flag[i] = 1;

		//Search the list for possible matches
		int t = 0;
		#pragma omp parallel for
		for (int j = i + 1; j < n; j++) {
		if(omp_get_thread_num() == 0)
			numthreads = omp_get_num_threads();
			if (!flag[j] && isEqual(files[i], files[j])) {
				#pragma omp critical
					clusterEles[clusters].push_back(files[j]);
				flag[j] = 1;
				
			}
		}
	}

	double end = omp_get_wtime();

	for (int j = 0; j <= clusters; j++) {
		printf("Cluster %d : \n", j + 1);
		for(string filename: clusterEles[j]) {
			printf("--> %s ", filename.c_str() );	
		}
		printf("\n");
	}
	printf("\nAverage execution time : %lf\nNumber of threads in outer level = %d\n\n", end - start, numthreads);

}



