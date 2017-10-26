// GenData.cpp
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <vector>
#include <stdlib.h>
using namespace std;

// GLOBAL VARIABLES
const int MIN_CONTOUR_AREA = 100; 
const int RESIZE_IMAGE_WIDTH = 20;
const int RESIZE_IMAGE_HEIGHT = 30;


/******************************************class*************************************************************/
class ImagesAndChars {

public:

	/*********************************declaration of data members******************************/

	cv::Mat imgTrainingNumbers; // 
	cv::Mat imgGrayscale;		//
	cv::Mat imgBlurred;			// declaration of various images needed
	cv::Mat imgThresh;			//
	cv::Mat imgThreshCopy;		//

	vector<vector<cv::Point>> ptContours;	// delcaration of contours 
	vector<cv::Vec4i> v4iHierarchy;

	
	cv::Mat matClassificationInts; // training classifications        //
															          // These are the files we need to import to testing program
	cv::Mat matTrainingImagesAsFlattenedFloats; // training images    //

	vector<int> intValidChars;					// characters we can accept		

	/***********************end of declaration of data members*************************/


	/***********************************constructor*************************************/

	ImagesAndChars() {                           // constructor to initialize valid chars
		intValidChars = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', // digits and capitals letters
			'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
			'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
			'U', 'V', 'W', 'X', 'Y', 'Z' };
	}


	/**************************member functions****************************/
	void readTrainingImage(){				
		imgTrainingNumbers = cv::imread("training_chars2.png"); // reading training image
		if (imgTrainingNumbers.empty()) {
			cout << "Error! Image not read\n" << endl;			// error checking
			exit(0);
		}
	}

	/********************************/


	void convert() {
		cv::cvtColor(imgTrainingNumbers, imgGrayscale, CV_BGR2GRAY); // Converting to grayscale
		cv::GaussianBlur(imgGrayscale, imgBlurred,cv::Size(5, 5), 0); // Blurring using gaussian filter 
		cv::adaptiveThreshold(imgBlurred, imgThresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2); //filter image from grayscale to black & white
	}


	/******************************/


	inline void showThresh() {
		cv::imshow("imgThresh", imgThresh);		// displaying threshold image
	}


	/******************************/

	inline void findContours() {
		cv::findContours(imgThreshCopy, ptContours, v4iHierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE); // Copy of threshold used since it will be modified
	}
	
	/******************************/


	void training() {

		for (int i = 0; i < ptContours.size(); i++) {                           // for each contour
			if (cv::contourArea(ptContours[i]) > MIN_CONTOUR_AREA) {                // if contour is big enough to consider
				cv::Rect boundingRect = cv::boundingRect(ptContours[i]);                // get the rectangle

				cv::rectangle(imgTrainingNumbers, boundingRect, cv::Scalar(0, 0, 255), 2);      // draw red rectangle around each contour as we train it

				cv::Mat matROI = imgThresh(boundingRect);           

				cv::Mat matROIResized;
				cv::resize(matROI, matROIResized, cv::Size(RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT));     // resize image, this will be easy to recognize and storage

				cv::imshow("matROI", matROI);                              
				cv::imshow("matROIResized", matROIResized);                
				cv::imshow("imgTrainingNumbers", imgTrainingNumbers);       

				int intChar = cv::waitKey(0);           // Only after pressing key it will proceed

				if (intChar == 27) {        // if esc key pressed
					exit(0);              // exit program
				}
				else if (std::find(intValidChars.begin(), intValidChars.end(), intChar) != intValidChars.end()) {     // else if the char is in the list of chars we are looking for . . .

					matClassificationInts.push_back(intChar);       // append classification char to integer list of chars

					cv::Mat matImageFloat;                          // now add the training image (some conversion is necessary first) . . .
					matROIResized.convertTo(matImageFloat, CV_32FC1);       // convert Mat to float

					cv::Mat matImageFlattenedFloat = matImageFloat.reshape(1, 1);       // flatten

					matTrainingImagesAsFlattenedFloats.push_back(matImageFlattenedFloat);       // add to Mat as though it was a vector, this is necessary due to the
																								// data types that KNearest.train accepts
				}   // end if
			}   // end if
		} // end for
	}

	/******************************/

	void saveClassification() {
		cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::WRITE);
		if (fsClassifications.isOpened() == false) {                                                        // if the file was not opened successfully
			std::cout << "error, unable to open training classifications file, exiting program\n\n";        // show error message
			exit(0);                                                                                      
		}

		fsClassifications << "classifications" << matClassificationInts;        // write classifications into classifications file
		fsClassifications.release();											// close
	}

	/******************************/

	void saveImages() {
		cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::WRITE);         // open the training images file

		if (fsTrainingImages.isOpened() == false) {                                                 // if the file was not opened successfully
			std::cout << "error, unable to open training images file, exiting program\n\n";         // show error message
			exit(0);                                                                              
		}

		fsTrainingImages << "images" << matTrainingImagesAsFlattenedFloats;         // write training images into images file
		fsTrainingImages.release();													// close

	}

	/************************end of member functions delcaration and definition***************************/

};

/************************************************end of class**************************************************/



int main() {
	ImagesAndChars i;
	i.readTrainingImage(); // Reading the training image
	i.convert();			  // Conversion into grayscale, blur & threshold
	i.showThresh();		  // display threshold image
	i.imgThreshCopy = i.imgThresh.clone(); // cloning thresh image because later imgThresh will be modified
	i.findContours();		// finding contours for training
	i.training();			// training process
	cout << "Training complete" << endl;


	// saving classification to file
	i.saveClassification();
	

	// saving images to file
	i.saveImages();

	return 0;
}
