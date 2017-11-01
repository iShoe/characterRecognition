// TrainAndTest.cpp

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <sstream>
#include <stdlib.h>

using namespace std;

//global variables
const int MIN_CONTOUR_AREA = 100;
const int RESIZE_IMAGE_WIDTH = 20;
const int RESIZE_IMAGE_HEIGHT = 30;


class ContourWithData {
public:
	vector<cv::Point> ptContour;
	cv::Rect boundingRect;
	float contArea; 

	bool checkIfContourIsValid() {
		if (contArea < MIN_CONTOUR_AREA) return false;
		return true;
	}

	static bool sortByBoundingRectXPosition(const ContourWithData &cwdLeft, const ContourWithData & cwdRight) {
		return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);
	}

};

class Image {
public:
	cv::Mat matClassifications;		//
	cv::Mat matImages;				//
	cv::Mat matTestingNumbers;		//
	cv::Mat matGrayscale;           //declarations of image variables
	cv::Mat matBlurred;				//
	cv::Mat matThresh;				//
	cv::Mat matThreshCopy;			//

	

	cv::Mat matROIResized;
	cv::Mat matROIFloat;
	cv::Mat matROIFlattenedFloat;


	//getting classifications
	void readClassifications() {
		cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::READ);

		if (fsClassifications.isOpened() == false) {                                                    // if the file was not opened successfully
			std::cout << "error, unable to open training classifications file, exiting program\n\n";    // show error message
			exit(0);                                                                                  // and exit program
		}

		fsClassifications["classifications"] >> matClassifications;      // read classifications section into Mat classifications variable
		fsClassifications.release();

	}

	//getting images
	void readImages() {
		cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::READ);          // open the training images file

		if (fsTrainingImages.isOpened() == false) {                                                 // if the file was not opened successfully
			std::cout << "error, unable to open training images file, exiting program\n\n";         // show error message
			exit(1);                                                                              // and exit program
		}

		fsTrainingImages["images"] >> matImages;           // read images section into Mat training images variable
		fsTrainingImages.release();
	}

	//opening test image 
	void readTest() {
		matTestingNumbers = cv::imread("test1.png");
		if (matTestingNumbers.empty()) {
			cout << "Error: Image not found from file\n\n";
			exit(1);
		}
	}

	//converts the training images to grayscale, blurred and threshold
	void converts() {
		cv::cvtColor(matTestingNumbers, matGrayscale, CV_BGR2GRAY);
		cv::GaussianBlur(matGrayscale, matBlurred, cv::Size(5, 5), 0); 
		cv::adaptiveThreshold(matBlurred, matThresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);
		matThreshCopy = matThresh.clone();
	
	}

	
};

int main() {
	Image i;

	vector<ContourWithData> allContoursWithData;
	vector<ContourWithData> validContoursWithData;

	i.readClassifications();
	i.readImages();
	
	//initializing k-nearest neighbours algorithm
	cv::Ptr<cv::ml::KNearest> KNearest(cv::ml::KNearest::create());
	KNearest->train(i.matImages, cv::ml::ROW_SAMPLE, i.matClassifications);


	i.readTest();
	i.converts();

	vector<vector<cv::Point>> ptContours;
	vector<cv::Vec4i> V4iHierarchy;

	cv::findContours(i.matThreshCopy, ptContours, V4iHierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < ptContours.size(); i++) {
		ContourWithData contourWithData;
		contourWithData.ptContour = ptContours[i];
		contourWithData.boundingRect = cv::boundingRect(contourWithData.ptContour);
		contourWithData.contArea = cv::contourArea(contourWithData.ptContour);
		allContoursWithData.push_back(contourWithData);
	}

	for (int i = 0; i < allContoursWithData.size(); i++) {                      // for all contours
		if (allContoursWithData[i].checkIfContourIsValid()) {                   // check if valid
			validContoursWithData.push_back(allContoursWithData[i]);            // if so, append to valid contour list
		}
	}

	// sort contours from left to right
	std::sort(validContoursWithData.begin(), validContoursWithData.end(), ContourWithData::sortByBoundingRectXPosition);


	string strFinalString; //final string in which resultant text is stored

	for (int j = 0; j < validContoursWithData.size(); j++) {

	cv::rectangle(i.matTestingNumbers, validContoursWithData[j].boundingRect, cv::Scalar(0, 255, 0), 2);

	cv::Mat matROI = i.matThresh(validContoursWithData[j].boundingRect); //region of interest ROI is got

	cv::Mat matROIResized; //ROI is resize to have unity
	cv::resize(matROI, matROIResized, cv::Size(RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT));

	cv::Mat matROIFloat;
	matROIResized.convertTo(matROIFloat, CV_32FC1);

	cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);

	cv::Mat matCurrentChar(0, 0, CV_32F);

	KNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);

	float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);
	strFinalString = strFinalString + char(int(fltCurrentChar));
}

	cout << "\nText: " << strFinalString << "\n\n"; // final string is printed to console
	cv::imshow("matTestingNumbers", i.matTestingNumbers);  //testing image is displayed
	cv::waitKey(0);
	



	return 0;

}
