#ifndef RGM_UTILOPENCV_HPP_
#define RGM_UTILOPENCV_HPP_

#include "common.hpp"

namespace RGM {

static const int rgbTableSz = 44;
static cv::Scalar rgbTable[44] = { cv::Scalar(255, 64, 64), cv::Scalar(242, 222,
		182), cv::Scalar(115, 153, 130), cv::Scalar(38, 45, 51), cv::Scalar(217,
		64, 255), cv::Scalar(191, 150, 143), cv::Scalar(153, 130, 38),
		cv::Scalar(0, 255, 170), cv::Scalar(61, 109, 242), cv::Scalar(226, 242,
				0), cv::Scalar(0, 153, 102), cv::Scalar(134, 140, 179),
		cv::Scalar(102, 0, 54), cv::Scalar(255, 0, 170), cv::Scalar(102, 27, 0),
		cv::Scalar(51, 23, 13), cv::Scalar(75, 77, 57), cv::Scalar(19, 77, 73),
		cv::Scalar(204, 102, 156), cv::Scalar(255, 162, 128), cv::Scalar(195,
				217, 108), cv::Scalar(23, 13, 51), cv::Scalar(128, 96, 113),
		cv::Scalar(191, 77, 0), cv::Scalar(0, 112, 140), cv::Scalar(97, 0, 242),
		cv::Scalar(217, 54, 98), cv::Scalar(64, 255, 64), cv::Scalar(143, 182,
				191), cv::Scalar(68, 45, 89), cv::Scalar(229, 149, 57),
		cv::Scalar(13, 51, 18), cv::Scalar(0, 68, 128), cv::Scalar(51, 0, 7),
		cv::Scalar(76, 51, 0), cv::Scalar(191, 255, 208), cv::Scalar(234, 191,
				255), cv::Scalar(140, 70, 79), cv::Scalar(0, 202, 217),
		cv::Scalar(46, 102, 26), cv::Scalar(127, 89, 64), cv::Scalar(255, 191,
				208), cv::Scalar(159, 96, 191), cv::Scalar(48, 124, 191) };

struct warpParam {
	warpParam();

	void init();
	void update();

	int num_;
	Scalar noise_;
	Scalar angle_;
	Scalar shift_;
	Scalar scale_;
};

class OpencvUtil {
public:
	/// Visualize HOG
	static Mat pictureHOG(cv::Mat_<Scalar> & filter, int bs);

	/// Rotates
	static Mat rotate(Mat m, float degree);

	/// Get subarray
	static Mat subarray(Mat img, cv::Rect roi, float padFactor, int padType);
	static Mat subarray(Mat img, cv::Rect roi, int padType);

	static vector<Mat> warp(Mat img, cv::Rect roi, const warpParam &param);
	static Mat warp(Mat img, const cv::Mat_<Scalar> & H,
			const cv::Rect_<Scalar> & bbox);

	static vector<Mat> showEigenMatrix(const vector<Matrix> & matrix,
			bool show = false);
	static Mat showEigenMatrix(const Matrix & m, bool show = false);

	static vector<Mat> showColorMap(vector<Matrix> & matrix, bool show = false);
	static Mat showColorMap(const Matrix & m, bool show = false);

	static void NCC(Mat &_imgI, Mat &_imgJ, vector<cv::Point2f> &_points0,
			vector<cv::Point2f> &_points1, vector<unsigned char> &_status,
			vector<float> &_match, int winsize, int method);

	static void euclideanDistance(vector<cv::Point2f> &_point1,
			vector<cv::Point2f> &_point2, vector<float> &match);

	static Mat floatColorImg(Mat img);

	// imgList: idx label imageName
	static Mat computeMeanImg(const string & rootDir, const string & imgList,
            cv::Size sz, string & delimeter);

};
// class OpencvUtil

template<typename T>
class OpencvUtil_ {
public:
	/// Resize a 3D matrix
	static cv::Mat_<T> resize(cv::Mat_<T> & m, float factor, int method);

	static cv::Mat_<T> linspace(const T s, const T e, const T step);
	static cv::Mat_<T> ntuples(const cv::Mat_<T> & mat1,
			const cv::Mat_<T> & mat2);

	/// Sample points
	static vector<cv::Point2f> getTrackPoints(cv::Rect_<T> bbox, int numRow,
			int numCol, int margin);

};
// class OpencvUtil_

}// namespace RGM

#endif // RGM_UTILOPENCV_HPP_
