#include <boost/math/special_functions.hpp>

#include "omp.h"

#include "util/UtilOpencv.hpp"

namespace RGM {

warpParam::warpParam() {
	init();
}

void warpParam::init() {
	num_ = 50;
	noise_ = 5;
	angle_ = 20;
	shift_ = 0.02F;
	scale_ = 0.02F;
}

void warpParam::update() {
	num_ = 20;
	noise_ = 5;
	angle_ = 10;
	shift_ = 0.02F;
	scale_ = 0.02F;
}

// -------- OpencvUtil -------

Mat OpencvUtil::pictureHOG(cv::Mat_<Scalar> & filter, int bs) {
	// construct a "glyph" for each orientaion
	cv::Mat_ < Scalar > bim1(bs, bs, Scalar(0));

	int hbs = floor(bs / 2.0f + 0.5f);
	cv::Range rg(hbs - 1, hbs + 1);
	bim1.colRange(rg) = 1;

	int ht = filter.size[0];
	int wd = filter.size[1];
	int numOri = filter.size[2];

	float degreeUnit = (float) 180.0F / numOri;

	vector < cv::Mat_<Scalar> > bim(numOri);
	bim[0] = bim1;

	for (int i = 1; i < numOri; ++i) {
		bim[i] = OpencvUtil::rotate(Mat(bim1), -i * degreeUnit);
	}

	// make pictures of positive weights bs adding up weighted glyphs
	cv::Mat_ < Scalar > posf = cv::max(filter, 0);
	cv::Mat_ < Scalar > img(bs * ht, bs * wd, Scalar(0));
	Scalar maxV = -1;
	for (int i = 0; i < ht; ++i) {
		for (int j = 0; j < wd; ++j) {
			for (int k = 0; k < numOri; ++k) {
				img(cv::Rect(j * bs, i * bs, bs, bs)) +=
						(bim[k] * posf(i, j, k));
				maxV = max < Scalar > (maxV, posf(i, j, k));
			}
		}
	}

	if (maxV > 1e-5) {
		img *= (255.0F / maxV);
	}

	/*Mat imgShow;
	 cv::normalize(img, imgShow, 255, 0.0, CV_MINMAX, CV_8UC1);

	 cv::imshow("Debug", imgShow);
	 cv::waitKey(0);*/

	return Mat(img);
}

Mat OpencvUtil::rotate(Mat m, float degree) {
	cv::Point2f src_center(m.cols / 2.0F, m.rows / 2.0F);
	Mat rot_mat = cv::getRotationMatrix2D(src_center, degree, 1.0);
	Mat dst;
	cv::warpAffine(m, dst, rot_mat, m.size());
	return dst;
}

Mat OpencvUtil::subarray(Mat img, cv::Rect roi, float padFactor, int padType) {

	int ht = ROUND(roi.height * (1 + padFactor));
	int wd = ROUND(roi.width * (1 + padFactor));

	int padx = (wd - roi.width) / 2;
	int padx1 = wd - roi.width - padx;
	int pady = (ht - roi.height) / 2;
	int pady1 = ht - roi.height - pady;

	int x1 = roi.x - padx;
	int x2 = roi.br().x + padx1;
	int y1 = roi.y - pady;
	int y2 = roi.br().y + pady1;

	cv::Rect newroi(x1, y1, x2 - x1, y2 - y1);

	return subarray(img, newroi, padType);
}

Mat OpencvUtil::subarray(Mat img, cv::Rect roi, int padType) {
	int x1 = roi.x;
	int x2 = roi.br().x;
	int y1 = roi.y;
	int y2 = roi.br().y;

	if (x1 >= 0 && y1 >= 0 && x2 <= img.cols && y2 <= img.rows) {
		return img(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
	} else {
		Mat subImg(y2 - y1, x2 - x1, img.type(), cv::Scalar::all(0));

		if (padType == 1) {
			for (int y = y1; y < y2; ++y) {
				int yy = min<int>(img.rows - 1, max<int>(0, y));
				for (int x = x1; x < x2; ++x) {
					int xx = min<int>(img.cols - 1, max<int>(0, x));

					if (img.channels() == 3) {
						subImg.at < cv::Vec3b > (y - y1, x - x1) = img.at
								< cv::Vec3b > (yy, xx);
					} else {
						subImg.at<unsigned char>(y - y1, x - x1) = img.at<
								unsigned char>(yy, xx);
					}
				}
			}
		} else {
			int xx1 = max<int>(0, x1);
			int xx2 = min<int>(img.cols, x2);
			int yy1 = max<int>(0, y1);
			int yy2 = min<int>(img.rows, y2);
			int wd = xx2 - xx1;
			int ht = yy2 - yy1;

			int padx = xx1 - x1;
			int pady = yy1 - y1;

			img(cv::Rect(xx2, yy1, wd, ht)).copyTo(
					subImg(cv::Rect(padx, pady, wd, ht)));
		}

		return subImg;
	}
}

vector<Mat> OpencvUtil::warp(Mat img, cv::Rect roi, const warpParam &param) {
	vector < Mat > wImgs(param.num_);

    Mat roiImg = subarray(img, roi, 1);
    wImgs[0] = roiImg.clone();

    if ( param.num_ == 1) return wImgs;

	int ctx = roi.x + ROUND(roi.width / 2.0F);
	int cty = roi.y + ROUND(roi.height / 2.0F);

	cv::Mat_ < Scalar > Sh1(3, 3);
	Sh1 << 1, 0, -ctx, 0, 1, -cty, 0, 0, 1;

	cv::Rect_ < Scalar
			> box(-roi.width / 2.0F, -roi.height / 2.0F, roi.width, roi.height);

	vector < cv::Mat_<Scalar> > allH(param.num_);

//	int baseSeed = rand();
//    srand(3);

    for (int i = 1; i < param.num_; ++i) {

//		srand(i * baseSeed);

		Scalar s = 1.0F - param.scale_ * (rand() / (Scalar) RAND_MAX - 0.5F);
		cv::Mat_ < Scalar > Sca(3, 3);
		Sca << s, 0, 0, 0, s, 0, 0, 0, 1.0F;

		Scalar a = 2 * M_PI / 360.0F * param.angle_
				* (rand() / (Scalar) RAND_MAX - 0.5F);
		Scalar ca = cos(a);
		Scalar sa = sin(a);

		cv::Mat_ < Scalar > Ang(3, 3);
		Ang << ca, -sa, 0, sa, ca, 0, 0, 0, 1;

		Scalar shR = param.shift_ * roi.height
				* (rand() / (Scalar) RAND_MAX - 0.5F);
		Scalar shC = param.shift_ * roi.width
				* (rand() / (Scalar) RAND_MAX - 0.5F);

		cv::Mat_ < Scalar > Sh2(3, 3);
		Sh2 << 1, 0, shC, 0, 1, shR, 0, 0, 1;

		allH[i] = Sh2 * Ang * Sca * Sh1;
	}

#pragma omp parallel for
    for (int i = 1; i < param.num_; ++i) {
		cv::Mat_ < Scalar > invH = allH[i].inv(cv::DECOMP_SVD);
		wImgs[i] = warp(img, invH, box);
	} // for i

#if 0
	for (int i=0; i< wImgs.size(); ++i ) {
		cv::imshow("warpped", wImgs[i]);
		cv::waitKey(0);
	}
#endif

	return wImgs;

}

Mat OpencvUtil::warp(Mat img, const cv::Mat_<Scalar> & H,
		const cv::Rect_<Scalar> & bbox) {
	int wd = bbox.width;
	int ht = bbox.height;
	int ch = img.channels();

	assert(ch == 1 || ch == 3);

	Mat warpedImg(ht, wd, img.type());

	Scalar curx, cury, curz, wx, wy, wz, ox, oy, oz;
	int x, y;
	float xx, yy;

	ox = H(0, 2);
	oy = H(1, 2);
	oz = H(2, 2);

	yy = bbox.y;
	for (int j = 0; j < ht; j++) {
		//calculate x, y for current row
		curx = H(0, 1) * yy + ox;
		cury = H(1, 1) * yy + oy;
		curz = H(2, 1) * yy + oz;
		xx = bbox.x;
		yy = yy + 1;
		for (int i = 0; i < wd; i++) {
			// calculate x, y in current column
			wx = H(0, 0) * xx + curx;
			wy = H(1, 0) * xx + cury;
			wz = H(2, 0) * xx + curz;
			wx /= wz;
			wy /= wz;
			xx = xx + 1;

			x = floor(wx);
			y = floor(wy);

			if (x >= 0 && y >= 0) {
				wx -= x;
				wy -= y;
				if (x + 1 == img.cols && wx == 1)
					x--;
				if (y + 1 == img.rows && wy == 1)
					y--;

				if ((x + 1) < img.cols && (y + 1) < img.rows) {
					if (ch == 1) {
						warpedImg.at < uchar > (j, i) = cv::saturate_cast
								< uchar
								> ((img.at < uchar > (y, x) * (1 - wx) + img.at
										< uchar > (y, x + 1) * wx) * (1 - wy)
										+ (img.at < uchar
												> (y + 1, x) * (1 - wx) + img.at
												< uchar > (y + 1, x + 1) * wx)
												* wy);
					} else {
						for (int c = 0; c < 3; ++c) {
							warpedImg.at < cv::Vec<uchar, 3> > (j, i)[c] =
									cv::saturate_cast < uchar
											> ((img.at < cv::Vec<uchar, 3>
													> (y, x)[c] * (1 - wx)
															+ img.at
													< cv::Vec<uchar, 3>
													> (y, x + 1)[c] * wx)
													* (1 - wy)
													+ (img.at
															< cv::Vec<uchar, 3>
															> (y + 1, x)[c]
																	* (1 - wx)
																	+ img.at
															< cv::Vec<uchar, 3>
															> (y + 1, x + 1)[c]
																	* wx) * wy);
						}
					}
				} //if
			} //if
		} //for i
	} //for j

	return warpedImg;
}

vector<Mat> OpencvUtil::showEigenMatrix(const vector<Matrix> & matrix,
		bool show) {
	vector < Mat > imgs(matrix.size());
	for (int i = 0; i < matrix.size(); ++i) {
		imgs[i] = showEigenMatrix(matrix[i], show);
	}
}

Mat OpencvUtil::showEigenMatrix(const Matrix & m, bool show) {
	Matrix m1 = m;

	Mat mat;
	cv::eigen2cv(m1, mat);

	Scalar Inf = numeric_limits < Scalar > ::infinity();

	Mat imgShow;
	cv::normalize(mat, imgShow, 255.0F, 0.0F, cv::NORM_MINMAX,
			CV_8UC(mat.channels()), mat > -Inf);

	if (show) {
		cv::namedWindow("EigenMatrix");
		cv::imshow("EigenMatrix", imgShow);
		cv::waitKey(0);
	}

	return imgShow;
}

vector<Mat> OpencvUtil::showColorMap(vector<Matrix> & matrix, bool show) {
	Scalar Inf = numeric_limits < Scalar > ::infinity();
	Scalar eps = numeric_limits < Scalar > ::epsilon();

	vector < pair<Scalar, Scalar> > minMaxVal(matrix.size());
	Scalar minv = Inf;
	Scalar maxv = -Inf;
	for (int i = 0; i < matrix.size(); ++i) {
		Matrix &m(matrix[i]);

		m = (m.array() == Inf).select(-Inf, m);
		minMaxVal[i].first = m.maxCoeff();
		maxv = max < Scalar > (maxv, minMaxVal[i].first);

		m = (m.array() == -Inf).select(Inf, m);
		minMaxVal[i].second = m.minCoeff();
		minv = min < Scalar > (minv, minMaxVal[i].second);
	}

	for (int i = 0; i < matrix.size(); ++i) {
		Matrix &m(matrix[i]);

		m = (m.array() == Inf).select(minv, m);
		m = (m.array() - minv) / (maxv - minv + eps) * 255;
	}

	vector < Mat > imgs(matrix.size());
	for (int i = 0; i < matrix.size(); ++i) {
		imgs[i] = showColorMap(matrix[i], show);
	}

	return imgs;
}

Mat OpencvUtil::showColorMap(const Matrix & m, bool show) {
	Mat mat;
	cv::eigen2cv(m, mat);

	Mat mat1;
	mat.convertTo(mat1, CV_8UC1);

	Mat cmap;
	cv::applyColorMap(mat1, cmap, cv::COLORMAP_JET);

	if (show) {
		cv::namedWindow("HotMap");
		cv::imshow("HotMap", cmap);
		cv::waitKey(0);
	}

	return cmap;
}

void OpencvUtil::NCC(Mat &_imgI, Mat &_imgJ, vector<cv::Point2f> &_points0,
		vector<cv::Point2f> &_points1, vector<unsigned char> &_status,
		vector<float> &_match, int winsize, int method) {
	cv::Size sz(winsize, winsize);

	Mat rec0(sz, CV_8UC1, cv::Scalar::all(0));
	Mat rec1(sz, CV_8UC1, cv::Scalar::all(0));
	Mat res(1, 1, CV_32FC1, cv::Scalar::all(0));

	int nPts = _points0.size();

	for (int i = 0; i < nPts; i++) {
		if (_status[i] == 1) {
			cv::getRectSubPix(_imgI, sz, _points0[i], rec0);
			cv::getRectSubPix(_imgJ, sz, _points1[i], rec1);
			cv::matchTemplate(rec0, rec1, res, method);

			_match[i] = res.at<float>(0, 0);

		} else {
			_match[i] = 0.0;
		}
	}
}

void OpencvUtil::euclideanDistance(vector<cv::Point2f> &_point1,
		vector<cv::Point2f> &_point2, vector<float> &match) {
	assert(_point1.size() == _point2.size());

	match.resize(_point1.size(), 0);

	for (int i = 0; i < _point1.size(); ++i) {
		match[i] = sqrt(
				(_point1[i].x - _point2[i].x) * (_point1[i].x - _point2[i].x)
						+ (_point1[i].y - _point2[i].y)
								* (_point1[i].y - _point2[i].y));
	}
}

Mat OpencvUtil::floatColorImg(Mat img) {

	Mat imgf;
	img.convertTo(imgf, CV_MAKE_TYPE(cv::DataDepth < Scalar > ::value, 3));

	return imgf;
}

Mat OpencvUtil::computeMeanImg(const string & rootDir, const string & imgList,
        cv::Size sz, string & delimeter) {

	const int numThreads = omp_get_max_threads();

	vector < string > imgNames;

	std::ifstream ifs(imgList.c_str(), std::ios::in);
	string line;
	vector < string > items;
	while (std::getline(ifs, line)) {
        boost::split(items, line, boost::is_any_of("\t"));
		imgNames.push_back(items.back());
	}

	int numImgs = imgNames.size();
	if (numImgs == 0) {
		return Mat();
	}

	vector < Mat > meanImgs(numThreads);
	vector<int> counts(numThreads, 1);

    Mat tmp;
	for (int i = 0; i < numThreads && i < numImgs; ++i) {
		string filename = rootDir + imgNames[i];
		Mat img = cv::imread(filename, cv::IMREAD_COLOR);        
		if (img.cols != sz.width || img.rows != sz.height) {
            cv::resize(img, tmp, sz, 0, 0, RGM_IMG_RESIZE);
        } else {
            tmp = img;
        }
        meanImgs[i] = floatColorImg(tmp);
	}

#pragma omp parallel for num_threads(numThreads)
    for (int i = numThreads; i < numImgs; ++i) {
		int tid = omp_get_thread_num();
		string filename = rootDir + imgNames[i];
		Mat img = cv::imread(filename, cv::IMREAD_COLOR);
        Mat tmp;
		if (img.cols != sz.width || img.rows != sz.height) {
            cv::resize(img, tmp, sz, 0, 0, RGM_IMG_RESIZE);
        } else {
            tmp = img;
        }
        meanImgs[tid] += floatColorImg(tmp);
		counts[tid]++;
	}

	Mat meanImg = meanImgs[0];
	float numTotal = counts[0];
	for (int i = 1; i < numThreads && i < numImgs; ++i) {
		meanImg += meanImgs[i];
		numTotal += counts[i];
	}

	meanImg /= numTotal;

	return meanImg;

}

// -------- OpencvUtil_ -------

template<typename T>
cv::Mat_<T> OpencvUtil_<T>::resize(cv::Mat_<T> & m, float factor, int method) {
	int ht = floor(m.size[0] * factor + 0.5f);
	int wd = floor(m.size[1] * factor + 0.5f);

	int dims[] = { ht, wd, m.size[2] };

	cv::Mat_ < T > result(3, dims);

	for (int d = 0; d < dims[2]; ++d) {
		Mat m1(m.size[0], m.size[1],
				CV_MAKE_TYPE(cv::DataDepth < T > ::value, 1));
		for (int r = 0; r < m1.rows; ++r) {
			for (int c = 0; c < m1.cols; ++c) {
				m1.at < T > (r, c) = m(r, c, d);
			}
		}

		Mat rm;
		cv::resize(m1, rm, cv::Size(wd, ht), 0, 0, method);

		for (int r = 0; r < ht; ++r) {
			for (int c = 0; c < wd; ++c) {
				result(r, c, d) = rm.at < T > (r, c);
			}
		}
	}

	return result;
}

template<typename T>
vector<cv::Point2f> OpencvUtil_<T>::getTrackPoints(cv::Rect_<T> bbox,
		int numRow, int numCol, int margin) {
	vector < cv::Point2f > pts;

	if (numRow <= 0 || numCol <= 0 || margin <= 0)
		return pts;

	cv::Rect_ < T
			> newBbox(bbox.x + margin, bbox.y + margin, bbox.width - margin * 2,
					bbox.height - margin * 2);

	if (newBbox.width <= 0 || newBbox.height <= 0)
		return pts;

	if (numRow == 1 && numCol == 1) {
		pts.resize(1);

		pts[0].x = float(newBbox.x + newBbox.br().x) / 2.0f;
		pts[0].y = float(newBbox.y + newBbox.br().y) / 2.0f;
		return pts;
	}

	cv::Mat_<float> ntuple;

	if (numRow == 1 && numCol > 1) {
		cv::Mat_<float> tmp(1, 2);
		tmp(0, 0) = float(newBbox.x + newBbox.br().x) / 2.0f;
		tmp(0, 1) = float(newBbox.y + newBbox.br().y) / 2.0f;
		float stepW = newBbox.width / float(numCol - 1);

		ntuple = OpencvUtil_<float>::ntuples(tmp,
				OpencvUtil_<float>::linspace(newBbox.x, newBbox.br().x, stepW));
	} else if (numRow > 1 && numCol == 1) {
		cv::Mat_<float> tmp(1, 2);
		tmp(0, 0) = float(newBbox.x + newBbox.br().x) / 2.0f;
		tmp(0, 1) = float(newBbox.y + newBbox.br().y) / 2.0f;
		float stepH = newBbox.height / float(numRow - 1);

		ntuple = OpencvUtil_<float>::ntuples(tmp,
				OpencvUtil_<float>::linspace(newBbox.y, newBbox.br().y, stepH));
	} else {
		float stepW = newBbox.width / float(numCol - 1);
		float stepH = newBbox.height / float(numRow - 1);

		ntuple = OpencvUtil_<float>::ntuples(
				OpencvUtil_<float>::linspace(newBbox.x, newBbox.br().x, stepW),
				OpencvUtil_<float>::linspace(newBbox.y, newBbox.br().y, stepH));
	}

	pts.resize(ntuple.cols);
	for (int i = 0; i < pts.size(); ++i) {
		pts[i].x = ntuple.at<float>(0, i);
		pts[i].y = ntuple.at<float>(1, i);
	}

	return pts;
}

template<typename T>
cv::Mat_<T> OpencvUtil_<T>::linspace(const T s, const T e, const T step) {
	cv::Mat_ < T > matRange;

	if (step <= numeric_limits < T > ::epsilon())
		return matRange;

	int num = 0;
	T start = s;
	for (; start <= e; start += step, ++num)
		;

	if (num == 0)
		return matRange;

	matRange.create(1, num);

	start = s;
	for (int i = 0; i < num; ++i, start += step) {
		matRange(0, i) = start;
	}

	return matRange;
}

template<typename T>
cv::Mat_<T> OpencvUtil_<T>::ntuples(const cv::Mat_<T> & mat1,
		const cv::Mat_<T> & mat2) {
	if (mat1.empty() || mat2.empty())
		return cv::Mat_<T>();

	int num_col = mat1.cols;
	int num_item = mat2.cols;

	int num_total = num_col * num_item;

	cv::Mat_ < T > ntuple(mat1.rows + mat2.rows, num_total);

	for (int i = 0, cIdx = 0; i < num_col; ++i) {
		Mat colmat1 = mat1.col(i);
		for (int j = 0; j < num_item; ++j, ++cIdx) {
			Mat subm = ntuple.rowRange(cv::Range(0, mat1.rows)).col(cIdx);
			colmat1.copyTo(subm);
		}
	}

	for (int i = 0; i < num_col; ++i) {
		Mat subm = ntuple.rowRange(cv::Range(mat1.rows, ntuple.rows)).colRange(
				cv::Range(i * num_item, (i + 1) * num_item));
		mat2.copyTo(subm);
	}

	return ntuple;
}

// instantiation
template class OpencvUtil_<int> ;
template class OpencvUtil_<float> ;
template class OpencvUtil_<double> ;

} // namespace RGM
