#include "pch.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <glog/logging.h>

#include "mynteye/api.h"

#include <fstream>

#include <cmath>

//#include "util/cv_painter.h"
//#include "util/pc_viewer.h"


namespace {

	class DepthRegion {
	public:
		explicit DepthRegion(std::int32_t n)
			: n_(std::move(n)), show_(false), selected_(false), point_(0, 0) {}

		~DepthRegion() = default;

		/**
		* 鼠标事件：默认不选中区域，随鼠标移动而显示。单击后，则会选中区域来显示。你可以再单击已选中区域或双击未选中区域，取消选中。
		*/
		void OnMouse(const int &event, const int &x, const int &y, const int &flags) {
			UNUSED(flags)
				if (event != CV_EVENT_MOUSEMOVE && event != CV_EVENT_LBUTTONDOWN) {
					return;
				}
			show_ = true;

			if (event == CV_EVENT_MOUSEMOVE) {
				if (!selected_) {
					point_.x = x;
					point_.y = y;
				}
			}
			else if (event == CV_EVENT_LBUTTONDOWN) {
				if (selected_) {
					if (x >= static_cast<int>(point_.x - n_) &&
						x <= static_cast<int>(point_.x + n_) &&
						y >= static_cast<int>(point_.y - n_) &&
						y <= static_cast<int>(point_.y + n_)) {
						selected_ = false;
					}
				}
				else {
					selected_ = true;
				}
				point_.x = x;
				point_.y = y;
			}
		}

		template <typename T>
		void ShowElems(
			const cv::Mat &depth,
			std::function<std::string(const T &elem)> elem2string,
			int elem_space = 40,
			std::function<std::string(
				const cv::Mat &depth, const cv::Point &point, const std::uint32_t &n)>
			getinfo = nullptr) {
			if (!show_)
				return;

			int space = std::move(elem_space);
			int n = 2 * n_ + 1;
			cv::Mat im(space * n, space * n, CV_8UC3, cv::Scalar(255, 255, 255));

			int x, y;
			std::string str;
			int baseline = 0;
			for (int i = - n_; i <= n; ++i) {
				x = point_.x + i;
				if (x < 0 || x >= depth.cols)
					continue;
				for (int j = - n_; j <= n; ++j) {
					y = point_.y + j;
					if (y < 0 || y >= depth.rows)
						continue;

					str = elem2string(depth.at<T>(y, x));

					cv::Scalar color(0, 0, 0);
					if (i == 0 && j == 0)
						color = cv::Scalar(0, 0, 255);

					cv::Size sz =
						cv::getTextSize(str, cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

					cv::putText(
						im, str, cv::Point(
						(i + n_) * space + (space - sz.width) / 2,
							(j + n_) * space + (space + sz.height) / 2),
						cv::FONT_HERSHEY_PLAIN, 1, color, 1);
				}
			}

			if (getinfo) {
				std::string info = getinfo(depth, point_, n_);
				if (!info.empty()) {
					cv::Size sz =
						cv::getTextSize(info, cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

					cv::putText(
						im, info, cv::Point(5, 5 + sz.height), cv::FONT_HERSHEY_PLAIN, 1,
						cv::Scalar(255, 0, 255), 1);
				}
			}

			cv::imshow("region", im);
		}

		//template <typename ushort>
		ushort GetDepthHelp(
			const cv::Mat &depth,
			cv::Point point
		) {
			int x = point.x;
			int y = point.y;
			return depth.at<ushort>(y, x);
		}
		ushort GetDepth(const cv::Mat &depth) {
			return GetDepthHelp(depth, point_);
		}



		template <typename T>
		void GetDepthWithGraph(
			const cv::Mat &depth,
			std::function<std::string(const T &elem)> elem2string,
			int elem_space = 40,
			std::function<std::string(
				const cv::Mat &depth, const cv::Point &point, const std::uint32_t &n)>
			getinfo = nullptr) {
			if (!show_)
				return;

			int space = std::move(elem_space);
			int n = 2 * n_ + 1;
			cv::Mat im(space * n, space * n, CV_8UC3, cv::Scalar(255, 255, 255));

			int x, y;
			std::string str;
			int baseline = 0;

			x = point_.x;
			y = point_.y;
			str = elem2string(depth.at<T>(y, x));
			cv::Scalar color = cv::Scalar(0, 0, 255);
			cv::Size sz =
				cv::getTextSize(str, cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

			cv::putText(
				im, str, cv::Point(
				(n_) * space + (space - sz.width) / 2,
				(n_) * space + (space + sz.height) / 2),
				cv::FONT_HERSHEY_PLAIN, 1, color, 1);


			if (getinfo) {
				std::string info = getinfo(depth, point_, n_);
				if (!info.empty()) {
					cv::Size sz =
						cv::getTextSize(info, cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

					cv::putText(
						im, info, cv::Point(5, 5 + sz.height), cv::FONT_HERSHEY_PLAIN, 1,
						cv::Scalar(255, 0, 255), 1);
				}
			}

			cv::imshow("test", im);
		}

		void DrawRect(cv::Mat &image) {  // NOLINT
			if (!show_)
				return;
			std::uint32_t n = (n_ > 1) ? n_ : 1;
			n += 1;  // outside the region
			cv::rectangle(
				image, cv::Point(point_.x - n, point_.y - n),
				cv::Point(point_.x + n, point_.y + n),
				selected_ ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 1);
		}

	private:
		std::int32_t n_;
		bool show_;
		bool selected_;
		cv::Point point_;
	};

	void OnDepthMouseCallback(int event, int x, int y, int flags, void *userdata) {
		DepthRegion *region = reinterpret_cast<DepthRegion *>(userdata);
		region->OnMouse(event, x, y, flags);
	}

}  // namespace

cv::Mat ConvertToBinary(cv::Mat in) {
	cv::Mat out;
	out = in > 200;
	return out;
}

bool GetFeaturePoints(cv::Mat in, std::vector<cv::Point> &pts) {
	cv::findNonZero(in, pts);
	//std::cout << "white: " << std::endl;
	return true;
}

bool CheckConnection(cv::Point p1, cv::Point p2) {
	if (abs(p1.x - p2.x) <= 1 && abs(p1.y - p2.y) <= 1) {
		return true;
	}
	else {
		return false;
	}
}

// Get adj pts from 8 direction
void GetAdjPts(cv::Point pt, std::vector<cv::Point> &pts) {
	pts.clear();
	// lu
	cv::Point lu;
	lu.x = pt.x - 1;
	lu.y = pt.y + 1;
	pts.push_back(lu);

	// u
	cv::Point u;
	u.x = pt.x;
	u.y = pt.y + 1;
	pts.push_back(u);

	//ru
	cv::Point ru;
	ru.x = pt.x + 1;
	ru.y = pt.y + 1;
	pts.push_back(ru);

	//l
	cv::Point l;
	l.x = pt.x - 1;
	l.y = pt.y ;
	pts.push_back(l);

	//r
	cv::Point r;
	r.x = pt.x + 1;
	r.y = pt.y;
	pts.push_back(r);

	//ld
	cv::Point ld;
	ld.x = pt.x - 1;
	ld.y = pt.y - 1;
	pts.push_back(ld);

	//d
	cv::Point d;
	d.x = pt.x;
	d.y = pt.y - 1;
	pts.push_back(d);

	//rd
	cv::Point rd;
	rd.x = pt.x + 1;
	rd.y = pt.y - 1;
	pts.push_back(rd);

	int p = pts.size();
	int q = p;
}

// Seperate dots and combine to groups
// Get the center point of a group
// using dfs
// Done

void SearchPts(cv::Point pt, std::vector<std::vector<cv::Point>> &group, int index, std::vector<cv::Point> &total) {

	// check if push back into vec
	bool isPush = false;

	// get adj pts
	std::vector<cv::Point> pts(8);	
	GetAdjPts(pt, pts);

	for (int i = 0; i < pts.size(); i++) {
		if (total.size() != 0) {

			// search adj pts in total
			std::vector<cv::Point>::iterator it = find(total.begin(), total.end(), pts[i]);
			if (it == total.end()) {

				// check if this is the last one
				if (i == pts.size() - 1 && isPush == false) {

					// update index
					if (index == -1) {
						index = group.size();
						std::vector<cv::Point> newVec;
						group.push_back(newVec);
					}

					// push back pt
					group[index].push_back(pt);
					isPush = true;

					// delete pt from total
					std::vector<cv::Point>::iterator ptit = find(total.begin(), total.end(), pt);
					if (ptit != total.end()) {
						total.erase(ptit);
					}
				}
				else {
				}
			}
			else {

				// update index
				if (index == -1) {
					index = group.size();
					std::vector<cv::Point> newVec;
					group.push_back(newVec);
				}

				// push back pt
				if (isPush == false) {
					group[index].push_back(pt);
					isPush = true;
				}

				// delete pt from total
				std::vector<cv::Point>::iterator ptit = find(total.begin(), total.end(), pt);
				if (ptit != total.end()) {
					total.erase(ptit);
				}

				// search adj pts in next round
				SearchPts(pts[i], group, index, total);
			}
		}
	}
}


int GetDepth(
	const cv::Mat &depth,
	cv::Point pt) {
	return (depth.at<ushort>(pt.y, pt.x));
}

void DeleteUselessPt(const cv::Mat &depth, std::vector<cv::Point> &pts) {

	auto it = pts.begin();
	while (it != pts.end()) {
		int ret = GetDepth(depth, *it);
		if (ret >= 10000) {
			it = pts.erase(it);
		}
		else {
			++it;
		}
	}
}

void OutputPts(const cv::Mat &depth, std::vector<cv::Point> &pts, std::string fileName) {
	std::ofstream saveFile;
	saveFile.open(fileName + ".txt");
	int size = pts.size();

	for (int i = 0; i < size; i++) {
		int ret = GetDepth(depth, pts[i]);
		saveFile << "x: " + std::to_string(pts[i].x) << " " << " y:" + std::to_string(pts[i].y)<<" depth: "<< ret << std::endl;
	}
	saveFile << "total: " << size << std::endl;

	saveFile << "   " << std::endl;

	DeleteUselessPt(depth, pts);

	std::vector<std::vector<cv::Point>> vecs;

	while (pts.size() != 0) {
		SearchPts(pts[0], vecs, -1, pts);
	}

	for (int i = 0; i < vecs.size(); i++) {
		float xSum = 0;
		float ySum = 0;
		float retSum = 0;
		for (int j = 0; j < vecs[i].size(); j++) {

			int ret = GetDepth(depth, vecs[i][j]);

			saveFile << "x: " + std::to_string(vecs[i][j].x) << " " << " y:" + std::to_string(vecs[i][j].y) << " depth: " << ret << std::endl;
			xSum += vecs[i][j].x;
			ySum += vecs[i][j].y;
			retSum += ret;
		}
		saveFile << (xSum / vecs[i].size()) <<" "<< (ySum / vecs[i].size())<<" "<< (retSum / vecs[i].size()) << std::endl;
		saveFile << "   " << std::endl;
	}


	saveFile.close();
}


MYNTEYE_USE_NAMESPACE

int main(int argc, char *argv[]) {
	auto &&api = API::Create(argc, argv);
	if (!api)
		return 1;

	api->SetOptionValue(Option::IR_CONTROL, 80);

	api->EnableStreamData(Stream::DISPARITY_NORMALIZED);
	api->EnableStreamData(Stream::DEPTH);

	api->Start(Source::VIDEO_STREAMING);

	//cv::namedWindow("frame");
	cv::namedWindow("depth");
	//cv::namedWindow("region");

	DepthRegion depth_region(3);
	auto depth_info = [](
		const cv::Mat &depth, const cv::Point &point, const std::uint32_t &n) {
		UNUSED(depth)
			std::ostringstream os;
		os << "depth pos: [" << point.y << ", " << point.x << "]"
			<< "±" << n << ", unit: mm";
		return os.str();
	};

	//CVPainter painter;
	//PCViewer pcviewer;
	
	int i = 0;

	while (true) {
		api->WaitForStreams();

		auto &&left_data = api->GetStreamData(Stream::LEFT);
		auto &&right_data = api->GetStreamData(Stream::RIGHT);

		//cv::Mat img;
		//cv::hconcat(left_data.frame, right_data.frame, img);

		cv::Mat img_left;
		cv::Mat img_right;
		img_left = left_data.frame;
		img_right = right_data.frame;

		cv::imshow("left_frame", img_left);
		cv::imshow("right_frame", img_right);

		cv::Mat img_left_operated;
		cv::Mat img_right_operated;

		img_left_operated = ConvertToBinary(img_left);
		img_right_operated = ConvertToBinary(img_right);

		cv::imshow("left_operated", img_left_operated);
		cv::imshow("right_operated", img_right_operated);



		//painter.DrawImgData(img, *left_data.img);

		//cv::imshow("frame", img);

		auto &&disp_data = api->GetStreamData(Stream::DISPARITY_NORMALIZED);
		auto &&depth_data = api->GetStreamData(Stream::DEPTH);
		if (!disp_data.frame.empty() && !depth_data.frame.empty()) {
			// Show disparity instead of depth, but show depth values in region.
			auto &&depth_frame = disp_data.frame;

//#ifdef USE_OPENCV3
			// ColormapTypes
			//   http://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html#ga9a805d8262bcbe273f16be9ea2055a65
			cv::applyColorMap(depth_frame, depth_frame, cv::COLORMAP_JET);
//#endif

			cv::setMouseCallback("depth", OnDepthMouseCallback, &depth_region);
			// Note: DrawRect will change some depth values to show the rect.
			depth_region.DrawRect(depth_frame);

			cv::imshow("depth", depth_frame);

			depth_region.GetDepthWithGraph<ushort>(
				depth_data.frame,
				[](const ushort &elem) {
				if (elem >= 10000) {
					return std::string("invalid");
				}
				return std::to_string(elem);
			},
				80, depth_info);
			//std::cout << depth_region.GetDepth(depth_data.frame) << std::endl;
			//std::cout << depth_data.frame.size()<<std::endl;
		}

		//auto &&points_data = api->GetStreamData(Stream::POINTS);

		char key = static_cast<char>(cv::waitKey(1));


		if (key == 'z' || key == 'Z') {  // z/Z
			i++;
			std::vector<cv::Point> pts;
			GetFeaturePoints(img_right_operated, pts);
			OutputPts(depth_data.frame, pts, "capture_" + std::to_string(i));
			std::cout << "captured: "<< pts.size() << std::endl;
		}

		if (key == 27 || key == 'q' || key == 'Q') {  // ESC/Q
			break;
		}
		
	}

	api->Stop(Source::VIDEO_STREAMING);
	return 0;
}