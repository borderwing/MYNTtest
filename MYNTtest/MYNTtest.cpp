#include "pch.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <glog/logging.h>

#include "mynteye/api.h"

//#include "util/cv_painter.h"
//#include "util/pc_viewer.h"


namespace {

	class DepthRegion {
	public:
		explicit DepthRegion(std::int32_t n)
			: n_(std::move(n)), show_(false), selected_(false), point_(0, 0) {}

		~DepthRegion() = default;

		/**
		* ����¼���Ĭ�ϲ�ѡ������������ƶ�����ʾ�����������ѡ����������ʾ��������ٵ�����ѡ�������˫��δѡ������ȡ��ѡ�С�
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
	out = in > 128;
	return out;
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

	cv::namedWindow("frame");
	cv::namedWindow("depth");
	cv::namedWindow("region");

	DepthRegion depth_region(3);
	auto depth_info = [](
		const cv::Mat &depth, const cv::Point &point, const std::uint32_t &n) {
		UNUSED(depth)
			std::ostringstream os;
		os << "depth pos: [" << point.y << ", " << point.x << "]"
			<< "��" << n << ", unit: mm";
		return os.str();
	};

	//CVPainter painter;
	//PCViewer pcviewer;

	while (true) {
		api->WaitForStreams();

		auto &&left_data = api->GetStreamData(Stream::LEFT);
		auto &&right_data = api->GetStreamData(Stream::RIGHT);

		cv::Mat img;
		cv::hconcat(left_data.frame, right_data.frame, img);

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

		cv::imshow("left", img_left_operated);
		cv::imshow("right", img_right_operated);

		//painter.DrawImgData(img, *left_data.img);

		cv::imshow("frame", img);

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

			depth_region.ShowElems<ushort>(
				depth_data.frame,
				[](const ushort &elem) {
				if (elem >= 10000) {
					// Filter errors, or limit to valid range.
					//
					// reprojectImageTo3D(), missing values will set to 10000
					//   https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga1bc1152bd57d63bc524204f21fde6e02
					return std::string("invalid");
				}
				return std::to_string(elem);
			},
				80, depth_info);
		}

		auto &&points_data = api->GetStreamData(Stream::POINTS);
		if (!points_data.frame.empty()) {
			//pcviewer.Update(points_data.frame);
		}

		char key = static_cast<char>(cv::waitKey(1));
		if (key == 27 || key == 'q' || key == 'Q') {  // ESC/Q
			break;
		}
		//if (pcviewer.WasStopped()) {
		//	break;
		//}
	}

	api->Stop(Source::VIDEO_STREAMING);
	return 0;
}