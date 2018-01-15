#include <iostream>
#include <string>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
//定义点云类型
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;
//相机内参
const double camera_factor = 1000; //代表1m
const double camera_cx = 325.5;
const double camera_cy = 253.5;
const double camera_fx = 518.0;
const double camera_fy = 519.0;

int main(int argc, char** argv) {
	cv::Mat rgb, depth; //图像矩阵
	rgb = cv::imread("../data/rgb.png"); //rgb图像是8UC3的彩色图像
	depth = cv::imread("../data/depth.png", -1); //depth是16UC1的单通道图像，flags取1，表示读取原始数据不做人和修改
	//点云变量
	//使用智能指针，创建一个空点云，用完指针自动释放。
	PointCloud::Ptr cloud(new PointCloud);
	for (int m = 0; m < depth.rows; m++) {
		for (int n = 0; n < depth.cols; n++) {
			ushort d = depth.ptr<ushort>(m)[n];
			if (d == 0)
				continue;
			PointT p;
			//计算这个点的空间坐标
			p.z = double(d) / camera_factor;
			p.x = (n - camera_cx) * p.z / camera_fx;
			p.y = (m - camera_cy) * p.z / camera_fy;
			//从rgb图像获取它的颜色
			//rgb是三通道的BGR歌诗图，所以按下面的顺序获取颜色
			p.b = rgb.ptr<uchar>(m)[n * 3];
			p.g = rgb.ptr<uchar>(m)[n * 3 + 1];
			p.r = rgb.ptr<uchar>(m)[n * 3 + 2];

			cloud->points.push_back(p);
		}
	}
	//设置并保存点云
	cloud->height = 1;
	cloud->width = cloud->points.size();
	cout << "point cloud size = " << cloud->points.size() << endl;
	cloud->is_dense = false;
	pcl::io::savePCDFile("../pointcloud.pcd", *cloud);
	cloud->points.clear();
	cout << "Point cloud saved." << endl;
	return 0;
}
