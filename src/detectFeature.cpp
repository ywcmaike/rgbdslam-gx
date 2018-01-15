#include <iostream>
#include "slamBase.h"

using namespace std;

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>

int main(int argc, char** argv) {
    //声明并从data文件夹里读取两个rgb与深度图
    cv::Mat rgb1, depth1, rgb2, depth2;
    rgb1 = cv::imread("../data/rgb1.png");
    depth1 = cv::imread("../data/depth1.png", -1);
    rgb2 = cv::imread("../data/rgb2.png");
    depth2 = cv::imread("../data/depth2.png", -1);
    if (!rgb1.data) {
        cout << "rgb1 is free" << endl;
        return -1;
    } else if (!rgb2.data) {
        cout << "rgb2 is free" << endl;
        return -1;
    } else if (!depth1.data) {
        cout << "depth1 is free" << endl;
        return -1;
    } else if (!depth2.data) {
        cout << "depth2 is free" << endl;
        return -1;
    }

    //声明特征提取器与描述子提取器
    cv::Ptr<cv::FeatureDetector> _detector;
    cv::Ptr<cv::DescriptorExtractor> _descriptor;

    //构建提取器，默认两者都为sift
    //构建sift,surf之前要初始化nonfree模块
    cv::initModule_nonfree();
    _detector = cv::FeatureDetector::create("SIFT");
    _descriptor = cv::DescriptorExtractor::create("SIFT");

//    _detector = cv::FeatureDetector::create("ORB");
//    _descriptor = cv::DescriptorExtractor::create("ORB");

    vector<cv::KeyPoint> kp1, kp2;//关键点
    _detector->detect(rgb1, kp1);//提取关键点
    _detector->detect(rgb2, kp2);

    cout << "Key points of two images: " << kp1.size() << " " << kp2.size() << endl;

    //可视化，显示关键点
    cv::Mat imgshow;
    cv::drawKeypoints(rgb1, kp1, imgshow, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("keypoints", imgshow);
    cv::imwrite("../data/keypoints.png", imgshow);
    cv::waitKey(0); //暂停等待一个按键

    //计算描述子
    cv::Mat desp1, desp2;
    _descriptor->compute(rgb1, kp1, desp1);
    _descriptor->compute(rgb2, kp2, desp2);

    //匹配描述子
    vector<cv::DMatch> matches;
    cv::FlannBasedMatcher matcher;
    matcher.match(desp1, desp2, matches);
    cout << "Find total " << matches.size() << " matches." << endl;

    //可视化：显示匹配的特征
    cv::Mat imgMatches;
    cv::drawMatches(rgb1, kp1, rgb2, kp2, matches, imgMatches);
    cv::imshow("matches", imgMatches);
    cv::imwrite("../data/matches.png", imgMatches);
    cv::waitKey(0);

    //筛选匹配，把距离太大的去掉
    //这里使用的准则是去掉大于四倍最小距离的匹配
    vector<cv::DMatch> goodMatches;
    double minDis = 9999;
    for (size_t i = 0; i < matches.size(); i++) {
        if (matches[i].distance < minDis) {
            minDis = matches[i].distance;
        }
    }
    cout << "min dis = " << minDis << endl;

    for (size_t i = 0; i < matches.size(); i++) {
        if (matches[i].distance < 4 * minDis) {
            goodMatches.push_back(matches[i]);
        }
    }

    //显示good matches
    cout << "good matches= " << goodMatches.size() << endl;
    cv::drawMatches(rgb1,  kp1, rgb2, kp2, goodMatches, imgMatches);
    cv::imshow("good matches ", imgMatches);
    cv::imwrite("../data/good_matches.png", imgMatches);
    cv::waitKey(0);

    //计算图像间的运动关系
    //cv::solvePnPRansac()
    //为调用此函数准备必要的参数

    //第一个帧的三维点
    vector<cv::Point3f> pts_obj;
    //第二个帧的图像点
    vector<cv::Point2f> pts_img;

    //相机内参
    CAMERA_INTRINSIC_PARAMETERS C;
    C.cx = 325.5;
    C.cy = 253.5;
    C.fx = 518.0;
    C.fy = 519.0;
    C.scale = 1000.0;

    for (size_t i = 0;  i < goodMatches.size(); i++) {
        //query 是第一个，train是第二个
        cv::Point2f p = kp1[goodMatches[i].queryIdx].pt;
        //获取d要小心，x是向右的，y是向下的，所以y才是行，x是列
        ushort d = depth1.ptr<ushort>(int(p.y))[int(p.x)];
        if (d == 0) {
            continue;
        }
        pts_img.push_back(cv::Point2f(kp2[goodMatches[i].trainIdx].pt));

        //将（u,v,d)转成（x, y,z);
        cv::Point3f pt(p.x, p.y, d);
        cv::Point3f pd = point2dTo3d(pt, C);
        pts_obj.push_back(pd);
    }

    double camera_matrix_data[3][3] = {
            {C.fx, 0, C.cx},
            {0, C.fy, C.cy},
            {0, 0, 1}
    };

    //构建相机矩阵
    cv::Mat camereMatrix(3, 3, CV_64F, camera_matrix_data);
    cv::Mat rvec, tvec, inliers;
    //求解PnP
    cv::solvePnPRansac(pts_obj, pts_img, camereMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 100, inliers);
    cout << "inliers: " << inliers.rows << endl;
    cout << "R= " << rvec << endl;
    cout << "t= " << tvec << endl;

    //画出inliers匹配
    vector<cv::DMatch> matchesShow;
    for (size_t i = 0; i < inliers.rows; i++) {
        matchesShow.push_back(goodMatches[inliers.ptr<int>(i)[0]]);
    }
    cv::drawMatches(rgb1, kp1, rgb2, kp2, matchesShow, imgMatches);
    cv::imshow("inlier matches", imgMatches);
    cv::imwrite("../data/inliers.png", imgMatches);
    cv::waitKey(0);
    return 0;
}