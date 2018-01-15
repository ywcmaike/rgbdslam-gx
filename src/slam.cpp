/*
 *
1关键帧的提取。把每一帧都拼到地图是去是不明智的。因为帧与帧之间距离很近，导致地图需要频繁更新，浪费时间与空间。所以，我们希望，当机器人的运动超过一定间隔，就增加一个“关键帧”。最后只需把关键帧拼到地图里就行了。
2回环的检测。回环的本质是识别曾经到过的地方。最简单的回环检测策略，就是把新来的关键帧与之前所有的关键帧进行比较，不过这样会导致越往后，需要比较的帧越多。所以，稍微快速一点的方法是在过去的帧里随机挑选一些，与之进行比较。更进一步的，也可以用图像处理/模式识别的方法计算图像间的相似性，对相似的图像进行检测。

把这两者合在一起，就得到了我们slam程序的基本流程。以下为伪码：
1初始化关键帧序列：F，并将第一帧f0放入F。
2对于新来的一帧I，计算F中最后一帧与I的运动，并估计该运动的大小e。有以下几种可能性：
  若e>Eerror，说明运动太大，可能是计算错误，丢弃该帧；
  若没有匹配上（match太少），说明该帧图像质量不高，丢弃；
  若e<Ekey，说明离前一个关键帧很近，同样丢弃；
  剩下的情况，只有是特征匹配成功，运动估计正确，同时又离上一个关键帧有一定距离，则把I
作为新的关键帧，进入回环检测程序：
3近距离回环：匹配I与F末尾m个关键帧。匹配成功时，在图里增加一条边。
4随机回环：随机在F里取n个帧，与I进行匹配。若匹配上，在图里增加一条边。
5将I放入F末尾。若有新的数据，则回2； 若无，则进行优化与地图拼接。
 */

//
// Created by ywc on 18-1-15.
//
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "slamBase.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>


//g2o头文件
#include <g2o/types/slam3d/types_slam3d.h> //顶点类型
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

//把g2o的定义放到前面
typedef g2o::BlockSolver_6_3 SlamBlockSolver;
typedef g2o::LinearSolverEigen<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

//给定index,读取一帧数据
FRAME readFrame(int index, ParameterReader& pd);
//度量运动的大小
double normTransform(cv::Mat rvec, cv::Mat tvec);

//检测两个帧，结果定义
enum CHECK_RESULT {
    NOT_MATCHED=0, TOO_FAR_AWAY, TOO_CLOSE, KEYFRAME
};
//函数声明
CHECK_RESULT checkKeyframes(FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool isloops=false);

//检测近距离的回环
void checkNearbyLoops(vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti);

//随机检测回环
void checkRandomLoops(vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti);

int main(int argc, char** argv) {
    ParameterReader pd;
    int startIndex = atoi(pd.getData("start_index").c_str());
    int endIndex = atoi(pd.getData("end_index").c_str());

    //所有的关键帧都放在这里
    vector<FRAME> keyframes;

    //initialize
    cout << "Initializing ..." << endl;
    int currIndex = startIndex; //当前索引为currIndex;
    FRAME currFrame = readFrame(currIndex, pd); //当前帧数据
    //我们总是在比较currFrame和lastFrame;
    string detector = pd.getData("detector");
    string descriptor = pd.getData("descriptor");
    CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
    computeKeyPointsAndDesp(currFrame, detector, descriptor);
    PointCloud::Ptr cloud = image2PointCloud(currFrame.rgb, currFrame.depth, camera);

//    pcl::visualization::CloudViewer viewer("viewer");
//
//    //是否显示点云
//    bool visualize = pd.getData("visualize_pointcloud")==string("yes");
//
//    int min_inliers = atoi(pd.getData("min_inliers").c_str());
//    double max_norm = atof(pd.getData("max_norm").c_str());


    //g2o图优化步骤：第一步，构建一个求解器：globalOptimizer
    //g2o的初始化
    //选择优化方法


    //初始化求解器
    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    linearSolver->setBlockOrdering(false);
    SlamBlockSolver* blockSolver = new SlamBlockSolver(std::unique_ptr<SlamLinearSolver>(linearSolver));
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<SlamBlockSolver>(blockSolver));

    g2o::SparseOptimizer globalOptimizer;
    globalOptimizer.setAlgorithm(solver);
    //不要输出调试信息
    globalOptimizer.setVerbose(false);

    //第二步：在求解器中添加点和边
    //向globalOptimizer增加第一个顶点
    g2o::VertexSE3* v = new g2o::VertexSE3();
    v->setId(currIndex);
    v->setEstimate(Eigen::Isometry3d::Identity()); //估计为单位矩阵
    v->setFixed(true); //第一个顶点固定，不用优化
    globalOptimizer.addVertex(v);

    keyframes.push_back(currFrame);

    double keyframe_threhold = atof(pd.getData("keyframe_threshold").c_str());
    bool check_loop_closure = pd.getData("check_loop_closure")==string("yes");

//    int lastIndex = currIndex; //上一帧的id

    for (currIndex=startIndex+1; currIndex<endIndex; currIndex++) {
        cout << "Reading files: " << currIndex << endl;
        FRAME currFrame = readFrame(currIndex, pd); //读取currFrame
        computeKeyPointsAndDesp(currFrame, detector, descriptor);
        CHECK_RESULT result = checkKeyframes(keyframes.back(), currFrame, globalOptimizer); //匹配该帧与keyframes最后一帧
        switch (result) {
            case NOT_MATCHED:
                cout << RED
                "Not enough inliers." << endl;
                break;
            case TOO_FAR_AWAY:
                cout << RED
                "Too far away, may be an error." << endl;
                break;
            case TOO_CLOSE:
                cout << RESET
                "Too close, not a keyframe" << endl;
                break;
            case KEYFRAME:
                cout << GREEN
                "This is a new keyframe" << endl;
                if (check_loop_closure) {
                    checkNearbyLoops(keyframes, currFrame, globalOptimizer);
                    checkRandomLoops(keyframes, currFrame, globalOptimizer);
                }
                keyframes.push_back(currFrame);
                break;
            default:
                break;

        }
    }
//        //比较currFrame和lastFrame
//        RESULT_OF_PNP result = estimateMotion(lastFrame, currFrame, camera);
//        if (result.inliers < min_inliers) { //inliers不够，放弃该帧
//            continue;
//        }
//        //计算运动范围是否太大
//        double norm = normTransform(result.rvec, result.tvec);
//        cout << "norm = " << norm << endl;
//        if (norm >= max_norm) {
//            continue;
//        }
//        Eigen::Isometry3d T = cvMat2Eigen(result.rvec, result.tvec);
//        cout << "T = " << T.matrix() << endl;
//
////        cloud = joinPointCloud(cloud, currFrame, T, camera);
//
//        // 去掉可视化的话，会快一些
//        if ( visualize == true )
//        {
//            cloud = joinPointCloud( cloud, currFrame, T, camera );
//            viewer.showCloud( cloud );
//        }
//
//        //向g2o中增加这个顶点与上一帧联系的边
//        //顶点部分
//        //顶点只需指定id即可
//        g2o::VertexSE3 *v = new g2o::VertexSE3();
//        v->setId(currIndex);
//        v->setEstimate(Eigen::Isometry3d::Identity());
//        globalOptimizer.addVertex(v);
//        //边部分
//        g2o::EdgeSE3* edge = new g2o::EdgeSE3();
//        //连接此边的两个顶点id
//        edge->vertices()[0] = globalOptimizer.vertex(lastIndex);
//        edge->vertices()[1] = globalOptimizer.vertex(currIndex);
//        //信息矩阵
//        Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
//
//        //信息矩阵是协方差矩阵的逆，表示我们对边的精度的预先估计
//        //因为pose是6D的，信息矩阵是6*6的阵，假设位置和角度的估计精度均为0.1且互相独立，
//        //那么协方差则为对角为0.01的矩阵，信息矩阵则为100的矩阵
//        information(0, 0) = information(1, 1) = information(2, 2) = 100;
//        information(3, 3) = information(4, 4) = information(5, 5) = 100;
//        //也可以将角度设大一些，表示对角度的估计更加准确
//        edge->setInformation(information);
//        //边的矩估计即是PnP求解的结果
//        edge->setMeasurement(T);
//        //将此边加入图中
//        globalOptimizer.addEdge(edge);
//
//        lastFrame = currFrame;
//        lastIndex = currIndex;
//    }
//    pcl::io::savePCDFile("../data/result.pcd", *cloud);
    //最后，完成优化并存储优化结果
    //优化所有边
    cout << "Optimizing pose graph, vertices: " << globalOptimizer.vertices().size() << endl;
    globalOptimizer.save("../data/result_before.g2o");
    globalOptimizer.initializeOptimization();
    globalOptimizer.optimize(100); //可以指定优化步数
    globalOptimizer.save("../data/result_after.g2o");
    cout << "Optimization done. " << endl;

    //拼接点云地图
    cout << "saving the point cloud map..." << endl;
    PointCloud::Ptr output(new PointCloud()); //全局地图
    PointCloud::Ptr tmp(new PointCloud());

    pcl::VoxelGrid<PointT> voxel; //网格滤波器，调整地图分辨率
    pcl::PassThrough<PointT> pass; //z方向区间滤波器，由于rgbd相机的有效深度区间有限，把太远的去掉
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0, 4.0); //4m以上就不要了

    double gridsize = atof(pd.getData("voxel_grid").c_str());
    voxel.setLeafSize(gridsize, gridsize, gridsize);

    for (size_t i = 0; i < keyframes.size(); i++) {
        //g2o里去除一帧
        g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex(keyframes[i].frameID));
        Eigen::Isometry3d pose = vertex->estimate(); //该帧优化后的位姿
        PointCloud::Ptr newCloud = image2PointCloud(keyframes[i].rgb, keyframes[i].depth, camera); //转成点云
        //以下是滤波
        voxel.setInputCloud(newCloud);
        voxel.filter(*tmp);
        pass.setInputCloud(tmp);
        pass.filter(*newCloud);
        //把点云变换后加入全局地图中
        pcl::transformPointCloud(*newCloud, *tmp, pose.matrix());
        *output += *tmp;
        tmp->clear();
        newCloud->clear();
    }

    voxel.setInputCloud(output);
    voxel.filter(*tmp);
    //存储
    pcl::io::savePCDFile("../data/result.pcd", *tmp);
    cout << "Final map is saved." << endl;
    globalOptimizer.clear();

    return 0;
}

FRAME readFrame(int index, ParameterReader& pd) {
    FRAME f;
    string rgbDir = pd.getData("rgb_dir");
    string depthDir = pd.getData("depth_dir");
    string rgbExt = pd.getData("rgb_extension");
    string depthExt = pd.getData("depth_extension");

    stringstream ss;
    ss << rgbDir << index << rgbExt;
    string filename;
    ss>>filename;
    f.rgb = cv::imread(filename);

    ss.clear();
    filename.clear();
    ss << depthDir << index << depthExt;
    ss >> filename;
    f.depth = cv::imread(filename, -1);
    f.frameID = index;
    return f;
}

double normTransform(cv::Mat rvec, cv::Mat tvec) {
    return fabs(min(cv::norm(rvec), 2 * M_PI - cv::norm(rvec))) + fabs(cv::norm(tvec));
}

//函数声明
CHECK_RESULT checkKeyframes(FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool isloops) {
    static ParameterReader pd;
    static int min_inliers = atoi(pd.getData("min_inliers").c_str());
    static double max_norm = atof(pd.getData("max_norm").c_str());
    static double keyframe_threshold = atof(pd.getData("keyframe_threshold").c_str());

    static double max_norm_lp = atof(pd.getData("max_norm_lp").c_str());
    static CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
    static g2o::RobustKernel* robustKernel = g2o::RobustKernelFactory::instance()->construct("Cauchy");
    //比较f1和f2
    RESULT_OF_PNP result = estimateMotion(f1, f2, camera);
    if (result.inliers < min_inliers) {
        return NOT_MATCHED;
    }
    double norm = normTransform(result.rvec, result.tvec);
    if (isloops == false) {
        if (norm >= max_norm) {
            return TOO_FAR_AWAY;
        }
    } else {
        if (norm >= max_norm_lp) {
            return TOO_FAR_AWAY;
        }
    }

    if (norm <= keyframe_threshold) {
        return TOO_CLOSE;
    }
    //顶点部分
    //顶点只需设定id即可
    if (isloops == false) {
        g2o::VertexSE3 *v = new g2o::VertexSE3();
        v->setId(f2.frameID);
        v->setEstimate(Eigen::Isometry3d::Identity());
        opti.addVertex(v);
    }
    //边部分
    g2o::EdgeSE3* edge = new g2o::EdgeSE3();
    //连接此边的两个顶点id
    edge->vertices()[0] = opti.vertex(f1.frameID);
    edge->vertices()[1] = opti.vertex(f2.frameID);
    edge->setRobustKernel(robustKernel);
    //信息矩阵
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
    information(0, 0) = information(1, 1) = information(2, 2) = 100;
    information(3, 3) = information(4, 4) = information(5, 5) = 100;
    edge->setInformation(information);
    //边的估计即是pnp求解的结果
    Eigen::Isometry3d T = cvMat2Eigen(result.rvec, result.tvec);
    edge->setMeasurement(T.inverse());
    //将此边加入图中
    opti.addEdge(edge);
    return KEYFRAME;

}

//检测近距离的回环
void checkNearbyLoops(vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti) {
    static ParameterReader pd;
    static int nearby_loops = atoi(pd.getData("nearby_loops").c_str());

    //就是把currFrame和frames里末尾几个测一遍
    if (frames.size() <= nearby_loops) {
        //no enough keyframes, check everyone
        for (size_t i = 0; i < frames.size(); i++) {
            checkKeyframes(frames[i], currFrame, opti, true);
        }
    } else {
        //check the nearest ones
        for (size_t i = frames.size() - nearby_loops; i < frames.size(); i++) {
            checkKeyframes(frames[i], currFrame, opti, true);
        }
    }
}

//随机检测回环
void checkRandomLoops(vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti) {
    static ParameterReader pd;
    static int random_loops = atoi(pd.getData("random_loops").c_str());
    srand((unsigned int)time(NULL));
    if (frames.size() <= random_loops) {
        //no enough keyframes, check everyone
        for (size_t i = 0; i < frames.size(); i++) {
            checkKeyframes(frames[i], currFrame, opti, true);
        }
    } else {
        //randomly check loops
        for (int i = 0; i < random_loops; i++) {
            int index = rand() % frames.size();
            checkKeyframes(frames[index], currFrame, opti, true);
        }
    }

}

