#define BOOST_BIND_NO_PLACEHOLDERS

#include <memory>

#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>
#include "lifecycle_msgs/srv/change_state.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_cone.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/features/normal_3d.h>

#include <unordered_set>

#include "cuda_runtime.h"
#include "../lib/cudaCluster.h"

using std::placeholders::_1;
using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

uint32_t HSVtoRGB(float H, float S, float V);

class ClusteringGPU : public rclcpp::Node
{
public:
    ClusteringGPU()
        : Node("lidar_clustering_gpu")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/livox/lidar", 10, std::bind(&ClusteringGPU::on_pc_update, this, _1));
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("lidar/colored_clusters", 10);

        capacityPoints = 300000;  // Initialize to 3,000,000 points

        cloud = new pcl::PCLPointCloud2();
        cloudNew = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        outputPC = new pcl::PointCloud<pcl::PointXYZRGB>();
        outputPCPtr = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(outputPC);
        outputPCPtr->points.reserve(capacityPoints);

        // init cuda
        cudaStreamCreate(&stream);
        this->gpumen_alloc(capacityPoints);
    }

    CallbackReturn on_shutdown(const rclcpp_lifecycle::State &) {
        cudaFree(inputEC);
        cudaFree(outputEC);
        cudaFree(indexEC);
        delete(cloud);
        delete(outputPC);
        return CallbackReturn::SUCCESS;
    }

private:
    unsigned int capacityPoints;
    pcl::PCLPointCloud2* cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudNew;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr outputPCPtr;
    pcl::PointCloud<pcl::PointXYZRGB> *outputPC;

    float *inputEC;
    float *outputEC;
    unsigned int *indexEC;
    cudaStream_t stream;

    void gpumen_alloc(unsigned int sizeEC) {
        cudaMallocManaged(&inputEC, sizeof(float) * 4 * sizeEC, cudaMemAttachHost);
        cudaStreamAttachMemAsync(stream, inputEC);

        cudaMallocManaged(&outputEC, sizeof(float) * 4 * sizeEC, cudaMemAttachHost);
        cudaStreamAttachMemAsync(stream, outputEC);

        cudaMallocManaged(&indexEC, sizeof(float) * 4 * sizeEC, cudaMemAttachHost);
        cudaStreamAttachMemAsync(stream, indexEC);
    }

    void on_pc_update(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) const
    {
        pcl_conversions::toPCL(*cloud_msg, *cloud);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudNew(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromPCLPointCloud2(*cloud, *cloudNew);

        /*add cuda cluster*/
        unsigned int sizeEC = cloudNew->size();
        // if (sizeEC > capacityPoints) {
        //     gpumen_alloc(sizeEC);
        //     capacityPoints = sizeEC;
        // }

        // Zero out GPU Mem
        cudaMemsetAsync(inputEC, 0, sizeof(float) * 4 * capacityPoints, stream);
        cudaMemsetAsync(outputEC, 0, sizeof(float) * 4 * capacityPoints, stream);
        cudaMemsetAsync(indexEC, 0, sizeof(float) * 4 * capacityPoints, stream);

        // Copy data to GPU Mem
        cudaMemcpyAsync(inputEC, cloudNew->points.data(), sizeof(float) * 4 * sizeEC, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(outputEC, cloudNew->points.data(), sizeof(float) * 4 * sizeEC, cudaMemcpyHostToDevice, stream);
        outputPCPtr->points.clear();
        cudaStreamSynchronize(stream);

        // Init Clustering
        extractClusterParam_t ecp;
        ecp.minClusterSize = 100;
        ecp.maxClusterSize = 2500000;
        ecp.voxelX = 0.01;
        ecp.voxelY = 0.01;
        ecp.voxelZ = 0.01;
        ecp.countThreshold = 20;
        cudaExtractCluster cudaec(stream);
        cudaec.set(ecp);

        cudaec.extract(inputEC, sizeEC, outputEC, indexEC);
        cudaStreamSynchronize(stream);
        
        auto output = sensor_msgs::msg::PointCloud2();
        float hue = 0;
        int num_cone_detected = 0;

        for (int cluster_index = 1; cluster_index <= indexEC[0]; cluster_index++)
        {
            unsigned int outoff = 0;
            for (int w = 1; w < cluster_index; w++)
            {
                if (cluster_index>1) {
                    outoff += indexEC[w];
                }
            }

            // Create cluster
            pcl::PointCloud<pcl::PointXYZ> *cluster = new pcl::PointCloud<pcl::PointXYZ>();
            pcl::PointCloud<pcl::PointXYZ>::Ptr clusterPtr (cluster);
            cluster->points.reserve(indexEC[cluster_index]);
            for (std::size_t k = 0; k < indexEC[cluster_index]; ++k)
            {
                pcl::PointXYZ point;
                point.x = outputEC[(outoff+k)*4+0];
                point.y = outputEC[(outoff+k)*4+1];
                point.z = outputEC[(outoff+k)*4+2];
                clusterPtr->points.push_back(point);
            }

            pcl::PointCloud<pcl::Normal> *clusterNorm = new pcl::PointCloud<pcl::Normal>();
            pcl::PointCloud<pcl::Normal>::Ptr clusterNormPtr (clusterNorm);
            pcl::search::KdTree<pcl::PointXYZ> *tree = new pcl::search::KdTree<pcl::PointXYZ> ();
            pcl::search::KdTree<pcl::PointXYZ>::Ptr treePtr (tree);
            pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_est;
            normal_est.setSearchMethod (treePtr);
            normal_est.setInputCloud (clusterPtr);
            normal_est.setKSearch (50);
            normal_est.compute (*clusterNormPtr);

            pcl::SampleConsensusModelCone<pcl::PointXYZ, pcl::Normal> *model_cone = new pcl::SampleConsensusModelCone<pcl::PointXYZ, pcl::Normal> (clusterPtr);
            pcl::SampleConsensusModelCone<pcl::PointXYZ, pcl::Normal>::Ptr model_cone_ptr (model_cone);
            model_cone_ptr->setInputNormals(clusterNormPtr);
            pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_cone_ptr);
            ransac.setDistanceThreshold (.01);
            ransac.computeModel();
            std::vector<int> inliers;
            ransac.getInliers(inliers);
            std::unordered_set<int> inliers_set;
            if (inliers.size() > 100) {
                num_cone_detected ++;
            }
            for (unsigned int i = 0; i < inliers.size(); i++) {
                inliers_set.insert(inliers[i]);
            } 

            
            // Add to pointcloud msg
            uint32_t black_rgb = HSVtoRGB(hue, 90, 50);
            uint32_t cone_rgb = HSVtoRGB(30, 100, 100);
            for (std::size_t k = 0; k < indexEC[cluster_index]; ++k)
            {
                pcl::PointXYZRGB point;
                if (inliers_set.find(*reinterpret_cast<int*>(&k)) != inliers_set.end()) {
                    point.rgb = *reinterpret_cast<float*>(&cone_rgb);
                } else {
                    point.rgb = *reinterpret_cast<float*>(&black_rgb);
                }
                
                point.x = outputEC[(outoff+k)*4+0];
                point.y = outputEC[(outoff+k)*4+1];
                point.z = outputEC[(outoff+k)*4+2];
                outputPCPtr->points.push_back(point);
            }

            // free object
            delete cluster;
            delete model_cone;
            delete clusterNorm;
        }

        // std::cout << "Num Cone Detected: " << num_cone_detected << std::endl;

        pcl::PCLPointCloud2 outputPCL;
        // convert to pcl::PCLPointCloud2
        pcl::toPCLPointCloud2(*outputPCPtr, outputPCL);

        // Convert to ROS data type
        pcl_conversions::fromPCL(outputPCL, output);

        // publish the clusters
        output.header.stamp.sec = cloud_msg->header.stamp.sec;
        output.header.stamp.nanosec = cloud_msg->header.stamp.nanosec;
        output.header.frame_id = cloud_msg->header.frame_id;
        publisher_->publish(output);
    }
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
};

uint32_t HSVtoRGB(float H, float S, float V)
{
    if (H > 360 || H < 0 || S > 100 || S < 0 || V > 100 || V < 0)
    {
        std::cout << "The givem HSV values are not in valid range" << std::endl;
        return 0;
    }
    float s = S / 100;
    float v = V / 100;
    float C = s * v;
    float X = C * (1 - abs(fmod(H / 60.0, 2) - 1));
    float m = v - C;
    float r, g, b;
    if (H >= 0 && H < 60)
    {
        r = C, g = X, b = 0;
    }
    else if (H >= 60 && H < 120)
    {
        r = X, g = C, b = 0;
    }
    else if (H >= 120 && H < 180)
    {
        r = 0, g = C, b = X;
    }
    else if (H >= 180 && H < 240)
    {
        r = 0, g = X, b = C;
    }
    else if (H >= 240 && H < 300)
    {
        r = X, g = 0, b = C;
    }
    else
    {
        r = C, g = 0, b = X;
    }
    uint8_t R = (r + m) * 255;
    uint8_t G = (g + m) * 255;
    uint8_t B = (b + m) * 255;
    return (uint32_t)R << 16 | (uint32_t)G << 8 | (uint32_t)B;
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ClusteringGPU>());
    rclcpp::shutdown();
    return 0;
}