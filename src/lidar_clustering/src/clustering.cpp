#define BOOST_BIND_NO_PLACEHOLDERS

#include <memory>

#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>
#include "custom_msgs/msg/cone_xyz.hpp"
#include "custom_msgs/msg/cones_xyz.hpp"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_cone.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/features/normal_3d.h>

#include <Eigen/Core>

using std::placeholders::_1;

uint32_t HSVtoRGB(float H, float S,float V);

class Clustering : public rclcpp::Node
{
  public:
    Clustering()
    : Node("lidar_clustering")
    {
      subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/livox/lidar", 10, std::bind(&Clustering::on_pc_update, this, _1));
      publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("lidar/colored_clusters", 10);
      cones_pc2_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("cone_points", 10);
      cones_xyz_publisher_ = this->create_publisher<custom_msgs::msg::ConesXYZ>("lidar/cones", 10);
    }

  private:
    void on_pc_update(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) const
    {
      // https://github.com/jupidity/PCL-ROS-cluster-Segmentation/blob/master/src/segmentation.cpp
      // Container for original & filtered data
      pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2;
      pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);
      pcl::PCLPointCloud2* cloud_filtered = new pcl::PCLPointCloud2;
      pcl::PCLPointCloud2Ptr cloudFilteredPtr (cloud_filtered);

      // Convert to PCL data type
      pcl_conversions::toPCL(*cloud_msg, *cloud);


      // Perform voxel grid downsampling filtering
      pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
      sor.setInputCloud (cloudPtr);
      sor.setLeafSize (0.02, 0.02, 0.02);
      sor.filter (*cloudFilteredPtr);


      pcl::PointCloud<pcl::PointXYZRGB> *xyz_cloud = new pcl::PointCloud<pcl::PointXYZRGB>;
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyzCloudPtr (xyz_cloud); // need a boost shared pointer for pcl function inputs

      // convert the pcl::PointCloud2 tpye to pcl::PointCloud<pcl::PointXYZRGB>
      pcl::fromPCLPointCloud2(*cloudFilteredPtr, *xyzCloudPtr);


      //perform passthrough filtering to remove top view

      // create a pcl object to hold the passthrough filtered results
      pcl::PointCloud<pcl::PointXYZRGB> *xyz_cloud_filtered = new pcl::PointCloud<pcl::PointXYZRGB>;
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyzCloudPtrFiltered (xyz_cloud_filtered);

      // Create the filtering object
      pcl::PassThrough<pcl::PointXYZRGB> pass;
      pass.setInputCloud (xyzCloudPtr);
      pass.setFilterFieldName ("z");
      pass.setFilterLimits (-0.15, 1.2-0.15);
      //pass.setFilterLimitsNegative (true);
      pass.filter (*xyzCloudPtrFiltered);


      // auto xyzCloudPtrFiltered = xyzCloudPtr;
      // create a pcl object to hold the ransac filtered results
      pcl::PointCloud<pcl::PointXYZRGB> *xyz_cloud_ransac_filtered = new pcl::PointCloud<pcl::PointXYZRGB>;
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyzCloudPtrRansacFiltered (xyz_cloud_ransac_filtered);


      // perform ransac planar filtration to remove table top
      pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
      pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
      // Create the segmentation object
      pcl::SACSegmentation<pcl::PointXYZRGB> seg1;
      // Optional
      seg1.setOptimizeCoefficients (true);
      // Mandatory
      seg1.setModelType (pcl::SACMODEL_PLANE);
      seg1.setMethodType (pcl::SAC_RANSAC);
      seg1.setDistanceThreshold (0.08);

      seg1.setInputCloud (xyzCloudPtrFiltered);
      seg1.segment (*inliers, *coefficients);


      // Create the filtering object
      pcl::ExtractIndices<pcl::PointXYZRGB> extract;

      //extract.setInputCloud (xyzCloudPtrFiltered);
      extract.setInputCloud (xyzCloudPtrFiltered);
      extract.setIndices (inliers);
      extract.setNegative (true);
      extract.filter (*xyzCloudPtrRansacFiltered);

      // perform euclidean cluster segmentation to seporate individual objects
      // Create the KdTree object for the search method of the extraction
      // auto xyzCloudPtrRansacFiltered = xyzCloudPtr;

      pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
      tree->setInputCloud (xyzCloudPtrRansacFiltered);

      // create the extraction object for the clusters
      std::vector<pcl::PointIndices> cluster_indices;
      pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
      // specify euclidean cluster parameters
      ec.setClusterTolerance (0.10); // 2cm
      ec.setMinClusterSize (25);
      ec.setMaxClusterSize (400);
      ec.setSearchMethod (tree);
      ec.setInputCloud (xyzCloudPtrRansacFiltered);
      // exctract the indices pertaining to each cluster and store in a vector of pcl::PointIndices
      ec.extract (cluster_indices);

      // declare an instance of the SegmentedClustersArray mecolcossage
      // obj_recognition::SegmentedClustersArray CloudClusters;

      // declare the output variable instances
      auto output = sensor_msgs::msg::PointCloud2();
      auto cones_pc_output = sensor_msgs::msg::PointCloud2();
      auto cones_output = custom_msgs::msg::ConesXYZ();
      pcl::PCLPointCloud2 outputPCL;
      pcl::PCLPointCloud2 cones_outputPCL;

      // create a pcl object to hold the extracted cluster
      pcl::PointCloud<pcl::PointXYZRGB> *cluster = new pcl::PointCloud<pcl::PointXYZRGB>;
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusterPtr (cluster);
      pcl::PointCloud<pcl::PointXYZ>::Ptr conesPtr (new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointXYZRGB point;
      float hue = 0;
      int num_cone_detected = 0;

      uint32_t cone_rgb = HSVtoRGB(30, 100, 100);
      // here, cluster_indices is a vector of indices for each cluster. iterate through each indices object to work with them seporately
      int point_processed = 0;
      bool cone_detected_ = false;
      for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end (); ++it)
      {
        uint32_t rgb = HSVtoRGB(hue, 90, 50);
        bool cone_detected=false;

        // Cone Detection
        pcl::PointCloud<pcl::PointXYZ> *cluster1 = new pcl::PointCloud<pcl::PointXYZ>();
        pcl::PointCloud<pcl::PointXYZ>::Ptr clusterPtr1 (cluster1);

        float num_points = 0;
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
        {
          pcl::PointXYZ *pointB = new pcl::PointXYZ();
          point = xyzCloudPtrRansacFiltered->points[*pit];
          pointB->x = point.x;
          pointB->y = point.y;
          pointB->z = point.z;
          clusterPtr1->points.push_back(*pointB);
          num_points++;
        }
        
        pcl::PointXYZ centroid;
        pcl::computeCentroid(*clusterPtr1, centroid);

        if (centroid.x > 15.0) {
          std::cout << "OOB (x): " << centroid.x << ", " << centroid.y <<  ", "<< centroid.z << " ; "<< std::endl;
        } else if (centroid.y > 4.5 || centroid.y < -4.5) {
          std::cout << "OOB (y): " << centroid.x << ", " << centroid.y <<  ", "<< centroid.z << " ; "<< std::endl;
        } else if (centroid.z > 1.5 || centroid.z < 0.0){
          std::cout << "OOB (z): " << centroid.x << ", " << centroid.y <<  ", "<< centroid.z << " ; " << std::endl;
        } else {
          std::cout << centroid.x << ", " << centroid.y <<  ", "<< centroid.z << " ; "<< std::endl;
          auto cone = custom_msgs::msg::ConeXYZ();
          cone.x = centroid.x;
          cone.y = centroid.y;
          cone.z = centroid.z;
          cone_detected=true;
          cone_detected_=true;
          num_cone_detected ++;
          cones_output.cones.push_back(cone);
          conesPtr->points.push_back(centroid);
        }

        // now we are in a vector of indices pertaining to a single cluster.
        // Assign each point corresponding to this cluster in xyzCloudPtrPassthroughFiltered a specific color for identification purposes
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
        {
          point = xyzCloudPtrRansacFiltered->points[*pit];
          if (cone_detected == true) {
            point.rgb = *reinterpret_cast<float*>(&cone_rgb);
          } else {
            point.rgb = *reinterpret_cast<float*>(&rgb);
          }
          clusterPtr->points.push_back(point);
          point_processed++;
        }

        hue += 360/cluster_indices.size();
      }
      std::cout << std::endl;
      std::cout << "Num Cone Detected: " << num_cone_detected << std::endl;
      std::cout << point_processed << std::endl;

      // convert to pcl::PCLPointCloud2
      pcl::toPCLPointCloud2( *clusterPtr ,outputPCL);
      pcl::toPCLPointCloud2( *conesPtr, cones_outputPCL);

      // Convert to ROS data type
      pcl_conversions::fromPCL(outputPCL, output);
      pcl_conversions::fromPCL(cones_outputPCL, cones_pc_output);

      // publish the clusters
      output.header.stamp.sec = cloud_msg->header.stamp.sec;
      output.header.stamp.nanosec = cloud_msg->header.stamp.nanosec;
      output.header.frame_id = cloud_msg->header.frame_id;

      cones_pc_output.header.stamp.sec = cloud_msg->header.stamp.sec;
      cones_pc_output.header.stamp.nanosec = cloud_msg->header.stamp.nanosec;
      cones_pc_output.header.frame_id = cloud_msg->header.frame_id;

      publisher_->publish(output);
      if (cone_detected_ == true) {
        cones_xyz_publisher_->publish(cones_output);
        cones_pc2_publisher_->publish(cones_pc_output);
      }
    }
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cones_pc2_publisher_;
    rclcpp::Publisher<custom_msgs::msg::ConesXYZ>::SharedPtr cones_xyz_publisher_; 
};

uint32_t HSVtoRGB(float H, float S,float V){
    if(H>360 || H<0 || S>100 || S<0 || V>100 || V<0){
        std::cout<<"The givem HSV values are not in valid range"<<std::endl;
        return 0;
    }
    float s = S/100;
    float v = V/100;
    float C = s*v;
    float X = C*(1-abs(fmod(H/60.0, 2)-1));
    float m = v-C;
    float r,g,b;
    if(H >= 0 && H < 60){
        r = C,g = X,b = 0;
    }
    else if(H >= 60 && H < 120){
        r = X,g = C,b = 0;
    }
    else if(H >= 120 && H < 180){
        r = 0,g = C,b = X;
    }
    else if(H >= 180 && H < 240){
        r = 0,g = X,b = C;
    }
    else if(H >= 240 && H < 300){
        r = X,g = 0,b = C;
    }
    else{
        r = C,g = 0,b = X;
    }
    uint8_t R = (r+m)*255;
    uint8_t G = (g+m)*255;
    uint8_t B = (b+m)*255;
    return (uint32_t)R << 16 | (uint32_t)G << 8 | (uint32_t)B;
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Clustering>());
  rclcpp::shutdown();
  return 0;
}
