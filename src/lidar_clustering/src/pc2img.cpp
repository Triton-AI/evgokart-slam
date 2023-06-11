#define BOOST_BIND_NO_PLACEHOLDERS
#include <memory>
// ROS core
#include "rclcpp/rclcpp.hpp"
// Image message
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
// pcl::toROSMsg
#include <pcl_conversions/pcl_conversions.h>
// stl stuff
#include <string>
#include <cmath>

using std::placeholders::_1;

class PointCloudToImage : public rclcpp::Node
{
public:
    PointCloudToImage() 
    : Node("pointcloud_to_image")
    {
        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("lidar/data", 30,
                             std::bind(&PointCloudToImage::cloud_cb, this, _1));
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("lidar/image", 30);
    }

private:
    void
    cloud_cb(const sensor_msgs::msg::PointCloud2::SharedPtr cloud) const
    {
        if ((cloud->width * cloud->height) == 0)
            return; // return if the cloud is not dense!
        sensor_msgs::msg::Image::SharedPtr image_(new sensor_msgs::msg::Image);
        try
        {
            pcl::toROSMsg(*cloud, *image_); // convert the cloud
            image_->header.stamp.sec = cloud->header.stamp.sec;
            image_->header.stamp.nanosec = cloud->header.stamp.nanosec;
            image_->header.frame_id = cloud->header.frame_id;
            int row = (int) std::sqrt(image_->width);
            // Very rude way of reshaping it
            image_->width = row;
            image_->height = row;
            image_->step = row * row;
            
            image_->data = image_->data;
        }
        catch (std::runtime_error e)
        {
            RCLCPP_ERROR_STREAM(this->get_logger(), "Error in converting cloud to image message: " << e.what());
        }
        image_pub_->publish(*image_); // publish our cloud image
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;  // cloud subscriber
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;     // image message publisher
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudToImage>());           // where she stops nobody knows
    rclcpp::shutdown();
    return 0;
}