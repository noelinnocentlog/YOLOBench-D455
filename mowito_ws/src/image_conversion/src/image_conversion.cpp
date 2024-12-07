#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <opencv2/opencv.hpp>

class ImageConversionNode : public rclcpp::Node {
public:
    ImageConversionNode() : Node("image_conversion_node"), mode_(1) {
        // Subscriber for input images
        image_subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10,
            std::bind(&ImageConversionNode::imageCallback, this, std::placeholders::_1));

        // Publisher for output images
        image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/image_converted", 10);

        // Service to change the mode
        mode_service_ = this->create_service<std_srvs::srv::SetBool>(
            "change_mode", std::bind(&ImageConversionNode::changeMode, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "ImageConversionNode started!");
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        // Convert ROS2 image to OpenCV format
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // Process the image based on the mode
        if (mode_ == 1) {
            cv::cvtColor(cv_ptr->image, cv_ptr->image, cv::COLOR_BGR2GRAY);
            cv_ptr->encoding = sensor_msgs::image_encodings::MONO8;
        }

        // Publish the converted image
        image_publisher_->publish(*cv_ptr->toImageMsg());
    }

    void changeMode(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                    std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
        mode_ = request->data ? 1 : 2;
        response->success = true;
        response->message = "Mode changed to " + std::string(mode_ == 1 ? "Grayscale" : "Color");
        RCLCPP_INFO(this->get_logger(), "Mode changed to: %s", mode_ == 1 ? "Grayscale" : "Color");
    }

    int mode_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscriber_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr mode_service_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageConversionNode>());
    rclcpp::shutdown();
    return 0;
}

