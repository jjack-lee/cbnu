#include "ros/ros.h"
int main(int argc, char **argv)
{
    // master에 등록
    ros::init(argc, argv, "test");
    // timestamp와 함께 출력.
    ROS_INFO("Hello ROS!");
    return 0;
}