#include "ros/ros.h"

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <image_transport/camera_common.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <nav_msgs/OccupancyGrid.h>
#include <sensor_msgs/LaserScan.h>

#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>

#include <iostream>
#include "occupancy.hpp"

int count( 0 );
occupancy2d map_;
sensor_msgs::LaserScan laserscan;

ros::Publisher occupancy2d_pub_;
ros::Publisher scan_pub;

std::shared_ptr< tf::TransformListener > tfl_;

int num_point;

std::string odom_frame_;
std::string base_frame_;

double tf_timeout_ = 5.0;
double local_radius = 0.0;
bool clear_global_map = false;
std::pair< float, float > range_ = {0.0, 0.17};

Eigen::Projective3d fromCameraInfo( const sensor_msgs::CameraInfoConstPtr& info )
{
    Eigen::Projective3d out = Eigen::Projective3d::Identity();
    out( 0, 0 ) = info->K[0] / info->width;
    out( 0, 2 ) = info->K[2] / info->width;
    out( 1, 1 ) = info->K[4] / info->height;
    out( 1, 2 ) = info->K[5] / info->height;
    return out;
}

Eigen::Isometry3d fromTf( const tf::Pose& p )
{
    Eigen::Isometry3d tmp;
    tf::poseTFToEigen( p, tmp );
    return tmp;
}

geometry_msgs::Pose toRosMsg( const Eigen::Isometry3d& p )
{
    tf::Pose tmp;
    tf::poseEigenToTF( p, tmp );

    geometry_msgs::Pose out;
    tf::poseTFToMsg( tmp, out );
    return out;
}

void publish( const std::string& frame_id, const occupancy2d& map, const ros::Time& stamp )
{
    nav_msgs::OccupancyGrid::Ptr occ( new nav_msgs::OccupancyGrid );

    occ->header.stamp = stamp;
    occ->header.frame_id = frame_id;
    occ->info.width = map.width_;
    occ->info.height = map.height_;
    occ->data.resize( occ->info.width * occ->info.height, -1 );
    occ->info.resolution = map.resolution_;

    occ->info.origin = toRosMsg( map.origin_ );

    for ( size_t i = 0; i < map.data_.size(); ++i )
    {
        if ( std::isfinite( map.data_[i] ) )
            occ->data[i] = 100 * probability( map.data_[i] );
        else
            occ->data[i] = -1;
    }

    // occupancy2d_pub_.publish(occ);
}

bool safe_lookup( tf::StampedTransform& trans, tf::TransformListener& L, const std::string& target_frame,
                  const std::string& source_frame, const ros::Time& time, const ros::Duration& timeout )
{
    try
    {
        if ( L.waitForTransform( target_frame, source_frame, time, timeout ) )
        {
            L.lookupTransform( target_frame, source_frame, time, trans );
            return true;
        }
        else
        {
            ROS_WARN_STREAM( "Could not lookup " << target_frame << " to " << source_frame << " tf, timeout expired ("
                                                 << timeout << "s)" );
            return false;
        }
    }
    catch ( tf::TransformException ex )
    {
        ROS_ERROR( "%s", ex.what() );
        return false;
    }
}

void on_new_depth_message( const sensor_msgs::Image::ConstPtr& depth_image_msg,
                           const sensor_msgs::CameraInfoConstPtr& depth_info )
{
    cv::Mat depth = cv_bridge::toCvShare( depth_image_msg )->image.clone();

    // Get the the transform that maps a camera point to the odom frame
    tf::StampedTransform camera_to_base;
    if ( !safe_lookup( camera_to_base, *tfl_, base_frame_, depth_image_msg->header.frame_id, depth_image_msg->header.stamp,
                       ros::Duration( tf_timeout_ ) ) )
        return;

    // Transform camera points to map
    Eigen::Isometry3d CamToBase = fromTf( camera_to_base );

    // Get normalized camera matrix
    Eigen::Projective3d K = fromCameraInfo( depth_info );

    // Get new scan //
    scan2d scan( depth, K, CamToBase, range_ );

    num_point = scan.data.size();

    laserscan.header.frame_id = "base_footprint";
    laserscan.header.stamp = depth_image_msg->header.stamp;
    laserscan.angle_min = -M_PI;
    laserscan.angle_max = M_PI;
    laserscan.angle_increment = scan.resolution;
    laserscan.range_max = 10;
    laserscan.range_min = 0.3;
    laserscan.ranges.resize( num_point );
    laserscan.time_increment = 0;

    for ( size_t i = 0; i < scan.data.size(); ++i )
    {
        /*float rho = scan.data[i].first;

        if(rho > 0.0)
        {
            // Polar to cart
            float theta = float(i)*scan.resolution - M_PI;
            float x = rho*std::cos(theta) + scan.origin(0);
            float y = rho*std::sin(theta) + scan.origin(1);

            float dist = sqrt(x*x + y*y);
            std::cout<<"dist(x,y) = "<<dist<<std::endl;
            laserscan.ranges[i] = dist;

        }*/

        laserscan.ranges[i] = scan.data[i].first;
    }

    scan_pub.publish( laserscan );
}

int main( int argc, char* argv[] )
{
    ros::init( argc, argv, "" );

    ros::NodeHandle nh;
    ros::NodeHandle nh_private( "~" );

    image_transport::ImageTransport depth_image_transport_( nh );

    scan_pub = nh.advertise< sensor_msgs::LaserScan >( "/scan180", 1 );

    std::string depth_topic;
    nh_private.param( "depth", depth_topic, std::string( "/camera/depth_registred/image_raw" ) );
    std::string depth_transport;
    nh_private.param( "depth_transport", depth_transport, std::string( "compressedDepth" ) );
    ROS_INFO_STREAM( "Transport " << depth_transport );

    int queue_size;
    nh_private.param( "queue", queue_size, 5 );
    nh_private.param( "radius", local_radius, 3.0 );
    nh_private.param( "resolution", map_.resolution_, 0.05f );
    nh_private.param( "odom_frame", odom_frame_, std::string( "odom" ) );
    nh_private.param( "base_frame", base_frame_, std::string( "base_footprint" ) );
    nh_private.param( "clear", clear_global_map, false );
    nh_private.param( "occupancy2d_zmin", range_.first, 0.01f );
    nh_private.param( "occupancy2d_zmax", range_.second, 0.17f );

    double pmiss, phit, plow, pup;

    nh_private.param( "pmiss", pmiss, 0.3 );
    nh_private.param( "phit", phit, 0.7 );

    nh_private.param( "plow", plow, 0.1 );
    nh_private.param( "pup", pup, 1.0 );

    ROS_INFO_STREAM( "Probabilities hit " << phit << " miss " << pmiss << " "
                                          << " low " << plow << " up " << pup );

    map_.set_prob_miss( pmiss );
    map_.set_prob_hit( phit );
    map_.set_prob_low( plow );
    map_.set_prob_up( pup );

    if ( local_radius > 0.0 )
        occupancy2d_pub_ = nh_private.advertise< nav_msgs::OccupancyGrid >( "local_map", 1, true );
    else
        occupancy2d_pub_ = nh_private.advertise< nav_msgs::OccupancyGrid >( "global_map", 1, true );

    tfl_ = std::make_shared< tf::TransformListener >();

    image_transport::TransportHints depth_hints( depth_transport, ros::TransportHints(), nh_private );
    image_transport::SubscriberFilter depth_image_subscriber_( depth_image_transport_, depth_topic, queue_size, depth_hints );
    std::string depth_info_topic = image_transport::getCameraInfoTopic( depth_topic );
    message_filters::Subscriber< sensor_msgs::CameraInfo > depth_camera_info_subscriber_( nh, depth_info_topic, 1 );

    typedef message_filters::sync_policies::ExactTime< sensor_msgs::Image, sensor_msgs::CameraInfo > image_info_sync;

    message_filters::Synchronizer< image_info_sync > depth_sync( image_info_sync( queue_size ), depth_image_subscriber_,
                                                                 depth_camera_info_subscriber_ );
    message_filters::Connection depth_connection =
        message_filters::Connection( depth_sync.registerCallback( boost::bind( on_new_depth_message, _1, _2 ) ) );

    ros::spin();

    return 0;
}
