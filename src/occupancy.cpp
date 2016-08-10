#include "occupancy.hpp"
#include <iostream>


scan2d::scan2d(const cv::Mat & depth,const Eigen::Projective3d & K, const Eigen::Isometry3d & warp,const std::pair<float,float> &range)
{

    if(depth.type()!=CV_16U)
        throw std::runtime_error("Invalid input depth format");

    float fx = K(0,0);
    int width = depth.cols;
    int height = depth.rows;

    // Camera aperture
    float ap = 2*std::atan(0.5/fx);

    resolution = ap/width; // rad/pixel


    int scan_size = 2.0*M_PI/resolution;

    data = std::vector<std::pair<float,float>>(scan_size,std::pair<float,float>(0.0,0.0));

    const uint16_t *depth_in = depth.ptr<uint16_t>();

    Eigen::Projective3f Kf =  K.cast<float>();
    Eigen::Isometry3f warpf = warp.cast<float>();

    // Store the 2d origin //
    origin = Eigen::Vector2f(warp(0,3),warp(1,3));



    // Convert depth image to 2D range
    for (int i = 0 ; i < height; ++i)
    {
        for (int j = 0 ; j < width; ++j)
        {
            float Z = float(depth_in[j+i*width])/1000.0;
            float w(1.0);

            if(Z > 0)
            {
                // Get 3D point
                Eigen::Vector3f v = backproject(Kf,float(j)/width,float(i)/height)*Z;

                // Warp to map //
                Eigen::Vector3f vw =  warpf.rotation()*v;


                /*std::cout<<width<<"  "<<height<<std::endl;
                std::cout<<Kf.matrix()<<std::endl;
                std::cout<<v.transpose()<<std::endl;
                std::cout<<vw.transpose()<<std::endl;
                getchar();*/

                // if point is in range
                if(vw(2) >= range.first && vw(2) < range.second)
                {
                    // Transform to range //
                    float theta = std::atan2(vw(1),vw(0))+M_PI;

                    float rho = std::sqrt(vw(0)*vw(0) + vw(1)*vw(1));
                    int thetai = std::round(theta/resolution);


                    if(data[thetai].first == 0.0 || (rho < data[thetai].first)) // Get the mindist
                    {
                        data[thetai].first = rho;
                        data[thetai].second = std::max(data[thetai].second,w);
                    }
                }
            }
        }
    }
}


void occupancy2d::update(const scan2d & scan)
{
    // Get min/max
    Eigen::Array2f scan_mini(1e9,1e9);
    Eigen::Array2f scan_maxi(-1e9,-1e9);


    // Get min/max of 2D scan within the map
    for(size_t i = 0 ; i < scan.data.size(); ++i)
    {
        float rho = scan.data[i].first;

        if(rho > 0.0)
        {
            // Polar to cart
            float theta = float(i)*scan.resolution - M_PI;
            float x = rho*std::cos(theta) + scan.origin(0);
            float y = rho*std::sin(theta) + scan.origin(1);

            scan_mini = scan_mini.min(Eigen::Array2f(x,y));
            scan_maxi = scan_maxi.max(Eigen::Array2f(x,y));
        }
    }


    // Add origin //
    scan_mini = scan_mini.min(scan.origin);
    scan_maxi = scan_maxi.max(scan.origin);

    Eigen::Array2f map_mini = Eigen::Array2f(origin_(0,3),origin_(1,3));
    Eigen::Array2f map_maxi = map_mini + Eigen::Array2f(float(width_)*resolution_,float(height_)*resolution_);

    // Extend Current map if scan is outside //
    if(scan_mini(0) < map_mini(0) || scan_mini(1) < map_mini(1) ||  scan_maxi(0) > map_maxi(0) || scan_maxi(1) > map_maxi(1))
    {
        Eigen::Array2f new_mini = map_mini.min(scan_mini);
        Eigen::Array2f new_maxi = map_maxi.max(scan_maxi);

        // Extend the original map //
        int new_width =  std::round((new_maxi(0)-new_mini(0))/resolution_);
        int new_height = std::round((new_maxi(1)-new_mini(1))/resolution_);

        // Initialize map with unknown //
        std::vector<float> update(new_width*new_height,std::numeric_limits<float>::quiet_NaN());

        int offset_x = std::round((map_mini(0) - new_mini(0))/resolution_);
        int offset_y = std::round((map_mini(1) - new_mini(1))/resolution_);

        new_mini(0) = map_mini(0) - resolution_*offset_x;
        new_mini(1) = map_mini(1) - resolution_*offset_y;


        offset_x = std::max(0,offset_x);
        offset_y = std::max(0,offset_y);

        // Copy the old map
        for(int i = 0 ; i < height_; ++i)
        {
            std::copy(&data_[i*width_], &data_[(i)*width_]+width_,  &update[offset_x + (offset_y + i)*new_width]);
        }

        data_ = update;
        width_ = new_width;
        height_ = new_height;

        // Update origin -> top left corner //
        origin_(0,3) = new_mini(0);
        origin_(1,3) = new_mini(1);
    }

    std::unordered_set<int> free_cells;
    std::unordered_set<int> occupied_cells;

    for(size_t i = 0 ; i < scan.data.size(); ++i)
    {
        float rho = scan.data[i].first;
        float w = scan.data[i].second;

        if(rho > 0.0)
        {
            float theta = float(i)*scan.resolution - M_PI;


            // Convert start and end points to map pixel coords //
            int x1 = std::round((rho*std::cos(theta) + scan.origin(0) - origin_(0,3))/resolution_);
            int y1 = std::round((rho*std::sin(theta) + scan.origin(1) - origin_(1,3))/resolution_);
            int x0 = std::round((scan.origin(0) - origin_(0,3))/resolution_);
            int y0 = std::round((scan.origin(1) - origin_(1,3))/resolution_);

            // Setup occupied cells
            if(w > 0.0)
                occupied_cells.insert(x1+y1*width_);

            // Raytrace the free cells
            int dx = std::abs(x1 - x0);
            int dy = std::abs(y1 - y0);
            int x = x0;
            int y = y0;
            int n = 1 + dx + dy;
            int x_inc = (x1 > x0) ? 1 : -1;
            int y_inc = (y1 > y0) ? 1 : -1;
            int error = dx - dy;
            dx *= 2;
            dy *= 2;

            for (; n > 1; --n)
            {
                free_cells.insert(x+y*width_);

                if (error > 0)
                {
                    x += x_inc;
                    error -= dy;
                }
                else
                {
                    y += y_inc;
                    error += dx;
                }
            }
        }
    }


    for(int s : free_cells)
    {
        if(size_t(s)>=data_.size() || s < 0)
        {
            continue;
        }


        if(!occupied_cells.count(s))
        {
            // Update Free space
            if(std::isnan(data_[s]))
            {
                data_[s]=log_prob_miss_;
            }
            else
            {
                data_[s]+=log_prob_miss_;
            }
        }
    }

    for(int s : occupied_cells)
    {
        if(size_t(s)>=data_.size() || s < 0)
        {
            continue;
        }

        // Update occupied space
        if(std::isnan(data_[s]))
        {
            data_[s]=log_prob_hit_;
        }
        else
        {
            data_[s]+=log_prob_hit_;
        }
    }
}

occupancy2d occupancy2d::warpFrom(const Eigen::Isometry3d & warp,double max_dist, bool clear)
{
    occupancy2d warped;
    warped.resolution_ = resolution_;
    warped.width_ = std::round(max_dist/resolution_);
    warped.height_ = std::round(max_dist/resolution_);
    warped.data_ = std::vector<float>(warped.width_*warped.height_,std::numeric_limits<float>::quiet_NaN());


    warped.origin_ = Eigen::Isometry3d::Identity();
    warped.origin_(0,3) = -float(resolution_*warped.width_)/2.0;
    warped.origin_(1,3) = -float(resolution_*warped.height_)/2.0;


    Eigen::Isometry3f warpf = warp.cast<float>();

    Eigen::Array2f new_mini(1e9,1e9);
    Eigen::Array2f new_maxi(-1e9,-1e9);


    for(int i = 0 ; i < warped.height_ ; ++i)
    {
        for(int j = 0 ; j < warped.width_ ; ++j)
        {
            // point in local space
            Eigen::Vector3f v(float(j)*resolution_+warped.origin_(0,3),float(i)*resolution_+warped.origin_(1,3),0.0);

            if(v.squaredNorm() < max_dist)
            {
                // Point in map space
                Eigen::Vector3f vw = warpf*v;

                // Convert to grid
                float x = (vw(0)-origin_(0,3))/resolution_;
                float y = (vw(1)-origin_(1,3))/resolution_;


                if(x >=0 && y >=0 && x < width_-1 &&  y < height_-1)
                {
                    new_mini = new_mini.min(Eigen::Array2f(vw(0),vw(1)));
                    new_maxi = new_maxi.max(Eigen::Array2f(vw(0),vw(1)));



                    warped.data_[j+i*warped.width_] = this->bilinear(x,y);
                }
            }
        }
    }




    if(clear)
    {
        Eigen::Array2f map_mini = Eigen::Array2f(origin_(0,3),origin_(1,3));
        Eigen::Array2f map_maxi = map_mini + Eigen::Array2f(float(width_)*resolution_,float(height_)*resolution_);

        int new_width =  std::round((new_maxi(0)-new_mini(0))/resolution_);
        int new_height = std::round((new_maxi(1)-new_mini(1))/resolution_);


        std::cout<<"New "<<new_width<<" "<<new_height<<std::endl;
        std::cout<<"Prev "<<width_<<" "<<height_<<std::endl;


        int offset_x = std::round((map_mini(0) - new_mini(0))/resolution_);
        int offset_y = std::round((map_mini(1) - new_mini(1))/resolution_);

        new_mini(0) = map_mini(0) - resolution_*offset_x;
        new_mini(1) = map_mini(1) - resolution_*offset_y;

        // Initialize map with unknown //
        std::vector<float> update(new_width*new_height,std::numeric_limits<float>::quiet_NaN());

        offset_x = std::max(0,offset_x);
        offset_y = std::max(0,offset_y);

        // Copy the old block
        for(int i = 0 ; i < new_height; ++i)
        {
            for(int j = 0 ; j < new_width; ++j)
            {
                update[j+i*new_width] = data_[j + offset_x + (offset_y + i)*width_];
            }
        }

        width_ = new_width;
        height_ = new_height;
        data_ = update;

        // Update origin -> top left corner //
        origin_(0,3) = new_mini(0);
        origin_(1,3) = new_mini(1);
    }



    return warped;
}

