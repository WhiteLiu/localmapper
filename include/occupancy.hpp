#ifndef LOCALMAPPER_OCCUPANCY_HPP
#define LOCALMAPPER_OCCUPANCY_HPP

#include <unordered_set>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>


template <class T>
inline Eigen::Matrix<T,3,1> backproject(const Eigen::Transform<T,3,Eigen::Projective> &K,T u,T v)
{
    return Eigen::Matrix<T,3,1>((u-K(0,2))/K(0,0),(v-K(1,2))/K(1,1),1.0);
}

/// compute log-odds from probability:
template <class T>
inline T logodds(T probability)
{
    return (float) std::log(probability/(1-probability));
}

/// compute probability from logodds:
template <class T>
inline T probability(T logodds){
    return 1. - ( 1. / (1. + std::exp(logodds)));

}


struct scan2d
{
    // Build a non oriented 2d scan
    scan2d(const cv::Mat & depth,const Eigen::Projective3d & K, const Eigen::Isometry3d & warp,const std::pair<float,float> &range);

    std::vector<std::pair<float,float>> data;
    Eigen::Array2f origin;
    float resolution;
};



struct occupancy2d
{
public:

    float default_prob_miss = 0.4;
    float default_prob_hit = 0.7;

public:

    occupancy2d():
        width_(0),
        height_(0),
        resolution_(0.0),
        origin_(Eigen::Isometry3d::Identity())
    {
        width_ = 1024;
        height_ = 1024;
        origin_(0,3) = -20.0;
        origin_(1,3) = -20.0;
        data_.resize(width_*height_,std::numeric_limits<float>::quiet_NaN());
    }


    occupancy2d(float resolution):
        width_(0),
        height_(0),
        resolution_(resolution)
    {
    }

    occupancy2d(const occupancy2d &rhs):
        width_(rhs.width_),
        height_(rhs.height_),
        resolution_(rhs.resolution_),
        data_(rhs.data_),
        origin_(rhs.origin_)
    {
    }

    void clear()
    {
        for(float &s : data_)
            s = std::numeric_limits<float>::quiet_NaN();
    }


    // Trace a ray from origin to points and recompute bounds
    void update(const scan2d & scan);


    // Rewarp the map to a new origin //
    occupancy2d warpFrom(const Eigen::Isometry3d & warp,double max_dist, bool clear=false);


    std::vector<float> probabilties() const
    {
        std::vector<float> res(data_.size(),-1.0);

        for(size_t i = 0 ; i < res.size(); ++i)
        {
            res[i] = probability(data_[i]);
        }
        return res;
    }

    inline float bilinear(float u,float v) const
    {
        int iu = int(u);
        int iv = int(v);

        float alpha_u = u - float(iu);
        float alpha_v = v - float(iv);
        int idx = iu + iv*width_;

        return (data_[idx]*(1.0f-alpha_u) + data_[idx+1]*alpha_u)*(1.0f-alpha_v) + (data_[idx+width_]*(1.0f-alpha_u) + data_[idx+width_+1]*alpha_u)*alpha_v;
    }

    inline float nearest(float u,float v) const
    {
        int iu = std::round(u);
        int iv = std::round(v);
        int idx = iu + iv*width_;
        return data_[idx];
    }


public:

    int width_ = 0;
    int height_ = 0;
    float resolution_ = 0.0;
    std::vector<float> data_; // We store the log of the probabilty
    Eigen::Isometry3d origin_ = Eigen::Isometry3d::Identity();
};


#endif
