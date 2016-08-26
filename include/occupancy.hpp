#ifndef LOCALMAPPER_OCCUPANCY_HPP
#define LOCALMAPPER_OCCUPANCY_HPP

#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <functional>

template < class T >
inline Eigen::Matrix< T, 3, 1 > backproject( const Eigen::Transform< T, 3, Eigen::Projective > &K, T u, T v )
{
    return Eigen::Matrix< T, 3, 1 >( ( u - K( 0, 2 ) ) / K( 0, 0 ), ( v - K( 1, 2 ) ) / K( 1, 1 ), 1.0 );
}

/// compute log-odds from probability:
template < class T >
inline constexpr T logodds( T probability )
{
    return static_cast< T >( std::log( probability / ( 1 - probability ) ) );
}

/// compute probability from logodds:
template < class T >
inline constexpr T probability( T logodds )
{
    return static_cast< T >( 1. - ( 1. / ( 1. + std::exp( logodds ) ) ) );
}

void initPixelRadialWeightLUT( int width, int height, std::vector< float > &lut, float per_cent_weight_at_edge = 0.05f );

class RhoWeightProbability
{
    private:
    struct
    {
        float min;
        float threshold;
        float max;
    } range_;
    float pc_at_threshold_;
    float lut_resolution_;
    std::vector< float > lut_;

    public:
    RhoWeightProbability( float range_min, float range_threshold, float range_max, float lut_resolution,
                          float per_cent_weight_at_threshold_range = 0.05 )
      : range_( {range_min, range_threshold, range_max} ), lut_resolution_( lut_resolution )
    {
        assert( 0 <= range_min && range_min <= range_threshold && range_threshold <= range_max );

        float sigma2w =
            -( range_.threshold * range_.threshold - 2. * range_.threshold * range_.min + range_.min * range_.min ) /
            std::log( per_cent_weight_at_threshold_range );
        int size = static_cast< int >( range_.max / lut_resolution_ );
        lut_.reserve( size );
        for ( int i = 0; i < size; ++i )
        {
            float dist = i * lut_resolution_ - range_.min;
            lut_.push_back( std::exp( -( dist * dist ) / sigma2w ) );
        }
    }

    inline float operator()( float rho ) const
    {
        assert( 0.f <= rho );
        return ( rho <= range_.max ) ? lut_[static_cast< size_t >( rho / lut_resolution_ )] : 0.f;
    }
};

struct Scan2d
{
    struct Sample
    {
        float rho;
        float weight;
    };
    // Build a non oriented 2d scan
    Scan2d( const cv::Mat &depth, const Eigen::Projective3d &K, const Eigen::Isometry3d &warp,
            const std::pair< float, float > &range, const cv::Mat &weight = cv::Mat() );

    std::vector< Sample > data;
    Eigen::Array2f origin;
    float resolution;
};

class Occupancy2d
{
    public:
    static const std::function< float(float)> rho_prob_func_default;
    Occupancy2d() = delete;
    Occupancy2d( float resolution, const std::function< float(float)> &rho_prob_func = rho_prob_func_default );
    Occupancy2d( int width, int height, float resolution, const Eigen::Isometry3d &origin,
                 const std::function< float(float)> &rho_prob_func = rho_prob_func_default );

    Occupancy2d( const Occupancy2d &rhs );

    inline void clear()
    {
        data_.resize( data_.size(), std::numeric_limits< float >::quiet_NaN() );
    }

    // Trace a ray from origin to points and recompute bounds
    void update( const Scan2d &scan );

    // Rewarp the map to a new origin //
    Occupancy2d warpFrom( const Eigen::Isometry3d &warp, double max_dist, bool clear = false );

    std::vector< float > probabilties() const
    {
        std::vector< float > res( data_.size(), -1.0 );

        for ( size_t i = 0; i < res.size(); ++i )
        {
            res[i] = probability( data_[i] );
        }
        return res;
    }

    inline float bilinear( float u, float v ) const
    {
        int iu = std::floor( u );
        int iv = std::floor( v );
        size_t idx = iu + iv * width_;
        assert( idx + width_ + 1 < data_.size() );

        float alpha_u = u - static_cast< float >( iu );
        float alpha_v = v - static_cast< float >( iv );
        return ( data_[idx] * ( 1.0f - alpha_u ) + data_[idx + 1] * alpha_u ) * ( 1.0f - alpha_v ) +
               ( data_[idx + width_] * ( 1.0f - alpha_u ) + data_[idx + width_ + 1] * alpha_u ) * alpha_v;
    }

    inline float nearest( float u, float v ) const
    {
        int iu = std::round( u );
        int iv = std::round( v );
        size_t idx = iu + iv * width_;
        assert( idx < data_.size() );

        return data_[idx];
    }

    inline void set_prob_miss_min( float p )
    {
        assert( 0.f < p && p <= 1.f );
        prob_miss_min_ = 1.f - p;
    }

    inline void set_prob_miss_max( float p )
    {
        assert( 0.f < p && p <= 1.f );
        prob_miss_max_ = 1.f - p;
    }

    inline void set_prob_hit_min( float p )
    {
        assert( 0.f < p && p <= 1.f );
        prob_hit_min_ = p;
    }

    inline void set_prob_hit_max( float p )
    {
        assert( 0.f < p && p <= 1.f );
        prob_hit_max_ = p;
    }

    inline void set_prob_lower_bound( float p )
    {
        assert( 0.f < p && p <= 1.f );
        log_prob_lower_bound_ = logodds( p );
    }
    inline void set_prob_upper_bound( float p )
    {
        assert( 0.f < p && p <= 1.f );
        log_prob_upper_bound_ = logodds( p );
    }

    private:
    inline float getProb( float confidence_prob_min, float confidence_prob_max, float dist, float scan_weight ) const;

    private:
    float prob_miss_min_ = 1.f - 0.5f;
    float prob_miss_max_ = 1.f - 0.7f;

    float prob_hit_min_ = 0.5f;
    float prob_hit_max_ = 0.7f;

    float log_prob_lower_bound_ = logodds( 0.1f );
    float log_prob_upper_bound_ = logodds( 1.f );

    std::vector< float > lut_rho_;

    public:
    int width_ = 0;
    int height_ = 0;
    float resolution_ = 0.f;
    Eigen::Isometry3d origin_ = Eigen::Isometry3d::Identity();
    std::function< float(float)> rho_prob_func_;

    std::vector< float > data_;  // We store the log of the probabilty
};

inline float Occupancy2d::getProb( float confidence_prob_min, float confidence_prob_max, float dist, float scan_weight ) const
{
    return confidence_prob_max - ( confidence_prob_max - confidence_prob_min ) * ( 1.f - rho_prob_func_( dist ) * scan_weight );
}

#endif
