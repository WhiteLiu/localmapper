#include "occupancy.hpp"

#include <Eigen/Core>

void initPixelRadialWeightLUT( int width, int height, std::vector< float > &lut, float per_cent_weight_at_edge )
{
    lut.clear();
    lut.reserve( height * width );
    float sigma2 = sqrt( height * height + width * width ) / ( -2. * std::log( per_cent_weight_at_edge ) );

    for ( int y = 0; y < height; ++y )
    {
        for ( int x = 0; x < width; ++x )
        {
            float y_rel = y - height / 2;
            float x_rel = x - width / 2;
            lut.push_back( std::exp( -sqrt( y_rel * y_rel + x_rel * x_rel ) / sigma2 ) );
        }
    }
}

const std::function< float(float)> Occupancy2d::rho_prob_func_default = []( float ) -> float
{
    return 1.f;
};

void initRhoWeightLUT( int range_min, int range_max, float resolution, std::vector< float > &lut,
                       float perCentWeightAtMaxRange )
{
    float sigma2w =
        -( range_max * range_max - 2. * range_max * range_min + range_min * range_min ) / std::log( perCentWeightAtMaxRange );
    int size = static_cast< int >( ( range_max - range_min ) / resolution );
    lut.clear();
    lut.reserve( size );
    for ( int i = 0; i < size; ++i )
    {
        float dist = range_min + i * resolution;
        lut.push_back( std::exp( -( dist * dist ) / sigma2w ) );
    }
}

Scan2d::Scan2d( const cv::Mat &depth, const Eigen::Projective3d &K, const Eigen::Isometry3d &warp,
                const std::pair< float, float > &range, const cv::Mat &weight )
{
    bool weighted = !weight.empty();
    if ( depth.type() != CV_16U )
    {
        throw std::runtime_error( "Invalid input depth format" );
    }
    else if ( weighted && weight.type() != CV_32F )
    {
        throw std::runtime_error( "Invalid input weight format" );
    }

    float cam_aperture = 2.f * std::atan( 0.5 / K( 0, 0 ) );
    resolution = cam_aperture / depth.cols;  // rad/pixel
    int scan_size = 2.f * M_PI / resolution;

    data = std::vector< Sample >( scan_size, {0.f, 0.f} );
    origin = Eigen::Vector2f( warp( 0, 3 ), warp( 1, 3 ) );

    // Convert depth image to 2D range
    Eigen::Projective3f Kf = K.cast< float >();
    Eigen::Isometry3f warpf = warp.cast< float >();
    const uint16_t *depth_in = depth.ptr< uint16_t >();

    for ( int y = 0; y < depth.rows; ++y )
    {
        float ybp = static_cast< float >( y ) / depth.rows;
        for ( int x = 0; x < depth.cols; ++x )
        {
            uint16_t zi = depth_in[x + y * depth.cols];
            if ( zi != 0 )
            {
                float z = static_cast< float >( zi ) / 1000.f;
                float xbp = static_cast< float >( x ) / depth.cols;
                // Get 3D point
                Eigen::Vector3f v = backproject( Kf, xbp, ybp ) * z;

                // Warp to map
                Eigen::Vector3f vw = warpf * v;

                // if point is in range
                if ( vw( 2 ) >= range.first && vw( 2 ) < range.second )
                {
                    vw -= warpf.translation();

                    // Transform to range
                    float theta = std::atan2( vw( 1 ), vw( 0 ) ) + M_PI;

                    float rho = std::sqrt( vw( 0 ) * vw( 0 ) + vw( 1 ) * vw( 1 ) );
                    int thetai = std::round( theta / resolution );

                    // Get the mindist
                    if ( ( rho < data[thetai].rho ) || ( data[thetai].rho == 0.f ) )
                    {
                        data[thetai].rho = rho;
                        data[thetai].weight = ( weighted ) ? weight.at< float >( y, x ) : 1.f;
                    }
                }
            }
        }
    }
}

Occupancy2d::Occupancy2d( int width, int height, float resolution, const Eigen::Isometry3d &origin,
                          const std::function< float(float)> &rho_prob_func )
  : width_( width )
  , height_( height )
  , resolution_( resolution )
  , origin_( origin )
  , data_( std::vector< float >( width_ * height_, std::numeric_limits< float >::quiet_NaN() ) )
  , rho_prob_func_( rho_prob_func )
{
}

Occupancy2d::Occupancy2d( float resolution, const std::function< float(float)> &rho_prob_func )
  : Occupancy2d( 0., 0., resolution, Eigen::Isometry3d::Identity(), rho_prob_func )
{
}

Occupancy2d::Occupancy2d( const Occupancy2d &rhs ) : Occupancy2d( rhs.width_, rhs.height_, rhs.resolution_, rhs.origin_ )
{
    data_ = rhs.data_;
}

void Occupancy2d::update( const Scan2d &scan )
{
    struct ScanPoint
    {
        Eigen::Array2f pt;
        float rho;
        float weight;
    };

    enum
    {
        unset = 0,
        occupied,
        free
    };

    Eigen::Array2f scan_mini( std::numeric_limits< float >::max(), std::numeric_limits< float >::max() );
    Eigen::Array2f scan_maxi( std::numeric_limits< float >::lowest(), std::numeric_limits< float >::lowest() );

    std::vector< ScanPoint > pts_sensor;
    for ( size_t i = 0; i < scan.data.size(); ++i )
    {
        float rho = scan.data[i].rho;

        if ( 0.f < rho )
        {
            // Polar to cart
            float theta = static_cast< float >( i ) * scan.resolution - M_PI;
            float x = rho * std::cos( theta );
            float y = rho * std::sin( theta );

            pts_sensor.push_back( {Eigen::Array2f( x, y ), rho, scan.data[i].weight} );

            scan_mini = scan_mini.min( pts_sensor.back().pt );
            scan_maxi = scan_maxi.max( pts_sensor.back().pt );
        }
    }
    // Add origin //
    scan_mini = scan_mini.min( Eigen::Array2f( 0, 0 ) );
    scan_maxi = scan_maxi.max( Eigen::Array2f( 0, 0 ) );

    int scan_width = std::round( ( scan_maxi( 0 ) - scan_mini( 0 ) ) / resolution_ );
    int scan_height = std::round( ( scan_maxi( 1 ) - scan_mini( 1 ) ) / resolution_ );

    Eigen::Array2f scan_mini_map = scan_mini + scan.origin;
    Eigen::Array2f scan_maxi_map = scan_maxi + scan.origin;

    Eigen::Array2f map_mini = Eigen::Array2f( origin_( 0, 3 ), origin_( 1, 3 ) );
    Eigen::Array2f map_maxi = map_mini + Eigen::Array2f( static_cast< float >( width_ ) * resolution_,
                                                         static_cast< float >( height_ ) * resolution_ );

    // Extend Current map if scan is outside //
    if ( scan_mini_map( 0 ) < map_mini( 0 ) || scan_mini_map( 1 ) < map_mini( 1 ) || scan_maxi_map( 0 ) > map_maxi( 0 ) ||
         scan_maxi_map( 1 ) > map_maxi( 1 ) )
    {
        Eigen::Array2f new_mini = map_mini.min( scan_mini_map );
        Eigen::Array2f new_maxi = map_maxi.max( scan_maxi_map );

        // Extend the original map //
        int new_width = std::round( ( new_maxi( 0 ) - new_mini( 0 ) ) / resolution_ );
        int new_height = std::round( ( new_maxi( 1 ) - new_mini( 1 ) ) / resolution_ );

        // Initialize map with unknown //
        std::vector< float > update( new_width * new_height, std::numeric_limits< float >::quiet_NaN() );

        int offset_x = std::round( ( map_mini( 0 ) - new_mini( 0 ) ) / resolution_ );
        int offset_y = std::round( ( map_mini( 1 ) - new_mini( 1 ) ) / resolution_ );

        new_mini( 0 ) = map_mini( 0 ) - resolution_ * offset_x;
        new_mini( 1 ) = map_mini( 1 ) - resolution_ * offset_y;

        offset_x = std::max( 0, offset_x );
        offset_y = std::max( 0, offset_y );

        // Copy the old map
        for ( int i = 0; i < height_; ++i )
        {
            std::copy( &data_[i * width_], &data_[(i)*width_] + width_, &update[offset_x + ( offset_y + i ) * new_width] );
        }

        data_ = update;
        width_ = new_width;
        height_ = new_height;

        // Update origin -> top left corner //
        origin_( 0, 3 ) = new_mini( 0 );
        origin_( 1, 3 ) = new_mini( 1 );
    }

    int x0 = std::round( ( scan.origin( 0 ) - origin_( 0, 3 ) ) / resolution_ );
    int y0 = std::round( ( scan.origin( 1 ) - origin_( 1, 3 ) ) / resolution_ );

    int xmap = std::round( ( scan_mini_map( 0 ) - origin_( 0, 3 ) ) / resolution_ );
    int ymap = std::round( ( scan_mini_map( 1 ) - origin_( 1, 3 ) ) / resolution_ );

    cv::Mat cell_status = cv::Mat::zeros( scan_height, scan_width, CV_8U );
    cv::Mat prob_update = cv::Mat::zeros( scan_height, scan_width, CV_32F );

    for ( ScanPoint sp : pts_sensor )
    {
        // Convert start and end points to map pixel coords //
        int x1 = std::round( ( sp.pt( 0 ) + scan.origin( 0 ) - origin_( 0, 3 ) ) / resolution_ );
        int y1 = std::round( ( sp.pt( 1 ) + scan.origin( 1 ) - origin_( 1, 3 ) ) / resolution_ );

        int x1_off = x1 - xmap;
        int y1_off = y1 - ymap;
        // Setup occupied cells
        size_t occ_index = x1 + y1 * width_;
        if ( occ_index < data_.size() && y1_off < prob_update.rows && x1_off < prob_update.cols )
        {
            float new_prob = getProb( prob_hit_min_, prob_hit_max_, sp.rho, sp.weight );
            unsigned char &status = cell_status.at< unsigned char >( y1_off, x1_off );
            float &prob = prob_update.at< float >( y1_off, x1_off );

            if ( status != occupied )
            {
                status = occupied;
                prob = new_prob;
            }
            else if ( new_prob > prob )
            {
                prob = new_prob;
            }
        }

        int dx = std::abs( x1 - x0 );
        int dy = std::abs( y1 - y0 );
        int nmax = dx + dy;
        int x_inc = ( x1 > x0 ) ? 1 : -1;
        int y_inc = ( y1 > y0 ) ? 1 : -1;
        int error = dx - dy;
        dx *= 2;
        dy *= 2;

        float dist_step = sp.rho / static_cast< float >( nmax );

        for ( int x = x0 - xmap, y = y0 - ymap, n = 0; n < nmax; ++n )
        {
            size_t occ_index = x + xmap + ( y + ymap ) * width_;
            if ( occ_index < data_.size() && y < prob_update.rows && x < prob_update.cols )
            {
                float dist = dist_step * static_cast< float >( n );

                float new_prob = getProb( prob_miss_min_, prob_miss_max_, dist, sp.weight );

                unsigned char &status = cell_status.at< unsigned char >( y, x );
                float &prob = prob_update.at< float >( y, x );

                if ( status == unset )
                {
                    status = free;
                    prob = new_prob;
                }
                else if ( status == free && new_prob < prob )
                {
                    prob = new_prob;
                }
            }

            if ( error > 0 )
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

    for ( int y = 0; y < cell_status.rows; ++y )
    {
        for ( int x = 0; x < cell_status.cols; ++x )
        {
            if ( cell_status.at< unsigned char >( y, x ) != 0 )
            {
                float &p = data_[x + xmap + ( y + ymap ) * width_];
                float log_prob = logodds( prob_update.at< float >( y, x ) );

                p = std::isnan( p ) ? log_prob : ( cell_status.at< unsigned char >( y, x ) == 1 ) ?
                                      std::min( p + log_prob, log_prob_upper_bound_ ) :
                                      std::max( p + log_prob, log_prob_lower_bound_ );
            }
        }
    }
}

Occupancy2d Occupancy2d::warpFrom( const Eigen::Isometry3d &warp, double max_dist, bool clear )
{
    int warped_size = std::round( max_dist / resolution_ );
    Eigen::Isometry3d warped_origin = Eigen::Isometry3d::Identity();
    warped_origin( 0, 3 ) = -static_cast< float >( resolution_ * warped_size ) / 2.f;
    warped_origin( 1, 3 ) = -static_cast< float >( resolution_ * warped_size ) / 2.f;

    Occupancy2d warped( warped_size, warped_size, resolution_, warped_origin );

    Eigen::Isometry3f warpf = warp.cast< float >();

    Eigen::Array2f new_mini( 1e9, 1e9 );
    Eigen::Array2f new_maxi( -1e9, -1e9 );

    for ( int i = 0; i < warped.height_; ++i )
    {
        for ( int j = 0; j < warped.width_; ++j )
        {
            // point in local space
            Eigen::Vector3f v( static_cast< float >( j ) * resolution_ + warped.origin_( 0, 3 ),
                               static_cast< float >( i ) * resolution_ + warped.origin_( 1, 3 ), 0.0 );

            if ( v.squaredNorm() < max_dist )
            {
                // Point in map space
                Eigen::Vector3f vw = warpf * v;

                // Convert to grid
                float x = ( vw( 0 ) - origin_( 0, 3 ) ) / resolution_;
                float y = ( vw( 1 ) - origin_( 1, 3 ) ) / resolution_;

                if ( x >= 0 && y >= 0 && std::ceil( x ) < width_ && std::ceil( y ) < height_ )
                {
                    new_mini = new_mini.min( Eigen::Array2f( vw( 0 ), vw( 1 ) ) );
                    new_maxi = new_maxi.max( Eigen::Array2f( vw( 0 ), vw( 1 ) ) );

                    warped.data_[j + i * warped.width_] = this->bilinear( x, y );
                }
            }
        }
    }

    if ( clear )
    {
        Eigen::Array2f map_mini = Eigen::Array2f( origin_( 0, 3 ), origin_( 1, 3 ) );
        Eigen::Array2f map_maxi = map_mini + Eigen::Array2f( static_cast< float >( width_ ) * resolution_,
                                                             static_cast< float >( height_ ) * resolution_ );

        int new_width = std::round( ( new_maxi( 0 ) - new_mini( 0 ) ) / resolution_ );
        int new_height = std::round( ( new_maxi( 1 ) - new_mini( 1 ) ) / resolution_ );

        int offset_x = std::round( ( map_mini( 0 ) - new_mini( 0 ) ) / resolution_ );
        int offset_y = std::round( ( map_mini( 1 ) - new_mini( 1 ) ) / resolution_ );

        new_mini( 0 ) = map_mini( 0 ) - resolution_ * offset_x;
        new_mini( 1 ) = map_mini( 1 ) - resolution_ * offset_y;

        // Initialize map with unknown //
        std::vector< float > update( new_width * new_height, std::numeric_limits< float >::quiet_NaN() );

        offset_x = std::max( 0, offset_x );
        offset_y = std::max( 0, offset_y );

        // Copy the old block
        for ( int i = 0; i < new_height; ++i )
        {
            for ( int j = 0; j < new_width; ++j )
            {
                update[j + i * new_width] = data_[j + offset_x + ( offset_y + i ) * width_];
            }
        }

        width_ = new_width;
        height_ = new_height;
        data_ = update;

        // Update origin -> top left corner //
        origin_( 0, 3 ) = new_mini( 0 );
        origin_( 1, 3 ) = new_mini( 1 );
    }

    return warped;
}
