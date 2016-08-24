#include "occupancy.hpp"

struct UpdateCellCandidate
{
    size_t cell_index;
    float log_prob_update;
};

inline bool operator==( const UpdateCellCandidate &lhs, const UpdateCellCandidate &rhs )
{
    return lhs.cell_index == rhs.cell_index;
}

inline bool operator!=( const UpdateCellCandidate &lhs, const UpdateCellCandidate &rhs )
{
    return !( lhs == rhs );
}

struct UpdateCellCandidateHash
{
    inline size_t operator()( const UpdateCellCandidate &v ) const
    {
        return v.cell_index;
    }
};

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

Scan2d::Scan2d( const cv::Mat &depth, const Eigen::Projective3d &K, const Eigen::Isometry3d &warp,
                const std::pair< float, float > &range, const cv::Mat &weight )
{
    bool weighted = !weight.empty();
    if ( depth.type() != CV_16U || ( weighted && weight.type() != CV_32F ) )
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
        for ( int x = 0; x < depth.cols; ++x )
        {
            float z = static_cast< float >( depth_in[x + y * depth.cols] ) / 1000.f;
            if ( 0 < z )
            {
                // Get 3D point
                Eigen::Vector3f v =
                    backproject( Kf, static_cast< float >( x ) / depth.cols, static_cast< float >( y ) / depth.rows ) * z;

                // Warp to map
                Eigen::Vector3f vw = warpf * v;

                /*
                std::cout << currentDepthSize.width << "  " << dSize.height << std::endl;
                std::cout << Kf.matrix() << std::endl;
                std::cout << v.transpose() << std::endl;
                std::cout << vw.transpose() << std::endl;
                getchar();
                //*/

                // if point is in range
                if ( vw( 2 ) >= range.first && vw( 2 ) < range.second )
                {
                    vw -= warpf.translation();

                    // Transform to range
                    float theta = std::atan2( vw( 1 ), vw( 0 ) ) + M_PI;

                    float rho = std::sqrt( vw( 0 ) * vw( 0 ) + vw( 1 ) * vw( 1 ) );
                    int thetai = std::round( theta / resolution );

                    // Get the mindist
                    if ( data[thetai].rho == 0.f || ( rho < data[thetai].rho ) )
                    {
                        data[thetai].rho = rho;
                        if ( weighted )
                        {
                            data[thetai].weight = weight.at< float >( y, x );
                        }
                        else
                        {
                            data[thetai].weight = 1.f;
                        }
                    }
                }
            }
        }
    }
}

Occupancy2d::Occupancy2d( int width, int height, float resolution, const Eigen::Isometry3d &origin )
  : width_( width )
  , height_( height )
  , resolution_( resolution )
  , origin_( origin )
  , data_( std::vector< float >( width_ * height_, std::numeric_limits< float >::quiet_NaN() ) )
{
    initLutRho( range_min_, range_max_, resolution_, lut_rho_ );
}

Occupancy2d::Occupancy2d( float resolution ) : Occupancy2d( 0., 0., resolution, Eigen::Isometry3d::Identity() )
{
}

Occupancy2d::Occupancy2d( const Occupancy2d &rhs ) : Occupancy2d( rhs.width_, rhs.height_, rhs.resolution_, rhs.origin_ )
{
    data_ = rhs.data_;
}

void Occupancy2d::getMinMaxFromScan( const Scan2d &scan, Eigen::Array2f &scan_mini, Eigen::Array2f &scan_maxi )
{
    // Get min/max
    scan_mini = Eigen::Array2f( 1e9, 1e9 );
    scan_maxi = Eigen::Array2f( -1e9, -1e9 );

    // Get min/max of 2D scan within the map
    for ( size_t i = 0; i < scan.data.size(); ++i )
    {
        float rho = scan.data[i].rho;

        if ( rho > 0.0 )
        {
            // Polar to cart
            float theta = static_cast< float >( i ) * scan.resolution - M_PI;
            float x = rho * std::cos( theta ) + scan.origin( 0 );
            float y = rho * std::sin( theta ) + scan.origin( 1 );

            scan_mini = scan_mini.min( Eigen::Array2f( x, y ) );
            scan_maxi = scan_maxi.max( Eigen::Array2f( x, y ) );
        }
    }
    // Add origin //
    scan_mini = scan_mini.min( scan.origin );
    scan_maxi = scan_maxi.max( scan.origin );
}

void Occupancy2d::initLutRho( int range_min, int range_max, float resolution, std::vector< float > &lut,
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

void Occupancy2d::update( const Scan2d &scan )
{
    Eigen::Array2f scan_mini, scan_maxi;
    getMinMaxFromScan( scan, scan_mini, scan_maxi );

    Eigen::Array2f map_mini = Eigen::Array2f( origin_( 0, 3 ), origin_( 1, 3 ) );
    Eigen::Array2f map_maxi = map_mini + Eigen::Array2f( static_cast< float >( width_ ) * resolution_,
                                                         static_cast< float >( height_ ) * resolution_ );

    // Extend Current map if scan is outside //
    if ( scan_mini( 0 ) < map_mini( 0 ) || scan_mini( 1 ) < map_mini( 1 ) || scan_maxi( 0 ) > map_maxi( 0 ) ||
         scan_maxi( 1 ) > map_maxi( 1 ) )
    {
        Eigen::Array2f new_mini = map_mini.min( scan_mini );
        Eigen::Array2f new_maxi = map_maxi.max( scan_maxi );

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

    std::unordered_set< UpdateCellCandidate, UpdateCellCandidateHash > free_cells, occupied_cells;


    for ( size_t i = 0; i < scan.data.size(); ++i )
    {
        float rho = scan.data[i].rho;

        if ( range_min_ <= rho && rho <= range_max_ )
        {
            float theta = static_cast< float >( i ) * scan.resolution - M_PI;
            // Convert start and end points to map pixel coords //
            int x1 = std::round( ( rho * std::cos( theta ) + scan.origin( 0 ) - origin_( 0, 3 ) ) / resolution_ );
            int y1 = std::round( ( rho * std::sin( theta ) + scan.origin( 1 ) - origin_( 1, 3 ) ) / resolution_ );
            int x0 = std::round( ( scan.origin( 0 ) - origin_( 0, 3 ) ) / resolution_ );
            int y0 = std::round( ( scan.origin( 1 ) - origin_( 1, 3 ) ) / resolution_ );

            // Setup occupied cells
            size_t occ_index = x1 + y1 * width_;
            if ( 0 <= occ_index && occ_index < data_.size() )
            {
                float map_dist = sqrt( ( x1 - x0 ) * ( x1 - x0 ) + ( y1 - y0 ) * ( y1 - y0 ) );
                float log_prob;
                if ( getLogProb( prob_hit_min_, prob_hit_max_, map_dist, scan.data[i].weight, log_prob ) )
                {
                    UpdateCellCandidate candidate = {occ_index, log_prob};

                    auto it = occupied_cells.find( candidate );
                    if ( it == occupied_cells.end() )
                    {
                        occupied_cells.insert( candidate );
                    }
                    else if ( candidate.log_prob_update > it->log_prob_update )
                    {
                        occupied_cells.insert( occupied_cells.erase( it ), candidate );
                    }
                }
            }

            // Raytrace the free cells
            int dx = std::abs( x1 - x0 );
            int dy = std::abs( y1 - y0 );
            int n = 1 + dx + dy;
            int x_inc = ( x1 > x0 ) ? 1 : -1;
            int y_inc = ( y1 > y0 ) ? 1 : -1;
            int error = dx - dy;
            dx *= 2;
            dy *= 2;

            for ( int x = x0, y = y0; n > 1; --n )
            {
                size_t occ_index = x + y * width_;
                if ( 0 <= occ_index && occ_index < data_.size() )
                {
                    float map_dist = sqrt( ( x - x0 ) * ( x - x0 ) + ( y - y0 ) * ( y - y0 ) );
                    float log_prob;
                    if ( getLogProb( prob_miss_min_, prob_miss_max_, map_dist, scan.data[i].weight, log_prob ) )
                    {
                        UpdateCellCandidate candidate = {occ_index, log_prob};
                        auto it = free_cells.find( candidate );
                        if ( it == free_cells.end() )
                        {
                            free_cells.insert( candidate );
                        }
                        else if ( candidate.log_prob_update < it->log_prob_update )
                        {
                            free_cells.insert( free_cells.erase( it ), candidate );
                        }
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
    }

    for ( UpdateCellCandidate cell : free_cells )
    {
        if ( occupied_cells.find( cell ) == occupied_cells.end() )
        {
            float &p = data_[cell.cell_index];
            p = std::isnan( p ) ? cell.log_prob_update : std::max( p + cell.log_prob_update, log_prob_lower_bound_ );
        }
    }

    for ( UpdateCellCandidate cell : occupied_cells )
    {
        float &p = data_[cell.cell_index];
        p = std::isnan( p ) ? cell.log_prob_update : std::min( p + cell.log_prob_update, log_prob_upper_bound_ );
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

        std::cout << "New " << new_width << " " << new_height << std::endl;
        std::cout << "Prev " << width_ << " " << height_ << std::endl;

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
