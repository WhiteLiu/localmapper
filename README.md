# localmapper

Generates a local 2d occupancy grid from a rgbd stream using laser raytracing 

## Params

Parameter                           |  Type | Default                      | Description
---------------------------------   | ------| ---------------------------- | -----------
_depth                              | string| /camera/depth_registered/image_raw | Input depth topic if =="none" disable
_depth_transport                    | string| raw                          | Depth image transport type
_queue                              | int   | 1                            | Queue size for the synchronizer
_radius                             | float | 0.0                          | Radius of the local map (if ==0 generates a global map)
_odom_frame                         | string| odom                         | Odom frame
_base_frame                         | string| base_footprint               | Base 2D footprint
_clear                              | bool  | false                        | Clear the global map when extracting the local map (currently broken)


## Usage
    rosrun localmapper localmapper _radius:=3
    
## Publishes to

/local_map or /global_map

