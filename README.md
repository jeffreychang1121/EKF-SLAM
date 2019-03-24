# EKF SLAM
### 1.1 Introduction
We will be implementing a full SLAM system based on an extended Kalman filter. The data we will be using was taken from a pickup truck equipped with a wheel encoder, GPS, and LIDAR scanner. The dataset we are using consists of this truck making several loops around a park with trees. From each laser scan, we begin by extracting tree features from the environment by detecting appropriately shaped clusters of laser measurements, estimating the trunk diameter, and finally estimating the range and bearing to the trunk center from the truck's position.

### 1.2 Algorithm
    while filter running do
        e <= next event to process
        if e is an odometry event then
            Perform EKF propagation with e
        else if e is a GPS measurement then
            Perform an EKF update with e
        else if e is a laser scan then
            Extract tree range, bearing measurements fzg from e
            Perform an EKF update with fzg
        end if
    end while
    
### 2.1 Odometry Propagation
 - ##### 2.1.1 Vehicle motion model
   > Implement the vehicle motion model and its Jacobian in the function **motion_model**
 - ##### 2.1.2 EKF Propagation
   > Implement the EKF odometry propagation in the function **odom_predict**

### 2.2 GPS Update
   > Implement the GPS update equations in the function **gps_update**

### 2.3 LIDAR update
- ##### 2.3.1 Range and bearing measurement model
  > Implement the range and bearing measurement model given above, as well as its Jacobian that you derived, in the function **laser_measurement_model**

- ##### 2.3.2 Landmark initialization
  > Implement the function **initialize_landmark** that initializes a new landmark in the state vector from a measurement z

- ##### 2.3.3 Data association
  > Implement the function **compute_data_association** that computes the measurement data association as discussed above

 - ##### 2.3.4 Laser update
   > Implement the EKF update from tree detections in the function **laser_update**
