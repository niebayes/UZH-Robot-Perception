1. Provide a definition of Visual Odometry?
VO is the process of incrementally estimating the pose of the "vehicle" by examing the changes that motion induces on the images of its onboard cameras.

2. Explain the most important differences between VO, VSLAM and SFM?
SFM > VSLAM > VO
  1. VO vs. VSLAM: 
     - VO focuses on incremental estimation and guarantees local consistency
     - VSLAM = VO + loop detecton & closing and guarantees global consistency
  2. VO vs. SFM: 
     - SFM is the more general VO that tackles the problem of 3D reconstruction and 6dof pose estimation jointly with unordered image sets.
     - VO is a particular case of SFM that 6dof pose estimation of the camera from image sequence and in real time.

3. Describe the needed assumptions for VO?
  - Sufficient illuminance 
  - Enough texture (and distinctive features)
  - Sufficient scene overlap between consecutive images
  - Dominance of static scene

4. Illustrate its building blocks?
Image sequence -> Feature detection -> Feature matching (tracking) -> Motion estimation (3D - 3D, 3D - 2D, 2D - 2D) -> local optimization