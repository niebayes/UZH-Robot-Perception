1. Describe the general PnP problem and derive the behavior of its solutions? 
  1. PnP: Given n point correspondences between 2D image points and 3D scene points, find the 6dof camera pose and optional calibration matrix K if the camera is not calibrated.
  2. > When n = 3, the PnP problem is in its minimal form of P3P and can be solved with three point correspondences. However, with just three point correspondences, P3P yields up to four real, geometrically feasible solutions. For low noise levels a fourth correspondence can be used to remove ambiguity. 

2. Explain the working principle of the P3P algorithm? 
  - Calibrated camera 
  - Non coplanar 3D scene points 

3. Explain and derive the DLT? What is the minimum number of point correspondences it requires? 
  - Derivation of DLT: starting from the perspective projection equation and progressively obtain the equation involving the rows of matrix M or H in planar case, and then rearrange the variables to form a linear matrix equation which can be solved by SVD.
  - Non coplanar correspondences case: 5 + 1/2; 6 in practice. 
  - Coplanar case: 4.

4. Define central and non central omnidirectional cameras? 

5. What kind of mirrors ensure central projection?