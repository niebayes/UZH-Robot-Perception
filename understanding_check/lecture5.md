1. Explain what is template matching and how it is implemented?

2. Explain what are the limitations of template matching? Can you use it to recognize cars?

3. Illustrate the similarity metrics SSD, SAD, NCC, and Census transform?

4. What is the intuitive explanation behind SSD and NCC?

5. Explain what are good features to track? In particular, can you explain what are corners and blobs together with their pros and cons?

6. Explain the Harris corner detector? In particular:
  - Use the Moravec definition of corner, edge and flat region. 

  - Show how to get the second moment matrix from the definition of SSD and first order approximation (show that this is a quadratic expression) and what is the intrinsic interpretation of the second moment matrix using an ellipse?

  - What is the M matrix like for an edge, for a flat region, for an axis-aligned 90-degree corner and for a non-axis-aligned 90-degree corner?

  - What do the eigenvalues of M reveal?

  - Can you compare Harris detection with Shi-Tomasi detection?

  - Can you explain whether the Harris detector is invariant to illumination or scale changes? Is it invariant to view point changes?

  - What is the repeatability of the Harris detector after rescaling by a factor of 3?