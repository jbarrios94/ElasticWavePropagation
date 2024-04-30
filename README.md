# Elastic Wave Propagation
Computational codes to model seismic wave propagation using Godunov-type finite volume methods and staggered finite differences.

The reference solutions used were calculated using a finite difference method of order 20 using the Python package **Devito** (https://www.devitoproject.org/).

In all tests presented, a ricker source was used.

Test 1 - Heterogeneous parallel velocity model.

Configurations:

  Final time (s)= 1000
  
  x = [0, 3000]
  
  y = [0, 1000]
  
  dx (m)= 10.0 -> ref_factor = 0
  
          5.0 -> ref_factor = 1
          
          2.5 -> ref_factor = 2
          
  dy (m)= 10.0 -> ref_factor = 0
  
          5.0 -> ref_factor = 1
          
          2.5 -> ref_factor = 2
          
  CFL = 0.50
  
  frequency (MGhz)= 0.015 (Used in the ricker source)
  


Test 2 - SEG-EAGE salt body velocity profile.
Configurations:
  Final time (s)= 2000
  x = [2000, 11000]
  y = [0, 3000]
  dx (m)= 20.0 -> ref_factor = 0
          10.0 -> ref_factor = 1
          5.0 -> ref_factor = 2
  dy (m)= 20.0 -> ref_factor = 0
          10.0 -> ref_factor = 1
          5.0 -> ref_factor = 2
  CFL = 0.50
  frequency (MGhz)= 0.005 (Used in the ricker source)

Test 3 - Marmousi velocity profile.
Configurations:
  Final time (s)= 2000
  x = [4000, 14000]
  y = [0, 3500]
  dx (m)= 20.0 -> ref_factor = 0
          10.0 -> ref_factor = 1
          5.0 -> ref_factor = 2
  dy (m)= 20.0 -> ref_factor = 0
          10.0 -> ref_factor = 1
          5.0 -> ref_factor = 2
  CFL = 0.50
  frequency (MGhz)= 0.005 (Used in the ricker source)

Test 4 - A typical velocity field of Santos Basin.
Configurations:
  Final time (s)= 2000
  x = [20000, 50000]
  y = [0, 8000]
  dx (m)= 50.0 -> ref_factor = 0
          25.0 -> ref_factor = 1
          12.5 -> ref_factor = 2
  dy (m)= 32.0 -> ref_factor = 0
          16.0 -> ref_factor = 1
          8.0 -> ref_factor = 2
  CFL = 0.50
  frequency (MGhz)= 0.005 (Used in the ricker source)
