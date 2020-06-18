## Work journal


### Week of June 15th - 17th, 2020


#### June 17th

- Taichi learning and experiments
    - Finished implementing vorticity confinement
    - Can now export PLY data to Houdini for rendering

        ![3D render](viz_results/houdini/out1.gif)

- TODOs
    - Better rendering
        - Size of particles
            - [Particle attributes](https://www.sidefx.com/docs/houdini/model/attributes.html)
            - This is a requisite for particle fluid surface in Houdini
        - Fluid surface rendering
            - Marching cubes in Taichi to create all the polygons?
            - Generate surface polygons in Houdini?
                - [Particle fluid surface](https://www.sidefx.com/docs/houdini/nodes/sop/particlefluidsurface.html)
        - Boundary walls
    - Modularize the simulation code for easier future use


#### June 16th

- Taichi learning and experiments
    - Watched Taichi lecture 3
    - Implemented XSPH viscosity in position-base fluids
    - Front view of the 3D simulation, horizontal is x, vertical is z
    
        ![Front view](viz_results/x_z/out.gif)
        
    - Top view of the 3D simulation, horizontal is x, vertical is y

        ![Top view](viz_results/x_y/out.gif)
        
    - Derived gradients for vorticity confinement
- TODOs
    - Implement vorticity confinement
    - Need to validate results in 3D, not sure what's a good quantitative evaluation yet, have to rely on visuals for now
    - Visualization limited in Taichi, try exporting particles and rendering with Houdini


#### June 15th

- Discussion with Fei, settled on some big directions for project
    - Build a blood suction fluid simulator first
    - Possible future extensions
        - Learning dynamics in real-time for surgical robotcs
        - Robot control such as optimal suction placement and orientation, finding source of flow, etc
- Taichi learning and experiments
    - Watched Taichi lectures 1-2
    - Installed Taichi, started experimenting with it
        - Modified 2D PBD based simulation to 3D
- TODOs
    - Camera mapping: how to visualize the 3D results
    - More complex scene and evironment, instead of a retangular box

