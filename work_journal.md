## Work journal


### Week of June 22nd - 23rd, 2020

#### June 23rd
- FleX in Unity
    - Generated a new set of renders
    - Successfully extracted particle position data and created ground truth masks in image space

    <img src="viz_results/unity/out3.gif" width="200" height="200" />

    <img src="viz_results/unity/out_truth.gif" width="200" height="200" />

    <img src="viz_results/unity/out_blended.gif" width="200" height="200" />


#### June 22nd
- FleX in Unity
    - Can export particle position 4-vectors in JSON, can be parsed in Python for ground truth
    - Finished building a scene using medical 3D printing models
    - Finished one render using this scene

        ![Render](viz_results/unity/out.gif)

- TODOs
    - Need to get camera view matrix to transform positions into pixel locations
    - Need to generate a ground truth plot with the pixel locationss


### Week of June 15th - 19th, 2020


#### June 19th

- Houdini rendering for fluids
    - Found out that Houdini crashes because geometry has infs, culprit is vorticity code
    - Turn off vorticity for now, will need to debug computations and hand-derived gradients
    - New render with fluid surfaces

        ![Fluid with surface](viz_results/houdini/out3.gif)

        ![Fluid emission](viz_results/houdini/out4.gif)

- Surgical scene rendering
    - Haven't found any good (and free) texture and mesh assets that can be used for a more realistic render
- PBF in Taichi
    - Put the simulation code into a class, should be easier to modify for future use
    - Set up particle emission for bleeding simulation

        ![Particle emission](viz_results/x_z_emit/out.gif)

- TODOs
    - Search more for surgical scene assets for rendering
    - Test more complex boundary conditions for Taichi PBF


#### June 18th

- Investigated interactive visualizaiton of point clouds
    - [pptk](https://heremaps.github.io/pptk/tutorials/viewer/tanks_and_temples.html)
        - Loading seems to have some issues
    - [Open3D](http://www.open3d.org/)
        - Can't create visualizer window!?!
- Houdini rendering for fluids
    - Crashes when trying to create polygon surfaces from particles
    - Most likly a hardware resource issue
- FleX in Unity
    - Can do a fluid simulation and render

        <img src="viz_results/unity/gif_animation_001.gif" width="200" height="200" />

        <img src="viz_results/unity/new.png" width="200" height="200" />
    
- TODOs
    - [Previous MPM work](https://github.com/yuanming-hu/taichi_mpm) rendered with Houdini, take a look at that again?
    - FleX Unity
        - Need better assets (surgical scenes?) for better simulation
        - Need to get particle positions as ground truth ([maybe this](https://forums.developer.nvidia.com/t/unity-flex-particles-position/64968/7))
    - Try Blender for rendering


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

