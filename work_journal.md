## Work journal


### Week of July 6th - July 9th, 2020


#### July 9th
- Suction control
    - Implemented new loss that penalizes every timestep and resulting tool movements are much more sensible
        - For simple case where the tool can act as soon as particles come out, the tool will stay at the emission point and leads to very good loss

        ![good case](viz_results/suction_tool/suction_tool_1/out.gif)

        - The gradients backprop through the particle positions back to the tool positions (solid-fluid one-way coupling) are not very good, it leads to spasmic movements
            - At this point not sure if the numerical values are just very unstable or if I have errors implementing the gradients

        ![bad case](viz_results/suction_tool/suction_tool_2/out.gif)

    - Distance based loss drives the tool to the centroid of the particles
        - This could be a problem when the centroid is not in the body of the fluid, resulting in no suction
        
        ![delayed case](viz_results/suction_tool/suction_tool_delayed/out.gif)
        


#### July 8th
- Differentiable simulation
    - Implmented proposed min distance loss and backpropagation to suction tool positions
    - Ran some simple tests
        - A trivial case of a few particles
            - The tool will converge to the center of the particles, which is what we expect since we want to minimize the distance between the tool and the particle in order to suction them
            - However the region in the center is empty, so the tool didn't do any suction
            - This is not that big of a problem because in a real case, the center of a group of particles should not be empty
        - A trajectory optimizatin test with 100 particles in 100 timesteps
            - Goal is to find optimal tool positions for all timesteps to remove as much particles as possible
            - We expect the tool to go to where the particles are emitted, to suction the particles as soon as they come out
            - Results show that the tool does tend to go to the emission point, but unintuitively, it doesn't stay there to suction all of the particles
    - Possibly new loss that penalizes all timeframes
        - Intuitively we want to suction all particles as fast as possible, so it makes sense to penalize a particle be existing
        - A trajectory optimizatin test with 100 particles in 100 timesteps
            - Result show that the tool is being driven towards the center of the fluid
            - Best results at around 200 iterations, in later iterations of gradient descent the trajectory gets worse
            - Even at the best iteration, there is a lot of spasmic movement, and has trouble going into the fluid


#### July 7th
- Differentiable simulation
    - Added in particle emission and particle state control for gradient computation
    - Formulated particle suction through deleteion as an optimization problem


#### July 6th
- Differentiable simulation
    - Fixed a mistake in propagating poly6 gradients
    - Put in clipping to ensure Spiky grad backward does not blow up
    - Put in clipping to ensure propagation through the lambdas path does not blow up
    - Finished multi-iteration solver and the backprop through it


### Week of June 29th - July 3rd, 2020


#### July 3rd

- Differentiable simulation
    - Finished most of gradients
        - Only one Jacobi iteration
            - Forward sim is slightly not stable because of this
            - Need to added the multi-iteration in
        - Need to add in the bound check after applying delta
    - Next check end-to-end optimiation on the full simulation
    - Next implement suction and its gradients


#### July 2nd
- Differentiable simulatioon
    - Finished deriving and implementing a draft of backprop for core solver iteration, need to test and debug
        - Preliminary tests seem ok, do gradient descent test tomorrow
    - Next steps
        - Include data structure to handle multi-step simulation
        - Include data structure to handle multi-iteration core solver
        

#### July 1st
- Differentiable simulation
    - Derived gradients for computing density constraint Lagrange multipliers
    - Implemented the above gradients
        - Seems correct so far, can use gradient descent to drive lambdas toward desired value
        - Need more extensive tests to make sure the gradients are correct, can try to use automatic numerical gradients for comparison
    - Next steps are implementing the rest of PBF


#### June 30th
- Taichi
    - Implementing suction
        - Suction apparatus is coded as a boundary condition
        - The apparatus rectangle can move and affect (push) the particles
        - Suction is coded as removing particles that are within some distance from the the bottom of the apparatus

        ![suction](viz_results/suction/out.gif)

    - Moving to use hand derived gradients for differentiable simulation
        - Implemented and testing backprop for gravity and position-based velocity update


#### June 29th
- Taichi
    - Implemented fix for "stmt ... cannot have operend ..." in update grid kernel
        - The return from atomic_add can not be used for indexing for some reason
        - Previous fix is not thread-safe (simultaneous read of index, and thus writting to the same index)
        - Changed grid2particle to an indicator tensor and use the particle ID as index
        - This is a messy work around
        - Might decrease performance since we have to do a longer loop when checking neighbors, but should be fine
    - compute_lambdas gradient accumulation also has "stmt ... cannot have operand ..." error
        - Can't add the results from spiky_gradient to grad accumulator?
        - Did some major rewrite, with global data structure and kernel simplicity rule
        - Seems like the problem is when trying to accumulate a Taichi function return?
        - Possibly a bug? Currently can't think of any work-arounds like the grid_update issue
    - Need to have some random offsets when handling boundary condition, but it seems to mess with auto-diff
    - Overall progress on differentiable simulation is hindered by Taichi issues
        - Somewhat hard to debug since this error message is not well documented
        - Statements that are similar can work or break, so it's hard to isolate issues


### Week of June 22nd - 26th, 2020


#### June 26th
- Taichi
    - Tested simple end-to-end optimization
        - Using gradient descent to select the right initial velocity to reach the target positions in 3D projecile motion


#### June 25th
- Taichi
    - Working more on end-to-end differentible PBF sim
    - Some simple propagations work
        - Propagating from initial position to end position by gravity only, without boundary, is verified with hand-derived computational graph

            <img src=simple_comp_graph.jpg width=400 />

    - Note gradients are just numbers, need to make sure the loss and corresponding gradients make physical sense and possibly reconsider if the operations can be differetiated in a sensible way
        - Example: particle falling from only gravity, with ground as boundary, for *t* steps, what is derivative of its final height with respect to its initial height? h_f = h_i + v_i t + 1/2 g t^2
            - If particle is still in free-fall at *t*
                - Increasing initial height will also increase height at *t* by same armound, derivative should be 1
                - Taichi AD matches in the discretized version
            - If particle hits the ground at exactly *t*
                - Left derivative is 0; decreasing initial height will lead to ball hitting ground earlier, but its height at *t* is still on the ground, so decreasing initial height does not change height at *t*
                - Right derivative is 1; increasing initial height will lead to ball hitting ground later, so its height at *t* will be increased by the same amount
                - Taichi AD gives 0 in the discretized version
            - If particle hits the ground some time before *t*
                - Derivative is similar to the previous case, but right-shifted by same amount as the time of impact
                - Taichi AD gives -1 in the discretized version
        - Example: [time discretization itself is not differentiable](https://arxiv.org/pdf/1910.00935.pdf)

    - More problems with AutoDiff that are not fully understood
        - ti.random() will not work
        - Emitting particles does not work
            - Unsupported Numpy operations?
            - Assigning particle active indicators does not work?
        - Update grid function does not work
            - Simply accessing and assigning indicator to a global tensor causes problem
        - Error in the form "stmt {} cannot have operand {}."
            - Seems to comes from some kind of [checker](https://github.com/taichi-dev/taichi/blob/master/taichi/analysis/verify.cpp) to make sure the operations to differentiate is supported

    - Consider different ways to do end-to-end gradient
        - If some sub-steps can't be done with AutoDiff, we can exclude it
        - For example, emitting and grid update doesn't work, but they simply set indicators and do house keeping
        - Just differentiate the sub-steps that make use of these indicators, which should be consisting of simpler differentiable operations
        - Backpropagated the gradients across the chain of sub-steps by hand (mostly just multiplying with chain rule)
        - Backpropagate across timesteps by hand (more chain rule)
        - This could give more control over how the overall gradient is computed, and allow us to handle special cases


#### June 24th
- Taichi
    - Watched Taichi Lecture 4
    - Worked on implementing differentiable simulation
        - Modified the data structure and class methods to keep tracks of particle information for each step
        - This is required for end-to-end differentiation, i.e, from initial conditions to loss evaluated on end state
        - For now can get one-step derivatives, still need to figure out how to do end-to-end, following [here](https://github.com/yuanming-hu/difftaichi/blob/master/examples/diffmpm.py)


#### June 23rd
- FleX in Unity
    - Generated several new renders
    - Implemented particle position extraction and mask creation

        <img src="viz_results/unity/out3.gif" width="350" height="200" />

        <img src="viz_results/unity/out_truth.gif" width="350" height="200" />

        <img src="viz_results/unity/out_blended.gif" width="350" height="200" />



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

