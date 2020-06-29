import taichi as ti

ti.init(arch=ti.gpu)

# params
num_particles = 10
grid_size = 3
max_num_particles_per_cell = 10

cell_size = 2.51
cell_recpr = 1.0 / cell_size

# variables
positions = ti.Vector(3, ti.f32)
grid_num_particles = ti.var(ti.i32)
grid2particles = ti.var(ti.i32)
loss = ti.var(ti.f32)

# layout
ti.root.dense(ti.i, num_particles).place(positions)
grid_snode = ti.root.dense(ti.ijk, grid_size)
grid_snode.place(grid_num_particles)
grid_snode.dense(ti.l, num_particles).place(grid2particles)
ti.root.place(loss)

ti.root.lazy_grad()


@ti.func
def get_cell(pos):
    return ti.cast(pos * cell_recpr, ti.i32)

@ti.kernel
def update_grid():
    for i in range(num_particles):
        cell = get_cell(positions[i])
        # ti.Vector doesn't seem to support unpacking yet
        # but we can directly use int Vectors as indices

        # This does not work (from pbf2d example)
        offs = grid_num_particles[cell].atomic_add(1)
        grid2particles[cell, i] = 1

        # This works
        # ind = grid_num_particles[cell]
        # grid2particles[cell, ind] = i
        # grid_num_particles[cell].atomic_add(1)


@ti.kernel
def init():
    for i in range(num_particles):
        for k in ti.static(range(3)):
            positions[i][k] = 1

def forward():
    grid_num_particles.fill(0)
    grid2particles.fill(0)
    update_grid()

@ti.kernel
def compute_loss():
    for i in range(num_particles):
        for k in ti.static(range(3)):
            loss += positions[i][k]


init()
loss[None] = 0

with ti.Tape(loss=loss):
    forward()
    compute_loss()
print("Loss:", loss[None])
print("Grad to initial pos: ", positions.grad[0][0], positions.grad[0][1], positions.grad[0][2])






