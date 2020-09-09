import trimesh

def convert(file_name, source_format, new_format):
    mesh = trimesh.load(file_name + "." + source_format)
    mesh.export(file_name + "." + new_format)

convert("heart2assem","PLY","obj")