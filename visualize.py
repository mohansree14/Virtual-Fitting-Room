import trimesh
import pymeshlab as ml

def compute_normals(input_file, output_file):
    # Load the mesh
    mesh = trimesh.load(input_file)
    
    # Compute normals
    mesh.compute_vertex_normals()
    
    # Save the mesh with normals
    mesh.export(output_file)

def poisson_remesh(input_file, output_file, depth=9):
    ms = ml.MeshSet()
    ms.load_new_mesh(input_file)
    ms.surface_reconstruction_screened_poisson(target_edge_length=depth)  # Perform Poisson surface reconstruction
    ms.save_current_mesh(output_file)


if __name__ == "__main__":
    # Paths
    input_ply = r"D:\vtryon_workout\M3D-VTON\results\aligned\pcd\test_pairs\BJ721E05W-J11@9=person.ply"
    output_normals_ply = "results/aligned/pcd/test_pairs/normals.ply"
    output_remeshed_ply = "results/aligned/pcd/test_pairs/remeshed.ply"

    # Compute normals
    compute_normals(input_ply, output_normals_ply)

    # Poisson remeshing
    poisson_remesh(output_normals_ply, output_remeshed_ply)

    print("Poisson remeshing completed.")
