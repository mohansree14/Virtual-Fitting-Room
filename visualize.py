"""Turn a predicted point cloud into a mesh: normal estimation + screened Poisson remeshing.

usage: python visualize.py results/aligned/pcd/test_pairs/<name>.ply
"""
import argparse
import os

import pymeshlab as ml
import trimesh


def compute_normals(input_file, output_file):
    mesh = trimesh.load(input_file)
    mesh.compute_vertex_normals()
    mesh.export(output_file)


def poisson_remesh(input_file, output_file, depth=9):
    ms = ml.MeshSet()
    ms.load_new_mesh(input_file)
    ms.surface_reconstruction_screened_poisson(target_edge_length=depth)
    ms.save_current_mesh(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ply', help='point cloud from rgbd2pcd.py')
    parser.add_argument('--out_dir', help='defaults to the folder of the input file')
    opt = parser.parse_args()

    out_dir = opt.out_dir or os.path.dirname(opt.ply)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(opt.ply))[0]
    normals_ply = os.path.join(out_dir, stem + '_normals.ply')
    remeshed_ply = os.path.join(out_dir, stem + '_remeshed.ply')

    compute_normals(opt.ply, normals_ply)
    poisson_remesh(normals_ply, remeshed_ply)
    print('Saved', remeshed_ply)
