# Copyright (c) Facebook, Inc. and its affiliates.

import os
import torch
import numpy as np
import cv2
import argparse
from model.mesh.mesh_model import FrankMocap


def load_mesh(input_path):
    base_img = cv2.imread(input_path)
    mesh_model = FrankMocap()
    
    exist, rot_mat = mesh_model.inference(base_img, 'pred_rotmat')
    
    if not exist:
        return False
    
    return rot_mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True, help='Path of input')
    args = parser.parse_args()
    load_mesh(args.input_path)


if __name__ == '__main__':
    main()