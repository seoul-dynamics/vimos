# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
from smplx import SMPL 

from vimos.base import Modifier

class SegmentModifier(Modifier):
    def load_smpl(self):
        smpl_dir = './vimos/model/mesh/extra_data'
        smplModelPath = smpl_dir + '/smpl.pkl'
        smpl = SMPL(smplModelPath, batch_size=1, create_transl=False)
        
        self.weights = smpl.lbs_weights
        self.faces = smpl.faces


    def __init__(self):
        self.load_smpl()
        self.segment_Jindex_map = {
            'left_arms' : [13, 16, 18, 20 ,22],
            'right_arms' : [23, 21, 19, 17, 14],
            'arms' : [23, 21, 19, 17, 14, 13, 16, 18, 20, 22],
            'legs':[2, 5, 8, 11, 1, 4, 7, 10], 
            'left_leg':[1, 4, 7, 10],
            'right_leg':[2, 5, 8, 11],
            'body':[0, 3, 6, 9, 12, 15]
        }
        
    def get_vertex_index(self, seg_name):
        index = np.where(self.weights[:,self.segment_Jindex_map[seg_name]] > 0.5)[0]
        return index

    def load_submesh_face(self, index):
        seg_faces = []
        flen = faces.shape[0]
        for j in range(flen):
            if faces[j,0] in index and faces[j,1] in index and faces[j,2] in index:
                f1 = np.where(index == faces[j,0])[0][0]
                f2 = np.where(index == faces[j,1])[0][0]
                f3 = np.where(index == faces[j,2])[0][0]
                seg_faces.append([f1,f2,f3])
        seg_faces = np.array(seg_faces)
        return seg_faces
    
    def _process(self, mesh, segment_name):
        if segment_name not in list(self.segment_Jindex_map.keys()):
            raise ValueError(f"Segment name {segment_name} not found")
        
        index = self.get_vertex_index(segment_name)
        seg_faces = self.load_submesh_face(index)
        
        sub_mesh = {}

        sub_mesh["face"] = seg_faces
        sub_mesh["vertex"] = mesh["vertex"][index]

        return sub_mesh
        
