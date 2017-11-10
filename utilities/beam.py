# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 17:03:00 2015

@author: wolfensb
"""

class Beam():
    def __init__(self, dic_values, mask, lats_profile,lons_profile, dist_ground_profile,heights_profile, elev_profile=[], GH_pt=[],GH_weight=1):
        self.mask=mask
        self.GH_pt=GH_pt
        self.GH_weight=GH_weight
        self.lats_profile=lats_profile
        self.lons_profile=lons_profile
        self.dist_profile=dist_ground_profile
        self.heights_profile=heights_profile
        self.elev_profile=elev_profile
        self.values=dic_values
    