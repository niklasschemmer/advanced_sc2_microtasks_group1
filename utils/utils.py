from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pysc2.lib import actions
from pysc2.lib import features

_SCREEN_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Array with all screen features that should be used for this training purpose
usable_screen_features = np.array([
    #  'placeholder',
    #  'height_map',
    #  'visibility_map',
    #  'creep',
    'power',
    'player_id',
    'player_relative',
    'unit_type',
    'selected',
    'unit_hit_points',
    'unit_hit_points_ratio',
    #  'unit_energy',
    #  'unit_energy_ratio',
    'unit_shields',
    'unit_shields_ratio',
    'unit_density',
    'unit_density_aa',
    'effects',
    #  'hallucinations',
    #  'cloaked',
    'blip',
    #  'buffs',
    #  'buff_duration',
    'active',
    #  'build_progress',
    #  'pathable',
    #  'buildable'
])


def preprocess_screen(screen):
    """
    Preprocess screen, to normalize it and putting all screen features into proper formats

    Parameter screen: The unprocessed screen, extracted from the observation of the environment.
    """
    layers = []
    assert screen.shape[0] == len(features.SCREEN_FEATURES)
    for i in range(len(features.SCREEN_FEATURES)):
        # Sort out not used screen features
        if features.SCREEN_FEATURES[i].name in usable_screen_features:
            if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE:
                layers.append(screen[i:i + 1] /
                              features.SCREEN_FEATURES[i].scale)
            elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
                layers.append(screen[i:i + 1] /
                              features.SCREEN_FEATURES[i].scale)
            else:
                layer = np.zeros([features.SCREEN_FEATURES[i].scale,
                                 screen.shape[1], screen.shape[2]], dtype=np.float32)
                for j in range(features.SCREEN_FEATURES[i].scale):
                    indy, indx = (screen[i] == j).nonzero()
                    layer[j, indy, indx] = 1
                layers.append(layer)
    return np.concatenate(layers, axis=0)


def screen_channel():
    """
    Get total amount of channel from all existing screen features
    """
    c = 0
    for i in range(len(features.SCREEN_FEATURES)):
        # Sort out not used screen features
        if features.SCREEN_FEATURES[i].name in usable_screen_features:
            # For player or scalar features increase channels by one and for
            # scalar values by the scale amount
            if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE:
                c += 1
            elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
                c += 1
            else:
                c += features.SCREEN_FEATURES[i].scale
    return c
