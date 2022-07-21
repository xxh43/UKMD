import random
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff 

import pandas as pd
import pyrender
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
# axes color

grey = (128/255, 128/255, 128/255)
black = (0, 0, 0)

red = (230/255, 25/255, 75/255)
green = (60/255, 180/255, 75/255)
blue = (0/255, 130/255, 200/255)
purple = (145/255, 30/255, 180/255)
orange = (245/255, 130/255, 48/255)
yellow = (255/255, 255/255, 25/255)
cyan = (70/255, 240/255, 240/255)
maroon = (128/255, 0, 0)
olive = (128/255, 128/255, 0)
teal = (0, 128/255, 128/255)
navy = (0, 0, 128/255)
lime = (210/255, 245/255, 60/255)
magenta = (240/255, 50/255, 230/255)
brown = (170/255, 110/255, 40/255)
pink = (250/255, 190/255, 212/255)
apricot = (255/255, 215/255, 180/255)
beige = (255/255, 250/255, 200/255)
mint = (170/255, 255/255, 195/255)
lavender = (220/255, 190/255, 255/255)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def to_numpy(item):
    if torch.is_tensor(item):
        if item.is_cuda:
            return item.cpu().detach().numpy()
        else:
            return item.detach().numpy()
    else:
        return item

def to_tensor(item):
    return torch.tensor(item, device="cuda:0", dtype=torch.float)

grey_color_rgb = (110,110,110)
color_palette_rgb = [(245,130,48), (0,130,200), (145,30,180), (128, 128, 0), (0, 128, 128), (70,240,240), (255,255,25), (0, 0, 128), (210, 245, 60), (240, 50, 230), (170, 110, 40)]

grey_color_str = 'rgb(110,110,110)'
color_palette_str = ['rgb(245,130,48)', 'rgb(0,130,200)', 'rgb(145,30,180)', 'rgb(128, 128, 0)', 'rgb(70,240,240)', 'rgb(0, 128, 128)', 'rgb(255,255,25)', 'rgb(0, 0, 128)', 'rgb(210, 245, 60)', 'rgb(240, 50, 230)', 'rgb(170, 110, 40)']
color_palette_str_dark = ['rgb(120,75,24)', 'rgb(0,75,100)', 'rgb(70,15,90)', 'rgb(64, 64, 0)', 'rgb(70,240,240)', 'rgb(0, 128, 128)', 'rgb(255,255,25)', 'rgb(0, 0, 128)', 'rgb(210, 245, 60)', 'rgb(240, 50, 230)', 'rgb(170, 110, 40)']

def display_pcs(pcs, filename=' ', save=False, has_bg=False):

    pcs = [ to_numpy(pcs[i]) for i in range(len(pcs))]

    traces = []
    all_x = []
    all_y = []
    all_z = []
    for i in range(len(pcs)):
        pcs[i] = to_numpy(pcs[i])
        x = []
        y = []
        z = []
        c = []
        for p in pcs[i]:
            c.append(color_palette_str[i])
            x.append(p[0])
            y.append(p[1])
            z.append(p[2])

        trace = go.Scatter3d(
            x=x, 
            y=y, 
            z=z, 
            mode='markers', 
            marker=dict(
                size=3,
                color=c,                
                colorscale='Viridis',   
                opacity=1.0
            )
        )
        traces.append(trace)
        all_x += x
        all_y += y
        all_z += z
    
    set_fig(traces, filename, save, has_bg)

def display_pcs_and_axes(pcs, axes, centers, filename=' ', save=False, has_bg=False):

    pcs = [ to_numpy(pcs[i]) for i in range(len(pcs))]
    axes = [ to_numpy(axes[i]) for i in range(len(axes))]
    centers = [ to_numpy(centers[i]) for i in range(len(centers))]

    pc_traces = []
    all_x = []
    all_y = []
    all_z = []
    for i in range(len(pcs)):
        pcs[i] = to_numpy(pcs[i])
        x = []
        y = []
        z = []
        c = []
        for p in pcs[i]:
            c.append(color_palette_str[i])
            x.append(p[0])
            y.append(p[1])
            z.append(p[2])

        trace = go.Scatter3d(
            x=x, 
            y=y, 
            z=z, 
            mode='markers', 
            marker=dict(
                size=3,
                color=c,                
                colorscale='Viridis',   
                opacity=1.0
            )
        )
        pc_traces.append(trace)
        all_x += x
        all_y += y
        all_z += z
    
    axis_traces = []
    for i in range(len(axes)):
        axis = axes[i]
        center = centers[i]
        if axis is None or center is None:
            continue
        
        axis_end1 = center - 4*axis
        axis_end2 = center + 4*axis

        trace = go.Scatter3d(
            x = [axis_end1[0], axis_end2[0]], 
            y = [axis_end1[1], axis_end2[1]],
            z = [axis_end1[2], axis_end2[2]],
            line=dict(
                color='black',
                width=15
            )
        )

        axis_traces.append(trace)
        all_x += x
        all_y += y
        all_z += z

    traces = pc_traces+axis_traces
    set_fig(traces, filename, save, has_bg)


def display_meshes_with_axes(meshes, axes, centers, filename=' ', save=False, has_bg=False):

    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    zmin = np.inf
    zmax = -np.inf

    mesh_traces = []
    for mesh_index in range(len(meshes)):
        print('mesh_index', mesh_index)
        color = colors[mesh_index]
        mesh = meshes[mesh_index]
        x = []
        y = []
        z = []
        i = []
        j = []
        k = []
        c = color_palette_str[mesh_index]
        for v in mesh.vertices:
            x.append(v[0])
            y.append(v[1])
            z.append(v[2])
        
        for f in mesh.faces:
            i.append(f[0])
            j.append(f[1])
            k.append(f[2])

        trace = go.Mesh3d(
            x=x, 
            y=y, 
            z=z, 
            i = i,
            j = j,
            k = k,
            color = color,
            opacity = 1.0
        )
        mesh_traces.append(trace)

    axis_traces = []
    for i in range(len(axes)):
        axis = axes[i]
        center = centers[i]
        if axis is None or center is None:
            continue
        
        axis_end1 = center
        axis_end2 = center + 2*axis

        trace = go.Scatter3d(
            x = [axis_end1[0], axis_end2[0]], 
            y = [axis_end1[1], axis_end2[1]],
            z = [axis_end1[2], axis_end2[2]],
            line=dict(
                color='black',
                width=15
            )
        )

        axis_traces.append(trace)

    traces = mesh_traces+axis_traces

    set_fig(traces, filename, save, has_bg)

def set_fig(traces, filename, save, has_bg=False):
    
    fig = go.Figure(data=traces)

    for trace in fig['data']: 
        trace['showlegend'] = False

    camera = dict(
    up=dict(x=0, y=1, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.0, y=1.0, z=1.0)
    )

    fig.update_layout(scene_camera=camera)

    range_min = -1.5
    range_max = 1.5

    if has_bg:
        fig.update_layout(
            scene = dict(xaxis = dict(range=[range_min,range_max],),
                        yaxis = dict(range=[range_min,range_max],),
                        zaxis = dict(range=[range_min,range_max],),
                        aspectmode='cube'),
            #paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            autosize=False,width=1000,height=1000)
    else:
        fig.update_layout(
            scene = dict(xaxis = dict(range=[range_min,range_max],),
                        yaxis = dict(range=[range_min,range_max],),
                        zaxis = dict(range=[range_min,range_max],),
                        aspectmode='cube'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            autosize=False,width=1000,height=1000)

    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
    
    if save is True:
        fig.write_image(filename)
    else:
        fig.show()