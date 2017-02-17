from __future__ import division
import os
from osgeo import gdal # must import osgeo first to avoid crash
from shapely.geometry import Point, Polygon, shape
import fiona
from rtree import index

from scipy.spatial import Voronoi
from scipy.spatial.distance import pdist, squareform
from sklearn.utils.extmath import cartesian
from itertools import combinations
import numpy as np
import pandas as pd

import sys

# to resample raster at lower resolutions
# gdalwarp -tr 100 100 -r sum SEN14adjv1_utm28.tif SEN14adjv1_utm28_100m.tif

def get_raster_data(raster_file):
    rast = gdal.Open(raster_file)
    geo_transform = rast.GetGeoTransform()
    rast_data = np.transpose(rast.ReadAsArray().astype(np.float))
    # change no-data value to -1, easier to read
    rast_data[rast_data<0] = -1
    rast = None
    return rast_data, geo_transform

""" GetGeoTransform returns
[0]  top left x
[1]  w-e pixel resolution
[2]  rotation, 0 if image is "north up"
[3]  top left y
[4]  rotation, 0 if image is "north up"
[5]  n-s pixel resolution
"""
def get_point_value(rast, point):
    rast_data, geo_transform = rast
    x = int((point[0] - geo_transform[0]) / geo_transform[1])
    y = int((point[1] - geo_transform[3]) / geo_transform[5])
    if x >= 0 and x < rast_data.shape[0] and y >= 0 and y < rast_data.shape[1]:
        return rast_data[x, y]
    else:
        return -1


def get_grid_points(rast):
    """get centre points of grid cells as array
    """
    r, gt = rast
    xstep = gt[1]
    xmin = gt[0] + xstep/2
    xmax = xmin + xstep*r.shape[0]
    ystep = gt[5]
    ymin = gt[3] + ystep/2
    ymax = ymin + ystep*r.shape[1]
    x = np.arange(xmin, xmax, xstep)
    y = np.arange(ymin, ymax, ystep)
    return cartesian((x,y))

def get_hex_grid_points(rast):
    square_shape = (rast[0].shape[0], rast[0].shape[1], 2)
    # hex shape has extra row of points
    hex_shape = (rast[0].shape[0], rast[0].shape[1] + 1, 2)
    grid = get_grid_points(rast).reshape(square_shape)
    # grid[i,:,0] ith column of xs
    # grid[i,:,1] ith column of ys (descending)

    hex_grid = np.zeros(hex_shape)
    y_step = rast[1][5]
    y_half_step = y_step / 2
    for i in range(square_shape[0]):
        # on even columns bump all the ys up half a step then add
        # new min y half a step lower
        if i % 2 == 0:
            hex_grid[i,:,1] = np.concatenate((grid[i,:,1] - y_half_step, grid[i,-1:,1] + y_half_step))
        else:
            hex_grid[i,:,1] = np.concatenate((grid[i,:1,1] - y_step, grid[i,:,1]))
        # add extra x coord to all columns
        hex_grid[i,:,0] = np.concatenate((grid[i,:,0], grid[i,:1,0]))
    return hex_grid.reshape((hex_shape[0]*hex_shape[1], 2))


def create_grids(loc_code, grid_sizes=[100, 200, 500, 1000, 2000, 5000, 10000], hexagon=False):
    rast_x = get_raster_data('../data/%s/geo/%s_pop/%s14adjv1_utm%d.tif' % (
        loc_code, loc_code, loc_code.upper(), 28 if loc_code=='sen' else 30))
    total_pop = rast_x[0][rast_x[0]>0].sum()
    print "total population: %f" % total_pop

    adm3 = read_shapefile('../data/%s/geo/%s_adm/%s_adm3_utm%d.shp' % (
        loc_code, loc_code.upper(), loc_code, 28 if loc_code=='sen' else 30))[0]
    for i, s in enumerate(grid_sizes):
        if hexagon:
            grid_fn = '../data/%s/geo/hex_grid_%d.csv' % (loc_code, s)
        else:
            grid_fn = '../data/%s/geo/grid_%d.csv' % (loc_code, s)
        if os.path.exists(grid_fn):
            grid_points = pd.read_csv(grid_fn)
        else:
            # buffer shps to capture points just outside
            shps = [shp.buffer(s/3) for shp in adm3.values()]
            rast_fn = '../data/%s/geo/%s_pop/%s14adjv1_utm%d_%d.tif' % (
                loc_code, loc_code, loc_code.upper(), 28 if loc_code=='sen' else 30, s)
            if not os.path.exists(rast_fn):
                # transform original raster file to new grid size
                os.system('gdalwarp -tr %d %d -r average ../data/%s/geo/%s_pop/%s14adjv1_utm%d.tif %s' % (
                    s, s, loc_code, loc_code, loc_code.upper(), 28 if loc_code=='sen' else 30, rast_fn))
            rast = get_raster_data(rast_fn)
            # population aggregation is average so scale up accordingly
            scale = total_pop / rast[0][rast[0]>0].sum()
            rast[0][rast[0]>0] *= scale
            grid_points = get_hex_grid_points(rast) if hexagon else get_grid_points(rast)
            ix = np.unique([i for _,i in point_in_polygon(shps, [Point(x, y) for x, y in grid_points])])
            grid_points = pd.DataFrame(grid_points[ix], columns=['x','y'])
            grid_points['population'] = [get_point_value(rast, p) for p in grid_points.values]
            grid_points.loc[grid_points['population']<0, 'population'] = 0
            grid_points.to_csv(grid_fn, index=False, encoding='utf8')
        print "total population %d: %f" % (s, grid_points['population'].sum())


def raster_to_polygons(rast):
    """convert raster to polygons
    """
    r, gt = rast
    xstep = gt[1]
    xmin = gt[0]
    xmax = xmin + xstep * r.shape[0]
    ystep = gt[5]
    ymin = gt[3]
    ymax = ymin + ystep * r.shape[1]
    x = np.arange(xmin, xmax, xstep)
    y = np.arange(ymin, ymax, ystep)
    # get bottom right of each cell
    points = cartesian((x,y))
    values = np.array([get_point_value(rast, (x+xstep/2, y+ystep/2)) for x, y in points])
    points = points[values>=0]
    values = values[values>=0]
    polys = [Polygon([(x,y), (x+xstep,y), (x+xstep,y+ystep), (x,y+ystep)]) for x, y in points]
    return polys, values

def points_from_csv(fn, sep=',', id_col=0, x_col=1, y_col=2, header=True):
    points = {}
    with open(fn, 'rb') as f:
        if header: f.next()
        i = 0
        for line in f:
            row = line.split(',')
            key = i if not id_col else int(row[id_col])
            i += 1
            x, y = (float(row[x_col]), float(row[y_col]))
            points[key] = Point(x, y)
    return points


def buffer_points(points, radius, resolution=32):
    return {key: points[key].buffer(radius, resolution) for key in points.keys()}


def voronoi_polygons(points_dict):
    ps = np.array([[p.x, p.y] for p in points_dict.values()])
    n = ps.shape[0]
    p_max = np.max(ps, axis=0)
    centre = (p_max + np.min(ps, axis=0)) / 2
    r = np.sqrt(((p_max - centre)**2).sum()) * 1.1 # max distance from centre plus 10%
    bounds = np.array(Point(centre).buffer(r, 16).exterior.coords.xy).T
    ps = np.concatenate((ps, bounds))
    vor = Voronoi(ps)
    polys = {
        key: Polygon(shell=vor.vertices[vor.regions[vor.point_region[i]]])
        for i, key in enumerate(points_dict.keys())
    }
    return polys

def read_shapefile(fn):
    shp_reader = fiona.open(fn, 'r')
    properties = {}
    shapes = {}
    for shp in shp_reader:
        id = int(shp['id'])
        properties[id] = shp['properties']
        shapes[id] = shape(shp['geometry'])
    shp_reader.close()
    return shapes, properties


def create_index(polys):
    tree = index.Index()
    iterator = None
    if type(polys) == list:
        iterator = enumerate(polys)
    elif type(polys) == dict:
        iterator = polys.iteritems()
    for i, poly in iterator:
        tree.insert(i, poly.bounds)
    return tree

def intersect_with_pop_polys(polys, pop_polys, populations, idx=None):
    if idx is None:
        idx = index.Index()
        for cell_id, cell in pop_polys.iteritems():
            idx.insert(cell_id, cell.bounds)
    poly_pop = {}
    for poly_id, poly in polys.iteritems():
        pop = 0.
        for k in idx.intersection(poly.bounds):
            cell = pop_polys[k]
            if poly.intersects(cell):
                inter_area = poly.intersection(cell).area
                if inter_area > 0 and populations[k] > 0:
                    pop += cell.area / inter_area * populations[k]
        poly_pop[poly_id] = pop
    return poly_pop

def intersect_polygons(polys_a, polys_b, idx_a=None):
    if idx is None:
        idx = index.Index()
        for cell_id, cell in pop_polys.iteritems():
            idx.insert(cell_id, cell.bounds)
    inter = {}
    for b, poly_b in polys_b.iteritems():
        for a in idx_a.intersection(poly_b.bounds):
            poly_a = polys_a[a]
            if poly_a.intersects(poly_b):
                inter[(a, b)] = poly_a.intersection(poly_b)
    return inter


def point_in_polygon(polys, points, poly_idx=None):
    if poly_idx is None:
        poly_idx = create_index(polys)
    matches = {}
    for i, p in points.iteritems():
        if i % 1000 == 0:
            print "%d of %d points done." % (i, len(points))
        for j in poly_idx.intersection((p.x, p.y)):
            # print "point %d within bbox of poly %d" % (i, j)
            if polys[j].contains(p):
                # print "point %d within poly %d" % (i, j)
                if not matches.get(j):
                    matches[j] = [i]
                else:
                    matches[j].append(i)
    return matches



def point_within_distance(polys, points, distance, poly_idx=None):
    if poly_idx is None:
        poly_idx = create_index(polys)
    matches = {}
    distances = {}
    buf_size = distance * 1.1
    if type(points) == list:
        iterator = enumerate(points)
    elif type(points) == dict:
        iterator = points.iteritems()
    for i, p in iterator:
        buf = p.buffer(buf_size)
        for j in poly_idx.intersection(buf.bounds):
            d = p.distance(polys[j])
            if d <= distance:
                if not matches.get(j):
                    matches[j] = [i]
                    distances[j] = [d]
                else:
                    matches[j].append(i)
                    distances[j].append(d)
    return matches, distances

def geom_distances(geom1, geom2):
    if type(geom1) == list:
        iterator1 = enumerate(geom1)
    elif type(geom1) == dict:
        iterator1 = geom1.iteritems()
    else:
        raise ValueError('Geometries must be in a list or dict, not %s.' % type(geom1))

    geom2_ix = geom2.keys()
    ds = {i: np.zeros(len(geom2)) for i in geom1.keys()}
    for i, a in iterator1:
        d = ds[i]
        for j, k in enumerate(geom2_ix):
            d[j] = a.distance(geom2[k])
    return ds, geom2_ix


def semivariogram_h(sqr_dif, d, h, bandwidth):
    x = sqr_dif[np.logical_and(d>=h-bandwidth, d<h+bandwidth)]
    return x.sum() / (2. * x.shape[0])

def semivariogram(points, y, lags, bandwidth):
    ds = pdist(points)
    sqr_dif = np.array([(i-j)**2 for i,j in combinations(y, 2)])
    sv = np.zeros((ds.shape[0],3))
    for i, d in enumerate(ds):
        mask = np.logical_and(ds>=d-bandwidth, ds<d+bandwidth)
        x = sqr_dif[mask]
        sv[i] = [d, mask.sum(), x.sum() / (2. * x.shape[0])]
    return sv[sv[:,1]>0]


if __name__ == '__main__':
    fn1 = sys.argv[1]
    fn2 = sys.argv[2]
    out_fn = sys.argv[3]

    shp1 = read_shapefile(fn1)
    shp2 = read_shapefile(fn2)
    ix = intersect_polys(shp1, shp2)

    with open(out_fn, 'rb') as f:
        for i, j in ix:
            f.write('%d,%d\n' % (i, j))




#
