import numpy as np
import cv2

DEBUG=False

def generate_char_maps(rec_rois_gt_chars, gt_box, polygon, rec_map_height, rec_map_width, shrink = 0.25, is_balanced = False):
    
    if DEBUG:
        print('rec_rois_gt_chars.shape -> ', rec_rois_gt_chars.shape)
        print('gt_box.shape -> ', gt_box.shape)
        print('polygon.shape -> ', polygon.shape)
        print('rec_map_height', rec_map_height)
        print('rec_map_width', rec_map_width)
        print('shrink', shrink)
        print('is_balanced', is_balanced)

    char_map = np.zeros((37, rec_map_height, rec_map_width))
    char_map_weight = np.zeros((37, rec_map_height, rec_map_width))
    xmin = gt_box[0]
    ymin = gt_box[1]
    xmax = gt_box[2]
    ymax = gt_box[3]
    gt_height = ymax - ymin
    gt_width = xmax - xmin
    width_ratio = float(rec_map_width) / gt_width
    height_ratio = float(rec_map_height) / gt_height
    rec_rois_gt_chars[0,:,0:8:2] = (rec_rois_gt_chars[0,:,0:8:2] - xmin) * width_ratio
    rec_rois_gt_chars[0,:,1:8:2] = (rec_rois_gt_chars[0,:,1:8:2] - ymin) * height_ratio
    polygon[0:8:2] = (polygon[0:8:2] - xmin) * width_ratio
    polygon[1:8:2] = (polygon[1:8:2] - ymin) * height_ratio
    polygon_reshape = polygon.reshape((4, 2)).astype(np.int32)
    cv2.fillPoly(char_map[0,:,:], [polygon_reshape], 1)
    char_map_weight[0,:,:] = np.ones((rec_map_height, rec_map_width))
    for i in range(rec_rois_gt_chars.shape[1]):  
        gt_poly = rec_rois_gt_chars[0,i,:8]
        gt_poly_reshape = gt_poly.reshape((4, 2))
        char_cls = int(rec_rois_gt_chars[0,i,8])
        if shrink>0:
            npoly = shrink_poly(gt_poly_reshape, shrink)
        else:
            npoly = gt_poly_reshape
        poly = npoly.astype(np.int32)
        cv2.fillPoly(char_map[char_cls,:,:], [poly], 1)
        if is_balanced:
            num_all = rec_map_height * rec_map_width
            num_pos = char_map[char_cls,:,:].sum()
            num_neg = num_all - num_pos
            weight_pos = float(num_neg) / num_all
            weight_neg = 1 - weight_pos
            map_tmp = np.ones((rec_map_height, rec_map_width), dtype=float) * weight_neg
            cv2.fillPoly(map_tmp, [polygon_reshape], weight_pos)
        else:
            map_tmp = np.ones((rec_map_height, rec_map_width), dtype=float)
        char_map_weight[char_cls,:,:] = map_tmp

    return char_map, char_map_weight

def generate_char_maps_and_polygon_map(rec_rois_gt_chars, gt_box, polygon, rec_map_height, rec_map_width, shrink = 0.25):
    char_map = np.zeros((2, rec_map_height, rec_map_width))
    char_map_weight = np.zeros((2, rec_map_height, rec_map_width))
    xmin = gt_box[0]
    ymin = gt_box[1]
    xmax = gt_box[2]
    ymax = gt_box[3]
    gt_height = ymax - ymin
    gt_width = xmax - xmin
    width_ratio = float(rec_map_width) / gt_width
    height_ratio = float(rec_map_height) / gt_height

    polygon[0:8:2] = (polygon[0:8:2] - xmin) * width_ratio
    polygon[1:8:2] = (polygon[1:8:2] - ymin) * height_ratio
    polygon_reshape = polygon.reshape((4, 2)).astype(np.int32)
    cv2.fillPoly(char_map[0,:,:], [polygon_reshape], 1)
    char_map_weight[0,:,:] = np.ones((rec_map_height, rec_map_width))
    if rec_rois_gt_chars.size > 0:
        rec_rois_gt_chars[0,:,0:8:2] = (rec_rois_gt_chars[0,:,0:8:2] - xmin) * width_ratio
        rec_rois_gt_chars[0,:,1:8:2] = (rec_rois_gt_chars[0,:,1:8:2] - ymin) * height_ratio
        for i in range(rec_rois_gt_chars.shape[1]):  
            gt_poly = rec_rois_gt_chars[0,i,:8]
            gt_poly_reshape = gt_poly.reshape((4, 2))
            char_cls = int(rec_rois_gt_chars[0,i,8])
            if shrink>0:
                npoly = shrink_poly(gt_poly_reshape, shrink)
            else:
                npoly = gt_poly_reshape
            poly = npoly.astype(np.int32)
            map_tmp = np.zeros((rec_map_height, rec_map_width))
            cv2.fillPoly(char_map[1,:,:], [poly], char_cls)
            char_map_weight[1,:,:] = np.ones((rec_map_height, rec_map_width))
    else:
        char_map[1, :, :].fill(-1)
        # assert(char_map[1, :, :].max() == -1)

    return char_map, char_map_weight

def shrink_poly(poly, shrink):
    # shrink ratio
    R = shrink
    r = [None, None, None, None]
    for i in range(4):
        r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                   np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
    # find the longer pair
    if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
                    np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
        ## p0, p3
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
    else:
        ## p0, p3
        # print poly
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
    return poly