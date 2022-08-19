from __future__ import absolute_import, division, print_function, unicode_literals

import scipy.ndimage
from scipy.sparse.linalg import spsolve
from scipy import sparse
import scipy.io as sio
import numpy as np
from PIL import Image
import copy
import cv2
import os
import argparse


def sub2ind(pi, pj, imgH, imgW):
    return pj + pi * imgW


def Poisson_blend(imgTrg, imgSrc_gx, imgSrc_gy, holeMask, edge=None):

    imgH, imgW, nCh = imgTrg.shape

    if not isinstance(edge, np.ndarray):
        edge = np.zeros((imgH, imgW), dtype=np.float32)

    # Initialize the reconstructed image
    imgRecon = np.zeros((imgH, imgW, nCh), dtype=np.float32)

    # prepare discrete Poisson equation
    A, b = solvePoisson(holeMask, imgSrc_gx, imgSrc_gy, imgTrg, edge)

    # Independently process each channel
    for ch in range(nCh):

        # solve Poisson equation
        x = scipy.sparse.linalg.lsqr(A, b[:, ch, None])[0]
        imgRecon[:, :, ch] = x.reshape(imgH, imgW)

    # Combined with the known region in the target
    holeMaskC = np.tile(np.expand_dims(holeMask, axis=2), (1, 1, nCh))
    imgBlend = holeMaskC * imgRecon + (1 - holeMaskC) * imgTrg

    # Fill in edge pixel
    pi = np.expand_dims(np.where((holeMask * edge) == 1)[0], axis=1) # y, i
    pj = np.expand_dims(np.where((holeMask * edge) == 1)[1], axis=1) # x, j

    for k in range(len(pi)):
        if pi[k, 0] - 1 >= 0:
            if edge[pi[k, 0] - 1, pj[k, 0]] == 0:
                imgBlend[pi[k, 0], pj[k, 0], :] = imgBlend[pi[k, 0] - 1, pj[k, 0], :]
                continue
        if pi[k, 0] + 1 <= imgH - 1:
            if edge[pi[k, 0] + 1, pj[k, 0]] == 0:
                imgBlend[pi[k, 0], pj[k, 0], :] = imgBlend[pi[k, 0] + 1, pj[k, 0], :]
                continue
        if pj[k, 0] - 1 >= 0:
            if edge[pi[k, 0], pj[k, 0] - 1] == 0:
                imgBlend[pi[k, 0], pj[k, 0], :] = imgBlend[pi[k, 0], pj[k, 0] - 1, :]
                continue
        if pj[k, 0] + 1 <= imgW - 1:
            if edge[pi[k, 0], pj[k, 0] + 1] == 0:
                imgBlend[pi[k, 0], pj[k, 0], :] = imgBlend[pi[k, 0], pj[k, 0] + 1, :]

    return imgBlend

def solvePoisson(holeMask, imgSrc_gx, imgSrc_gy, imgTrg, edge):

    # Prepare the linear system of equations for Poisson blending
    imgH, imgW = holeMask.shape
    N = imgH * imgW

    # Number of unknown variables
    numUnknownPix = holeMask.sum()

    # 4-neighbors: dx and dy
    dx = [1, 0, -1,  0]
    dy = [0, 1,  0, -1]

    #      3
    #      |
    # 2 -- * -- 0
    #      |
    #      1
    #

    # Initialize (I, J, S), for sparse matrix A where A(I(k), J(k)) = S(k)
    I = np.empty((0, 1), dtype=np.float32)
    J = np.empty((0, 1), dtype=np.float32)
    S = np.empty((0, 1), dtype=np.float32)

    # Initialize b
    b = np.empty((0, 2), dtype=np.float32)

    # Precompute unkonwn pixel position
    pi = np.expand_dims(np.where(holeMask == 1)[0], axis=1) # y, i
    pj = np.expand_dims(np.where(holeMask == 1)[1], axis=1) # x, j
    pind = sub2ind(pi, pj, imgH, imgW)

    # |--------------------|
    # |        y (i)       |
    # |   x (j)  *         |
    # |                    |
    # |--------------------|

    qi = np.concatenate((pi + dy[0],
                         pi + dy[1],
                         pi + dy[2],
                         pi + dy[3]), axis=1)

    qj = np.concatenate((pj + dx[0],
                         pj + dx[1],
                         pj + dx[2],
                         pj + dx[3]), axis=1)

    # Handling cases at image borders
    validN = (qi >= 0) & (qi <= imgH - 1) & (qj >= 0) & (qj <= imgW - 1)
    qind = np.zeros((validN.shape), dtype=np.float32)
    qind[validN] = sub2ind(qi[validN], qj[validN], imgH, imgW)

    e_start = 0  # equation counter start
    e_stop  = 0  # equation stop

    # 4 neighbors
    I, J, S, b, e_start, e_stop = constructEquation(0, validN, holeMask, edge, imgSrc_gx, imgSrc_gy, imgTrg, pi, pj, pind, qi, qj, qind, I, J, S, b, e_start, e_stop)
    I, J, S, b, e_start, e_stop = constructEquation(1, validN, holeMask, edge, imgSrc_gx, imgSrc_gy, imgTrg, pi, pj, pind, qi, qj, qind, I, J, S, b, e_start, e_stop)
    I, J, S, b, e_start, e_stop = constructEquation(2, validN, holeMask, edge, imgSrc_gx, imgSrc_gy, imgTrg, pi, pj, pind, qi, qj, qind, I, J, S, b, e_start, e_stop)
    I, J, S, b, e_start, e_stop = constructEquation(3, validN, holeMask, edge, imgSrc_gx, imgSrc_gy, imgTrg, pi, pj, pind, qi, qj, qind, I, J, S, b, e_start, e_stop)

    nEqn = len(b)
    # Construct the sparse matrix A
    A = sparse.csr_matrix((S[:, 0], (I[:, 0], J[:, 0])), shape=(nEqn, N))

    return A, b


def constructEquation(n, validN, holeMask, edge, imgSrc_gx, imgSrc_gy, imgTrg, pi, pj, pind, qi, qj, qind, I, J, S, b, e_start, e_stop):

    # Pixel that has valid neighbors
    validNeighbor = validN[:, n]

    # Change the out-of-boundary value to 0, in order to run edge[y,x]
    # in the next line. It won't affect anything as validNeighbor is saved already

    qi_tmp = copy.deepcopy(qi)
    qj_tmp = copy.deepcopy(qj)
    qi_tmp[np.invert(validNeighbor), n] = 0
    qj_tmp[np.invert(validNeighbor), n] = 0

    # Not edge
    NotEdge = (edge[pi[:, 0], pj[:, 0]] == 0) * (edge[qi_tmp[:, n], qj_tmp[:, n]] == 0)

    # Boundary constraint
    Boundary = holeMask[qi_tmp[:, n], qj_tmp[:, n]] == 0
    valid = validNeighbor * NotEdge * Boundary
    J_tmp = pind[valid, :]

    # num of equations: len(J_tmp)
    e_stop = e_start + len(J_tmp)
    I_tmp = np.arange(e_start, e_stop, dtype=np.float32).reshape(-1, 1)
    e_start = e_stop

    S_tmp = np.ones(J_tmp.shape, dtype=np.float32)

    if n == 0:
        b_tmp = - imgSrc_gx[pi[valid, 0], pj[valid, 0], :] + imgTrg[qi[valid, n], qj[valid, n], :]
    elif n == 2:
        b_tmp = imgSrc_gx[pi[valid, 0], pj[valid, 0] - 1, :] + imgTrg[qi[valid, n], qj[valid, n], :]
    elif n == 1:
        b_tmp = - imgSrc_gy[pi[valid, 0], pj[valid, 0], :] + imgTrg[qi[valid, n], qj[valid, n], :]
    elif n == 3:
        b_tmp = imgSrc_gy[pi[valid, 0] - 1, pj[valid, 0], :] + imgTrg[qi[valid, n], qj[valid, n], :]

    I = np.concatenate((I, I_tmp))
    J = np.concatenate((J, J_tmp))
    S = np.concatenate((S, S_tmp))
    b = np.concatenate((b, b_tmp))


    # Non-boundary constraint
    NonBoundary = holeMask[qi_tmp[:, n], qj_tmp[:, n]] == 1
    valid = validNeighbor * NotEdge * NonBoundary

    J_tmp = pind[valid, :]

    # num of equations: len(J_tmp)
    e_stop = e_start + len(J_tmp)
    I_tmp = np.arange(e_start, e_stop, dtype=np.float32).reshape(-1, 1)
    e_start = e_stop

    S_tmp = np.ones(J_tmp.shape, dtype=np.float32)

    if n == 0:
        b_tmp = - imgSrc_gx[pi[valid, 0], pj[valid, 0], :]
    elif n == 2:
        b_tmp = imgSrc_gx[pi[valid, 0], pj[valid, 0] - 1, :]
    elif n == 1:
        b_tmp = - imgSrc_gy[pi[valid, 0], pj[valid, 0], :]
    elif n == 3:
        b_tmp = imgSrc_gy[pi[valid, 0] - 1, pj[valid, 0], :]

    I = np.concatenate((I, I_tmp))
    J = np.concatenate((J, J_tmp))
    S = np.concatenate((S, S_tmp))
    b = np.concatenate((b, b_tmp))

    S_tmp = - np.ones(J_tmp.shape, dtype=np.float32)
    J_tmp = qind[valid, n, None]

    I = np.concatenate((I, I_tmp))
    J = np.concatenate((J, J_tmp))
    S = np.concatenate((S, S_tmp))

    return I, J, S, b, e_start, e_stop


def gradient_mask(mask):  #产生梯度的mask

    gradient_mask = np.logical_or.reduce((mask,
        np.concatenate((mask[1:, :], np.zeros((1, mask.shape[1]), dtype=np.bool)), axis=0),
        np.concatenate((mask[:, 1:], np.zeros((mask.shape[0], 1), dtype=np.bool)), axis=1)))

    return gradient_mask

    
    
if __name__ == '__main__':
    import cvbase
    from skimage.feature import canny
    import argparse
    import imageio

    parser = argparse.ArgumentParser()
    parser.add_argument('--flow', type=str, default='../../test_blending/flow/00000.flo')
    parser.add_argument('--mask', type=str, default='../../test_blending/mask/00000.png')
    parser.add_argument('--width', type=int, default=432)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--output', type=str, default='../../test_blending/ret')
    args = parser.parse_args()
    flow, mask = args.flow, args.mask
    width, height = args.width, args.height
    output = args.output
    
    if not os.path.exists(output):
        os.makedirs(output)

    flow = cvbase.read_flow(flow)
    mask = cv2.imread(mask, 0)
    h, w, c = flow.shape
    flow_resized = np.zeros((height, width, 2))
    flow_resized[:, :, 0] = cv2.resize(flow[:, :, 0], (width, height), cv2.INTER_LINEAR) * width / w
    flow_resized[:, :, 1] = cv2.resize(flow[:, :, 1], (width, height), cv2.INTER_LINEAR) * height / h
    flow = flow_resized

    mask = cv2.resize(mask, (width, height), cv2.INTER_NEAREST)
    mask_gradient = gradient_mask(mask)

    flow_gray = (flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2) ** 0.5
    flow_gray = flow_gray / flow_gray.max()
    edge = canny(flow_gray, sigma=1, low_threshold=0.1, high_threshold=0.2)
    edge = edge.astype(np.bool)
    masked_edge = edge * mask

    # gradients
    gradient_x = np.concatenate((np.diff(flow, axis=1), np.zeros((height, 1, 2), dtype=np.float32)), axis=1)
    gradient_y = np.concatenate((np.diff(flow, axis=0), np.zeros((1, width, 2), dtype=np.float32)), axis=0)
    gradient = np.concatenate((gradient_x, gradient_y), axis=2)
    gradient[mask_gradient, :] = 0  # 把中间的梯度设置成了0

    # complete flow
    imgSrc_gy = gradient[:, :, 2: 4]
    imgSrc_gy = imgSrc_gy[0: h - 1, :, :]
    imgSrc_gx = gradient[:, :, 0: 2]
    imgSrc_gx = imgSrc_gx[:, 0: w - 1, :]
    compFlow = Poisson_blend(flow, imgSrc_gx, imgSrc_gy, mask, masked_edge)  # todo: edge or masked_edge ?

    # save flow
    flow_n = cvbase.flow2rgb(flow)
    compFlow_n = cvbase.flow2rgb(compFlow)
    imageio.imwrite(os.path.join(output, 'flow.png'), flow_n)
    imageio.imwrite(os.path.join(output, 'compFlow.png'), compFlow_n)
    imageio.imwrite(os.path.join(output, 'edge.png'), masked_edge)
    # imageio.imwrite(os.path.join(output, 'gx.png'), imgSrc_gx)
    # imageio.imwrite(os.path.join(output, 'gy.png'), imgSrc_gy)
    imageio.imwrite(os.path.join(output, 'mask.png'), mask)
    imageio.imwrite(os.path.join(output, 'grad0.png'), gradient[:, :, 0])
    imageio.imwrite(os.path.join(output, 'grad1.png'), gradient[:, :, 1])

