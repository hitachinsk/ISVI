import numpy as np
import cv2
from scipy import sparse
from scipy.sparse.linalg import spsolve


def regionfill(I, mask, factor=1.0):  # I -> flow, mask -> flow mask
    if np.count_nonzero(mask) == 0:
        return I.copy()  # All of the regions in mask has been filled
    resize_mask = cv2.resize(
        mask.astype(float), (0, 0), fx=factor, fy=factor) > 0
    resize_I = cv2.resize(I.astype(float), (0, 0), fx=factor, fy=factor)
    maskPerimeter = findBoundaryPixels(resize_mask)  # boundary pixels in the resized mask
    regionfillLaplace(resize_I, resize_mask, maskPerimeter)
    resize_I = cv2.resize(resize_I, (I.shape[1], I.shape[0]))
    resize_I[mask == 0] = I[mask == 0]
    return resize_I


def findBoundaryPixels(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    maskDilated = cv2.dilate(mask.astype(float), kernel)
    return (maskDilated > 0) & (mask == 0)


def regionfillLaplace(I, mask, maskPerimeter):
    height, width = I.shape
    rightSide = formRightSide(I, maskPerimeter)

    # Location of mask pixels
    maskIdx = np.where(mask)

    # Only keep values for pixels that are in the mask
    rightSide = rightSide[maskIdx]

    # Number the mask pixels in a grid matrix
    grid = -np.ones((height, width))
    grid[maskIdx] = range(0, maskIdx[0].size)
    # Pad with zeros to avoid "index out of bounds" errors in the for loop
    grid = padMatrix(grid)
    gridIdx = np.where(grid >= 0)

    # Form the connectivity matrix D=sparse(i,j,s)
    # Connect each mask pixel to itself
    i = np.arange(0, maskIdx[0].size)
    j = np.arange(0, maskIdx[0].size)
    # The coefficient is the number of neighbors over which we average
    numNeighbors = computeNumberOfNeighbors(height, width)
    s = numNeighbors[maskIdx]
    # Now connect the N,E,S,W neighbors if they exist
    for direction in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        # Possible neighbors in the current direction
        neighbors = grid[gridIdx[0] + direction[0], gridIdx[1] + direction[1]]
        # ConDnect mask points to neighbors with -1's
        index = (neighbors >= 0)
        i = np.concatenate((i, grid[gridIdx[0][index], gridIdx[1][index]]))
        j = np.concatenate((j, neighbors[index]))
        s = np.concatenate((s, -np.ones(np.count_nonzero(index))))

    D = sparse.coo_matrix((s, (i.astype(int), j.astype(int)))).tocsr()
    sol = spsolve(D, rightSide)
    I[maskIdx] = sol
    return I


def formRightSide(I, maskPerimeter):
    height, width = I.shape
    perimeterValues = np.zeros((height, width))
    assert perimeterValues.shape == I.shape, 'P shape: {}, I shape: {}'.format(perimeterValues.shape, I.shape)
    perimeterValues[maskPerimeter] = I[maskPerimeter]
    rightSide = np.zeros((height, width))

    rightSide[1:height - 1, 1:width - 1] = (
        perimeterValues[0:height - 2, 1:width - 1] +
        perimeterValues[2:height, 1:width - 1] +
        perimeterValues[1:height - 1, 0:width - 2] +
        perimeterValues[1:height - 1, 2:width])

    rightSide[1:height - 1, 0] = (
        perimeterValues[0:height - 2, 0] + perimeterValues[2:height, 0] +
        perimeterValues[1:height - 1, 1])

    rightSide[1:height - 1, width - 1] = (
        perimeterValues[0:height - 2, width - 1] +
        perimeterValues[2:height, width - 1] +
        perimeterValues[1:height - 1, width - 2])

    rightSide[0, 1:width - 1] = (
        perimeterValues[1, 1:width - 1] + perimeterValues[0, 0:width - 2] +
        perimeterValues[0, 2:width])

    rightSide[height - 1, 1:width - 1] = (
        perimeterValues[height - 2, 1:width - 1] +
        perimeterValues[height - 1, 0:width - 2] +
        perimeterValues[height - 1, 2:width])

    rightSide[0, 0] = perimeterValues[0, 1] + perimeterValues[1, 0]
    rightSide[0, width - 1] = (
        perimeterValues[0, width - 2] + perimeterValues[1, width - 1])
    rightSide[height - 1, 0] = (
        perimeterValues[height - 2, 0] + perimeterValues[height - 1, 1])
    rightSide[height - 1, width - 1] = (perimeterValues[height - 2, width - 1] +
                                        perimeterValues[height - 1, width - 2])
    return rightSide


def computeNumberOfNeighbors(height, width):
    # Initialize
    numNeighbors = np.zeros((height, width))
    # Interior pixels have 4 neighbors
    numNeighbors[1:height - 1, 1:width - 1] = 4
    # Border pixels have 3 neighbors
    numNeighbors[1:height - 1, (0, width - 1)] = 3
    numNeighbors[(0, height - 1), 1:width - 1] = 3
    # Corner pixels have 2 neighbors
    numNeighbors[(0, 0, height - 1, height - 1), (0, width - 1, 0,
                                                  width - 1)] = 2
    return numNeighbors


def padMatrix(grid):
    height, width = grid.shape
    gridPadded = -np.ones((height + 2, width + 2))
    gridPadded[1:height + 1, 1:width + 1] = grid
    gridPadded = gridPadded.astype(grid.dtype)
    return gridPadded
