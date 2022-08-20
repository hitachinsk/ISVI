from distutils.command.config import config
from importlib import import_module
from multiprocessing.sharedctypes import Value
import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', 'IGFC')))
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', 'ASFN')))

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import cv2
import glob
import yaml
import copy
import numpy as np
import torch
import imageio
from PIL import Image
import scipy.ndimage
from skimage.feature import canny
import torchvision.transforms.functional as F
import torch.nn.functional as F2
from RAFT import utils
from RAFT import RAFT
import imageio

import utils.region_fill as rf
# from utils.Poisson_blend import Poisson_blend
from utils.Poisson_blend_img import Poisson_blend_img, getUnfilledMask
from get_flowNN_gradient import get_flowNN_gradient
from spatial_inpaint import spatial_inpaint_single
from frame_inpaint import DeepFillv1
from torchvision.transforms import ToTensor
import cvbase


def diffusion(flows, masks):
    flows_filled = []
    for i in range(flows.shape[0]):
        flow, mask = flows[i], masks[i]
        flow_filled = np.zeros(flow.shape)
        flow_filled[:, :, 0] = rf.regionfill(flow[:, :, 0], mask[:, :, 0])
        flow_filled[:, :, 1] = rf.regionfill(flow[:, :, 1], mask[:, :, 0])
        flows_filled.append(flow_filled)
    return flows_filled


def np2tensor(array, near='c'):
    if isinstance(array, list):
        array = np.stack(array, axis=0)  # [t, h, w, c]
    if near == 'c':
        array = torch.from_numpy(np.transpose(array, (3, 0, 1, 2))).unsqueeze(0).float()
    elif near == 't':
        array = torch.from_numpy(np.transpose(array, (0, 3, 1, 2))).unsqueeze(0).float()
    else:
        raise ValueError(f'Unknown near type: {near}')
    return array


def tensor2np(array):
    array = torch.stack(array, dim=-1).squeeze(0).permute(1, 2, 0, 3).cpu().numpy()
    return array


def gradient_mask(mask):

    gradient_mask = np.logical_or.reduce((mask,
                                          np.concatenate((mask[1:, :], np.zeros((1, mask.shape[1]), dtype=np.bool)),
                                                         axis=0),
                                          np.concatenate((mask[:, 1:], np.zeros((mask.shape[0], 1), dtype=np.bool)),
                                                         axis=1)))

    return gradient_mask


def indicesGen(pivot, interval, frames, t):
    singleSide = frames // 2
    results, relativeResults = [], []
    for i in range(-singleSide, singleSide + 1):
        index = pivot + interval * i
        if index < 0:
            index = abs(index)
        if index > t - 1:
            index = 2 * (t - 1) - index
        results.append(index)
        relativeResults.append(pivot - index)
    return results, relativeResults


def save_flows(output, videoFlowF, videoFlowB):
    create_dir(os.path.join(output, 'completed_flow', 'forward_flo'))
    create_dir(os.path.join(output, 'completed_flow', 'backward_flo'))
    create_dir(os.path.join(output, 'completed_flow', 'forward_png'))
    create_dir(os.path.join(output, 'completed_flow', 'backward_png'))
    N = videoFlowF.shape[-1]
    for i in range(N):
        forward_flow = videoFlowF[..., i]
        backward_flow = videoFlowB[..., i]
        forward_flow_vis = cvbase.flow2rgb(forward_flow)
        backward_flow_vis = cvbase.flow2rgb(backward_flow)
        cvbase.write_flow(forward_flow, os.path.join(output, 'completed_flow', 'forward_flo', '{:05d}.flo'.format(i)))
        cvbase.write_flow(backward_flow, os.path.join(output, 'completed_flow', 'backward_flo', '{:05d}.flo'.format(i)))
        imageio.imwrite(os.path.join(output, 'completed_flow', 'forward_png', '{:05d}.png'.format(i)), forward_flow_vis)
        imageio.imwrite(os.path.join(output, 'completed_flow', 'backward_png', '{:05d}.png'.format(i)), backward_flow_vis)


def create_dir(dir):
    """Creates a directory if not exist.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def initialize_RAFT(args, device):
    """Initializes the RAFT model.
    """
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(device)
    model.eval()

    return model


def initialize_models(args, device, model_type='IGFC'):
    assert model_type in ['IGFC', 'ASFN']
    assert len(os.listdir(args.igfc_ckpts)) == 2 and len(os.listdir(args.asfn_ckpts)) == 2
    if model_type == 'IGFC':
        checkpoint, config_file = glob.glob(os.path.join(args.igfc_ckpts, '*.tar'))[0], \
                              glob.glob(os.path.join(args.igfc_ckpts, '*.yaml'))[0]
    elif model_type == 'ASFN':
        checkpoint, config_file = glob.glob(os.path.join(args.asfn_ckpts, '*.tar'))[0], \
                              glob.glob(os.path.join(args.asfn_ckpts, '*.yaml'))[0]
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    with open(config_file, 'r') as f:
        configs = yaml.load(f)
    model = configs['model']
    pkg = import_module('{}.models.{}'.format(model_type, model))
    model = pkg.Model(configs)
    state = torch.load(checkpoint, map_location=lambda storage, loc: storage.cuda(device))
    model.load_state_dict(state['model_state_dict'])
    model = model.to(device)
    return model, configs


def calculate_flow(args, model, video, mode):
    """Calculates optical flow.
    """
    if mode not in ['forward', 'backward']:
        raise NotImplementedError

    imgH, imgW = args.imgH, args.imgW
    Flow = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)

    if args.vis_flows:
        create_dir(os.path.join(args.outroot, 'flow', mode + '_flo'))
        create_dir(os.path.join(args.outroot, 'flow', mode + '_png'))

    with torch.no_grad():
        for i in range(video.shape[0] - 1):
            print("Calculating {0} flow {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')
            if mode == 'forward':
                # Flow i -> i + 1
                image1 = video[i, None]
                image2 = video[i + 1, None]
            elif mode == 'backward':
                # Flow i + 1 -> i
                image1 = video[i + 1, None]
                image2 = video[i, None]
            else:
                raise NotImplementedError

            _, flow = model(image1, image2, iters=20, test_mode=True)
            flow = flow[0].permute(1, 2, 0).cpu().numpy()
            h, w = flow.shape[:2]
            if h != imgH or w != imgW:
                # resize optical flows
                flow = cv2.resize(flow, (imgW, imgH), cv2.INTER_LINEAR)
                flow[:, :, 0] *= (float(imgW) / float(w))
                flow[:, :, 1] *= (float(imgH) / float(h))

            Flow = np.concatenate((Flow, flow[..., None]), axis=-1)
            
            if args.vis_flows:
                # Flow visualization.
                flow_img = utils.flow_viz.flow_to_image(flow)
                flow_img = Image.fromarray(flow_img)

                # Saves the flow and flow_img.
                flow_img.save(os.path.join(args.outroot, 'flow', mode + '_png', '%05d.png' % i))
                utils.frame_utils.writeFlow(os.path.join(args.outroot, 'flow', mode + '_flo', '%05d.flo' % i), flow)

    return Flow


def extrapolation(args, video_ori, corrFlowF_ori, corrFlowB_ori):
    """Prepares the data for video extrapolation.
    """
    imgH, imgW, _, nFrame = video_ori.shape

    # Defines new FOV.
    imgH_extr = int(args.H_scale * imgH)
    imgW_extr = int(args.W_scale * imgW)
    H_start = int((imgH_extr - imgH) / 2)
    W_start = int((imgW_extr - imgW) / 2)

    # Generates the mask for missing region.
    flow_mask = np.ones(((imgH_extr, imgW_extr)), dtype=np.bool)
    flow_mask[H_start: H_start + imgH, W_start: W_start + imgW] = 0

    mask_dilated = gradient_mask(flow_mask)

    # Extrapolates the FOV for video.
    video = np.zeros(((imgH_extr, imgW_extr, 3, nFrame)), dtype=np.float32)
    video[H_start: H_start + imgH, W_start: W_start + imgW, :, :] = video_ori

    for i in range(nFrame):
        print("Preparing frame {0}".format(i), '\r', end='')
        video[:, :, :, i] = cv2.inpaint((video[:, :, :, i] * 255).astype(np.uint8), flow_mask.astype(np.uint8), 3,
                                        cv2.INPAINT_TELEA).astype(np.float32) / 255.

    # Extrapolates the FOV for flow.
    corrFlowF = np.zeros(((imgH_extr, imgW_extr, 2, nFrame - 1)), dtype=np.float32)
    corrFlowB = np.zeros(((imgH_extr, imgW_extr, 2, nFrame - 1)), dtype=np.float32)
    corrFlowF[H_start: H_start + imgH, W_start: W_start + imgW, :] = corrFlowF_ori
    corrFlowB[H_start: H_start + imgH, W_start: W_start + imgW, :] = corrFlowB_ori

    return video, corrFlowF, corrFlowB, flow_mask, mask_dilated, (W_start, H_start), (W_start + imgW, H_start + imgH)


def complete_flow(config, flow_model, flows, flow_masks, mode, device):
    if mode not in ['forward', 'backward']:
        raise NotImplementedError(f'Error flow mode {mode}')
    flow_masks = np.moveaxis(flow_masks, -1, 0)  # [N, H, W]
    flows = np.moveaxis(flows, -1, 0)  # [N, H, W, 2]
    if len(flow_masks.shape) == 3:
        flow_masks = flow_masks[:, :, :, np.newaxis]  # [N, H, W, 1]
    if mode == 'forward':
        flow_masks = flow_masks[0:-1]
    else:
        flow_masks = flow_masks[1:]
    try:
        frames, interval = config['num_frames'], config['interval']
    except:
        frames, interval = 3, 1

    diffused_flows = diffusion(flows, flow_masks)

    if mode == 'backward':
        flows = flows[::-1, ...].copy()
        diffused_flows.reverse()
        flow_masks = flow_masks[::-1, ...].copy()

    flows = np2tensor(flows)
    flow_masks = np2tensor(flow_masks)
    diffused_flows = np2tensor(diffused_flows)

    flows = flows.to(device)
    flow_masks = flow_masks.to(device)
    diffused_flows = diffused_flows.to(device)

    t = diffused_flows.shape[2]
    filled_flows = [None] * t
    pivot = frames // 2
    for i in range(t):
        print(i)
        indices, relativeIndices = indicesGen(i, interval, frames, t)
        cand_flows = flows[:, :, indices]
        cand_masks = flow_masks[:, :, indices]
        inputs = diffused_flows[:, :, indices]
        pivot_mask = cand_masks[:, :, pivot]
        pivot_flow = cand_flows[:, :, pivot]
        with torch.no_grad():
            outputs = flow_model(inputs, cand_masks, None, relativeIndices)
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            outputs = outputs[0]
        comp = outputs * pivot_mask + pivot_flow * (1 - pivot_mask)
        filled_flows[i] = comp
    if mode == 'backward':
        filled_flows.reverse()
    return filled_flows


def asfn_processing(model, gradx, grady, mask, device):
    assert gradx.shape[-1] == grady.shape[-1] and gradx.shape[-1] == mask.shape[-1]
    video_len = mask.shape[-1]
    refined_gradx, refined_grady = [], []
    for i in range(video_len):
        gx, gy, m = gradx[..., i], grady[..., i], mask[..., i]
        if len(m.shape) != 3:
            m = m[:, :, np.newaxis]
        gx_t = torch.from_numpy(np.transpose(gx, (2, 0, 1)).copy()).float().unsqueeze(0)
        gy_t = torch.from_numpy(np.transpose(gy, (2, 0, 1)).copy()).float().unsqueeze(0)
        m_t = torch.from_numpy(np.transpose(m, (2, 0, 1)).copy()).float().unsqueeze(0)
        gx_t = gx_t.to(device)
        gy_t = gy_t.to(device)
        m_t = m_t.to(device)
        with torch.no_grad():
            rf_gx = model(gx_t, m_t)
            rf_gy = model(gy_t, m_t)
        rf_gx = rf_gx[0].permute(1, 2, 0).contiguous().cpu().numpy()
        rf_gy = rf_gy[0].permute(1, 2, 0).contiguous().cpu().numpy()
        rf_gx = rf_gx * m + gx * (1 - m)
        rf_gy = rf_gy * m + gy * (1 - m)
        refined_gradx.append(rf_gx)
        refined_grady.append(rf_gy)
    refined_gradx = np.stack(refined_gradx, axis=-1)
    refined_grady = np.stack(refined_grady, axis=-1)
    return refined_gradx, refined_grady


def read_flow(flow_dir, video):
    nFrame, _, imgH, imgW = video.shape
    Flow = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)
    flows = sorted(glob.glob(os.path.join(flow_dir, '*.flo')))
    for flow in flows:
        flow_data = cvbase.read_flow(flow)
        h, w = flow_data.shape[:2]
        flow_data = cv2.resize(flow_data, (imgW, imgH), cv2.INTER_LINEAR)
        flow_data[:, :, 0] *= (float(imgW) / float(w))
        flow_data[:, :, 1] *= (float(imgH) / float(h))
        Flow = np.concatenate((Flow, flow_data[..., None]), axis=-1)
    return Flow


def crop_blending(frame, gradx, grady, mask):
    mask_dilate = scipy.ndimage.binary_dilation(mask, iterations=2)
    if np.sum(mask[:, 0]) + np.sum(mask[:, -1]) + np.sum(mask[0, :]) + np.sum(mask[-1, :]) > 0:
        return Poisson_blend_img(frame, gradx, grady, mask)[0]
    mask_dilate = mask_dilate.astype(np.uint8)
    ret, labels = cv2.connectedComponents(mask_dilate)
    for i in range(1, ret):
        position = np.where(labels == i)
        hmax = position[0].max()
        hmin = position[0].min()
        wmax = position[1].max()
        wmin = position[1].min()
        frame_cropped = frame[hmin:hmax, wmin:wmax, :]
        mask_cropped = mask[hmin:hmax, wmin:wmax]
        gradx_cropped = gradx[hmin:hmax, wmin:wmax, :]
        grady_cropped = grady[hmin:hmax, wmin:wmax, :]
        result = Poisson_blend_img(frame_cropped, gradx_cropped, grady_cropped, mask_cropped)[0]
        frame[hmin:hmax, wmin:wmax, :] = result
    return frame


def video_inpainting(args):
    device = torch.device('cuda:{}'.format(args.gpu))

    if args.opt is not None:
        with open(args.opt, 'r') as f:
            opts = yaml.load(f)

    for k in opts.keys():
        if k in args:
            setattr(args, k, opts[k])

    # Flow model.
    RAFT_model = initialize_RAFT(args, device)
    # IGFC
    IGFC_model, IGFC_config = initialize_models(args, device, 'IGFC')
    # ASFN
    ASFN_model, ASFN_config = initialize_models(args, device, 'ASFN')

    # Loads frames.
    filename_list = glob.glob(os.path.join(args.path, '*.png')) + \
                    glob.glob(os.path.join(args.path, '*.jpg'))

    # Obtains imgH, imgW and nFrame.
    imgH, imgW = args.imgH, args.imgW
    nFrame = len(filename_list)

    if imgH < 350:
        flowH, flowW = imgH * 2, imgW * 2
    else:
        flowH, flowW = imgH, imgW

    # Load video.
    video, video_flow = [], []
    for filename in sorted(filename_list):
        frame = torch.from_numpy(np.array(Image.open(filename)).astype(np.uint8)).permute(2, 0, 1).float().unsqueeze(0)
        frame = F2.upsample(frame, size=(imgH, imgW), mode='bilinear', align_corners=False)
        frame_flow = F2.upsample(frame, size=(flowH, flowW), mode='bilinear', align_corners=False)
        video.append(frame)
        video_flow.append(frame_flow)

    video = torch.cat(video, dim=0)
    video_flow = torch.cat(video_flow, dim=0)
    video = video.to('cuda')
    video_flow = video_flow.to(device)

    # Calcutes the corrupted flow.
    forward_flows = calculate_flow(args, RAFT_model, video_flow, 'forward')
    backward_flows = calculate_flow(args, RAFT_model, video_flow, 'backward')

    # Makes sure video is in BGR (opencv) format.
    video = video.permute(2, 3, 1, 0).cpu().numpy()[:, :, ::-1, :] / 255.

    if args.mode == 'video_extrapolation':

        # Creates video and flow where the extrapolated region are missing.
        video, corrFlowF, corrFlowB, flow_mask, mask_dilated, start_point, end_point = extrapolation(args, video,
                                                                                                     corrFlowF,
                                                                                                     corrFlowB)
        imgH, imgW = video.shape[:2]

        # mask indicating the missing region in the video.
        mask = np.tile(flow_mask[..., None], (1, 1, nFrame))
        flow_mask = np.tile(flow_mask[..., None], (1, 1, nFrame))
        mask_dilated = np.tile(mask_dilated[..., None], (1, 1, nFrame))

    else:
        # Loads masks.
        filename_list = glob.glob(os.path.join(args.path_mask, '*.png')) + \
                        glob.glob(os.path.join(args.path_mask, '*.jpg'))

        mask = []
        mask_dilated = []
        flow_mask = []
        for filename in sorted(filename_list):
            mask_img = np.array(Image.open(filename).convert('L'))
            mask_img = cv2.resize(mask_img, dsize=(imgW, imgH), interpolation=cv2.INTER_NEAREST)

            if args.flow_mask_dilates > 0:
                flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=args.flow_mask_dilates)
            else:
                flow_mask_img = mask_img
            flow_mask.append(flow_mask_img)

            if args.frame_dilates > 0:
                mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=args.frame_dilates)
            mask_img = scipy.ndimage.binary_fill_holes(mask_img).astype(np.bool)
            mask.append(mask_img)
            mask_dilated.append(gradient_mask(mask_img))

        # mask indicating the missing region in the video.
        mask = np.stack(mask, -1).astype(np.bool)  # [H, W, C, N]
        mask_dilated = np.stack(mask_dilated, -1).astype(np.bool)
        flow_mask = np.stack(flow_mask, -1).astype(np.bool)

    # Completes the flow.
    videoFlowF = complete_flow(IGFC_config, IGFC_model, forward_flows, flow_mask, 'forward', device)
    videoFlowB = complete_flow(IGFC_config, IGFC_model, backward_flows, flow_mask, 'backward', device)
    videoFlowF = tensor2np(videoFlowF)
    videoFlowB = tensor2np(videoFlowB)
    print('\nFinish flow completion.')

    if args.vis_completed_flows:
        save_flows(args.outroot, videoFlowF, videoFlowB)

    # Prepare gradients
    gradient_x = np.empty(((imgH, imgW, 3, 0)), dtype=np.float32)
    gradient_y = np.empty(((imgH, imgW, 3, 0)), dtype=np.float32)

    for indFrame in range(nFrame):
        img = video[:, :, :, indFrame]
        img[mask[:, :, indFrame], :] = 0
        img = cv2.inpaint((img * 255).astype(np.uint8), mask[:, :, indFrame].astype(np.uint8), 3,
                          cv2.INPAINT_TELEA).astype(np.float32) / 255.

        gradient_x_ = np.concatenate((np.diff(img, axis=1), np.zeros((imgH, 1, 3), dtype=np.float32)),
                                     axis=1)
        gradient_y_ = np.concatenate((np.diff(img, axis=0), np.zeros((1, imgW, 3), dtype=np.float32)), axis=0)
        gradient_x = np.concatenate((gradient_x, gradient_x_.reshape(imgH, imgW, 3, 1)), axis=-1)
        gradient_y = np.concatenate((gradient_y, gradient_y_.reshape(imgH, imgW, 3, 1)), axis=-1)

        gradient_x[mask_dilated[:, :, indFrame], :, indFrame] = 0
        gradient_y[mask_dilated[:, :, indFrame], :, indFrame] = 0
        
    gradient_x_filled = gradient_x  
    gradient_y_filled = gradient_y
    mask_gradient = mask_dilated
    video_comp = video

    # Image inpainting model.
    deepfill = DeepFillv1(pretrained_model=args.deepfill_model, image_shape=[imgH, imgW], device=device)

    gradx_dir = os.path.join(args.outroot, 'x')
    grady_dir = os.path.join(args.outroot, 'y')
    if not os.path.exists(gradx_dir):
        os.makedirs(gradx_dir)
    if not os.path.exists(grady_dir):
        os.makedirs(grady_dir)

    mask_bkp = copy.deepcopy(mask)

    while (np.sum(mask) > 0):

        gradient_x_filled, gradient_y_filled, mask_gradient = \
            get_flowNN_gradient(args,
                                gradient_x_filled,
                                gradient_y_filled,
                                mask,
                                mask_gradient,
                                videoFlowF,
                                videoFlowB,
                                None,
                                None)

        # if there exist holes in mask, Poisson blending will fail. So I did this trick. I sacrifice some value. Another solution is to modify Poisson blending.
        for indFrame in range(nFrame):
            mask_gradient[:, :, indFrame] = scipy.ndimage.binary_fill_holes(mask_gradient[:, :, indFrame]).astype(
                np.bool)

        if np.all(mask_gradient == False):
        	break
                
        # get the unfilled masks for all of the frames in the video
        for indFrame in range(nFrame):
            UnfilledMask = getUnfilledMask(mask[:, :, indFrame], mask_gradient[:, :, indFrame])
            mask[:, :, indFrame] = UnfilledMask

        keyFrameInd = np.argmax(np.sum(np.sum(mask, axis=0), axis=0))
        try:
            keyFrameBlend, keyFrameMask = Poisson_blend_img(video_comp[:, :, :, keyFrameInd],
                                                        gradient_x_filled[:, 0: imgW - 1, :, keyFrameInd],
                                                        gradient_y_filled[0: imgH - 1, :, :, keyFrameInd],
                                                        mask_bkp[:, :, keyFrameInd], mask_gradient[:, :, keyFrameInd])
        except:
            keyFrameBlend, keyFrameMask = video_comp[:, :, :, keyFrameInd], mask[:, :, keyFrameInd]
        keyFrameBlend = np.clip(keyFrameBlend, 0, 1.0)

        # inpaint the key frame only
        keyFrame = spatial_inpaint_single(deepfill, keyFrameMask, keyFrameBlend)
        mask[:, :, keyFrameInd] = False

        # Recalculate the gradient_x/y_filled and mask_fradient for the key frame
        for indFrame in range(nFrame):
            mask_gradient[:, :, indFrame] = gradient_mask(mask[:, :, indFrame])
        gradient_x_filled[:, :, :, keyFrameInd] = np.concatenate(
            (np.diff(keyFrame, axis=1), np.zeros((imgH, 1, 3), dtype=np.float32)), axis=1
        )
        gradient_y_filled[:, :, :, keyFrameInd] = np.concatenate(
            (np.diff(keyFrame, axis=0), np.zeros((1, imgW, 3), dtype=np.float32)), axis=0
        )

    # ASFN processing
    gradient_x_refined, gradient_y_refined = asfn_processing(ASFN_model, gradient_x_filled, gradient_y_filled, mask_bkp, device)
    frameBlends = []
    for indFrame in range(nFrame):
        frameBlend = crop_blending(video_comp[:, :, :, indFrame], gradient_x_refined[:, 0: imgW - 1, :, indFrame],
                                                                 gradient_y_refined[0: imgH - 1, :, :, indFrame],
                                                                 mask_bkp[:, :, indFrame])
        frameBlends.append(frameBlend)

    create_dir(os.path.join(args.outroot, 'frame_vis'))
    frameBlends_vis = copy.deepcopy(frameBlends)
    for indFrame in range(len(frameBlends)):
        cv2.imwrite(os.path.join(args.outroot, 'frame_vis', '%05d.png' % indFrame),
                    frameBlends_vis[indFrame] * 255.)
    cmd = 'ffmpeg -i {}/%05d.png -c:v libx264 -pix_fmt yuv420p {}'.format(os.path.join(args.outroot, 'frame_vis'), os.path.join(args.outroot, 'result.mp4'))
    os.system(cmd)
    print(f'Done, please check your result in {args.outroot}')


def main(args):
    assert args.mode in ('object_removal', 'video_extrapolation'), (
                                                                       "Accepted modes: 'object_removal', 'video_extrapolation', but input is %s"
                                                                   ) % args.mode

    video_inpainting(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', default='configs/object_removal.yaml', help='The config file for inference')
    # video completion
    parser.add_argument('--mode', default='object_removal', help="modes: object_removal / video_extrapolation")
    parser.add_argument('--path', default='../data/frames/bmx-bumps', help="dataset for evaluation")
    parser.add_argument('--path_mask', default='../data/masks/bmx-bumps', help="mask for object removal")
    parser.add_argument('--outroot', default='../data/results/bmx-bumps_2', help="output directory")
    parser.add_argument('--consistencyThres', dest='consistencyThres', default=np.inf, type=float,
                        help='flow consistency error threshold')
    parser.add_argument('--alpha', dest='alpha', default=0.1, type=float)
    parser.add_argument('--Nonlocal', dest='Nonlocal', default=False, type=bool)

    # RAFT
    parser.add_argument('--model', default='../weights/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    # Deepfill
    parser.add_argument('--deepfill_model', default='../weights/imagenet_deepfill.pth', help="restore checkpoint")

    # IGFC
    parser.add_argument('--igfc_ckpts', type=str, default='../weights/IGFC')

    # ASFN
    parser.add_argument('--asfn_ckpts', type=str, default='../weights/ASFN')

    # extrapolation
    parser.add_argument('--H_scale', dest='H_scale', default=2, type=float, help='H extrapolation scale')
    parser.add_argument('--W_scale', dest='W_scale', default=2, type=float, help='W extrapolation scale')

    # Image basic information
    parser.add_argument('--imgH', type=int, default=256)
    parser.add_argument('--imgW', type=int, default=432)
    parser.add_argument('--flow_mask_dilates', type=int, default=8)
    parser.add_argument('--frame_dilates', type=int, default=0)

    parser.add_argument('--gpu', type=int, default=0)

    # Visualization
    parser.add_argument('--vis_flows', action='store_true', help='Visualize the initialzed flows')
    parser.add_argument('--vis_completed_flows', action='store_true', help='Visualize the completed flows')

    args = parser.parse_args()

    main(args)
