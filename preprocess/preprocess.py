# -*- coding:UTF-8 -*-
import os
import pickle
import numpy as np
import time

import pandas
from scipy.ndimage.interpolation import zoom
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from skimage.morphology import convex_hull_image
from multiprocessing import Pool
from functools import partial
from step1 import step1_python
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def resample(imgs, spacing, new_spacing, order=2):
    if len(imgs.shape) == 3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode='nearest', order=order)
        return imgs, true_spacing
    elif len(imgs.shape) == 4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:, :, :, i]
            newslice, true_spacing = resample(slice, spacing, new_spacing)
            newimg.append(newslice)
        newimg = np.transpose(np.array(newimg), [1, 2, 3, 0])
        return newimg, true_spacing
    else:
        raise ValueError('wrong shape')


def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord


def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any(transformM != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))  # 猜测是[x,y,z]->[z,y,x] reversed的作用
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing, isflip


def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1 = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1) > 0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2) > 1.5 * np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3, 1)
    dilatedMask = binary_dilation(convex_mask, structure=struct, iterations=10)
    return dilatedMask


def lumTrans(img):
    lungwin = np.array([-1200., 600.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


def savenpy(id, annos, filelist, data_path, prep_folder):
    try:
        start = time.time()
        # id是range(N=888)形成的列表，annos是大的csv，一行表示一个CT数据的标注信息
        # filelist是数据路径，data_path是filelist的路径前缀，两者要拼起来，看step_1_python的调用
        # prep_folder是结果存放的路径
        resolution = np.array([1, 1, 1])
        name = filelist[id]

        im, m1, m2, spacing, origin = step1_python(os.path.join(data_path, name))
        Mask = m1 + m2  # 得到了肺实质掩码
        newshape = np.round(np.array(Mask.shape) * spacing / resolution)
        xx, yy, zz = np.where(Mask)
        box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
        box = box * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
        box = np.floor(box).astype('int')
        margin = 5
        extendbox = np.vstack(
            [np.max([[0, 0, 0], box[:, 0] - margin], 0), np.min([newshape, box[:, 1] + 2 * margin], axis=0).T]).T
        extendbox = extendbox.astype('int')
        np.save(os.path.join(prep_folder, name + '_spacing.npy'), spacing)  # 间距
        np.save(os.path.join(prep_folder, name + '_origin.npy'), origin)  # 原点中心
        np.save(os.path.join(prep_folder, name + '_extendbox.npy'), extendbox)  # 从原图裁剪的extendbox
        print('shape', im.shape, 'spacing', spacing, 'origin', origin)
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1 + dm2
        Mask = m1 + m2
        extramask = dilatedMask ^ Mask  # - => ^
        bone_thresh = 210
        pad_value = 170
        im[np.isnan(im)] = -2000
        sliceim = lumTrans(im)
        sliceim = sliceim * dilatedMask + pad_value * (1 - dilatedMask).astype('uint8')
        bones = sliceim * extramask > bone_thresh
        sliceim[bones] = pad_value
        sliceim1, _ = resample(sliceim, spacing, resolution, order=1)
        sliceim2 = sliceim1[extendbox[0, 0]:extendbox[0, 1],
                   extendbox[1, 0]:extendbox[1, 1],
                   extendbox[2, 0]:extendbox[2, 1]]
        sliceim = sliceim2[np.newaxis, ...]
        np.save(os.path.join(prep_folder, name + '_clean.npy'), sliceim)  # 经过处理的图片

        this_nodule = annos[annos[:, 4] == name]
        label = []
        # origin是z,y,x
        for nod in this_nodule:
            # 如果标签文件给的是世界坐标，则用这一行
            # pos = worldToVoxelCoord(nod[:3][::-1], origin=origin, spacing=spacing)
            # label.append(np.concatenate([pos, [nod[3] / spacing[1]]]))
            # 如果标签文件给的是像素坐标，做一下xyz转换到zyx即可
            pos = nod[:3][::-1]
            label.append(np.concatenate([pos, [nod[3]]]))
        if len(label) == 0:
            label2 = np.array([[0, 0, 0, 0]])
        else:
            label2 = np.copy(label).T
            label2[:3] = label2[:3] * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
            label2[3] = label2[3] * spacing[1] / resolution[1]
            label2[:3] = label2[:3] - np.expand_dims(extendbox[:, 0], 1)
            label2 = label2[:4].T
        np.save(os.path.join(prep_folder, name + '_label.npy'), label2)  # 经过处理的label标签

        end = time.time()
        print(name, end="  ")
        print(end - start, "ms")
    except Exception as e:
        print(e.args)
        print(filelist[id])


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # 处理后的结果存放路径
    prep_folder = "/media/data1/LC015preprocess/test_set/DSBmask"
    # 数据存放位置，文件夹下是每一个图像的文件夹，如LC015001/、LC01501002/
    data_path = "/media/data1/LC015-feichuang"

    if not os.path.exists(prep_folder):
        os.mkdir(prep_folder)
    # 标签文件
    alllabelfiles = "/media/data1/LC015share/coord.xls"
    tmp = []

    content = np.array(pandas.read_excel(alllabelfiles))
    content = content[content[:, 0] != np.nan]
    tmp.append(content[:, 1:6])
    alllabel = np.concatenate(tmp, 0)
    filelist = list(alllabel[:, -1])
    # filelist是所有的CT数据对应的路径，这里应该是将所有的数据放到同一个文件夹下
    print('starting preprocessing')

    filelist = sorted(filelist)
    pool = Pool(8)
    partial_savenpy = partial(savenpy, annos=alllabel, filelist=filelist, data_path=data_path,
                              prep_folder=prep_folder)

    N = len(filelist)
    _ = pool.map(partial_savenpy, range(N))
    pool.close()
    pool.join()

