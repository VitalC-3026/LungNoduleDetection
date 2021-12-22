# -*- coding:UTF-8 -*-
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import SimpleITK as sitk

from skimage import measure, morphology


def load_scan(path):
    next_path = os.path.join(path, os.listdir(path)[0])
    if os.path.isdir(next_path):
        doc = os.listdir(path)
        while doc[0].find("dcm") == -1:  # and os.path.isdir(doc[0])
            path = os.path.join(path, doc[0])
            doc = os.listdir(path)
    slices = [pydicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    # 有可能是论文用的数据集的问题，可能会出现同一个切片有多张图片的结果，这里是去重
    # LIDC的图片不存在这个问题，所以直接读取file然后排序后返回即可
    if slices[0].ImagePositionPatient[2] == slices[1].ImagePositionPatient[2]:
        sec_num = 2
        while slices[0].ImagePositionPatient[2] == slices[sec_num].ImagePositionPatient[2]:
            sec_num = sec_num + 1
        slice_num = int(len(slices) / sec_num)
        slices.sort(key=lambda x: float(x.InstanceNumber))
        slices = slices[0:slice_num]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(slices):
    # 进行矩阵的堆叠，将2D的图像变成3D的图像，image的维度是[切片数, 512, 512]
    # TODO: 出现的问题是有的dicom无法提取pixel_array，与期待的大小有出入。这是LIDC数据集中有些dcm大小不对的原因=>重新下载数据集或者跳过这套CT数据
    origin = [float(slices[0].ImagePositionPatient[2]), float(slices[0].ImagePositionPatient[1]),
              float(slices[0].ImagePositionPatient[0])]  # z,y,x
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)
    # 第二个返回值的作用: 需要进行归一化
    spacing = [slices[0].SliceThickness, slices[0].PixelSpacing[0], slices[0].PixelSpacing[1]]  # z,y,x
    return np.array(image, dtype=np.int16), np.array(spacing, dtype=np.float32), np.array(origin, dtype=np.float32)


def binarize_per_slice(image, spacing, intensity_th=-600, sigma=2, area_th=30, eccen_th=0.99, bg_patch_size=10):
    # 提取肺实质，将图片两极化，非肺实质是一个大的连通区域，其他的是小的连通区域，将这些其他的小连通区域进行提取之后并以与大连通区域对立的像素点进行标识
    # 具体的intensity_th, sigma, area_th, eccen_th都需要根据经验值得到
    # TODO: nan_mask的依据
    bw = np.zeros(image.shape, dtype=bool)

    # prepare a mask, with all corner values set to nan
    if image.shape[1] == image.shape[2]:
        image_size = image.shape[1]
        # 如果image_size为512，则得到[-255.5, -254.5, ..., 254.5, 255.5]
        grid_axis = np.linspace(-image_size / 2 + 0.5, image_size / 2 - 0.5, image_size)
        # 得到两个矩阵，x是以[-255.5, -254.5, ..., 254.5, 255.5]为行，有512行；y是以[-255.5, -254.5, ..., 254.5, 255.5]为列，有512列
        x, y = np.meshgrid(grid_axis, grid_axis)
    else:
        image_size = max(image.shape[1], image.shape[2])
        # 如果image_size为512，则得到[-255.5, -254.5, ..., 254.5, 255.5]
        grid_axis_x = np.linspace(-image.shape[1] / 2 + 0.5, image.shape[1] / 2 - 0.5, image.shape[1])
        grid_axis_y = np.linspace(-image.shape[2] / 2 + 0.5, image.shape[2] / 2 - 0.5, image.shape[2])
        # 得到两个矩阵，x是以[-255.5, -254.5, ..., 254.5, 255.5]为行，有512行；y是以[-255.5, -254.5, ..., 254.5, 255.5]为列，有512列
        x, y = np.meshgrid(grid_axis_x, grid_axis_y)
    d = (x ** 2 + y ** 2) ** 0.5
    nan_mask = (d < image_size / 2).astype(float)
    nan_mask[nan_mask == 0] = np.nan
    # image.shape[0]是切片数
    for i in range(image.shape[0]):
        # Check if corner pixels are identical, if so the slice  before Gaussian filtering
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            # 这里使用了高斯滤波，是否应该改成如下方式？ 可以避免RuntimeWarning
            # current_bw = scipy.ndimage.filters.gaussian_filter(np.multiply(image[i].astype('float32'), nan_mask), sigma, truncate=2.0)
            # current_bw = current_bw < intensity_th
            # 这里会有问题，因为nan_mask设置的都是nan，所以得到的结果也是存在nan的矩阵，和任何数进行比较的时候会有RuntimeError
            current_bw = scipy.ndimage.filters.gaussian_filter(np.multiply(image[i].astype('float32'), nan_mask), sigma,
                                                               truncate=2.0) < intensity_th
        else:
            current_bw = scipy.ndimage.filters.gaussian_filter(image[i].astype('float32'), sigma,
                                                               truncate=2.0) < intensity_th

        # select proper components
        label = measure.label(current_bw)
        properties = measure.regionprops(label)
        valid_label = set()
        # spacing[1]和spacing[2]分别是x,y的spacing值
        for prop in properties:
            # eccentricity离心率
            # prop.area * spacing[1] * spacing[2] > area_th表示连通区域有一定面积的时候才将记录到valid_label中
            # TODO: 为什么需要乘上x, y的spacing, spacing的作用?
            if prop.area * spacing[1] * spacing[2] > area_th and prop.eccentricity < eccen_th:
                valid_label.add(prop.label)
        current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
        bw[i] = current_bw

    return bw


def all_slice_analysis(bw, spacing, cut_num=0, vol_limit=[0.68, 8.2], area_th=6e3, dist_th=62):
    # in some cases, several top layers need to be removed first
    if cut_num > 0:
        # 去除了几张没有肺实质的切片，也就是理解成往更中心取了切片
        bw0 = np.copy(bw)
        bw[-cut_num:] = False
    # 得到的是一个和bw大小一样的label, 不同的连通区域会被标注成不同的数字，从0开始
    # connectivity表示的是4点联通
    label = measure.label(bw, connectivity=1)
    # remove components access to corners
    mid = int(label.shape[2] / 2)  # 256
    # 获取的是第一张切片和倒数第cut_num张切片的六个点所在的连通区域，把它们视作边缘，这些连通区域视为0（最大的那个外围连通区域）
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1 - cut_num, 0, 0], label[-1 - cut_num, 0, -1], label[-1 - cut_num, -1, 0],
                    label[-1 - cut_num, -1, -1], \
                    label[0, 0, mid], label[0, -1, mid], label[-1 - cut_num, 0, mid], label[-1 - cut_num, -1, mid]])
    for l in bg_label:
        label[label == l] = 0

    # select components based on volume, 体积太小的话认为这个连通区域不是我们希望提取的肺实质
    properties = measure.regionprops(label)
    for prop in properties:
        if prop.area * spacing.prod() < vol_limit[0] * 1e6 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
            label[label == prop.label] = 0
        print(prop.label, prop.area * spacing.prod())

    # prepare a distance map for further analysis
    x_axis = np.linspace(-label.shape[1] / 2 + 0.5, label.shape[1] / 2 - 0.5, label.shape[1]) * spacing[1]
    y_axis = np.linspace(-label.shape[2] / 2 + 0.5, label.shape[2] / 2 - 0.5, label.shape[2]) * spacing[2]
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x ** 2 + y ** 2) ** 0.5
    # d是所有像素点到中心点的距离所组成的矩阵
    vols = measure.regionprops(label)
    valid_label = set()
    # select components based on their area and distance to center axis on all slices
    for vol in vols:
        single_vol = label == vol.label
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            # 连通区域的面积大小，像素点之间的spacing是x,y(PixelSpacing)
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            # 每一张切片距离中心点的最短的距离
            min_distance[i] = np.min(single_vol[i] * d + (1 - single_vol[i]) * np.max(d))
        # 判断当前每一张切片的面积是否足够大，同时连通区域距离中心点的距离平均值会不会过大
        if np.average([min_distance[i] for i in range(label.shape[0]) if slice_area[i] > area_th]) < dist_th:
            valid_label.add(vol.label)

    bw = np.in1d(label, list(valid_label)).reshape(label.shape)

    # fill back the parts removed earlier
    if cut_num > 0:
        # bw1 is bw with removed slices, bw2 is a dilated version of bw, part of their intersection is returned as final mask
        bw1 = np.copy(bw)
        # 在这里会补回来之前舍弃了的切片
        bw1[-cut_num:] = bw0[-cut_num:]
        bw2 = np.copy(bw)
        # 4个方向的dilation
        bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num)
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})
        valid_l3 = set()
        # label3是经过扩充的mask和补回切片的mask取交之后的连通区mask，
        # 如果缩小的mask的连通区在label3上也有同样的连通区，那么这个连通区就被选择了
        for l in l_list:
            indices = np.nonzero(label == l)
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)
    # len(valid_label)这个是用来标识有没有连通区出现，和valid_label3谁大谁小没关系，只要不为0就可以让调用该函数的所在循环停止
    return bw, len(valid_label)


def fill_hole(bw):
    # 希望把肺实质可以连起来，不要独立成很多小的连通区域（在3d层面上）
    # fill 3d holes
    label = measure.label(~bw)
    # idendify corner components
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1, 0, 0], label[-1, 0, -1], label[-1, -1, 0], label[-1, -1, -1]])
    bw = ~np.in1d(label, list(bg_label)).reshape(label.shape)

    return bw


def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):
    def extract_main(bw, cover=0.95):
        # 根据得到的bw，一张一张切片去框定bounding box，然后最后找到占最大面积的bounding box
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            properties.sort(key=lambda x: x.area, reverse=True)
            area = [prop.area for prop in properties]
            count = 0
            sum = 0
            while sum < np.sum(area) * cover:
                sum = sum + area[count]
                count = count + 1
            filter = np.zeros(current_slice.shape, dtype=bool)
            for j in range(count):
                bb = properties[j].bbox
                filter[bb[0]:bb[2], bb[1]:bb[3]] = filter[bb[0]:bb[2], bb[1]:bb[3]] | properties[j].convex_image
            bw[i] = bw[i] & filter

        label = measure.label(bw)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        bw = label == properties[0].label

        return bw

    def fill_2d_hole(bw):
        # 将所有的连通区域都连起来，形成一个最大限度的肺实质区域
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            for prop in properties:
                bb = prop.bbox
                current_slice[bb[0]:bb[2], bb[1]:bb[3]] = current_slice[bb[0]:bb[2], bb[1]:bb[3]] | prop.filled_image
            bw[i] = current_slice

        return bw

    found_flag = False
    iter_count = 0
    bw0 = np.copy(bw)
    while not found_flag and iter_count < max_iter:
        label = measure.label(bw, connectivity=2)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        # 为了防止左肺和右肺差别太大，不断地进行向四周腐蚀，保证两个肺的面积比例不会有太大差距
        if len(properties) > 1 and properties[0].area / properties[1].area < max_ratio:
            found_flag = True
            bw1 = label == properties[0].label
            bw2 = label == properties[1].label
        else:
            bw = scipy.ndimage.binary_erosion(bw)
            iter_count = iter_count + 1

    if found_flag:
        d1 = scipy.ndimage.morphology.distance_transform_edt(bw1 == False, sampling=spacing)
        d2 = scipy.ndimage.morphology.distance_transform_edt(bw2 == False, sampling=spacing)
        # 进一步切分出分出左右肺
        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)

        bw1 = extract_main(bw1)
        bw2 = extract_main(bw2)

    else:
        bw1 = bw0
        bw2 = np.zeros(bw.shape).astype('bool')

    bw1 = fill_2d_hole(bw1)
    bw2 = fill_2d_hole(bw2)
    bw = bw1 | bw2

    return bw1, bw2, bw


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

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing, isflip


def step1_python(case_path):
    case = load_scan(case_path)
    case_pixels, spacing, origin = get_pixels_hu(case)
    case_pixels = np.array(case_pixels)
    bw = binarize_per_slice(case_pixels, spacing)

    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    while flag == 0 and cut_num < bw.shape[0]:  # bw => bw0
        bw = np.copy(bw0)
        bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.6, 7.5], area_th=50)
        cut_num = cut_num + cut_step

    bw = fill_hole(bw)
    bw1, bw2, bw = two_lung_only(bw, spacing)
    return case_pixels, bw1, bw2, spacing, origin

