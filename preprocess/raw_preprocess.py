import numpy as np
import pandas
import pydicom
import os
import time
from scipy.ndimage.interpolation import zoom
from multiprocessing import Pool
from functools import partial
import warnings

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


def lumTrans(img):
    lungwin = np.array([-1200., 600.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


def get_pixels_hu(slices):
    # 进行矩阵的堆叠，将2D的图像变成3D的图像，image的维度是[切片数, 512, 512]
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


def preprocess_image(case_path):
    case = load_scan(case_path)
    case_pixels, spacing, origin = get_pixels_hu(case)
    case_pixels = np.array(case_pixels)
    return case_pixels, spacing, origin


def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord


def preprocess_pipeline(id, annos, filelist, data_path, prep_folder):
    try:
        start = time.time()
        name = filelist[id]
        resolution = np.array([1, 1, 1])
        im, spacing, origin = preprocess_image(os.path.join(data_path, name))
        im[np.isnan(im)] = -2000
        sliceim = lumTrans(im)
        sliceim1, _ = resample(sliceim, spacing, resolution, order=1)
        sliceim = sliceim1[np.newaxis, ...]
        np.save(os.path.join(prep_folder, name + '_clean.npy'), sliceim)  # 经过处理的图片

        this_nodule = annos[annos[:, 4] == name]
        label = []
        # origin是z,y,x
        for nod in this_nodule:
            # 如果标签文件是世界坐标，需要进行转换
            # pos = worldToVoxelCoord(nod[:3][::-1], origin=origin, spacing=spacing)
            # label.append(np.concatenate([pos, [nod[3] / spacing[1]]]))
            # 如果标签文件是像素坐标，只需要对xyz换成zyx即可
            pos = nod[:3][::-1]
            label.append(np.concatenate([pos, [nod[3]]]))
        if len(label) == 0:
            label2 = np.array([[0, 0, 0, 0]])
        else:
            label2 = np.copy(label).T
            label2[:3] = label2[:3] * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
            label2[3] = label2[3] * spacing[1] / resolution[1]
            label2 = label2[:4].T
        np.save(os.path.join(prep_folder, name + '_label.npy'), label2)  # 经过处理的label标签

        end = time.time()
        print(name, end="  ")
        print(end - start, "ms")
    except Exception as e:
        print(filelist[id])
        print(e.args)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # 处理后的结果存放路径
    prep_folder = "/media/data1/LC015preprocess/test_set/DSBmask_tmp"
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
    partial_savenpy = partial(preprocess_pipeline, annos=alllabel, filelist=filelist, data_path=data_path,
                              prep_folder=prep_folder)

    N = len(filelist)
    _ = pool.map(partial_savenpy, range(N))
    pool.close()
    pool.join()
