#!/usr/bin/python
# -*- coding: utf-8 -*-
from librarys import *
from config import *

def itk_read(img_path):
    image = sitk.ReadImage(img_path) # z, y, x
    image = np.squeeze(sitk.GetArrayFromImage(image)) + 1024
    image[image == -2000] = 0
    return image


def itk_read_(img_path):
    image = sitk.ReadImage(img_path) # z, y, x
    image = np.squeeze(sitk.GetArrayFromImage(image))
    # image[image == -2000] = 0
    return image
# trans = transforms.Compose([
#     transforms.ToPILImage(),
#     # transforms.RandomCrop(),
#     # transforms.RandomApply([transforms.RandomAffine(90., [0.2, 0.2], scale=(0.9, 1.15))], 0.8),
#     # transforms.Lambda()
#     # transforms.RandomVerticalFlip(0.5),
#     transforms.RandomHorizontalFlip(0.5),
#     # transforms.TenCrop(), # TODO
#     # transforms.RandomRotation(90.),
#     transforms.ToTensor()
#     # transforms.ToTensor(),
#     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
# ])

# def get_pixel_hu(slice):
#     image = slice.pixel_array
#     # Convert to int16 (from sometimes int16),
#     # should be possible as values should always be low enough (<32k)
#     image = image.astype(np.int16)
#     image[image == -2000] = 0
#     # Convert to Hounsfield units (HU)
#
#     intercept = slice.RescaleIntercept
#     slope = slice.RescaleSlope
#
#     if slope != 1:
#         image = slope * \
#                 image.astype(np.float64)
#         image = image.astype(np.int16)
#
#     image += np.int16(intercept)
#
#     return np.array(image, dtype=np.int16)


# def get_pixels_hu(slices):
#     image = np.stack([s.pixel_array for s in slices])
#     # Convert to int16 (from sometimes int16),
#     # should be possible as values should always be low enough (<32k)
#     image = image.astype(np.int16)
#     image[image == -2000] = 0
#     # Convert to Hounsfield units (HU)
#     for slice_number in range(len(slices)):
#         intercept = slices[slice_number].RescaleIntercept
#         slope = slices[slice_number].RescaleSlope
#
#         if slope != 1:
#             image[slice_number] = slope * \
#                                   image[slice_number].astype(np.float64)
#             image[slice_number] = image[slice_number].astype(np.int16)
#
#         image[slice_number] += np.int16(intercept)
#
#     return np.array(image, dtype=np.int16)  # , np.array([slices[0].SliceThickness] + slices[0].PixelSpacing,
#     #     dtype=np.float32)


def process_data(path):
    data = pd.DataFrame([{
        'ID': filePath.split('/')[-3],
        'phase': filePath.split('/')[-2],
        'path_dcm': filePath
    } for filePath in sorted(glob.iglob(DATA_PATH + path))])

    # data['path_contour'] = data['path_dcm'].map(lambda x: os.path.dirname(
    #     x) + '/' + x.split('/')[-1].split('.')[0] + '_contour.png')
    data['path_mask'] = data['path_dcm'].map(lambda x: os.path.dirname(
        x) + '/' + x.split('/')[-1].split('.')[0] + '_mask.png')
    df = data.merge(_read_csv('临床数据.csv'), on='ID')
    # TODO
    if phase != 'all':
        df = df[df.phase == phase].reset_index(drop=True)


    if SHUFFLE:
        df = df.sample(frac=1).reset_index(drop=True)
    return df


def _read_csv(path):
    data = pd.read_csv(DATA_PATH + path, encoding='gb2312')
    data['ID'] = data['ID'].astype('str')
    return data


def read_dcm(file_path):
    dcm_data = pydicom.read_file(file_path)
    im = dcm_data.pixel_array
    return im


def read_mask(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
