import os, sys, glob

import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon

ROI_ORDER = ['SpinalCord', 'Lung_R', 'Lung_L', 'Heart', 'Esophagus']


def read_data(path, train_flag, floder_flag, out_path):
    '''
            read training set and validation set according to directory structure
    :param path:
    :param train_flag: future work for semi-supervised method to generate new masks
    :param floder_flag: output folder
    :param out_path: output path
    :return:
    '''
    for root, dirs, file in os.walk(path):
        for i in dirs:
            # get the number of current path
            dicoms = glob.glob(os.path.join(root + '/' + i, '*.dcm'))

            # read RT structure files
            if train_flag == 'train_set' and 'simplified' in os.path.split(i)[1]:
                rt_dicom = pydicom.dcmread(dicoms[0])
                contours = get_contours(rt_dicom)

            # read dicoms
            elif len(dicoms) > 0:
                scans = [None] * len(dicoms)
                for i in range(len(dicoms)):
                    scans[i] = pydicom.dcmread(dicoms[i])
                    # print(scans[i].pixel_array.shape)
                    # plt.imshow(scans[i].pixel_array, cmap=plt.cm.bone)
                    # plt.show()

                # ['ImagePositionPatient', 'PatientPosition', 'PositionReferenceIndicator']
                print(scans[i].dir('position'))
                # print(scans[i].ImagePositionPatient)
                # sort by z
                scans.sort(key=lambda z: float(z.ImagePositionPatient[2]))
                #
                images = np.stack([s.pixel_array for s in scans])
                images = images.astype(np.int16)
                print(images.shape)

                # HU
                for s_num in range(len(scans)):
                    intercept = scans[s_num].RescaleIntercept
                    slope = scans[s_num].RescaleSlope
                    # print('inter:', intercept, 'slope', slope)

                    if slope != 1:
                        images[s_num] = slope * images[s_num].astype(np.float64)
                        images[s_num] = images[s_num].astype(np.int16)

                    images[s_num] += np.int16(intercept)
                    # plot HU for every image
                    # plt.hist(images[s_num].flatten(), bins=80, color='c')
                    # plt.show()

    # normalization HU
    images = normalization_hu(images)
    # plt.hist(images[s_num].flatten(), bins=80, color='c')
    # plt.show()

    if train_flag == 'train_set':

        # save images as npy
        for img in range(0, len(images)):
            total_imgs = []
            folder = os.path.exists(out_path + '/train_' + str(floder_flag) + '/images')
            if not folder:
                os.mkdir(out_path + '/train_' + str(floder_flag))
                os.mkdir(out_path + '/train_' + str(floder_flag) + '/images')

            if img == 0 or img == len(images) - 1:
                continue
            # print(img)
            # 2.5D data, using adjacent 3 images
            cur_img = images[img - 1:img + 2, :, :].astype('uint8')
            np.save(out_path + '/train_' + str(floder_flag) + '/images/' + str(img) + '_image.npy', cur_img)


        masks = get_masks(contours, images.shape, scans)

        # save masks as npy
        for msk in range(0, len(masks)):
            # save as png
            # labels_output_path = os.path.join(os.path.join(root, os.path.pardir), os.path.pardir)
            # print('base name', os.path.basename(root))
            folder = os.path.exists(out_path + '/train_' + str(floder_flag) + "/masks")
            if not folder:
                os.mkdir(out_path + '/train_' + str(floder_flag) + "/masks")
            if msk == 0 or msk == len(images) - 1:
                continue
            # cur_mask = masks[msk - 1: msk + 2, :, :].astype('uint8')
            # cur_mask = masks[msk, :, :].astype('uint8')
            np.save(out_path + '/train_' + str(floder_flag) + '/masks/' + str(msk) + '_mask.npy', masks[msk])

        return images, masks

    # future work for semi-supervised learning
    # elif train_flag == 'semi_supervised_set':
    #     # save images as png
    #     # need to be un comment
    #     for img in range(0, len(images)):
    #         # images_output_path = os.path.join(os.path.join(root, os.path.pardir), os.path.pardir)
    #         folder = os.path.exists(out_path + '/test_' + str(floder_flag) + "/images")
    #         if not folder:
    #             os.mkdir(out_path + "/images")
    #         else:
    #             np.save(out_path + '/train_' + str(floder_flag) + '/images/' + str(img) + '_image.npy', cur_img)
    #     return images


# some code imported from http://aapmchallenges.cloudapp.net/forums/3/2/
def get_contours(structure):
    '''
    get structure and reference information from RT Structure DICOM file

    :param structure: RT Structure file
    :return: contour
    '''
    contours = []
    for i in range(len(structure.ROIContourSequence)):
        contour = {}
        contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
        contour['number'] = structure.ROIContourSequence[i].ReferencedROINumber
        contour['name'] = structure.StructureSetROISequence[i].ROIName
        assert contour['number'] == structure.StructureSetROISequence[i].ROINumber
        contour['contours'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
        contours.append(contour)
    return contours


def get_masks(contours, shape, scans):
    '''
    convert RT Structure to label map mask

    :param contours
    :param shape:
    :param scans:
    :return:
    '''
    z = [np.around(s.ImagePositionPatient[2], 1) for s in scans]
    pos_row = scans[0].ImagePositionPatient[1]
    # print(scans[0].ImagePositionPatient)
    spacing_row = scans[0].PixelSpacing[1]
    pos_column = scans[0].ImagePositionPatient[0]
    spacing_column = scans[0].PixelSpacing[0]

    mask = np.zeros(shape, dtype=np.float32)
    for con in contours:
        num = ROI_ORDER.index(con['name']) + 1
        for c in con['contours']:
            nodes = np.array(c).reshape((-1, 3))
            assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
            z_index = z.index(np.around(nodes[0, 2], 1))
            # print('z_index: ', z_index)
            r = (nodes[:, 1] - pos_row) / spacing_row
            c = (nodes[:, 0] - pos_column) / spacing_column
            rr, cc = polygon(r, c)
            mask[z_index, rr, cc] = int(num)
    # print('mask shape: ', mask[0], 'num shape', num)
    return mask
# import code end


def normalization_hu(images):
    '''
    normalization to 0-255. and CT value from -1000 to 800 HU

    :param images:
    :return: normalized image
    '''
    MIN = -1000
    MAX = 800
    images = (images - MIN) / (MAX - MIN)
    images[images > 1] = 1.
    images[images < 0] = 0.
    return images * 255


def plot_ct_scan(scan):
    '''
    plot a few more images of the slices

    :param scan:
    :return:
    '''
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(30, 30))  # set 7 for all images
    # print('shape[0]', scan.shape[0])
    for i in range(0, scan.shape[0], 5):
        for j in range(4):
            # plots[int(i / 20), j].axis('off')
            plots[int(i / 20), j].imshow(scan[i], cmap=plt.cm.bone)
    plt.show()


if __name__ == '__main__':
    # set the path
    total_data_path = ""
    total_data_path_val = ""

    images_output_path = ""
    images_output_path_val = ""

    cases = [os.path.join(total_data_path_val, name)
             for name in sorted(os.listdir(total_data_path_val)) if
             os.path.isdir(os.path.join(total_data_path_val, name))]
    print('Patient number: ', len(cases))

    # # train sets
    for c in cases:
        # for folder_flag in range(3):
        print('c index:', cases.index(c))
        folder_flag = cases.index(c)
        print('C: ', os.path.basename(os.path.dirname(c)))
        images, masks = read_data(c, 'train_set', folder_flag, images_output_path_val)

    plot_ct_scan(images)
