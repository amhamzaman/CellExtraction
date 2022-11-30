"""
A class named "CellDetection"
A class named "CellFilter"
"""


from pathlib import Path
import torch
import numpy as np
import pandas as pd
import cv2 as cv


class CellDetection:
    """
    CellDetection class for extracting cells from a stacked image and filtering them.

    Constructor arguments:
        path : path/filename of the stacked image
        out_path : path of the output directory

    Output:
        A directory called "DAPI_Cells/"
            "DAPI_Cells/" contains the following:
                "DAPI_cells.csv" containing information of all the extracted cells
                "accepted/" directory which contains all the cells that were accepted after filtering
                if "keep_rejects" flag is set to True (default) in the class constructor, a directory 
                called "rejected/" is created which contains all the rejected cells  
    """

    def __init__(self, path, out_path, signals, fileinfo, debug=False, url='', frame_size=1000, conf_thresh=0.2, bbox_margin=0.05, aspect_thresh=0.5, keep_rejects=True):
        """
        Constructor

        Args:
            path (str): Path of stacked DAPI image
            out_path (str): Path of output directory
            signals (list of str): Type of signals to extract
            url (str, optional): (Future feature) URL of stacked DAPI image. Defaults to ''.
            frame_size (int, optional): Size of frame to process at a time. Defaults to 1000.
            conf_thresh (float, optional): Confidence threshold for confidence filter. Defaults to 0.2.
            bbox_margin (float, optional): Bounding box margin for bounding box filter. Defaults to 0.1.
            aspect_thresh (float, optional): Aspect ratio threshold for aspect ratio filter. Defaults to 0.6.
            keep_rejects (bool, optional): Specify whether to store rejected cells or not. Defaults to True.
        """
        self._URL = url
        self._path = path
        self.signals = signals
        self.fileinfo = fileinfo
        self.debug = debug
        self.model = self.__load_model()
        self.conf_thresh = conf_thresh
        self.classes = self.model.names
        self.cellcount = 0
        self.frame_size = frame_size
        self.keep_rejects = keep_rejects
        self.filter = CellFilter(
            conf_thresh=conf_thresh, bbox_margin=bbox_margin, aspect_thresh=aspect_thresh)
        self.out_path = out_path
        self.df = pd.DataFrame(columns=[
                               'Name', 'Xmin', 'Ymin', 'Xmax', 'Ymax', 'Confidence', 'Informative', 'Accepted', 'Conf_Filter', 'Aspect_Filter', 'Boundary_Filter', 'Contour_Filter', 'Frame_Row', 'Frame_Col', 'Xmin_abs', 'Ymin_abs', 'Xmax_abs', 'Ymax_abs'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:", self.device)

    def __get_image_from_url(self):
        """
        [Not implemented] Future feature to take DAPI image from a URL
        """
        pass

    def __parse_path(self):
        """
        [TO BE IMPLEMENTED] Parse DAPI image path to extract Case number, Probe type etc 
        """
        pass

    def _get_stacked_image(self, file):
        """
        Read image from file
        Returns:
            numpy array: DAPI image from file
        """
        img = cv.imread(self._path + file)
        return img

    def _get_frames(self, img):
        """
        Extracts frames from the complete DAPI image to be processed individually

        Args:
            img (2D numpy array): Complete DAPI image

        Returns:
            frames (list of 2D numpy arrays): List of all the frames extracted
            locs (list): List of 2D indeces for all the frames
        """

        chunk = self.frame_size
        frames = []
        locs = []
        x = img.shape[1]
        y = img.shape[0]

        for j in range(0, y, chunk):
            for i in range(0, x, chunk):
                frame = img[j:j+chunk, i:i+chunk]
                if not self.__isempty(frame):
                    frames.append(frame)
                    locs.append([int(j/chunk), int(i/chunk)])

        return frames, locs

    def __isempty(self, frame):
        """
        Check if a frame is empty

        Args:
            frame (2D numpy array): _description_

        Returns:
            bool: True if empty
        """
        return np.sum(frame) == 0

    def __load_model(self):
        """
        Loads YOLOv5 Model

        Returns:
            model: YOLOv5 pretrained model
        """
        model = torch.hub.load('yolov5', 'custom',
                               path='models/detector.pt', source='local', verbose=False)
        return model

    def __score_frame(self, frame):
        """
        Feedforward a frame through the model

        Args:
            frame (2D numpy array): frame to pass through the model

        Returns:
            labels (tensor): Class label for each detected cell. 0 for informative, 1 for uninformative
            cord (tensor): Relative coordinates and confidence for each detected cell 
        """

        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def __class_to_label(self, x):
        """
        Return name of class corresponding to numeric tag

        Args:
            x (int): Numeric class tag

        Returns:
            str: Class name
        """

        return self.classes[int(x)]

    def __extract_cells(self, frame, results):
        """
        Extract cells along with their filtering status. 

        Args:
            results (tensor): results from YOLOv5 model
            frame (2D numpy array): frame

        Returns:
            : list of cells along with filtering status
        """

        return self.filter.get_cells(frame, results)

    def __plot_boxes(self, frame, results):
        """
        Plot boxes and class names around cells

        Args:
            results (_type_): _description_
            frame (_type_): _description_

        Returns:
            _type_: _description_
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= self.conf_thresh:
                x1, y1, x2, y2 = int(
                    row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv.putText(frame, self.__class_to_label(
                    labels[i]), (x1, y1), cv.FONT_HERSHEY_COMPLEX, 0.9, bgr)

        return frame

    def __populate_df(self, name, cord,  label, accepted, loc, abscord):
        """
        Add a row to the object dataframe

        Args:
            name (str): Name of the cell
            cord (tensor): 
            loc (list): _description_
            label (int): _description_
            accepted (bool): Whether the cell is accepted
        """
        label = int(1 - label)
        self.df.loc[len(self.df.index)] = [
            name, *cord,  label, int(all(accepted)), *[int(x) for x in accepted], *loc, *abscord]

    def __get_pixel_cord(self, cordi):
        """
        Convert relative coordinates to frame-relative pixel coordinates

        Args:
            cordi (tensor): Relative coordinates
        Returns:
            cord (list): Pixel coordinates
        """
        cord = cordi.detach().clone()
        # print(cord)
        x_shape = int(self.frame_size - 1)
        y_shape = int(self.frame_size - 1)
        cord[0], cord[2] = int(cord[0]*x_shape), int(cord[2]*x_shape)
        cord[1], cord[3] = int(cord[1]*y_shape), int(cord[3]*y_shape)
        return cord.tolist()

    def __get_abs_cord(self, cordi, loc):
        """
        Gets absolute coordinates relative to the stacked image

        Args:
            cordi (list): Pixel coordinates relative to frame
            loc (list): 2D Location of the frame on the stacked image

        Returns:
            cord (list): A list of absolute coordinates
        """
        cord = cordi.copy()
        fact = int(self.frame_size)
        cord[0], cord[2] = int(cord[0] + (loc[1]*fact)
                               ), int(cord[2] + (loc[1]*fact))
        cord[1], cord[3] = int(cord[1] + (loc[0]*fact)
                               ), int(cord[3] + (loc[0]*fact))

        return cord[:4]

    def __get_filepath(self, loc, label, signal, accepted, count):
        """
        Automatically generate filename
        """
        case = self.fileinfo
        lab = 'i' if label == 0 else 'u'
        file_name = f'{case}_{loc[0]}_{loc[1]}_{count}_{signal}_' + lab
        dir = signal + '_cells/'

        if self.keep_rejects:
            acc = "accepted/" if accepted else "rejected/"
        else:
            acc = ''

        dir_name = self.out_path + dir + acc
        path_name = dir_name + file_name + '.jpg'

        return path_name, file_name

    def __make_dir(self):
        """
        Make output directory tree
        """

        for signal in self.signals:
            out = self.out_path + signal + '_cells/'
            acc = out + 'accepted/'
            rej = out + 'rejected/'

            outpath = Path(out)
            accpath = Path(acc)
            rejpath = Path(rej)
            if not outpath.exists():
                outpath.mkdir(parents=True, exist_ok=True)
            if not accpath.exists() and self.keep_rejects:
                accpath.mkdir(parents=True, exist_ok=True)
            if not rejpath.exists() and self.keep_rejects:
                rejpath.mkdir(parents=True, exist_ok=True)

    def __mark_reject(self, rejected_cell, reason):
        """
        Mark rejected cells with reasons for their rejection

        Args:
            cell (_type_): _description_
            reason (_type_): _description_

        Returns:
            _type_: _description_
        """
        cell = rejected_cell.copy()

        if not reason[0]:
            cv.putText(cell, 'Conf', (1, 1),
                       cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        if not reason[1]:
            cv.putText(cell, 'Asp', (1, 15),
                       cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        if not reason[2]:
            cv.putText(cell, 'Bound', (1, 30),
                       cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

        return cell

    def __save_cells(self, cells, acceptance, coords, results, loc):
        """
        Generate output images for each cell and populate the dataframe with relevant data

        Args:
            cells (list of 2D nympy arrays): A list of cell images
            acceptance (list): A matrix with filtering information for each cell
            coords (list): A list of frame-relative coordinates 
            results (tensor): Results from model
            loc (list): Index of the current frame
        """
        labels, cord = results
        for i, cell in enumerate(cells):

            accepted = all(acceptance[i])
            if not accepted and not self.keep_rejects:
                continue
            elif not accepted:
                cell = self.__mark_reject(cell, acceptance[i])

            path_name, file_name = self.__get_filepath(
                loc, labels[i], 'DAPI', accepted, self.cellcount)

            pixcord = coords[i]
            pixcord.append(cord[i].tolist()[-1])
            abscord = self.__get_abs_cord(pixcord, loc)
            self.__populate_df(file_name, pixcord,
                               labels[i], acceptance[i], loc, abscord)
            cv.imwrite(path_name, cell)
            cell = None
            self.cellcount += 1

    def __save_signals(self):
        """
        Generate output images for each signal other than DAPI
        """

        celldf = self.df[[
            'Xmin_abs', 'Ymin_abs', 'Xmax_abs', 'Ymax_abs', 'Accepted', 'Frame_Row', 'Frame_Col', 'Informative']]

        for signal in self.signals:
            if signal == 'DAPI':
                continue
            count = 0
            img = self._get_stacked_image(signal + '.jpg')
            for i in range(celldf.shape[0]):
                x1, y1, x2, y2, accepted, fr, fc, label = celldf.iloc[i]
                if not accepted and not self.keep_rejects:
                    continue
                cell = img[y1:y2+1, x1:x2+1]
                path_name, _ = self.__get_filepath(
                    [fr, fc], int(1-label), signal, accepted, count)

                cv.imwrite(path_name, cell)
                cell = None
                count += 1
                print(
                    f'Generating corresponding {signal} cells: {count}/{celldf.shape[0]}', end='\r')
            print(
                f'Generating corresponding {signal} cells: {count}/{celldf.shape[0]}')

    def __call__(self):
        """
        Call function responsible for reading DAPI image, create frames, detect cells, write cell images, signal images and the dataframe file.
        """

        self.__make_dir()
        img = self._get_stacked_image('DAPI.jpg')
        frames, locs = self._get_frames(img)
        for i, frame in enumerate(frames):
            if i > 2 and self.debug:
                break

            results = self.__score_frame(frame)
            cells, acceptance, coords = self.__extract_cells(frame, results)
            self.__save_cells(cells, acceptance, coords, results, locs[i])

            print(
                f"{i+1}/{len(frames)} frames processed. Total cells Extracted: {self.cellcount}", end='\r')

        print(
            f"{i+1}/{len(frames)} frames processed. Total cells Extracted: {self.cellcount}")

        self.__save_signals()

        fname = 'cellinfo'
        self.df.to_csv(self.out_path + fname + '.csv')


class CellFilter():
    """
    Filters cells based on confidence, boundary case and aspect ratio.
    """

    def __init__(self, conf_thresh=0.2, bbox_margin=0.1, aspect_thresh=0.625):
        """
        Args:
            conf_thresh (float, optional): _description_. Defaults to 0.2.
            bbox_margin (float, optional): _description_. Defaults to 0.1.
            aspect_thresh (float, optional): _description_. Defaults to 0.625.
        """
        self.conf_thresh = conf_thresh
        self.bbox_margin = bbox_margin
        self.aspect_thresh = aspect_thresh
        self.image = []
        self.results = []
        self.labels = []
        self.n = 0
        self.cord = []

    def get_cells(self, image, results):
        """
        Takes a frame and YOLO5 cell detection results and returns all the cells along with filtering results
        Args:
            image (numpy array): frame to extract cells from
            results (tensor): results (labels, coordinates, confidence) from YOLO5 model

        Returns:
            cells (list of numpy arrays): a list of all the cells
            acceptance (list): N x M boolean matrix. acceptance[n][m] is True if n_th cell passed the m_th filter otherwise False
            coords (list): frame relative coordinates
        """
        self.image = image
        labels, cord = results
        self.results = results
        self.labels = labels
        self.n = len(labels)
        self.cord = cord

        cells, acceptance, coords = self.__apply_filters()

        return cells, acceptance, coords

    def __apply_filters(self):
        """
        Internal function to apply all the filters

        Returns:
            cells (list of numpy arrays): a list of all the cells
            acceptance (list) : N x M boolean matrix. acceptance[n][m] is True if n_th cell passed the m_th filter otherwise False
            coords (list): frame-relative coordinates
        """
        n = self.n
        cells = []
        acceptance = []
        coords = []
        good = True
        for i in range(n):
            good = [self.__conf_thresholding(i), self.__aspect_check(
                i), self.__boundary_check(i), self.__contour_check(i)]
            acceptance.append(good)
            cell, cord = self.__cell_box(i)
            cells.append(cell)
            coords.append(cord)
        return cells, acceptance, coords

    def __conf_thresholding(self, i):
        """
        Private function to apply confidence thresholding to i_th cell

        Args:
            i (int): cell index

        Returns:
            bool: True if passed, False if failed
        """
        return self.results[1][i][4] >= self.conf_thresh

    def __boundary_check(self, i):
        """
        Private function to apply boundary check on i_th cell

        Args:
            i (int): cell index

        Returns:
            bool: True if passed, False if failed
        """

        x_shape, y_shape = self.image.shape[1], self.image.shape[0]
        row = self.results[1][i]
        marg = self.bbox_margin
        x1, y1, x2, y2 = int(
            row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
        dx, dy = (x2 - x1), (y2 - y1)
        x1, y1, x2, y2 = int(x1 - dx * marg), int(y1 - dy *
                                                  marg), int(x2 + dx*marg), int(y2 + dy*marg)
        return x1 > 0 and y1 > 0 and x2 < x_shape and y2 < y_shape

    def __aspect_check(self, i):
        """        
        Private function to apply aspect ratio check on i_th cell

        Args:
            i (int): cell index

        Returns:
            bool: True if passed, False if failed
        """
        row = self.results[1][i]
        x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
        dx, dy = x2 - x1, y2 - y1
        ratio = min(dx/dy, dy/dx)

        return ratio > self.aspect_thresh

    def __contour_check(self, i):
        """
        [TO BE IMPLEMENTED]        
        Private function to apply countour check on i_th cell

        Args:
            i (int): cell index

        Returns:
            bool: True if passed, False if failed
        """
        return True

    def __cell_box(self, i):
        """
        Private function to create i_th cell image with margin

        Args:
            i (int): cell index

        Returns:
            cell (numpy array): cell image
        """
        frame = self.image
        x_shape, y_shape = frame.shape[1]-1, frame.shape[0]-1
        row = self.results[1][i]
        marg = self.bbox_margin
        x1, y1, x2, y2 = int(
            row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
        dx, dy = (x2 - x1), (y2 - y1)
        x1, y1, x2, y2 = int(x1 - dx*marg), int(y1 - dy *
                                                marg), int(x2 + dx*marg), int(y2 + dy*marg)

        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(
            x2, x_shape), min(y2, y_shape)
        return frame[y1:y2+1, x1:x2+1], [x1, y1, x2, y2]


### TEST CODE ###
# image_path = "C:/PythonProjects/FISH Patterns/CellExtraction/test/"
# output_dir = "C:/PythonProjects/FISH Patterns/CellExtraction/test/output/"
# signals = ['DAPI', 'GFP', 'Orange']
# fileprefix = 'Case-ID-this-that'
# fileinfo = {'Case': 'C22', 'id': '522322', 'Probe': 'BCRABL'}


# detection = CellDetection(
#     path=image_path, out_path=output_dir, signals=signals, fileinfo=fileprefix, debug=False)
# detection()
