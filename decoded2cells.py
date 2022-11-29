import cv2 as cv
import os
from optim_focus_stacking import init_stacking
from cellDetect import CellDetection


signal_dict = ['DAPI', 'GFP', 'Orange', 'Aqua']
name_patterns = {
    0: 'V-Case-ID-Tech-Probe-Zoom-Date'
}


def get_signals(path):
    signals = []
    dir_list = next(os.walk(path))[1]
    for dir in dir_list:
        if dir in signal_dict:
            signals.append(dir)
    return signals


def stack(read_path, write_path, signals):

    for signal in signals:
        canvas = init_stacking(read_path + signal + '/', write_path, signal)
        stacked_fname = write_path + signal + '.jpg'
        cv.imwrite(stacked_fname, canvas)
        canvas = None


def generate_cells(read_path, write_path, signals):
    detection = CellDetection(read_path, write_path, signals)
    detection()


def parse_dir(dir, patternID):

    pattern = name_patterns[patternID]
    pattern = pattern.split('-')
    case_ind = pattern.index('Case')
    ID_ind = pattern.index('ID')
    probe_ind = pattern.index('Probe')

    dir = dir.split('-')
    case = dir[case_ind]
    id = dir[ID_ind]
    probe = dir[probe_ind]

    return case, id, probe


#input_path = 'Z:/TEST_VSI\V-C22-08533-AC-D13CEN12-60X-09212022/_V-C22-08533-AC-D13CEN12-60X-09212022_/Decoded/'
stacking_input_path = 'C:/PythonProjects/FISH Patterns/FISH Stacks/'
stacking_output_path = 'C:/PythonProjects/FISH Patterns/FISH Stacks/Stacked/'
cell_output_path = 'C:/PythonProjects/FISH Patterns/Detected Cells/'

signals = get_signals(stacking_input_path)

stack(stacking_input_path, stacking_output_path, signals)

generate_cells(stacking_output_path, cell_output_path, signals)
