import numpy as np
import warnings
import os

import pandas as pd

# Read in the filenames and 
with open('./Experimental_setup/Knot_Tying/Balanced/GestureClassification/OneTrialOut/10_Out/itr_1/Train.txt') as f:
     file_names = f.readlines()
file_data = np.loadtxt('./Knot_Tying/kinematics/AllGestures/Knot_Tying_B001.txt')

# 0 -> novice, 1 -> intermediate, 2 -> expert
class_values = {'B': 0, 'C': 1, 'D': 2, 'E': 2, 'F': 1, 'G': 0, 'H': 0, 'I': 0}
DATA_INDEX = 0
LABEL_TYPE_INDEX = 1
SURGEON_INDEX = 2
CLASS_INDEX = 3

class PreProcess(): 
    def __init__(self, directory):
        self.directory = directory
        
    def get_from_labelType(self, directory, data_type, labels, leave_out):
        '''
        Helper function for parse_experiment setup
        data_type: "Knot_Tying", "Suturing", "Needle_Passing"
        Labels: "GestureClassification", "GestureRecognition", "SkillDetection"
        Leave_out: "OneTrialOut", "SuperTrialOut", "UserOut"
        '''
        rootdir = os.path.join(directory, "Experimental_setup", data_type, "unBalanced", labels, leave_out)
        output = None
        if (labels == "SkillDetection"):
            form = 'i4'
        else:
            form = 'U64'
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                f = os.path.join(subdir, file)           
                if output is None:
                    output = np.loadtxt(f, dtype={'names': ('fileName', 'gesture'), 
                                             'formats': ('U64',  form)})
                else:
                    a = np.loadtxt(f, dtype={'names': ('fileName', 'gesture'), 
                                             'formats': ('U64', form)})
                    output = np.append(output, a)
        return output
    def parse_experimental_setup(self, data_type):
        '''
        Will iterate throught the experimental setup txt files
        @param data_type: will be either knot_tying, suturing, or needle_passing 
        @return: will return a list of three numpy objects:
            1). Gesture classification 
            2). Gesture Recognition
            3). Skill detection
        '''  
        leave_out="OneTrialOut"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gClass = self.get_from_labelType(self.directory, data_type, "GestureClassification", leave_out)
            gRec = self.get_from_labelType(self.directory, data_type, "GestureRecognition", leave_out)

        if (data_type == "Suturing"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                skillDet = self.get_from_labelType(self.directory, data_type, "SkillDetection", leave_out)
                return gClass, gRec, skillDet
        return gClass, gRec
    
    
    def get_start_and_end(self, txt_file, data_type):
        try:
            # Gets rid of '.txt' for the end of the row
            if data_type == 'Suturing':
                txt_file[3] = txt_file[3][0:-4]
                return int(txt_file[2]), int(txt_file[3])
            txt_file[4] = txt_file[4][0:-4]
            return int(txt_file[3]), int(txt_file[4])
        except IndexError as error:
            print(error)
            print('Text file', txt_file, data_type)

    def get_file_name(self, txt_file, data_type):
        if data_type == 'Suturing':
            return txt_file[0] + '_' + txt_file[1] + '.txt'
        return txt_file[0] + '_' + txt_file[1] + '_' + txt_file[2] + '.txt'
    
    def read_data(self, data, data_type):
        """
        Will read the data by it's appropriate line number
        @param data: a numpy array of size two containing a txt file name & 
        class value
        @param data_type: string that is either Knot_Tying, Suturing, or Needle_Passing
        @return output_data: the correct time series data
        """
        output_data = []
        print("Printing data, ")
        print(data)
        for i in range(len(data)):
            row_data = []
            txt_file = data[i][0].split('_')
            label = data[i][1]
            # Find out which surgeon it is
            if data_type == 'Suturing':
                surgeon = txt_file[1][0]
            else:
                surgeon = txt_file[2][0]
            class_value = class_values[surgeon]
            file_name = self.get_file_name(txt_file, data_type)
            row_start, row_end = self.get_start_and_end(txt_file, data_type)
            # Read in the appropriate files for the kinematics
            with open('./' + data_type + '/kinematics/AllGestures/' + file_name) as f:
                rows = f.readlines()
            for j in range(row_start, row_end):
                current_row = rows[j].split()
                row_data.append(current_row)
            output_data.append([row_data, label, surgeon, class_value])
        return output_data

    def parse(self):
        """
        Provides the entire functionality to parse our data

        @param: 
        @return: 
        """
        table = {}

        knot_tying_data = self.parse_experimental_setup('Knot_tying')
        suturing_data = self.parse_experimental_setup('Suturing')
        needle_passing = self.parse_experimental_setup('Needle_passing')
        
        # Iterate through the knot_tying data
        kt_gest_class_data = self.read_data(knot_tying_data[0], 'Knot_Tying')
        #kt_gest_rec_data = self.read_data(knot_tying_data[1], 'Knot_Tying')
        table['knot_tying'] = {'gesture_classification': kt_gest_class_data} #, 
                               #'gesture_recognition': kt_gest_rec_data}
        '''
        # Iterate through the suturing data
        sut_gest_class_data = self.read_data(suturing_data[0], 'Suturing')
        sut_gest_rec_data = self.read_data(suturing_data[1], 'Suturing')
        sut_skill_data = self.read_data(suturing_data[2], 'Suturing')
        table['suturing'] = {'gesture_classification': sut_gest_class_data, 
                               'gesture_recognition': sut_gest_rec_data,
                               'skill_detection': sut_skill_data}

        # Iterate through the needle passing data
        np_gest_class_data = self.read_data(needle_passing[0], 'Needle_Passing')
        np_gest_rec_data = self.read_data(needle_passing[1], 'Needle_Passing')
        table['needle_passing'] = {'gesture_classification': np_gest_class_data, 
                               'gesture_recognition': np_gest_rec_data}'''
        
        return table

p = PreProcess( "./")
d = p.parse()

csv_columns = ['Task','Label_Type','Data', 'Gesture/Skill', 'Surgeon', 'Class_Value']

df = pd.DataFrame(columns=csv_columns)
data_ = { 
            'Task': [],
            'Label_Type': [],
            'Data': [], 
            'Gesture/Skill': [], 
            'Surgeon': [], 
            'Class_Value': []
        }
try:
    for task in d.keys():
        for label_type in d[task].keys():
            for row in d[task][label_type]:
                data_['Task'].append(task)
                data_['Label_Type'].append(label_type)
                data_['Data'].append(row[0])
                data_['Gesture/Skill'].append(row[1])
                data_['Surgeon'].append(row[2])
                data_['Class_Value'].append(row[3])

except IOError:
    print("I/O error") 

print("Saving full dataset into numpy object")
np.save('data_full.npy', data_)
print("Finished saving full dataset")
