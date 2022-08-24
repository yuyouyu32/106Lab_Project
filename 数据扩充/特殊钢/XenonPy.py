import os
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm


class DataTransform:
    """This class defines the data transform methods.
    Attributes:
        data_path: The file path of data.
        save_path: The direction path of saving.
        ELEMENTTABLE: Legal element names(1-111)
    """

    def __init__(self, data_path: str, save_path: str) -> None:
        if os.path.isfile(data_path) and os.path.isdir(save_path):
            self.data_path = data_path
            self.save_path = save_path
        else:
            raise Exception(
                "Sorry, the path of data or path of saving is illegal.")
        self.ELEMENTTABLE = {'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                             'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
                             'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
                             'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
                             'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At',
                             'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No',
                             'Lw', 'Rf', 'Db', 'Sg ', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg'}

    def _get_out_columns(self, img_template_path: str):
        column_names = pd.read_excel(img_template_path).values.flatten()
        heads = ['sum:', 'var:', 'gmean:', 'ave:', 'hmean:', 'max:', 'min:']
        out_columns = [None]*504
        for raw_index in range(0, 7):
            for column_index in range(0, 8):
                column_name = column_names[column_index + raw_index * 8]
                out_columns[raw_index * 72 + column_index *
                            3] = f'{heads[0]}{column_name}'
                out_columns[raw_index * 72 + 1 + column_index*3] = 'zeros'
                out_columns[raw_index * 72 + 2 + column_index *
                            3] = f'{heads[1]}{column_name}'
                out_columns[raw_index * 72 + 24 +
                            column_index*3] = f'{heads[2]}{column_name}'
                out_columns[raw_index * 72 + 25 +
                            column_index*3] = f'{heads[3]}{column_name}'
                out_columns[raw_index * 72 + 26 +
                            column_index*3] = f'{heads[4]}{column_name}'
                out_columns[raw_index * 72 + 48 +
                            column_index*3] = f'{heads[5]}{column_name}'
                out_columns[raw_index * 72 + 49 + column_index*3] = 'zeros'
                out_columns[raw_index * 72 + 50 +
                            column_index*3] = f'{heads[6]}{column_name}'
        return out_columns

    def set_data_path(self, data_path: str) -> None:
        """Set new path of data.
        Arguments:
            data_path: The file path of data.
        """
        if os.path.isfile(data_path):
            self.save_path = data_path
        else:
            raise Exception("Sorry, the path of data is illegal.")

    def set_save_path(self, save_path: str) -> None:
        """Set new path of saving.
        Arguments:
            save_path: The direction path of saving.
        """
        if os.path.isdir(save_path):
            self.save_path = save_path
        else:
            raise Exception("Sorry, the path of saving is illegal.")

    def generate_original_img(self, chemical_compositions: List[str], file_name = 'original_images.csv') -> pd.DataFrame:
        """Generate original image with XenonPy.
        Arguments:
            chemical_compositions: The names of chemical compositions.
        Return:
            The result of transforming with XenonPy(pd.DataFrame).
        """
        # Import Xenonpy packages.
        from xenonpy.datatools import preset
        from xenonpy.descriptor import Compositions
        preset.sync('atom_init')

        if chemical_compositions is None or len(chemical_compositions) == 0:
            raise Exception("Chemical elements cannot be empty.")
        if not (set(chemical_compositions) <= self.ELEMENTTABLE):
            raise Exception("There are illegal element names in the list.")
        original_data = pd.read_csv(self.data_path).astype('float64')
        # Print original data's describe.
        print(original_data.describe().T)
        # Get chemical composition for each data item.
        comp = pd.DataFrame(
            original_data, columns=chemical_compositions).values
        compObj = []
        for i in range(len(original_data)):
            tmp = {}
            for j in range(len(chemical_compositions)):
                if comp[i][j] != 0:
                    tmp[chemical_compositions[j]] = comp[i][j]
            compObj.append(tmp)
        # XenonPy transform
        cal = Compositions(featurizers=['WeightedAverage', 'WeightedSum',
                                        'WeightedVariance', 'MaxPooling', 'MinPooling', 'GeometricMean', 'HarmonicMean'])
        compRes = cal.transform(compObj)
        # Saving
        compRes.to_csv(f'{self.save_path}/{file_name}', index=False)
        return compRes

    def generate_img(self, original_img_path: str = None, img_template_path: str = './ImgTemplate.xlsx',  chemical_compositions: List[str] = None, file_name = 'Images.csv') -> pd.DataFrame:
        """Generate image from orginal image or chemical compositions.
        Arguments:
            original_img_path: The path of original image.
            chemical_compositions: The names of chemical compositions.
        Return:
            The result of transform(pd.DataFrame).
        """

        if original_img_path is None and chemical_compositions is None:
            raise Exception(
                'Missing parameters: original_img_path or chemical_compositions.')
        if original_img_path is not None and os.path.isfile(original_img_path):
            original_pic = pd.read_csv(original_img_path).astype('float64')
        elif chemical_compositions is not None:
            original_pic = self.generate_original_img(chemical_compositions)
        else:
            raise Exception(
                'Error parameters: original_img_path or chemical_compositions.')
        def max_min_scaler(x): return (x - np.min(x)) / \
            (np.max(x) - np.min(x))
        image = original_pic.apply(max_min_scaler)
        def gray_scaler(x): return round(x * 255)
        image = image.apply(gray_scaler)
        image.fillna(0, inplace=True)
        image.replace(np.inf, 255, inplace=True)
        image.astype('float64')
        out_columns = self._get_out_columns(img_template_path)
        image['zeros'] = 0
        image.to_csv(f'{self.save_path}/{file_name}',
                     index=False, columns=out_columns)
        image = pd.read_csv(f'{self.save_path}/{file_name}')
        return image


def transfet_img(img_path: str, save_path: str, save_prename: str) -> None:
    from PIL import Image
    original_pics = pd.read_csv(f'./{img_path}').astype('double').values.reshape(-1, 1, 24, 21).astype(np.uint8)
    for index, pic in enumerate(tqdm(original_pics)):
        im = Image.fromarray(np.squeeze(pic.transpose(2,1,0)))
        im.save(f'{save_path}/{save_prename}_{index}.png')

# Test
if __name__ == '__main__':
    datatransformer = DataTransform(
        data_path='./特殊钢-全链条计算数据_插值_中位数版_formatted2.csv', save_path='./')
    original_data = datatransformer.generate_original_img(
        chemical_compositions=['C','Mn','Si','P','S','Cr','Mo','V','Ni','W','Ce','La','Ca','Co','Nb','Ti','B','Cu','Sr','Se','Na','Hg','Cd','Ag','As','Bi','Zn','Sb','Sn','Pb','Fe','N','O','H','Mg'], file_name='xeonpy_e_1.csv')
    pic_1 = datatransformer.generate_img(img_template_path = './ImgTemplate.xlsx', original_img_path='xeonpy_e_1.csv', file_name='./Images.csv')
    transfet_img(img_path = './Images.csv', save_path = './pics_1', save_prename = 'test')
    pic_2 = datatransformer.generate_img(img_template_path = './PICTURE.xlsx', original_img_path='xeonpy_e_1.csv', file_name='./Images_2.csv')
    transfet_img(img_path = './Images_2.csv', save_path = './pics_2', save_prename = 'test_2')
    
