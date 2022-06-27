import math
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

head_list = ['Eea', 'I1', 'I2', 'Tm', 'AW', 'AN', 'Rm', 'Rc', 'Gp', 'P',
             'VEC', 'sVEC', 'pVEC', 'dVEC', 'XP', 'XM', 'Cp', 'K', 'W', 'D',
             'Hf', 'LP', 'Tb']


def findHeadIndex(_head_name):
    for i in range(len(head_list)):
        if head_list[i] == _head_name:
            return i
    return -1

elem_list = ['Ag', 'Al', 'Au', 'B', 'Ba', 'Be', 'C', 'Ca', 'Ce', 'Co',
             'Cr', 'Cu', 'Dy', 'Er', 'Fe', 'Ga', 'Gd', 'Ge', 'Hf', 'Ho',
             'In', 'Ir', 'La', 'Li', 'Lu', 'Mg', 'Mn', 'Mo', 'Nb', 'Nd',
             'Ni', 'P', 'Pb', 'Pd', 'Pr', 'Pt', 'Rh', 'Ru', 'Sb', 'Sc',
             'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Ti', 'Tm', 'U', 'V',
             'W', 'Y', 'Yb', 'Zn', 'Zr']


column_names = ['AN1', 'AN2', 'AND', 'ANd', 'Rc1', 'Rc2', 'RcD', 'Rcd', 'Vmc', 'Rm1', 'Rm2', 'RmD', 'Rmd', 'Vmm', 'AW1', 'AW2', 'AWD', 'AWd', 'D1', 'D2', 'DD', 'Dd', 'Eea1', 'Eea2', 'EeaD', 'Eead', 'XM1', 'XM2', 'XMD', 'XMd', 'XP1', 'XP2', 'XPD', 'XPd', 'Gp1', 'Gp2', 'GpD', 'Gpd', 'Hf1', 'Hf2', 'HfD', 'Hfd', 'I11', 'I12', 'I1D', 'I1d', 'I21', 'I22', 'I2D', 'I2d', 'LP1', 'LP2', 'LPD', 'LPd', 'Tm1', 'Tm2', 'TmD', 'Tmd', 'P1', 'P2', 'PD', 'Pd', 'Cp1', 'Cp2', 'CpD', 'Cpd', 'K1', 'K2', 'KD', 'Kd', 'VEC1', 'VEC2', 'VECD', 'VECd', 'Tb1', 'Tb2', 'TbD', 'Tbd', 'sVEC1', 'pVEC1', 'dVEC1', 'fs', 'fp', 'fd', 'W1', 'W2', 'WD', 'Wd', 'Hmix', 'Smix/R', 'Smix/kb', 'PHSmis', 'PHSmix', 'PHSS']

def findElemIndex(_elem):
    for i in range(len(elem_list)):
        if elem_list[i] == _elem:
            return i
    return -1


def parseCompoStr(_compo_str: str):
    result = np.zeros(len(elem_list))
    for it in re.findall(r'[A-Z][a-z]?[\d.]+', _compo_str):
        elem = re.findall(r'[A-Z][a-z]?', it)[0]
        cent = re.findall(r'[\d.]+', it)[0]
        result[findElemIndex(elem)] = float(cent)
    return result / np.sum(result)


s1Data = np.loadtxt('./adata/0_S1data.csv', delimiter=',')
deltaH = np.loadtxt('./adata/0_deltaH.csv', delimiter=',')


def function1(_composition: np.ndarray, _head_name: str):
    """
        @ref equation (1)

        @example
            function1(compo, 'Eea') is Eea1
    """
    result = 0.0
    for i in range(_composition.shape[0]):
        result += _composition[i] * s1Data[i, findHeadIndex(_head_name)]
    return result


def function2(_composition: np.ndarray, _head_name: str):
    """
        @ref equation (2)

        @example
            function2(compo, 'Eea') is Eea2
    """
    result = 0.0
    for i in range(_composition.shape[0]):
        result += _composition[i] / s1Data[i, findHeadIndex(_head_name)]
    return 1 / result


def functionD(_composition: np.ndarray, _head_name: str):
    """
        @ref equation (3)

        @example
            functionD(compo, 'Eea') is EeaD
    """
    x1 = function1(_composition, _head_name)
    result = 0.0
    for i in range(_composition.shape[0]):
        result += _composition[i] * ((s1Data[i, findHeadIndex(_head_name)] - x1)**2)
    return math.sqrt(result)


def functiond(_composition: np.ndarray, _head_name: str):
    """
        @ref equation (4)

        @example
            functiond(compo, 'Eea') is Eead
    """
    x1 = function1(_composition, _head_name)
    result = 0.0
    for i in range(_composition.shape[0]):
        result += _composition[i] * ((1 - s1Data[i, findHeadIndex(_head_name)] / x1)**2)
    return math.sqrt(result)


def functionHmix(_composition: np.ndarray):
    """
        @ref equation (5)

        @example
            functionHmix(compo) is Hmix
    """
    result = 0.0
    for i in range(_composition.shape[0]):
        for j in range(i + 1, _composition.shape[0]):
            result += deltaH[i, j] * _composition[i] * _composition[j]
    return 4 * result


def functionSmixR(_composition: np.ndarray):
    """
        @ref equation (6)

        @example
            functionSmixR(compo) is Smix/R
    """
    result = 0.0
    for i in range(_composition.shape[0]):
        if _composition[i] > 0.0:
            result += _composition[i] * math.log(_composition[i], math.e)
    return -result


def functionSmixkb(_composition: np.ndarray):
    """
        @ref equation (7)

        @example
            functionSmixkb(compo) is Smix/kb (Smix/k)
    """
    return (functiond(_composition, 'Rm') / 21.92)**2 * 10000


def functionPHSmis(_composition: np.ndarray):
    """
        @ref equation (8)

        @example
            functionPHSmis(compo) is PHSmis (PHS/k)
    """
    return functionHmix(_composition) * functionSmixkb(_composition)


def functionPHSmix(_composition: np.ndarray):
    """
        @ref equation (9)

        @example
            functionPHSmix(compo) is PHSmix (PHS/R)
    """
    return functionHmix(_composition) * functionSmixR(_composition)


def functionPHSS(_composition: np.ndarray):
    """
        @ref equation (10)

        @example
            functionPHSS(compo) is PHSS
    """
    return functionHmix(_composition) * functionSmixkb(_composition) * functionSmixR(_composition)


def functionf(_composition: np.ndarray, _head_name: str):
    """
        @ref equation (11)

        @example
            functionf(compo, 'sVEC') is fs
            functionf(compo, 'pVEC') is fp
            functionf(compo, 'dVEC') is fd
    """
    return function1(_composition, _head_name) / function1(_composition, 'VEC')


def functionV(_composition: np.ndarray, _head_name: str):
    """
        @ref equation (12)

        @example
            functionV(compo, 'Rm') is Vmm
            functionV(compo, 'Rc') is Vmc
    """
    result = 0.0
    for i in range(_composition.shape[0]):
        result += _composition[i] * 4 / 3 * math.pi * s1Data[i, findHeadIndex(_head_name)]**3
    return result


def get_features(raw):
    compo = parseCompoStr(raw['formula'])
    features = []

    for head in ['AN', 'Rc']:
        features.append(function1(compo, head))
        features.append(function2(compo, head))
        features.append(functionD(compo, head))
        features.append(functiond(compo, head))

    features.append(functionV(compo, 'Rc'))

    for head in ['Rm']:
        features.append(function1(compo, head))
        features.append(function2(compo, head))
        features.append(functionD(compo, head))
        features.append(functiond(compo, head))

    features.append(functionV(compo, 'Rm'))

    for head in ['AW', 'D', 'Eea', 'XM', 'XP', 'Gp', 'Hf', 'I1', 'I2', 'LP', 'Tm', 'P', 'Cp', 'K', 'VEC', 'Tb']:
        features.append(function1(compo, head))
        features.append(function2(compo, head))
        features.append(functionD(compo, head))
        features.append(functiond(compo, head))

    for head in ['sVEC', 'pVEC', 'dVEC']:
        features.append(function1(compo, head))

    for head in ['sVEC', 'pVEC', 'dVEC']:
        features.append(functionf(compo, head))

    for head in ['W']:
        features.append(function1(compo, head))
        features.append(function2(compo, head))
        features.append(functionD(compo, head))
        features.append(functiond(compo, head))

    features.append(functionHmix(compo))
    features.append(functionSmixR(compo))
    features.append(functionSmixkb(compo))
    features.append(functionPHSmix(compo))
    features.append(functionPHSmis(compo))
    features.append(functionPHSS(compo))
    
    return features

if __name__ == '__main__':
    chem_data = pd.read_csv('../负膨胀/xenonpy_input.csv')
    exp_data = pd.DataFrame()
    tqdm.pandas(desc="get feature in input_data")
    exp_data[column_names] = chem_data.progress_apply(get_features, axis=1, result_type="expand")
    exp_data.to_csv('./original_94.csv',columns=column_names, index=False)