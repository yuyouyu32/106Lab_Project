import math
import re

import numpy as np

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


if __name__ == '__main__':
    # parse composition to vector first
    # compo_str = 'Ag27Cu73'
    compo_str = 'Cu45Zr45Ag10'

    compo = parseCompoStr(compo_str)

    # features in full dataset
    george_features = [33.86, 32.34456573, 7.991270237, 0.236009162, 0.13551, 0.135274578,
                       0.005771473, 0.042590753, 0.010480795, 0.132309, 0.131916337,
                       0.007414123, 0.056036422, 0.009795063, 75.51348, 71.47577582,
                       19.678059, 0.260590016, 9.3758, 9.329446064, 0.683697565,
                       0.072921518, 1.25436, 1.2536511, 0.030189243, 0.024067447,
                       1.8446, 1.844557045, 0.008879189, 0.004813612, 1.9081,
                       1.908007701, 0.013318784, 0.006980129, 11, 11,
                       0, 0, 12.5775, 12.52622638, 0.776929051,
                       0.061771342, 7.6855, 7.684917744, 0.066593919, 0.008664878,
                       20.60596984, 20.59338451, 0.515809247, 0.02503203, 13.52791,
                       13.07445002, 2.722803354, 0.201273024, 1324.6032, 1322.257817,
                       54.53597977, 0.04117156, 4.27, 4.22832981, 0.443959458,
                       0.10397177, 24.6857, 24.67919838, 0.404003106, 0.016365876,
                       408.56, 408.1933371, 12.43086481, 0.030426045, 11,
                       11, 0, 0, 1, 0,
                       10, 0.090909091, 0, 0.909090909, 4.6838,
                       4.68364761, 0.026637567, 0.00568717, 1.5768, 0.58325884,
                       0.653521219, 0.919682539, 1.030472258, 0.601032054, 2727.15,
                       2714.74963, 177.583783, 0.065116984]

    # calculated by program
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

    # judge
    for i in range(len(features)):
        if abs(features[i] - george_features[i]) > 1e-6:
            print(i, features[i], george_features[i])

    print(features)
    print(len(features))