
"""
Test import from import data saved with XSens-sensors, through subclassing 'IMU_Base'
"""

# Author: Thomas Haslwanter

import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath, '..', 'src','skinematics'))

import unittest
import imus
from time import sleep
from sensors.xsens import XSens
from sensors.manual import MyOwnSensor
import pdb
from skinematics import quat, rotmat


def test_import_xsens():
    # Get data, with a specified input from an XSens system
    in_file = os.path.join(myPath, 'data', 'data_xsens.txt')

    sensor = XSens(in_file=in_file)
    rate = sensor.rate
    acc = sensor.acc
    omega = sensor.omega




    transfer_data = {'rate':sensor.rate,
                   'acc': sensor.acc,
                   'omega':sensor.omega,
                   'mag':sensor.mag}
    my_sensor = MyOwnSensor(in_file='My own 123 sensor.', in_data=transfer_data)
    pdb.set_trace()




def test_IMU_xsens(self):
    # Get data, with a specified input from an XSens system
    in_file = os.path.join(myPath, 'data', 'data_xsens.txt')
    my_IMU = XSens(in_file=in_file)


if __name__ == '__main__':
    test_import_xsens()
