import numpy as np
from tinyekf import EKF
import matplotlib.pyplot as plt
import csv
from decimal import Decimal

''' notes
    ADXL is noisier than MPU
    mpu_az: g around 10.1 m/s^2
    adxl_ay: g bleh around -10 -11 m/s^2
    baro_press in mbar i believe
    baro_temp in C
    altitudeMSL in m
'''

file_name = "demo2_pyxflash.csv"
TEMP_C = 33.145     # THIS SHOULD NOT BE HARD CODED BUT I WAS LAZY I'M SORRY

def press_to_alt(press_mbar):   # currently only for <11km, <mach1
    press_pa = press_mbar * 100.0
    temp_k = 273.15 + TEMP_C
    Tb = temp_k     # temp at sea level
    Lb = -0.0065    # std temp lapse rate
    Pb = 101325.0   # std press sea level
    R = 8.31432     # universal gas const
    g0 = 9.80665
    M = 0.0289644   # molar mass of earth air
    alt = Tb/Lb*((press_pa/Pb)**(-R*Lb/(g0*M)) - 1)
    return alt

def alt_to_press(alt):     # <11km, <mach1
    temp_k = 273.15 + TEMP_C
    Tb = temp_k     # temp at sea level
    Lb = -0.0065    # std temp lapse rate
    Pb = 101325.0   # std press sea level
    R = 8.31432     # universal gas const
    g0 = 9.80665
    M = 0.0289644   # molar mass of earth air
    press_pa = (alt*Lb/Tb + 1)**(g0*M/(-R*Lb)) * Pb
    press_mbar = press_pa / 100.0
    deriv = (alt*Lb/Tb + 1)**(g0*M/(-R*Lb) - 1)*(g0*M/(-R*Lb))*Lb/Tb*Pb/100.0
    return press_mbar, deriv

class alt_EKF(EKF):
    def __init__(self):
        EKF.__init__(self, 3, 4, rval=.1)   # covariance arbitrary rn lol

    def f(self, x):
        # dt = 27-28 ms
        # s = s + v*dt + a*dt^2/2
        # v = v + a*dt
        # a = a
        dt = 0.0275 # in seconds
        x = np.array(x)
        transition = np.array([[1.0, dt, 0.5*dt*dt],[0.0, 1.0, dt],[0.0, 0.0, 1.0]])
        x_hat = transition.dot(x)
        return x_hat, transition

    def h(self, x):
        alt = x[0]
        v = x[1]
        a = x[2]

        gps_alt = alt
        adxl = a
        mpu = a
        baro, db  = alt_to_press(alt)

        sensors = np.array([gps_alt, adxl, mpu, baro])

        H = np.array([[1.0, 0, 0],[0, 0, 1.0],[0, 0, 1.0],[db, 0, 0]])

        return sensors, H

estimates = []
baro_alts = []
gps_alts = []
row_count = 0
altimeter = alt_EKF()

with open('../demo2_pyxflash.csv') as data_file:
    file_reader = csv.reader(data_file, delimiter=',')
    for row in file_reader:
        if (row_count == 0):
            row_count += 1
            continue
        row_count += 1

        gps = float(row[5])
        adxl = float(row[2])
        mpu = float(row[1])
        baro = float(row[3])
        est = altimeter.step((gps, adxl, mpu, baro))[0]

        gps_alts.append(gps)
        baro_alts.append(press_to_alt(baro))
        estimates.append(est)

steps = len(estimates)
time = [0.0275*i for i in range(steps)]
plt.plot(time, estimates)
plt.plot(time, gps_alts)
plt.plot(time, baro_alts)

plt.show()
