import numpy as np
from scipy.spatial.transform import Rotation as R
from ahrs.filters import (
    AngularRate, 
    Tilt, 
    AQUA, 
    Complementary, 
    FQA,
    Madgwick,
    Mahony,
)


class AHRSApplication:
    def __init__(self, filter_type, gyr=None, acc=None, frequency=50, **kw):
        self.filter_type = filter_type

        if self.filter_type == 'angularrate':
            self.filter = AngularRate(gyr=gyr, frequency=frequency, **kw)
        elif self.filter_type == 'aqua':
            self.filter = AQUA(gyr=gyr, acc=acc, frequency=frequency, **kw)
        elif self.filter_type == 'complementary':
            self.filter = Complementary(gyr=gyr, acc=acc, frequency=frequency)
        elif self.filter_type == 'madgwick':
            self.filter = Madgwick(gyr=gyr, acc=acc, frequency=frequency, **kw)
        elif self.filter_type == 'mahony':
            self.filter = Mahony(gyr=gyr, acc=acc, frequency=frequency, **kw)
        elif self.filter_type == 'tilt':
            self.filter = Tilt(acc=acc, representation='quaternion', **kw)
        else:
            raise ValueError('Invalid filter_type.', flush=True)
            
        self.Q = self.filter.Q.copy()
        self.Q = self.Q[:, [1, 2, 3, 0]]
    
    def get_Q(self):
        return self.Q.copy()
    
    # def apply_quaternion(self, data):
    #     return R.from_quat(self.get_Q()).apply(data, inverse=True)