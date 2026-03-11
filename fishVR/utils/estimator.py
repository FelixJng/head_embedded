import numpy as np
from fishVR.core import FishVRConfig


class Estimator:

    def __init__(self, config: FishVRConfig = FishVRConfig()):
        self.config = config

    def estimate(self):
        pass
    
    def calculate_lighthill_force_torque(self, tail_points_list):
        '''
        it calculates metrics proportional to force and torque according to lighthill's model,
        it only considers the components of the speed of the last segment, perpendicular and parallel to the segment themself
        '''
        last_segment_center=(tail_points_list[-1][-2]+tail_points_list[-1][-1])/2
        last_segment_center_d1=(tail_points_list[-2][-2]+tail_points_list[-2][-1])/2
        last_segment_center_d2=(tail_points_list[-3][-2]+tail_points_list[-3][-1])/2
        vel_tail_tip=0.5*self.config.acquisition_rate*(last_segment_center-last_segment_center_d2)
        tip_direction=tail_points_list[-2][-1]-tail_points_list[-2][-2]
        ns_perp=np.transpose((-tip_direction[1],tip_direction[0])/np.sqrt(tip_direction[0]**2+tip_direction[1]**2))
        ns_par=np.transpose((-tip_direction[0],-tip_direction[1])/np.sqrt(tip_direction[0]**2+tip_direction[1]**2))
        vel_perp=np.sum(vel_tail_tip*ns_perp)

        force_lighthill=[-vel_perp*vel_tail_tip[1]-0.5*vel_perp**2*ns_par[0],vel_perp*vel_tail_tip[0]-0.5*vel_perp**2*ns_par[1]]
        torque_lighthill=np.cross(last_segment_center_d1,force_lighthill)

        return force_lighthill,torque_lighthill

    def raise_to_power(self, signal, exponent):
        '''
        raise an array to a non integer power by first taking the absolute values and then remultiplying by the signs
        '''
        return np.sign(signal)*np.abs(signal)**exponent

    def lowpass_filter(self, signal_now, signal_lowpass):
        '''
        it implements an online lowpass filter where signal_now is the last instance of the actual signal and
        signal_lowpass is the last instance of the lowpassed signal
        '''
        if self.config.tau_lp_now==0:
            return signal_now
        else:
            beta_lp=np.exp(-1/(self.config.acquisition_rate*self.config.tau_lp_now))
            return  signal_lowpass*beta_lp+signal_now*(1-beta_lp)