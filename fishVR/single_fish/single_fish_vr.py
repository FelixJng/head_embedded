
from fishVR.single_fish.core import SingleFishVR, SingleFishVRConfig



class SingleFishVR_CPU(SingleFishVR):
    def process(self, frame, first_frame=False):
        
        tail_points_list = self.state_handler.state.tail_points_list
        tail_points = self.state_handler.state.tail_points

        # if not first_frame:
        #     pass

        # todo: i > 1!!!
        tail_points_transformed = self.tail_tracker.track_tail(frame, tail_points)  # track tail and update state
        tail_points_list.append(np.array(tail_points_transformed))

        
        force_lighthill,torque_lighthill=calculate_lighthill_force_torque(tail_points_list, vr_dict['acquisition_rate']) 

        return res
    


    
