
from fishVR.single_fish.core import SingleFishVR, SingleFishVRConfig



class SingleFishVR_CPU(SingleFishVR):
    def process(self, frame):

        # todo: i > 1!!!
        mask = self.tail_tracker.get_tail_mask(frame)  # get mask of tail for tracking
        tail_points = self.tail_tracker.segment_tail(mask, tail_points)
        tail_points_transformed = self.tail_tracker.transform_tail_points(tail_points)


        return tracking_results
    
