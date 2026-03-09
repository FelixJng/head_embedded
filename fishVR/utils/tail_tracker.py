import cv2
import numpy as np
from numba import njit
from fishVR.core import FishVRConfig


class TailTracker:

    def __init__(self, config: FishVRConfig):
        self.config = config


    def get_tail_mask(self, cropped: np.ndarray) -> np.ndarray:

        kernel_size = self.config.kernel_size
        n_interations = self.config.n_interations

        # cropped = crop_along_direction(frame, (centroid_x, centroid_y), angle, crop_w=200, crop_h=400)
        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        # cropped_gray = cv2.GaussianBlur(cropped_gray, (3,3), 0) 
        cropped_gray = 255 - cropped_gray  # invert grayscale so that tail is bright on dark background

        edges = cv2.Canny(cropped_gray, 50, 150)
        kernel = np.ones((kernel_size,kernel_size), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=n_interations)
        mask = np.zeros_like(cropped_gray)
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

        return mask


    @njit
    def next_segment(self, im, x0, y0, dx, dy, r, segment_length, scalar_product_threshold=0.7):
        '''
        function to segment the tail, it gives the next point by calculating the center of mass in a circular window
        of radius r around (x0+dx,y0+dy) and taking the point at a distance |dx,dy| form (x0,y0) that is closest to that center of mass,
        returns nans in case of problems

        adapted from Demarchi et al. 2025, PNAS
        '''
        r2=r**2
        y_max,x_max=im.shape
        xs=min(max(int(round(x0+dx-r)),0),x_max)
        xl=min(max(int(round(x0+dx+r)),0),x_max)
        ys=min(max(int(round(y0+dy-r)),0),y_max)
        yl=min(max(int(round(y0+dy+r)),0),y_max)
        if xs==xl or ys==yl:
            return np.nan,np.nan,np.nan,np.nan
        acc=0.0
        acc_x=0.0
        acc_y=0.0
        for x in range(xs,xl):
            for y in range(ys,yl):
                lx=(x0+dx-x)**2
                ly=(y0+dy-y)**2
                if lx+ly<=r2:
                    acc_x+=x*im[y,x]
                    acc_y+=y*im[y,x]
                    acc+=im[y,x]
        if acc==0:
            return np.nan,np.nan,np.nan,np.nan
        dx_new=acc_x/acc-x0
        dy_new=acc_y/acc-y0
        norm=np.sqrt(dx_new**2+dy_new**2)
        if norm==0:
            return np.nan,np.nan,np.nan,np.nan
        dx_new=dx_new/norm*segment_length
        dy_new=dy_new/norm*segment_length
        if (dx*dx_new+dy*dy_new)/segment_length**2<scalar_product_threshold:
            return np.nan,np.nan,np.nan,np.nan
        return x0+dx_new,y0+dy_new,dx_new,dy_new

    def segment_tail(self, frame_p, tail_points, vr_dict):
        '''
        segment the tail by repeatedly calling the next_segment function,
        in case of nans it linearly extrapolate to get the remaining tail points

        adapted from Demarchi et al. 2025, PNAS
        '''

        start_x,start_y=vr_dict['initial_pos']
        dx,dy=vr_dict['initial_delta']

        for s in range(1,vr_dict['n_segments']+1):
            start_x,start_y,dx,dy=self.next_segment(frame_p,start_x,start_y,dx,dy,vr_dict['window_radius'],vr_dict['segment_length'])
            if np.isnan(start_x):
                # print(f'Error at segment {s}')
                tail_points[s:]=np.nan
                for sp in range(s,vr_dict['n_segments']+1):
                    if sp==1:
                        tail_points[1]=vr_dict['initial_pos']+vr_dict['initial_delta']
                    else:
                        tail_points[sp]=tail_points[sp-1]+tail_points[sp-1]-tail_points[sp-2]
                break
            tail_points[s]=[start_x,start_y]
        
        return tail_points