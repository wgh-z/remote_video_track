# 停车检测类
from processing.regional_judgment import point_in_polygon, ploy_in_polygon
import time


# 停车类
class Car():
    def __init__(self, parking_area, car_id, car_center, car_outline, can_park_time):
        self.car_id = car_id  # 车辆ID
        self.car_center = car_center  # 车辆中心点
        self.car_outline = car_outline  # 车辆轮廓
        # self.parking_area = parking_area  # 停车区域
        self.can_park_time = can_park_time  # 可停车时间

        self.parking = False  # 是否已停车
        self.parking_standard = False  # 停车规范

        self.parking_time_start = 0  # 停车开始时间
        self.parking_time_now = 0  # 当前停车时间
        self.parking_time_end = 0  # 停车结束时间


    # 是否停车检测
    def is_park(self, car, parking_area):
        if point_in_polygon(self.car_center, parking_area):  # 车辆中心点在停车区域内，即认为停车
            self.parking = True  # 是否已停车
            self.parking_time_start = time.time()
        else:
            self.parking = False
            self.parking_time_end = time.time()
        if ploy_in_polygon(self.car_outline, parking_area):  # 车辆轮廓在停车区域内，即认为停车规范
            # car.parking = True
            self.parking_standard = True                                                                             
        # # 停车检测
        # if car.parking:
        #     if not self.parking:
        #         self.parking_time_start = time.time()
        #         self.parking = True
        # else:
        #     if self.parking:
        #         self.parking_time_end = time.time()
        #         self.parking_time = self.parking_time_end - self.parking_time_start
        #         self.parking = False
        #         if self.parking_time > 5:
        #             print("停车时间：", self.parking_time)
        #             # 保存图片
        #             cv2.imwrite("parking.jpg", parking_area)
                
    
    # # 停车规范检测
    # def is_standard(self):
    #     pass


    # # 停车计时
    # def parking_timer(self):
    #     pass



# 停车位类
class ParkingLot():
    def __init__(self, regin_id, lot_points):
        self.regin_id = regin_id  # 停车区域编号
        self.parking_area = lot_points  # 停车区域

        self.parking = False  # 已停有车辆
        self.parking_standard = False  # 停车是否规范

        # self.parking_time = 0
        self.parking_car_id = None  # 已停车车辆ID
        self.parking_time_start = 0.0  # 停车开始时间
        self.parking_time_now = 0.0  # 当前停车时间
        self.parking_time_end = 0.0  # 停车结束时间
    
    # 停车检测
    def is_parked(self, car_id, car_center):
        # if self.parking:
        #     print("已停有车辆")
        if point_in_polygon(car_center, self.parking_area):
            if self.parking_car_id is None:
                self.parking_car_id = car_id
                self.parking_time_start = time.time()  # 记录停车开始时间
            self.parking = True  # 车辆中心点在停车区域内，即认为停车
            
        else:
            if car_id == self.parking_car_id:
                self.parking_car_id = None
                self.parking_time_start = 0.0
            self.parking = False
        return self.parking


    # 停车规范检测
    def is_parked_standard(self, car_outline):
        if ploy_in_polygon(car_outline, self.parking_area):
            self.parking_standard = True
            # self.parking_car_id = car_id
        else:
            self.parking_standard = False
            # self.parking_car_id = None
        return self.parking_standard
    
    # 停车计时
    def parking_timer(self):
        if self.parking_time_start != 0.0:
            self.parking_time_end = time.time()
            self.parking_time_now = self.parking_time_end - self.parking_time_start
        return self.parking_time_now
