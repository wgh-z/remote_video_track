class Timer:
    """
    这是一个计时器类，用于将(data_dict.keys() - data_dict.keys() ∩ data_set)的元素延迟delay_count次后删除
    """
    def __init__(self, delay_count:int = 30):
        self.delay_count = delay_count
    
    def add_delay(self, data_dict:dict, id:int):
        data_dict[id] = self.delay_count
        return data_dict

    def __call__(self, data_set:set, data_dict:dict):
        temp_dict = data_dict.copy()
        for element in data_dict.keys():
            if element not in data_set:
                temp_dict[element] -= 1
                if temp_dict[element] == 0:
                    del temp_dict[element]
            else:
                data_dict[element] = self.delay_count  # 重置计时器
        return temp_dict