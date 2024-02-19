#coding:utf-8

def RGB_to_Hex(rgb):
    # rgb = tmp.split(',')  # 将RGB格式划分开来
    # rgb.reverse()
    strs = '#'
    for i in rgb:
        num = int(i)  # 将str转int
        # 将R、G、B分别转化为16进制拼接转换并大写
        strs += str(hex(num))[-2:].replace('x', '0').upper()

    return strs


if __name__ == '__main__':
    print(RGB_to_Hex((0,255,255)))
