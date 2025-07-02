import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import simpson
from dataclasses import dataclass

####################################波形有关的参数放在一起（内含4个待优化参数）#################################################
@dataclass
class WaveConfig:
    ############################待优化参数#########################
    amplitude: float #幅值
    frequency: float #频率
    start_time: float #起始时间
    duty_cycle: float = 0.5 #占空比

    ###########################非优化参数#########################
    offset: float =850#低电平值
    end_time: float = None #终止时间
    rise_ratio: float = 0 #上升沿比例
    fall_ratio: float = 0 #下降沿比例 两者为0表示方波
    label: str = ""  #不同波的名称

##########################################温度有关的参数放在一起（非优化参数）#################################################
@dataclass
class ThermalConfig:
    ##########非优化参数##########
    T0: float = 26.0 #起始温度
    dt: float = 1.0 #时间间隔
    A: float = 60.0 #整机散热能力
    R: float = 8.0 #热阻
    C: float = 35.0 #热容

######################################## 所有配置参数整合（内含材料给出的4个波形参数）##########################################
config = {
    ####################时间设置###################
    "time": {
        "sum_time": 60 * 100,  # 总模拟时间 (秒)
        "sampling_rate": 20000,  # 采样率
    },
    "wave_configs": [
        ##############基础热负载##############
        WaveConfig(
            amplitude=150,
            frequency=1.667e-4,
            offset=850,
            start_time=16 * 60,
            duty_cycle=0.345,
            label="no_pwm"
        ),
        ############### PWM 1###############
        WaveConfig(
            amplitude=120,
            frequency=1.667e-4,
            offset=850,
            start_time=16 * 60,
            duty_cycle=0.43,
            label="pwm_1"
        ),
        ############### PWM 2###############
        WaveConfig(
            amplitude=150,
            frequency=2.778e-3,
            offset=850,
            start_time=30 * 60,
            end_time=92 * 60,
            duty_cycle=0.5,
            label="pwm_2"
        ),
        ############### PWM 3###############
        WaveConfig(
            amplitude=120,
            frequency=1.111e-3,
            offset=850,
            start_time=18 * 60,
            end_time=95 * 60,
            duty_cycle=0.5,
            label="pwm_3"
        )
    ],
    "thermal": ThermalConfig()
}

###########################################输入波形参数生成波形##############################################################
class EnhancedWaveGenerator:
    def __init__(self, amplitude=1, frequency=1, offset=0, start_time=0, end_time=None):
        """
        初始化波形生成器
        
        参数:
            amplitude: 波形峰值幅度 (高电平值)
            frequency: 波形频率 (Hz)
            offset: 波形偏移量 (起始值/低电平值)
            start_time: 波形起始时间 (秒), 此时间点前输出恒为offset
            end_time: 波形终止时间 (秒), 此时间点后输出恒为offset
        """
        self.amplitude = amplitude
        self.frequency = frequency
        self.offset = offset
        self.start_time = start_time
        self.end_time = end_time
    
    def generate_trapezoid(self, t, duty_cycle=0.5, rise_ratio=0.1, fall_ratio=0.1):
        """
        生成带起始和终止时间控制的梯形波
        
        参数:
            t: 时间向量 (numpy数组)
            duty_cycle: 平稳段占周期的比例 (0~1)
            rise_ratio: 上升段占周期的比例 (0~1)
            fall_ratio: 下降段占周期的比例 (0~1)
            
        返回:
            梯形波形的数值数组
        """
        # 参数有效性验证
        assert 0 <= duty_cycle <= 1, "占空比必须在0~1之间"
        assert rise_ratio + duty_cycle + fall_ratio <= 1, "上升+平稳+下降比例总和不能超过1"
        if self.end_time is not None:
            assert self.end_time >= self.start_time, "终止时间必须大于或等于起始时间"
        
        # 初始化波形数组（默认全为offset）
        waveform = np.full_like(t, self.offset)
        
        # 创建有效时间掩码（仅当时间在[start_time, end_time]区间内计算波形）
        active_mask = (t >= self.start_time)
        if self.end_time is not None:
            active_mask &= (t <= self.end_time)
        
        # 仅对有效时间点计算波形
        t_active = t[active_mask] - self.start_time  # 时间归零处理
        period = 1 / self.frequency
        cycle_pos = t_active % period
        
        # 计算各阶段时间长度
        t_rise = period * rise_ratio
        t_high = period * duty_cycle
        t_fall = period * fall_ratio
        
        # 计算各阶段掩码
        mask_rise = (cycle_pos < t_rise)
        mask_high = (t_rise <= cycle_pos) & (cycle_pos < t_rise + t_high)
        mask_fall = (t_rise + t_high <= cycle_pos) & (cycle_pos < t_rise + t_high + t_fall)
        mask_low = (cycle_pos >= t_rise + t_high + t_fall)
        
        # 计算各阶段波形值
        waveform_active = np.zeros_like(t_active)
        waveform_active[mask_rise] = self.offset + self.amplitude * (cycle_pos[mask_rise] / t_rise)
        waveform_active[mask_high] = self.offset + self.amplitude
        waveform_active[mask_fall] = self.offset + self.amplitude * (1 - (cycle_pos[mask_fall] - t_rise - t_high) / t_fall)
        waveform_active[mask_low] = self.offset
        
        # 将计算后的波形值赋给原始波形数组
        waveform[active_mask] = waveform_active
        
        return waveform

#######################################利用递推计算温度##########################################################
def calculate_temperatures(P, thermal_cfg):
    """
    根据热功耗计算温度变化
    
    参数:
        P: 热功耗时间序列列表
        thermal_cfg: 热特性配置对象
        
    返回:
        温度时间序列列表
    """
    T0 = thermal_cfg.T0
    dt = thermal_cfg.dt
    A = thermal_cfg.A
    R = thermal_cfg.R
    C = thermal_cfg.C
    
    T = [T0]  # 初始化温度数组，包含初始温度
    
    # 遍历每个时间点，计算温度变化
    for i in range(len(P) - 1):
        next_temp = (T0 + P[i] / A) + (T[-1] - T0 - P[i] / A) * math.exp(-dt/(R * C))
        T.append(next_temp)
    
    return T

#########################################计算过温积分##########################################################
def over_temperature_integration(x, y):
    """
    计算温度超过35℃的面积积分
    
    参数:
        x: 时间数组
        y: 温度数组
        
    返回:
        超过35℃的温度积分面积
    """
    y1 = np.array(y) - 35  # 计算超过35℃的温差
    return simpson(np.maximum(y1, 0), x)  # 仅计算正温差（超过35℃部分）

##########################################热耗散功率##########################################################
def heat_consumption_integration(x, y):
    """
    计算热功耗的积分值
    
    参数:
        x: 时间数组
        y: 热功耗数组
        
    返回:
        热功耗的积分值
    """
    return simpson(y, x)

def main():
    ############## 准备时间数据##############
    time_cfg = config["time"]
    t = np.linspace(0, time_cfg["sum_time"], time_cfg["sampling_rate"])
    
    ################ 存储结果###############
    P_results = {}
    T_results = {}
    color_map = {
        "no_pwm": "black",
        "pwm_1": "blue",
        "pwm_2": "orange",
        "pwm_3": "green"
    }
    
    ####### 生成所有PWM波形和温度曲线################
    for wave_cfg in config["wave_configs"]:
        # 创建波形生成器
        wave_gen = EnhancedWaveGenerator(
            amplitude=wave_cfg.amplitude,
            frequency=wave_cfg.frequency,
            offset=wave_cfg.offset,
            start_time=wave_cfg.start_time,
            end_time=wave_cfg.end_time
        )
        
        # 生成梯形波（耗散功率）
        P = wave_gen.generate_trapezoid(
            t, 
            duty_cycle=wave_cfg.duty_cycle,
            rise_ratio=wave_cfg.rise_ratio,
            fall_ratio=wave_cfg.fall_ratio
        )
        
        # 计算温度变化
        temperature = calculate_temperatures(P, config["thermal"])
        
        # 存储结果
        P_results[wave_cfg.label] = P
        T_results[wave_cfg.label] = temperature
    
    ###################################################创建绘图##################################################
    plt.figure(figsize=(14, 10))
    
    ############################## 1. 温度曲线图##########################################
    plt.subplot(2, 1, 1)
    for label, T in T_results.items():
        plt.plot(t, T, color=color_map[label], linewidth=2, label=label)
    
    plt.axhline(y=35, color='red', linestyle='--', alpha=0.7, label='(35℃)')
    plt.title("the temperature of system")
    plt.xlabel("time (s)")
    plt.ylabel("temperature (℃)")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, time_cfg["sum_time"])
    plt.ylim(20, 45)
    
    ############################## 2. 热耗散曲线图##########################################
    plt.subplot(2, 1, 2)
    for label, P in P_results.items():
        plt.plot(t, P, color=color_map[label], linewidth=2, label=label)
    
    plt.title("the thermal consumption")
    plt.xlabel("time (s)")
    plt.ylabel("thermal consumption")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, time_cfg["sum_time"])
    plt.ylim(800, 1050)
    
    plt.tight_layout()
    plt.savefig("thermal_analysis_results.png", dpi=300)
    plt.show()
    ####################################################################################################################

    ##############################################计算并打印超温积分结果#############################################
    print("\n过温积分结果:")
    for label, T in T_results.items():
        over_temp_area = over_temperature_integration(t, T)
        heat_consumption = heat_consumption_integration(t, P_results[label])
        if label == "no_pwm":
            over_temp_area0 = over_temp_area
        print(f"{label}: {over_temp_area:.2f} °C·s (总热耗散: {heat_consumption:.2e}) 过温积分优化的百分比：{(100-over_temp_area/over_temp_area0*100):.2f}%")

#############################################执行函数#############################################
if __name__ == "__main__":
    main()

#################################################################################################################################
"""
尚未实现的部分：
1.确保热耗散积分为定值的模块
2.接收非优化参数的模块
3.动态调整优化参数的模块

"""