import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import simpson

amplitude0=150 
frequency0=1.667*pow(10,-4)
offset0=850
start_time0=16*60
duty_cycle0=0.33
rise_ratio0=0
fall_ratio0=0
T0=26
dt0=1.0
A0=60
R0=8
C0=35
sampling_rate=6000
sum_time=60*100
t = np.linspace(0, sum_time, sampling_rate)
amplitude1=100
duty_cycle1=0.396
start_time2=32*60
frequency2=2.778*pow(10,-3)
duty_cycle2=0.5
end_time2=88*60
end_time3=98*60
frequency3=1.111*pow(10,-3)


class EnhancedWaveGenerator:
    def __init__(self, amplitude=1, frequency=1, offset=0, start_time=0, end_time=None):
        """
        amplitude: 波形峰值幅度 (高电平值)
        frequency: 波形频率 (Hz)
        offset: 波形偏移量 (起始值/低电平值)
        start_time: 波形起始时间 (秒),此时间点前输出恒为offset
        end_time: 波形终止时间 (秒),此时间点后输出恒为offset
        """
        self.amplitude = amplitude
        self.frequency = frequency
        self.offset = offset
        self.start_time = start_time
        self.end_time = end_time  # 新增终止时间参数
    
    def generate_trapezoid(self, t, duty_cycle=0.5, rise_ratio=0.1, fall_ratio=0.1):
        """
        生成带起始和终止时间控制的梯形波
        t: 时间向量(numpy数组)
        duty_cycle: 平稳段占周期的比例 (0~1)
        rise_ratio: 上升段占周期的比例 (0~1)
        fall_ratio: 下降段占周期的比例 (0~1)
        """
        # 参数有效性验证
        assert 0 <= duty_cycle <= 1, "占空比必须在0~1之间"
        assert rise_ratio + duty_cycle + fall_ratio <= 1, "上升+平稳+下降比例总和不能超过1"
        if self.end_time is not None:
            assert self.end_time >= self.start_time, "终止时间必须大于或等于起始时间"
        
        # 初始化波形数组（默认全为offset）
        waveform = np.full_like(t, self.offset)
        
        # 创建有效时间掩码（仅当时间在[start_time, end_time]区间内计算波形）
        if self.end_time is None:
            active_mask = t >= self.start_time
        else:
            active_mask = (t >= self.start_time) & (t <= self.end_time)
        
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

def calculate_temperatures(P, T0=26, dt=1.0,A=A0,R=R0,C=C0):
    """
    p:heat_consumption_list
    T0:initial_temperature
    dt:delta_time
    A:over_heat_dissipation
    R:thermal resistance
    C:thermal_capacity
    """
    T = [T0]  # 初始化温度数组，包含初始温度
    
    # 遍历每个时间点，计算温度变化
    for i in range(0,len(P)-1):
        next_temp = (T[0]+P[i]/A)+(T[-1]-T[0]-P[i]/A )* math.exp(dt/(R*C*(-1)))
        T.append(next_temp)
    
    return T

#积分
def over_temperature_integration(x,y):
    """
    x:time_list
    y:temperature_list

    return the over_temperature_area

    """
    y1=[]
    for i in y:
        y1.append(i-35)
    return simpson(y1,x)

def heat_consumption_integration(x,y):
    """
    return heat_consumption_integration
    x:time_list
    y:heat_consumption
    """
    return simpson(y,x)


# # 创建波形生成器（振幅150，频率0.5Hz，起始值850，起始时间分别=1s和3s）
# wave_gen1 = AdvancedWaveGenerator(amplitude=150, frequency=0.5, offset=850, start_time=1.0)
# wave_gen2 = AdvancedWaveGenerator(amplitude=150, frequency=0.5, offset=850, start_time=3.0)

# t = np.linspace(0, 6, 1000)  # 0~6秒时长

# # 生成波形（占空比40%，上升/下降各占15%）
# y_delay1 = wave_gen1.generate_trapezoid(t, duty_cycle=0.4, rise_ratio=0.15, fall_ratio=0.15)
# y_delay2 = wave_gen2.generate_trapezoid(t, duty_cycle=0.4, rise_ratio=0.15, fall_ratio=0.15)

# # 可视化对比
# plt.figure(figsize=(10, 5))
# plt.plot(t, y_delay1, 'b-', linewidth=2, label='起始时间=1s')
# plt.plot(t, y_delay2, 'r--', linewidth=2, label='起始时间=3s')

# # 标记起始时间线
# plt.axvline(x=1.0, color='b', linestyle=':', alpha=0.7, label='起始时间线')
# plt.axvline(x=3.0, color='r', linestyle=':', alpha=0.7)

# plt.axhline(y=850, color='gray', linestyle='-', alpha=0.5, label='基线值850')
# plt.axhline(y=1000, color='green', linestyle='-', alpha=0.5, label='峰值1000')

# plt.title("带起始时间控制的梯形波")
# plt.xlabel("时间 (s)")
# plt.ylabel("幅值")
# plt.grid(True)
# plt.legend()
# plt.xlim(0,10)
# plt.ylim(800, 1050)
# plt.show()

def wave_generating(amplitude,frequency,offset,start_time,t, duty_cycle, rise_ratio,fall_ratio,end_time=None):
    """
    return a list of heat consumption
    """
    pwm_wave=EnhancedWaveGenerator(amplitude, frequency, offset, start_time,end_time)
    P = pwm_wave.generate_trapezoid(t, duty_cycle, rise_ratio,fall_ratio)
    return P

# 可视化对比
# plt.figure(figsize=(10, 5))
# plt.plot(t, P0, 'b-', linewidth=2, label='起始时间=1s')

# 标记起始时间线
# plt.axvline(x=1.0, color='b', linestyle=':', alpha=0.7, label='起始时间线')
# plt.axvline(x=3.0, color='r', linestyle=':', alpha=0.7)

# plt.axhline(y=850, color='gray', linestyle='-', alpha=0.5, label='基线值850')
# plt.axhline(y=1000, color='green', linestyle='-', alpha=0.5, label='峰值1000')

# plt.title("带起始时间控制的梯形波")
# plt.xlabel("time(s)")
# plt.ylabel("amplitude")
# plt.grid(True)
# plt.legend()
# plt.xlim(0,100*60)
# plt.ylim(700,1100)
# plt.show()


# plt.figure(figsize=(10, 5))
# plt.plot(t, T_0, 'b-', linewidth=2, label='no_pwm')
# plt.xlabel("time(s)")
# plt.ylabel("tempreture")
# plt.grid(True)
# plt.legend()
# plt.xlim(0,100*60)
# plt.ylim(20,45)
# plt.show()

P0=wave_generating(amplitude0,frequency0,offset0,start_time0,t,duty_cycle0,rise_ratio0,fall_ratio0)
P1=wave_generating(amplitude1,frequency0,offset0,start_time0,t,duty_cycle1,rise_ratio0,fall_ratio0)
P2=wave_generating(amplitude0,frequency2,offset0,start_time2,t,duty_cycle2,rise_ratio0,fall_ratio0,end_time2)
P3=wave_generating(amplitude1,frequency3,offset0,start_time0,t,duty_cycle2,rise_ratio0,fall_ratio0,end_time3)
temperature0=calculate_temperatures(P0,T0=26,dt=1.0,A=A0,R=R0,C=C0)
temperature1=calculate_temperatures(P1,T0=26,dt=1.0,A=A0,R=R0,C=C0)
temperature2=calculate_temperatures(P2,T0=26,dt=1.0,A=A0,R=R0,C=C0)
temperature3=calculate_temperatures(P3,T0=26,dt=1.0,A=A0,R=R0,C=C0)

# 温度曲线
plt.figure(figsize=(10, 5))
plt.plot(t, temperature0, 'black', linewidth=2, label='no_pwm')
plt.plot(t, temperature1, 'blue', linewidth=2, label='pwm_1')
plt.plot(t, temperature2, 'orange', linewidth=2, label='pwm_2')
plt.plot(t, temperature3, 'green', linewidth=2, label='pwm_3')
plt.xlabel("time(s)")
plt.ylabel("tempreture")
plt.grid(True)
plt.legend()
plt.xlim(0,100*60)
plt.ylim(20,45)
plt.show()

#热耗散曲线
plt.figure(figsize=(10, 5))
plt.plot(t, P0, 'black', linewidth=2, label='no_pwm')
plt.plot(t, P1, 'blue', linewidth=2, label='pwm_1')
plt.plot(t, P2, 'orange', linewidth=2, label='pwm_2')
plt.plot(t, P3, 'green', linewidth=2, label='pwm_3')
plt.xlabel("time(s)")
plt.ylabel("heat_consumption")
plt.grid(True)
plt.legend()
plt.xlim(0,100*60)
plt.ylim(800,1050)
plt.show()