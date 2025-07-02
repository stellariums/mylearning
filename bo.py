import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
import os
from matplotlib.font_manager import FontProperties

# ===================== 参数配置 =====================
# 热模型参数
THERMAL_PARAMS = {
    'T0': 30,       # 起始温度 (°C)
    'A': 210,       # 整机散热能力 (W/°C)
    'R': 8,         # 热阻 (°C/W)
    'C': 50,        # 热容 (J/°C)
    'THRESHOLD': 34 # 超温阈值 (°C)
}

# 波形参数调整范围
PARAM_BOUNDS = [
    Real(0, 100, name='start_time'),    # 起始时间 (s)
    Real(200, 3000, name='amplitude'),     # 振幅 (W)
    Real(0.1, 100, name='period'),           # 周期 (s)
    Real(0.01, 0.9, name='duty_cycle')      # 占空比 (0-1)
]
# ==================================================

# 读取CSV数据
def read_csv_data(file_path):
    """从CSV文件中读取波形数据"""
    try:
        # 使用pandas读取CSV文件
        df = pd.read_csv(file_path)
        
        # 提取所需列 - 根据实际CSV列名修改
        t = df['Time(s)'].values
        combined_power = df['Combined_Power'].values
        
        # 确保数据完整
        if len(t) == 0 or len(combined_power) == 0:
            raise ValueError("CSV file missing required data columns")
            
        return t, combined_power
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None, None

# 计算温度响应的函数
def calculate_temperature(t, P):
    """计算温度响应"""
    T = np.zeros_like(t)
    T[0] = THERMAL_PARAMS['T0']
    over_temp_area = 0  # 超温面积 (°C·s)
    
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        # 热传导方程
        T[i] = (THERMAL_PARAMS['T0'] + P[i]/THERMAL_PARAMS['A']) + \
               (T[i-1] - THERMAL_PARAMS['T0'] - P[i]/THERMAL_PARAMS['A']) * \
               np.exp(-dt/(THERMAL_PARAMS['R']*THERMAL_PARAMS['C']))
        
        # 计算超温面积
        if T[i] > THERMAL_PARAMS['THRESHOLD']:
            over_temp_area += (T[i] - THERMAL_PARAMS['THRESHOLD']) * dt
    
    return T, over_temp_area/60  # 转换为°C·min

# 生成调整后的波形函数（保持总功不变）
def generate_adjusted_waveform(t, start_time, amplitude, period, duty_cycle, original_energy):
    """生成调整后的波形（保持总功不变）"""
    # 创建基础波形（方波）
    base_wave = np.zeros_like(t)
    for i, time in enumerate(t):
        # 计算在周期中的位置
        if time < start_time:
            base_wave[i] = 0
        else:
            phase = (time - start_time) % period
            if phase < period * duty_cycle:
                base_wave[i] = amplitude
    
    # 计算当前参数下的总功
    dt = t[1] - t[0] if len(t) > 1 else 0.1
    current_energy = np.sum(base_wave) * dt
    
    # 计算缩放因子以保持总功不变
    scale_factor = original_energy / current_energy if current_energy > 0 else 1
    
    return base_wave * scale_factor,scale_factor

# 执行贝叶斯优化
def optimize_waveform(t, original_wave, original_energy):
    """使用贝叶斯优化调整波形参数"""
    # 包装目标函数
    @use_named_args(dimensions=PARAM_BOUNDS)
    def wrapped_objective(start_time, amplitude, period, duty_cycle):
        # 生成调整后的波形（保持总功不变）
        adjusted_wave,scale_factor = generate_adjusted_waveform(t, start_time, amplitude, period, duty_cycle, original_energy)
        
        # 检查是否超过范围
        amplitude_param = next(param for param in PARAM_BOUNDS if param.name == 'amplitude')
        min_amplitude=amplitude_param.low
        max_amplitude=amplitude_param.high
        if np.max(adjusted_wave) > max_amplitude or np.max(adjusted_wave) < min_amplitude:
            return 1e10 
        
        # 计算温度响应
        _, over_temp_area = calculate_temperature(t, adjusted_wave)
        
        return over_temp_area
    
    # 执行优化
    result = gp_minimize(
        func=wrapped_objective,
        dimensions=PARAM_BOUNDS,
        n_calls=200,          # 评估次数
        n_random_starts=100,  # 随机起始点数量
        random_state=5,      # 随机种子
        n_jobs=-1             # CPU并行计算
    )
    
    # 提取最佳参数
    best_params = {
        'start_time': result.x[0],
        'amplitude': result.x[1],
        'period': result.x[2],
        'duty_cycle': result.x[3]
    }
    
    return best_params, result.fun

# 可视化优化结果
def visualize_results(t, original_wave, optimized_wave, best_params, min_area):
    """可视化优化结果"""
    # 计算原始波形和优化波形的温度响应
    T_original, original_area = calculate_temperature(t, original_wave)
    T_optimized, optimized_area = calculate_temperature(t, optimized_wave)

    # 计算优化后超温面积占原始的百分比
    percent_reduction = (original_area - min_area) / original_area * 100
    
    # 创建图表
    plt.figure(figsize=(15, 10))
    
    # 设置英文字体
    font = FontProperties()
    font.set_family('sans-serif')
    font.set_size(12)
    
    # 功率曲线比较
    plt.subplot(2, 1, 1)
    plt.plot(t, original_wave, 'b-', label='Original Waveform')
    plt.plot(t, optimized_wave, 'r-', label='Optimized Waveform')
    plt.xlabel('Time (s)', fontproperties=font)
    plt.ylabel('Power (W)', fontproperties=font)
    plt.title('Waveform Comparison', fontproperties=font)
    plt.grid(True, alpha=0.3)
    plt.legend(prop=font, loc='best', framealpha=0.5, shadow=True,bbox_to_anchor=(1.02, 1))
    
    # 温度响应比较
    plt.subplot(2, 1, 2)
    plt.plot(t, T_original, 'b-', alpha=0.7, label='Original Temperature')
    plt.plot(t, T_optimized, 'r-', label='Optimized Temperature')
    plt.axhline(THERMAL_PARAMS['THRESHOLD'], color='g', linestyle='--', 
                label=f'Threshold ({THERMAL_PARAMS["THRESHOLD"]}°C)')
    
    # 填充超温区域
    plt.fill_between(t, T_optimized, THERMAL_PARAMS['THRESHOLD'], 
                     where=(T_optimized > THERMAL_PARAMS['THRESHOLD']),
                     color='red', alpha=0.3,
                     label=f'Over-Temperature Area: {min_area:.2f} °C·min\nVS\nOriginal-Temperature Area:{original_area:.2f} °C·min\n({percent_reduction:.1f}% reduction)')
    
    plt.xlabel('Time (s)', fontproperties=font)
    plt.ylabel('Temperature (°C)', fontproperties=font)
    plt.title(f'Temperature Response (Min Over-Temperature Area: {min_area:.2f} °C·min)', fontproperties=font)
    plt.grid(True, alpha=0.3)
    
    # 优化图例显示 [8,10](@ref)
    plt.legend(prop=font, loc='lower right', framealpha=0.7, shadow=True, 
               fancybox=True, borderpad=1, labelspacing=1)
    
    plt.tight_layout()
    plt.savefig('optimization_results.png', dpi=300)
    plt.close()

# 主函数
def main():
    # 读取CSV数据 - 替换为您的实际文件路径
    csv_file = "thermal_simulation_data.csv"  # 示例文件名
    print(f"Reading CSV file: {csv_file}")
    t, combined_power = read_csv_data(csv_file)
    
    if t is None:
        print("Unable to read CSV data, please check file path and format")
        return
    
    # 使用组合功率作为原始波形
    original_wave = combined_power
    
    # 计算原始总功
    dt = t[1] - t[0] if len(t) > 1 else 0.1
    original_energy = np.sum(original_wave) * dt
    print(f"Original waveform total energy: {original_energy:.2f} J")
    
    # 计算原始波形的超温面积
    _, original_area = calculate_temperature(t, original_wave)
    print(f"Original waveform over-temperature area: {original_area:.2f} °C·min")
    
    # 执行优化
    print("Starting Bayesian optimization...")
    best_params, min_area = optimize_waveform(t, original_wave, original_energy)
    
    # 生成优化后的波形
    optimized_wave,scale_factor = generate_adjusted_waveform(
        t, 
        best_params['start_time'], 
        best_params['amplitude'], 
        best_params['period'], 
        best_params['duty_cycle'],
        original_energy
    )
    
    # 计算优化后波形的总功（验证总功不变）
    optimized_energy = np.sum(optimized_wave) * dt
    print(f"Optimized waveform total energy: {optimized_energy:.2f} J (Difference: {abs(original_energy-optimized_energy):.2e} J)")
    
    # 打印结果
    print("\n=== Optimization Results ===")
    print(f"Optimal Parameters:")
    print(f"  Start Time: {best_params['start_time']} s")
    print(f"  Amplitude: {best_params['amplitude']*scale_factor:.1f} W")
    print(f"  Period: {best_params['period']:.1f} s")
    print(f"  Duty Cycle: {best_params['duty_cycle']:.3f}")
    print(f"Min Over-Temperature Area: {min_area:.4f} °C·min (Reduction: {original_area-min_area:.4f} °C·min)")
    
    # 可视化结果
    visualize_results(t, original_wave, optimized_wave, best_params, min_area)
    print("\nVisualization results saved as 'optimization_results.png'")

if __name__ == "__main__":
    main()

