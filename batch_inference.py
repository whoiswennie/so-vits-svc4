import logging
import os
import shutil
from tqdm import tqdm
import soundfile
from inference import infer_tool
from inference.infer_tool import Svc
from spkmix import spk_mix_map

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")

def process_all_audio(model_path, config_path, trans, spk_list, clip=0, device=None, f0_predictor="pm"):
    # 初始化模型 - 简化版本
    svc_model = Svc(model_path,
                    config_path,
                    device=device,
                    cluster_model_path="")
    
    # 确保结果目录存在
    infer_tool.mkdir(["raw", "results"])
    
    # 获取raw文件夹中的所有音频文件
    raw_files = [f for f in os.listdir("raw") if f.lower().endswith(('.wav', '.flac', '.mp3', '.ogg'))]
    
    if not raw_files:
        print("raw文件夹中没有找到音频文件")
        return
    
    # 创建进度条
    pbar = tqdm(raw_files, desc="处理音频文件")
    
    for clean_name in pbar:
        pbar.set_description(f"正在处理: {clean_name}")
        raw_audio_path = f"raw/{clean_name}"
        
        try:
            # 格式化音频文件
            infer_tool.format_wav(raw_audio_path)
            
            for spk in spk_list:
                # 设置推理参数
                kwarg = {
                    "raw_audio_path": raw_audio_path,
                    "spk": spk,
                    "tran": trans,
                    "slice_db": -40,
                    "cluster_infer_ratio": 0,
                    "auto_predict_f0": False,
                    "noice_scale": 0.4,
                    "pad_seconds": 0.5,
                    "clip_seconds": clip,
                    "lg_num": 0,
                    "lgr_num": 0.75,
                    "f0_predictor": f0_predictor  # 使用传入的f0_predictor参数
                }
                
                # 执行推理
                audio = svc_model.slice_inference(**kwarg)
                
                # 保存结果
                key = f"{trans}key"
                res_path = f'results/{clean_name}_{key}_{spk}_sovits_{f0_predictor}.flac'
                soundfile.write(res_path, audio, svc_model.target_sample, format='flac')
                
                # 清空缓存
                svc_model.clear_empty()
            
            # 处理完成后移动原文件到processed文件夹
            processed_dir = "raw/processed"
            os.makedirs(processed_dir, exist_ok=True)
            shutil.move(raw_audio_path, f"{processed_dir}/{clean_name}")
            
        except Exception as e:
            tqdm.write(f"处理文件 {clean_name} 时出错: {str(e)}")
            # 移动出错文件到error文件夹
            error_dir = "raw/error"
            os.makedirs(error_dir, exist_ok=True)
            shutil.move(raw_audio_path, f"{error_dir}/{clean_name}")
    
    print("所有音频文件处理完成")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='批量音频推理')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('-c', '--config_path', type=str, required=True, help='配置文件路径')
    parser.add_argument('-t', '--trans', type=int, default=0, help='音高调整')
    parser.add_argument('-s', '--spk_list', type=str, nargs='+', required=True, help='合成目标说话人名称')
    parser.add_argument('-cl', '--clip', type=float, default=0, help='音频强制切片时长')
    parser.add_argument('-d', '--device', type=str, default=None, help='推理设备')
    parser.add_argument('-f0p', '--f0_predictor', type=str, default="pm", 
                       help='选择F0预测器,可选择crepe,pm,dio,harvest,rmvpe,fcpe默认为pm(注意：crepe为原F0使用均值滤波器)')
    
    args = parser.parse_args()
    
    # 验证f0_predictor参数是否有效
    valid_f0_predictors = ["crepe", "pm", "dio", "harvest", "rmvpe", "fcpe"]
    if args.f0_predictor not in valid_f0_predictors:
        raise ValueError(f"无效的F0预测器: {args.f0_predictor}, 请选择其中之一: {', '.join(valid_f0_predictors)}")
    
    process_all_audio(
        model_path=args.model_path,
        config_path=args.config_path,
        trans=args.trans,
        spk_list=args.spk_list,
        clip=args.clip,
        device=args.device,
        f0_predictor=args.f0_predictor
    )