import io

import numpy as np
import soundfile
from flask import Flask, request, send_file, jsonify
import argparse
from inference import infer_tool, slicer

app = Flask(__name__)


@app.route("/wav2wav", methods=["POST"])
def wav2wav():
    request_form = request.form
    audio_path = request_form.get("audio_path", None)  # wav文件地址
    tran = int(float(request_form.get("tran", 0)))  # 音调
    spk = request_form.get("spk", 0)  # 说话人(id或者name都可以,具体看你的config)
    wav_format = request_form.get("wav_format", 'wav')  # 范围文件格式
    infer_tool.format_wav(audio_path)
    chunks = slicer.cut(audio_path, db_thresh=-40)
    audio_data, audio_sr = slicer.chunks2audio(audio_path, chunks)

    audio = []
    for (slice_tag, data) in audio_data:
        print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')

        length = int(np.ceil(len(data) / audio_sr * svc_model.target_sample))
        if slice_tag:
            print('jump empty segment')
            _audio = np.zeros(length)
        else:
            # padd
            pad_len = int(audio_sr * 0.5)
            data = np.concatenate([np.zeros([pad_len]), data, np.zeros([pad_len])])
            raw_path = io.BytesIO()
            soundfile.write(raw_path, data, audio_sr, format="wav")
            raw_path.seek(0)
            out_audio, out_sr, _ = svc_model.infer(spk, tran, raw_path)
            svc_model.clear_empty()
            _audio = out_audio.cpu().numpy()
            pad_len = int(svc_model.target_sample * 0.5)
            _audio = _audio[pad_len:-pad_len]

        audio.extend(list(infer_tool.pad_array(_audio, length)))
    out_wav_path = io.BytesIO()
    soundfile.write(out_wav_path, audio, svc_model.target_sample, format=wav_format)
    out_wav_path.seek(0)
    return send_file(out_wav_path, download_name=f"temp.{wav_format}", as_attachment=True)

@app.route("/update_model", methods=["POST"])
def update_model():
    global svc_model  # 声明全局变量，以便在函数内部修改它
    request_form = request.form
    model_name = request_form.get("model_name", None)  # 新模型文件地址
    config_name = request_form.get("config_name", None)  # 新config文件地址

    if model_name is None or config_name is None:
        return jsonify({"status": "error", "message": "缺少模型或配置文件路径"}), 400

    try:
        # 更新模型和配置文件
        svc_model = infer_tool.Svc(model_name, config_name)
        return jsonify({"status": "success", "message": "模型和配置文件已更新"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": f"更新模型时发生错误: {str(e)}"}), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process audio separation parameters')

    parser.add_argument('-mn', '--model_name', type=str, help='put your model name')
    parser.add_argument('-cn', '--config_name', type=str, help='put your config name')

    args = parser.parse_args()
    # model_name = "trained/G_流萤.pth"  # 模型地址
    # config_name = "trained/config.json"  # config地址
    svc_model = infer_tool.Svc(args.model_name, args.config_name)
    app.run(port=1145, host="127.0.0.1", debug=False, threaded=False)
