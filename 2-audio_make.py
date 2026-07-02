import os
import csv
import torch
import openai
import torchaudio
from moviepy import VideoFileClip
from pydub import AudioSegment
from openai import OpenAI
import re
import sys
sys.path.append('./CosyVoice')
sys.path.append('./CosyVoice/third_party/Matcha-TTS')
from CosyVoice.cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from CosyVoice.cosyvoice.utils.file_utils import load_wav
import torchaudio
from funasr import AutoModel
from pydub import AudioSegment
import subprocess
from pypinyin import pinyin, Style
from align.utils.text.text_encoder import is_sil_phoneme
from align.utils.tts.base_preprocess_ai3 import BasePreprocessor
from textgrid import TextGrid

# 设置DeepSeek API密钥
deepseek_api_key = ""

cosyvoice = CosyVoice2('./CosyVoice/pretrained_models/CosyVoice2-0___5B', load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)

# 创建文件夹
def create_directories_if_not_exist(directory_list):
    for dir_path in directory_list:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"已创建文件夹：{dir_path}")
        else:
            print(f"文件夹已存在：{dir_path}")

# 步骤1: 读取视频文件夹，拆分视频和音频
def split_video(video_path, video_name, output_audio_folder, output_video_folder):

    video = VideoFileClip(video_path)

    # 提取音频并保存
    audio = video.audio
    audio_path = os.path.join(output_audio_folder, f"{os.path.splitext(video_name)[0]}.mp3")
    audio.write_audiofile(audio_path)

    # 提取无声视频并保存
    video_no_audio = video.without_audio()
    video_no_audio_path = os.path.join(output_video_folder, f"{os.path.splitext(video_name)[0]}_no_audio.mp4")
    video_no_audio.write_videofile(video_no_audio_path, codec="libx264", audio_codec="aac")

    video.close()
    audio.close()


# 步骤2: 使用FunASR提取音频中的文本并获取时间戳
def transcribe_audio_with_timestamps(audio_path):
    model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                    vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                    punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                    # spk_model="cam++", spk_model_revision="v2.0.2",
                    )
    res = model.generate(audio_path, 
                batch_size_s=300, 
                hotword='魔搭')
    print(res[0]['text'])
    
    # 获取转录的完整文本
    transcribed_text = res[0]["text"]

    
    return transcribed_text

# 生成拼音
def generate_pinyin(text, preprocessor):
    # 去掉标点符号
    text = re.sub(r'[^\w\s]', '', text)

    *_, ph_gb_word = preprocessor.txt_to_ph(text)

    # 处理拼音，分开声母和韵母，并使用 _ 隔开
    ph_gb_word_nosil = " ".join(["_".join([p for p in w.split("_") if not is_sil_phoneme(p)])
                                    for w in ph_gb_word.split(" ") if not is_sil_phoneme(w)])
    print(ph_gb_word_nosil)

    return ph_gb_word_nosil
    

# 根据拼音去匹配对应的开始时间和结束时间
def find_interval_xmin_xmax(textgrid_file, search_text):
    # 加载TextGrid文件
    tg = TextGrid.fromFile(textgrid_file)
    
    # 假设文本标注在第一个层级的 interval 中
    tier = tg[0]  # 获取第一个层级
    
    # 遍历所有 intervals，匹配文本
    for interval in tier:
        if interval.mark == search_text:
            return interval.minTime, interval.maxTime
    
    # 如果没有找到匹配的文本，返回 None
    return None, None

# 步骤3: 调用DeepSeek API对文本进行修改，并记录修改前后的词语
def modify_text_with_deepseek(output_audio_folder, basename, original_text, preprocessor):
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个词语修改机器"},
            {"role": "user", "content": f"请修改下面的句子中的一个字或一个词，在修改前后字数不变的条件下使其原意发生改变，输出修改前后的字或词，不要修改数字，不需要多余解释，不需要有引号，回答格式为[修改前的字或词,修改后的字或词]：\n{original_text}"},
        ],
        stream=False
    )
    
    # 获取DeepSeek返回的修改前后词语和修改后的文本
    response_content = response.choices[0].message.content.strip()
    print('response_content: ',response_content)

    # 提取修改前后的词语
    old_word, new_word = response_content.strip('[]').split(',')
    old_word = old_word.strip()
    new_word = new_word.strip()

    print('old_word: ',old_word)
    print('new_word: ',new_word)

    # 防止bug，移动对应音频至临时文件夹单独处理
    command_cp = 'cp ' + os.path.join(output_audio_folder, f"{basename}.mp3") + ' ./align/mfa_model/mfa_inputs'
    os.system(command_cp)

    # 保存对应音素文件lab，用于MFA音频匹配
    original_pinyin = generate_pinyin(original_text, preprocessor)
    with open(os.path.join('./align/mfa_model/mfa_inputs', f"{basename}.lab"), 'w') as f_txt:
        f_txt.write(original_pinyin)

    # 调用MFA进行匹配，生成对应TextGrid文件，包含音素和对应时间戳
    command = 'mfa align -j 4 --clean ./align/mfa_model/mfa_inputs ./align/mfa_model/mfa_dict.txt ./align/mfa_model/mfa_model.zip ./align/mfa_model/mfa_outputs'
    os.system(command)
    
    # 用修改后的词替换原文中的词
    modified_text = original_text.replace(old_word, new_word)

    # 将旧词新词都转换为拼音，用于后续匹配
    old_pinyin = generate_pinyin(old_word, preprocessor)
    new_pinyin = generate_pinyin(new_word, preprocessor)
    print('old_pinyin: ', old_pinyin)
    print('new_pinyin: ', new_pinyin)

    # 根据拼音去找词的起止时间
    start_time, _ = find_interval_xmin_xmax(os.path.join('./align/mfa_model/mfa_outputs', f'{basename}.TextGrid'), old_pinyin.split()[0])
    _, end_time = find_interval_xmin_xmax(os.path.join('./align/mfa_model/mfa_outputs', f'{basename}.TextGrid'), old_pinyin.split()[-1])

    # 防止bug
    command_rm = 'rm -rf ./align/mfa_model/mfa_outputs ./align/mfa_model/mfa_inputs/*'
    os.system(command_rm)
    
    # 如果找到了原词，计算其持续时间
    if start_time and end_time:
        old_duration = end_time - start_time
        return old_word, new_word, old_pinyin, new_pinyin, modified_text, start_time, end_time, old_duration
    else:
        # 如果未找到原词，返回相应的错误信息
        print(f"Word '{old_word}' not found in the audio transcript.")
        return old_word, new_word, old_pinyin, new_pinyin, modified_text, None, None, None


# 步骤4: 使用语音合成模型合成修改后的音频，并返回音频数据
def synthesize_modified_audio(modified_text, transcribed_text, original_audio_path, output_generated_audio_folder):
    """
    生成修改后的音频并返回 AudioSegment 对象
    :param modified_text: 修改后的文本
    :param transcribed_text: 原始转录文本
    :param original_audio_path: 原始音频文件路径
    :return: 返回合成的音频（AudioSegment 对象）
    """
    # 载入原始音频文件对应的音频数据
    prompt_speech_16k = load_wav(original_audio_path, 16000)

    # 调用语音合成模型生成语音
    result = list(cosyvoice.inference_zero_shot(modified_text, transcribed_text, prompt_speech_16k, stream=False))
    generated_audio = result[0]['tts_speech']

    base_name = os.path.splitext(os.path.basename(original_audio_path))[0]
    generated_audio_path = os.path.join(output_generated_audio_folder, f"{base_name}.wav")
    torchaudio.save(generated_audio_path, generated_audio, cosyvoice.sample_rate)
    
    # 返回合成的音频地址
    return generated_audio_path


# 步骤5: 提取新生成音频中的新词部分
def extract_new_word_audio(generated_audio_path, basename, tran_text, new_word, new_pinyin, preprocessor, sample_rate=16000):
    
    # 防止bug，移动对应音频至临时文件夹单独处理
    command_cp = 'cp ' + generated_audio_path + ' ./align/mfa_model/mfa_inputs'
    os.system(command_cp)

    # 保存对应音素文件lab，用于MFA音频匹配
    generated_pinyin = generate_pinyin(tran_text, preprocessor)
    with open(os.path.join('./align/mfa_model/mfa_inputs', f"{basename}.lab"), 'w') as f_txt:
        f_txt.write(generated_pinyin)

    # 调用MFA进行匹配，生成对应TextGrid文件，包含音素和对应时间戳
    command = 'mfa align -j 4 --clean ./align/mfa_model/mfa_inputs ./align/mfa_model/mfa_dict.txt ./align/mfa_model/mfa_model.zip ./align/mfa_model/mfa_outputs'
    os.system(command)

    # 根据拼音去找词的起止时间
    start_time, _ = find_interval_xmin_xmax(os.path.join('./align/mfa_model/mfa_outputs', f'{basename}.TextGrid'), new_pinyin.split()[0])
    _, end_time = find_interval_xmin_xmax(os.path.join('./align/mfa_model/mfa_outputs', f'{basename}.TextGrid'), new_pinyin.split()[-1])

    # 防止bug
    command_rm = 'rm -rf ./align/mfa_model/mfa_outputs ./align/mfa_model/mfa_inputs/*'
    os.system(command_rm)
    
    # 如果找到了原词，计算其持续时间
    if start_time and end_time:
        word_duration = end_time - start_time

        # 将生成的音频数据转换为AudioSegment（pydub对象）
        generated_audio_segment = AudioSegment.from_wav(generated_audio_path)
        
        # 提取新词部分的音频片段（以毫秒为单位）
        start_ms = start_time * 1000  # 转换为毫秒
        end_ms = end_time * 1000      # 转换为毫秒
        segment = generated_audio_segment[start_ms:end_ms]  # 提取片段

        return segment, word_duration  # 返回音频片段和时长
    else:
        print(f"新词 '{new_word}' 的时间戳未找到。")
        return None, None  # 如果没有找到新词的时间戳，返回None


# 调整时长函数
def adjust_audio_duration(input_audio, modified_duration, original_duration):
    """
    将音频从 modified_duration 时长调整为 original_duration 时长
    :param input_audio: 输入音频片段
    :param modified_duration: 当前音频时长（需要被调整的原始时长）
    :param original_duration: 目标音频时长（调整后的新时长）
    :return: 调整后的音频
    """
    if original_duration <= 0:
        raise ValueError("original_duration 必须大于零")
    
    # 计算变速因子：当前时长 / 目标时长
    speed_change_factor = modified_duration / original_duration
    
    if speed_change_factor >= 1.0:
        # 加速处理（使用 speedup）
        return input_audio.speedup(playback_speed=speed_change_factor, crossfade=0)
    else:
        # 减速处理（通过修改帧率实现）
        new_frame_rate = int(input_audio.frame_rate * speed_change_factor)
        slowed_audio = input_audio._spawn(
            input_audio.raw_data,
            overrides={'frame_rate': new_frame_rate}
        ).set_frame_rate(input_audio.frame_rate)
        return slowed_audio

# 步骤6: 调整新词的持续时长并保存调整后的音频
def adjust_speed(audio_segment, original_duration, modified_duration, adjusted_audio_folder, original_audio_path):
    print("original_duration: ", original_duration)
    print("modified_duration: ", modified_duration)
    
    # 计算语速变化因子
    speed_change_factor = original_duration / modified_duration
    
    # 调整新词的音频播放速度，使其持续时长和原词一致
    seep_audio = adjust_audio_duration(audio_segment, modified_duration, original_duration)
    
    # 获取原音频文件名并替换扩展名为 '.wav'
    base_name = os.path.splitext(os.path.basename(original_audio_path))[0]
    adjusted_audio_path = os.path.join(adjusted_audio_folder, f"{base_name}.wav")
    
    # 导出调整后的音频文件
    seep_audio.export(adjusted_audio_path, format="wav")
    
    # 返回调整后的音频数据（AudioSegment对象）
    return seep_audio




# 步骤7: 提取原词音频片段，去除人声并提取背景音
def extract_background_audio(original_audio_path, start_time, end_time):
    # 加载音频
    if original_audio_path.endswith('.mp3'):
        original_audio = AudioSegment.from_mp3(original_audio_path)
    elif original_audio_path.endswith('.wav'):
        original_audio = AudioSegment.from_wav(original_audio_path)
    else:
        raise ValueError("不支持的文件格式")
    
    start_ms = start_time * 1000  # 毫秒
    end_ms = end_time * 1000      # 毫秒
    
    # 提取音频片段（原词部分）
    word_audio_segment = original_audio[start_ms:end_ms]
    
    # 提取开始前的音频片段
    pre_audio_segment = original_audio[:start_ms]
    
    # 提取结束后的音频片段
    post_audio_segment = original_audio[end_ms:]
    
    # 临时保存提取的原音频片段
    temp_file_path = "temp_word_audio.wav"
    word_audio_segment.export(temp_file_path, format="wav")
    
    # 使用 Demucs 分离人声和背景音
    # 运行 Demucs 命令行来分离音频
    command = ['demucs', '--two-stems=vocals', temp_file_path]  # 使用2stems模型（人声和伴奏分离）
    subprocess.run(command, check=True)  # 调用命令行并等待分离完成
    
    # 从分离后的音频中读取背景音（即伴奏）
    background_audio_path = 'separated/htdemucs/temp_word_audio/no_vocals.wav'
    
    # 加载音频
    if background_audio_path.endswith('.mp3'):
        background_audio_segment = AudioSegment.from_mp3(background_audio_path)
    elif background_audio_path.endswith('.wav'):
        background_audio_segment = AudioSegment.from_wav(background_audio_path)
    else:
        raise ValueError("不支持的文件格式")
    
    # 删除临时文件和分离后的文件
    os.remove(temp_file_path)
    os.remove(background_audio_path)  # 删除分离后的伴奏文件
    
    # 返回背景音频数据以及开始前和结束后的音频段
    return background_audio_segment, pre_audio_segment, post_audio_segment


# 步骤8: 合成背景音和修改后的音频片段
def combine_with_adjusted_background_audio(background_audio_segment, adjusted_audio_segment, adjusted_audio_folder, original_audio_path):
    # 合成背景音和修改后的音频片段
    combined_audio = adjusted_audio_segment.overlay(background_audio_segment)

    # 获取原音频文件名并替换扩展名为 '.wav'
    base_name = os.path.splitext(os.path.basename(original_audio_path))[0]
    combined_audio_path = os.path.join(adjusted_audio_folder, f"{base_name}.wav")
    
    # 保存合成后的音频到指定路径
    combined_audio.export(combined_audio_path, format="wav")

    # 返回合成后的音频和保存路径
    return combined_audio

# 步骤9：合成最终音频
def combine_all_audio(pre_audio_segment, combined_audio, post_audio_segment, final_audio_folder, original_audio_path):
    # 合并前段音频、合成音频和后段音频
    final_audio = pre_audio_segment + combined_audio + post_audio_segment

    # 获取原音频文件名并替换扩展名为 '.wav'
    base_name = os.path.splitext(os.path.basename(original_audio_path))[0]
    final_audio_path = os.path.join(final_audio_folder, f"{base_name}_final.wav")

    # 保存最终合成的音频到指定路径
    final_audio.export(final_audio_path, format="wav")

    # 返回最终合成后的音频和路径
    return final_audio, final_audio_path


# 步骤10: 保存到 CSV
def save_to_csv(data, csv_filename):
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['音频名称', '最后保存路径', '原完整文本', '修改后完整文本', '待修改词/字', '修改后词/字', '待修改词/字时间戳'])
        for row in data:
            writer.writerow(row)
          
# 主函数，遍历视频文件夹，执行所有操作
def process_videos(input_folder, preprocessor, output_audio_folder, output_video_folder, output_csv_file, output_generated_audio_folder, output_speed_adjusted_audio_folder, output_final_audio_folder):
    data = []
    with open('audio_error.txt', 'w') as error_log:
        
        for video_name in os.listdir(input_folder):
            if video_name.endswith(('.mp4', '.avi', '.mov')):  # 可根据实际情况修改
                try:
                    video_path = os.path.join(input_folder, video_name)
                    basename = os.path.splitext(video_name)[0]
                    audio_path = os.path.join(output_audio_folder, f"{basename}.mp3")

                    # 步骤1: 拆分视频和音频
                    split_video(video_path, basename, output_audio_folder, output_video_folder)

                    # 步骤2: 使用FunASR提取音频中的文本并获取时间戳
                    original_text = transcribe_audio_with_timestamps(audio_path)

                    # 步骤3: 修改文本
                    old_word, new_word, old_pinyin, new_pinyin, modified_text, start_time_old, end_time_old, duration_old = modify_text_with_deepseek(output_audio_folder, basename, original_text, preprocessor)

                    # 步骤4: 使用语音合成模型合成修改后的音频
                    generated_audio_path = synthesize_modified_audio(modified_text, original_text, audio_path, output_generated_audio_folder)

                    # 步骤5: 提取修改词/字片段音频
                    tran_text = transcribe_audio_with_timestamps(generated_audio_path)
                    new_word_audio, duration_new = extract_new_word_audio(generated_audio_path, basename, tran_text, new_word, new_pinyin, preprocessor)
                    
                    # 步骤6: 调整语速并保存调整后的音频            
                    seep_audio = adjust_speed(new_word_audio, duration_old, duration_new, output_speed_adjusted_audio_folder, audio_path)

                    # 步骤7: 去除人声，保存背景音           
                    background_audio_segment, pre_audio_segment, post_audio_segment = extract_background_audio(audio_path, start_time_old, end_time_old)

                    # 步骤8: 合成背景音和修改后的音频片段
                    combined_audio = combine_with_adjusted_background_audio(background_audio_segment, seep_audio, output_noise_adjusted_audio_folder, audio_path)

                    # 步骤9：合成最终音频
                    final_audio, final_audio_path = combine_all_audio(pre_audio_segment, combined_audio, post_audio_segment, output_final_audio_folder, audio_path)

                    # 步骤10: 保存信息到 CSV
                    data.append([audio_path, final_audio_path, original_text, modified_text, old_word, new_word, f'"{start_time_old}-{end_time_old}"'])
                    

                except Exception as e:
                    # 防止bug
                    command_rm = 'rm -rf ./align/mfa_model/mfa_outputs ./align/mfa_model/mfa_inputs/*'
                    os.system(command_rm)
                    error_log.write(f"处理文件 {video_name} 时发生错误: {str(e)}\n")
                    continue  # 跳过当前迭代，继续下一个文件的处理
    #  写入 CSV
    save_to_csv(data, output_csv_file)

# 调用主函数
if __name__ == "__main__":
    preprocessor = BasePreprocessor()
    input_folder = "./data/temp_1000/input_videos/"  # 视频文件夹路径
    output_audio_folder = "./data/temp_1000/output_audio"  # 音频文件夹路径
    output_video_folder = "./data/temp_1000/output_video"  # 无声视频文件夹路径
    output_csv_file = "./data/temp_1000/output.csv"  # 输出 CSV 文件路径
    output_generated_audio_folder = "./data/temp_1000/adjust_audio"
    output_noise_adjusted_audio_folder = "./data/temp_1000/output_noise_adjusted_audio"  # 有噪音的修改片段音频文件夹路径
    output_speed_adjusted_audio_folder = "./data/temp_1000/output_speed_adjusted_audio"  # 调整语速后的音频文件夹路径
    output_final_audio_folder = "./data/temp_1000/output_final_audio"  # 最后生成音频文件夹路径
    directories = [output_audio_folder, output_video_folder, output_generated_audio_folder, output_noise_adjusted_audio_folder, output_speed_adjusted_audio_folder, output_final_audio_folder]
    create_directories_if_not_exist(directories)

    process_videos(input_folder, preprocessor, output_audio_folder, output_video_folder, output_csv_file, output_generated_audio_folder, output_speed_adjusted_audio_folder, output_final_audio_folder)
