import os
import re
import cv2
import subprocess
import logging
from tqdm import tqdm
import numpy as np
from moviepy import VideoFileClip
from funasr import AutoModel
import argparse
import random
import dlib
from insightface.app import face_analysis
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor

# 全局变量，用于保存模型实例
engine = None

def init_worker(model_name='buffalo_s', ctx_id=0, det_size=(320, 320)):
    global engine
    global detector
    global predictor
    global model
    # 加载 dlib 的人脸检测器和关键点预测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
    engine = FaceRecognitionEngine(model_name=model_name, ctx_id=ctx_id, det_size=det_size)
    # 加载模型并生成识别结果
    model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                      vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                      punc_model="ct-punc-c", punc_model_revision="v2.0.4")
    print(f"[PID: {os.getpid()}] 模型已初始化")

# -----------------------------
# 断点续跑相关函数
# -----------------------------

def read_completed_list(completed_file):
    """读取已完成视频列表"""
    if not os.path.exists(completed_file):
        return set()
    with open(completed_file, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f)


def write_completed_video(completed_file, video_name):
    """写入已完成视频名称"""
    with open(completed_file, 'a', encoding='utf-8') as f:
        f.write(video_name + '\n')

# -----------------------------
# 配置日志
# -----------------------------
def setup_logger(log_file):
    logger_name = f"video_processor_{hash(log_file)}"  # 确保唯一性
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # 防止重复添加 handler
    if logger.handlers:
        return logger, None  # 已经初始化过了，直接返回

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 控制台输出
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件输出
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger, fh  # 返回 logger 和 handler，用于后续关闭

def process_single_video(video_file, input_dir, result_dir, log_dir, checkpoint_file):
    global engine

    video_path = os.path.join(input_dir, video_file)
    base_name = os.path.splitext(os.path.basename(video_file))[0]
    video_result_dir = os.path.join(result_dir, base_name)
    log_file = os.path.join(log_dir, f"{base_name}.log")

    print(f"\n🚀 开始处理视频：{video_file}")
    logger, file_handler = setup_logger(log_file)
    logger.info(f"开始处理视频：{video_file}")

    try:
        process_video(video_path, result_dir, logger)
        write_completed_video(checkpoint_file, video_file)
    except Exception as e:
        logger.error(f"处理失败：{e}", exc_info=True)
        print(f"❌ 处理失败：{e}")
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()

# -----------------------------
# 辅助函数：供每个进程处理自己的视频块
# -----------------------------
def process_chunk(chunk, input_dir, result_dir, log_dir, checkpoint_file):
    for video_file in chunk:
        process_single_video(video_file, input_dir, result_dir, log_dir, checkpoint_file)

# -----------------------------
# Step 1: 提取音频
# -----------------------------
def extract_audio(video_path, output_audio_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_audio_path)
    video.close()
    return output_audio_path


# -----------------------------
# Step 2: 分句并获取时间戳
# -----------------------------
def split_sentences_with_positions(text, cleaned_text, timestamps):
    punctuations = set('。！？；，、：＂＇（）《》〈〉【】「」『』…—～﹏｟｠「・」')

    # 建立原始文本到 clean 版本的位置映射
    char_map = []
    for i, c in enumerate(text):
        if c not in punctuations:
            char_map.append(i)

    # 找出原始文本中标点位置
    sentence_boundaries = [i for i, c in enumerate(text) if c in punctuations]

    if not sentence_boundaries:
        raise ValueError("未检测到任何标点，无法进行语义分句")

    # 找出 clean 版本中的句子结束位置
    clean_sentence_ends = []
    j = 0  # char_map 的指针
    for i in range(len(text)):
        if text[i] in punctuations:
            if j > 0:  # 只有当已经有字符被映射才考虑
                clean_sentence_ends.append(j - 1)  # clean 版本的最后一个字符
        else:
            j += 1  # 非标点字符才移动指针

    if not clean_sentence_ends:
        raise ValueError("未找到有效的分句结尾位置，请检查标点分布和映射关系")

    # 添加最后一段
    if clean_sentence_ends[-1] != len(cleaned_text) - 1:
        clean_sentence_ends.append(len(cleaned_text) - 1)

    # 生成时间段落
    segments = []
    prev_end = 0
    for end in clean_sentence_ends:
        if end >= len(timestamps):
            continue
        start_time = timestamps[prev_end][0] / 1000
        end_time = timestamps[end][1] / 1000
        duration = end_time - start_time
        segments.append((start_time, end_time, duration))
        prev_end = end + 1

    return segments


# -----------------------------
# Step 3: 合并/拆分句子以满足时长要求
# -----------------------------
def optimize_segments(segments, min_duration=3.0, max_duration=7.0):
    optimized = []
    current_start = None
    current_end = None
    current_duration = 0.0

    for start, end, dur in segments:
        if current_start is None:
            # 初始化第一个片段
            current_start = start
            current_end = end
            current_duration = dur
        else:
            tentative_duration = end - current_start

            if tentative_duration >= min_duration:
                # 当前累积已经满足最小长度，立即保存
                optimized.append((current_start, end))
                # 开启新的合并段
                current_start = start
                current_end = end
                current_duration = dur
            elif tentative_duration <= max_duration:
                # 还未达到最小长度，继续合并
                current_end = end
                current_duration = tentative_duration
            else:
                # 累计超出了最大长度，但还没满足最小长度（罕见情况）
                # 此时也强制保存这个段落
                optimized.append((current_start, end))
                # 开启新段
                current_start = start
                current_end = end
                current_duration = dur

    # 处理最后剩余的部分
    if current_start is not None and current_duration >= min_duration:
        optimized.append((current_start, current_end))

    return optimized


# -----------------------------
# Step 4: 切割视频
# -----------------------------
def cut_video_into_clips(video_path, segments, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print(f"🎬 开始切割视频：{video_path}，共 {len(segments)} 段")

    for i, (start, end) in enumerate(segments):
        print(f"正在切割第 {i+1} 段：{start:.2f}s - {end:.2f}s")
        output_path = f"{output_dir}/clip_{i+1}.mp4"

        # 使用 FFmpeg 精确剪辑并保留音频
        cmd = [
            "ffmpeg",
            "-y",  # 自动覆盖已存在的文件
            "-i", video_path,
            "-ss", str(start),
            "-to", str(end),
            "-c:v", "libx264",   # 视频编码器
            "-preset", "fast",    # 编码速度
            "-crf", "23",         # 视频质量（值越小越好，默认23）
            "-c:a", "aac",        # 音频编码器
            "-b:a", "192k",       # 音频比特率
            "-strict", "experimental",
            output_path
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg 执行失败：{result.stderr.decode()}")

        print(f"✅ 第 {i+1} 段保存至：{output_path}")


# -----------------------------
# Step 5: 人脸检测与过滤模块
# -----------------------------

class FaceRecognitionEngine:
    def __init__(self, model_name='buffalo_l', ctx_id=0, det_size=(320, 320)):
        """
        初始化人脸识别引擎

        :param model_name: 使用的模型名称，如 'buffalo_l', 'buffalo_s'
        :param ctx_id: 使用的设备 ID，0 表示 GPU，-1 表示 CPU
        :param det_size: 检测分辨率，例如 (640, 640)
        """
        self.app = face_analysis.FaceAnalysis(name=model_name, root='./models')
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)  # 初始化检测器

    def get_face_embedding(self, frame_bgr):
        """
        提取人脸 embedding（标准化后的）

        :param frame_bgr: OpenCV 读取的 BGR 图像帧 
        :return: embedding 向量 或 None
        """
        faces = self.app.get(frame_bgr)
        return faces[0].normed_embedding if faces else None


def cosine_similarity(a, b):
    """
    计算两个向量之间的余弦相似度
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def extract_face_embeddings_from_video(video_path, engine, sample_rate=10):
    """
    从视频中提取人脸 embedding 特征

    :param video_path: 视频文件路径
    :param engine: 已初始化的人脸识别引擎
    :param sample_rate: 每秒采样多少帧
    :return: 所有人脸 embedding 列表
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件：{video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, fps // sample_rate)

    all_embeddings = []

    for frame_idx in range(0, frame_count, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        embedding = engine.get_face_embedding(frame)
        if embedding is not None:
            all_embeddings.append(embedding)

    cap.release()
    return all_embeddings


def is_single_stable_face(video_path, engine, tolerance=0.5, sample_rate=10):
    embeddings = extract_face_embeddings_from_video(video_path, engine, sample_rate)

    if not embeddings:
        print(f"⚠️ 视频 {video_path} 中无人脸")
        return False

    first_face = embeddings[0]

    for i, emb in enumerate(embeddings[1:]):
        sim = cosine_similarity(first_face, emb)
        if sim < tolerance:
            print(f"⚠️ 视频 {video_path} 中人脸发生变化（第 {i + 1} 帧差异较大）")
            return False

    return True


def filter_videos_by_face(clips_dir, filtered_dir):
    os.makedirs(filtered_dir, exist_ok=True)

    global engine

    valid_files = []
    files = sorted([f for f in os.listdir(clips_dir) if f.endswith(".mp4")])

    print(f"🔍 开始过滤 {len(files)} 个视频片段（单线程顺序处理）")

    for file in files:
        path = os.path.join(clips_dir, file)
        print(f"🔍 正在分析视频：{file}")
        if is_single_stable_face(path, engine, tolerance=0.6, sample_rate=10):
            valid_files.append(file)
            # 直接移动文件，避免一次性加载所有文件到内存
            dst = os.path.join(filtered_dir, file)
            os.rename(path, dst)

    print(f"✅ 过滤完成，共保留 {len(valid_files)} 个含单一稳定人脸的视频")


# -----------------------------
# Step 6: 人脸张嘴检测
# -----------------------------
def is_mouth_open(face_landmarks, threshold_ratio=0.15):
    """
    判断嘴巴是否张开：
    lips_distance / face_height > 阈值（默认 0.15）
    """
    upper_lip = (face_landmarks.part(51).x, face_landmarks.part(51).y)  # 上唇中心
    lower_lip = (face_landmarks.part(57).x, face_landmarks.part(57).y)  # 下唇中心
    nose_top = (face_landmarks.part(28).x, face_landmarks.part(28).y)  # 鼻梁顶部
    chin_bottom = (face_landmarks.part(9).x, face_landmarks.part(9).y)  # 下巴底部
    
    lip_dist = np.linalg.norm(np.array(upper_lip) - np.array(lower_lip))
    face_h = np.linalg.norm(np.array(nose_top) - np.array(chin_bottom))
    
    if face_h == 0:
        return False, 0
    
    ratio = lip_dist / face_h
    return ratio > threshold_ratio, ratio


def get_face_landmarks(frame_bgr):
    """使用 Dlib 获取面部关键点"""
    global detector
    global predictor
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        return None
    
    landmarks = predictor(gray, faces[0])
    return landmarks


def filter_videos_by_mouth_open(filtered_dir, final_dir):
    """遍历视频文件夹，检查每段视频是否有嘴巴张开的画面"""
    os.makedirs(final_dir, exist_ok=True)

    valid_videos = []

    files = [f for f in os.listdir(filtered_dir) if f.endswith(".mp4")]
    print(f"🔍 开始分析 {len(files)} 个视频是否包含张嘴画面...")

    for file in tqdm(files, desc="Analyzing Mouth Open"):
        video_path = os.path.join(filtered_dir, file)
        cap = cv2.VideoCapture(video_path)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if frame_count <= 0 or fps <= 0:
            print(f"⚠️ 跳过无效视频：{file}")
            continue

        # 随机选取 5 个帧位置
        sample_frames = sorted(random.sample(range(frame_count), min(5, frame_count)))

        mouth_opened = False

        for frame_idx in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            landmarks = get_face_landmarks(frame)
            if landmarks is not None:
                opened, ratio = is_mouth_open(landmarks)
                if opened:
                    mouth_opened = True
                    break  # 只要有一帧张嘴就满足条件

        cap.release()

        if mouth_opened:
            valid_videos.append(file)
            dst_path = os.path.join(final_dir, file)
            os.rename(video_path, dst_path)

    print(f"✅ 筛选完成，共找到 {len(valid_videos)} 个包含张嘴动作的视频。")

# -----------------------------
# 主函数：单个视频处理
# -----------------------------
def process_video(video_path, result_dir, logger):
    global model
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_result_dir = os.path.join(result_dir, base_name)
    clips_dir = os.path.join(video_result_dir, "output_clips")
    filtered_dir = os.path.join(video_result_dir, "filtered_clips")
    final_dir = os.path.join(video_result_dir, "final_clips")
    audio_output_path = os.path.join(video_result_dir, "audio.wav")

    os.makedirs(video_result_dir, exist_ok=True)

    # Step 1: 提取音频
    logger.info("Step 1: 正在提取音频...")
    extract_audio(video_path, audio_output_path)

    res = model.generate(input=audio_output_path, batch_size_s=300, hotword='魔搭')

    # 获取文本和时间戳
    text = res[0]['text']
    print('带标点：', text)
    timestamps = res[0]['timestamp']
    print('时间戳：', timestamps)
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    print('不带标点：', cleaned_text)

    # Step 2: 分句
    logger.info("Step 2: 正在根据标点分句...")
    segments = split_sentences_with_positions(text, cleaned_text, timestamps)

    # Step 3: 优化切片
    logger.info("Step 3: 正在优化视频切片时长（4~7秒）...")
    optimized_segments = optimize_segments(segments)

    # Step 4: 切割视频
    logger.info(f"Step 4: 正在切割视频为 {len(optimized_segments)} 段...")
    cut_video_into_clips(video_path, optimized_segments, clips_dir)

    # Step 5: 人脸过滤
    logger.info("Step 5: 正在对视频片段进行人脸过滤...")
    filter_videos_by_face(clips_dir, filtered_dir)

    # Step 6: 张嘴检测 + 最终筛选
    logger.info("Step 6: 正在进行张嘴检测，筛选最终视频...")
    filter_videos_by_mouth_open(filtered_dir, final_dir)

    logger.info("🎉 处理完成！最终保留的视频位于 filtered_clips 文件夹中。")


# -----------------------------
# 主程序入口
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量处理视频文件夹")
    parser.add_argument("--input_dir", type=str, default="/home/ubuntu/mnt_4T/why/AV/original_videos/mingren", help="包含视频的文件夹路径")
    parser.add_argument("--result_dir", type=str, default="video_segments/mingren", help="输出结果文件夹")
    parser.add_argument("--log_dir", type=str, default="clip_logs/mingren", help="日志文件夹")
    parser.add_argument("--workers", type=int, default=4, help="并发进程数")
    parser.add_argument("--checkpoint_file", type=str, default="clip_logs/mingren/completed_videos.txt", help="已完成视频记录文件")
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 获取所有视频文件
    all_videos = [f for f in os.listdir(args.input_dir) if f.lower().endswith(".mp4")]
    completed_videos = read_completed_list(args.checkpoint_file)
    pending_videos = [v for v in all_videos if v not in completed_videos]

    print(f"✅ 总视频数：{len(all_videos)}")
    print(f"🟡 已完成：{len(completed_videos)}")
    print(f"🟢 待处理：{len(pending_videos)}")

    if not pending_videos:
        print("🎉 所有视频均已处理完毕！无需操作。")
        exit(0)

    # 将待处理视频平均分配给各个进程
    chunk_size = len(pending_videos) // args.workers
    video_chunks = [
        pending_videos[i:i + chunk_size]
        for i in range(0, len(pending_videos), chunk_size)
    ]

    print(f"🔄 启动 {args.workers} 个进程，共分割成 {len(video_chunks)} 个任务块")

    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as executor:
        futures = []
        for chunk in video_chunks:
            future = executor.submit(
                process_chunk,
                chunk=chunk,
                input_dir=args.input_dir,
                result_dir=args.result_dir,
                log_dir=args.log_dir,
                checkpoint_file=args.checkpoint_file
            )
            futures.append(future)

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"⚠️ 子任务异常：{e}")

    print("🏁 所有视频处理完成！")