#!/usr/bin/env python3
# coding: utf-8
"""
batch_video_replace.py  —  精确截取 + 替换 + 拼接（filter_complex 版本）
"""

import csv
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple


# ---------- 工具函数 ----------

def extract_filename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def run_ffmpeg(cmd: List[str]) -> None:
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed:\n{e.stderr.strip()}") from e


# ---------- 核心：filter_complex 拼接 ----------

def build_filter_complex(start: float, end: float) -> str:
    return (
        f"[0:v]trim=end={start},setpts=PTS-STARTPTS[v0];"
        f"[0:a]atrim=end={start},asetpts=PTS-STARTPTS[a0];"
        f"[1:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v1];"
        f"[1:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a1];"
        f"[0:v]trim=start={end},setpts=PTS-STARTPTS[v2];"
        f"[0:a]atrim=start={end},asetpts=PTS-STARTPTS[a2];"
        "[v0][v1][v2]concat=n=3:v=1:a=0[vout];"
        "[a0][a1][a2]concat=n=3:v=0:a=1[aout]"
    )


def process_video_replacement(
    row: List[str],
    folder_a: str,
    folder_b: str,
    output_dir: str,
) -> Tuple[bool, str, str]:
    try:
        if len(row) < 7:
            return False, f"列数不足：{row}", "unknown"

        filename = extract_filename(row[0])
        video_a = os.path.join(folder_a, f"{filename}.mp4")
        video_b = os.path.join(folder_b, f"{filename}.mp4")

        missing = [p for p in (video_a, video_b) if not os.path.exists(p)]
        if missing:
            return False, f"缺少文件：{', '.join(missing)}", filename

        try:
            start, end = map(float, row[6].strip().split(','))
            if end <= start:
                raise ValueError
        except ValueError:
            return False, f"时间戳格式错误：{row[6]} ({filename})", filename

        output_path = os.path.join(output_dir, f"{filename}.mp4")
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-i", video_b,
            "-i", video_a,
            "-filter_complex", build_filter_complex(start, end),
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            output_path,
        ]
        run_ffmpeg(cmd)
        return True, f"完成 {filename}", filename

    except Exception as e:
        return False, f"处理 {filename} 出错：{e}", filename


# ---------- 批量并行 ----------

def batch_process_videos(
    csv_path: str,
    folder_a: str,
    folder_b: str,
    output_dir: str,
    max_workers: int = 4,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    failures: List[str] = []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)            # 跳过表头
        tasks = [row for row in reader if len(row) >= 7]

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(process_video_replacement, row, folder_a, folder_b, output_dir): row
            for row in tasks
        }
        for fut in as_completed(futures):
            ok, msg, _ = fut.result()
            print(("[SUCCESS] " if ok else "[FAIL]    ") + msg)
            if not ok:
                failures.append(msg)

    log = os.path.join(output_dir, "final_error.txt")
    with open(log, 'w', encoding='utf-8') as f:
        f.write("\n".join(failures))
    print(f"\n全部完成！失败 {len(failures)} 条，见日志：{log}")


# ---------- 入口 ----------

if __name__ == "__main__":
    # ✏️ 在这里写死配置路径
    CSV_PATH   = "/home/ubuntu/mnt_4T/why/AV/data/temp_1000/output.csv"
    FOLDER_A   = "/home/ubuntu/mnt_4T/why/AV/data/temp_1000/all_fake_results"
    FOLDER_B   = "/home/ubuntu/mnt_4T/why/AV/data/temp_1000/input_videos"
    OUTPUT_DIR = "/home/ubuntu/mnt_4T/why/AV/data/temp_1000/final_results"
    MAX_WORKERS = 4     # 根据 CPU 调整

    batch_process_videos(
        csv_path=CSV_PATH,
        folder_a=FOLDER_A,
        folder_b=FOLDER_B,
        output_dir=OUTPUT_DIR,
        max_workers=MAX_WORKERS,
    )
