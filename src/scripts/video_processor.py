import asyncio
import json
import random
import subprocess
from typing import List, Optional
import PIL
import cv2
from tqdm.asyncio import tqdm
import yt_dlp
import os
from dotenv import load_dotenv
from scenedetect import detect, ContentDetector, split_video_ffmpeg
from faster_whisper import WhisperModel
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
SEM = asyncio.Semaphore(4)

class SubtitleResult(BaseModel):
    is_academic: bool
    refined_content: str

class PPTContent(BaseModel):
    text_hierarchy: str  # 必须有，即便只有一句话
    latex_formulas: Optional[List[str]] = None  # 允许为空
    table_markdown: Optional[str] = None        # 允许为空
    visual_description: Optional[str] = None    # 允许为空

class VideoProcessor:

    @staticmethod
    def download_bilibili_video(url, save_path='../../data/videos/'):

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': f'{save_path}%(title)s.%(ext)s',
            'merge_output_format': 'mp4',
            'nocheckcertificate': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=True)
                video_title = info.get('title', 'video')
                file_path = ydl.prepare_filename(info)
                
                print(f"下载成功: {video_title}")
                return {"status": "success", "file_path": file_path, "title": video_title}
            except Exception as e:
                return {"status": "error", "message": str(e)}
    
    @staticmethod
    def keyframe_extract(video_path, output_dir, threshold=6.0):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        # 1. 使用内容检测算法
        print("正在检测场景，请稍候...")
        scene_list = detect(
            video_path, 
            ContentDetector(threshold=threshold, min_scene_len=fps*3),
            show_progress=True, # 开启进度条，让你看到它是不是真的卡住了
        )

        keyframes_info = []

        # 2. 遍历检测到的每一个场景
        for i, scene in enumerate(scene_list):
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            
            # 【优化】：取 end_time 前 0.5 秒，确保这一页的内容已经完全“弹出”显示完整
            capture_time = max(0, end_time - 3.1)
            
            img_name = f"ppt_page_{i+1:03d}_at_{capture_time:.2f}s.jpg"
            img_path = os.path.join(output_dir, img_name)
            
            # 3. 使用 ffmpeg 提取单帧
            cmd = [
                'ffmpeg', '-ss', str(capture_time), 
                '-i', video_path, 
                '-frames:v', '1', 
                '-q:v', '2', 
                img_path, '-y', '-loglevel', 'error'
            ]
            subprocess.run(cmd)

            keyframes_info.append({
                "page": i + 1,
                "start": start_time,
                "end": end_time,
                "file": img_name
            })
            print(f"成功提取第 {i+1} 页: {start_time:.2f}s -> {end_time:.2f}s (采样点: {capture_time:.2f}s)")

        # --- 重点：以下代码必须在 for 循环结束后执行 ---
        result_json_path = os.path.join(output_dir, "metadata.json")
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(keyframes_info, f, ensure_ascii=False, indent=4)
        
        print(f"处理完成！总计 {len(keyframes_info)} 页 PPT。元数据已保存至 {result_json_path}")
        return keyframes_info
    
    @staticmethod
    def transcribe_extract(video_path, output_dir, model_size="large-v3-turbo"):
        model = WhisperModel(model_size, device="cuda", compute_type="float16")

        print(f"正在使用 {model_size} 识别语音，请稍候...")
        
        segments, info = model.transcribe(
            video_path, 
            beam_size=5, 
            word_timestamps=False,
            vad_filter=True,
            initial_prompt="This is an academic paper sharing session, and the content may include a mix of Chinese and English."
        )

        print(f"检测到语言: {info.language} (置信度: {info.language_probability:.2f})")

        transcript_data = []
        for segment in segments:
            transcript_data.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })

        with open(os.path.join(output_dir, "subscribe.json"), 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=4)
        return transcript_data
    
    @staticmethod
    def merge_keyframe_and_subtitles(ppt_json_path, sub_json_path, output_path, tolerance=3.0):
        with open(ppt_json_path, 'r', encoding='utf-8') as f:
            ppt_data = json.load(f)
        with open(sub_json_path, 'r', encoding='utf-8') as f:
            sub_data = json.load(f)

        for ppt in ppt_data:
            ppt_start = max(0, ppt["start"] - tolerance)
            ppt_end = ppt["end"] + tolerance
            
            # 筛选逻辑：只要字幕的 [start, end] 区间与 PPT 的 [ppt_start, ppt_end] 有重叠即可
            matched_texts = []
            for sub in sub_data:
                # 判断两个区间是否有交集：子集 start < 页面 end 且 子集 end > 页面 start
                if sub["start"] < ppt_end and sub["end"] > ppt_start:
                    matched_texts.append(sub["text"])
            
            # 拼接字幕内容
            ppt["subscribe"] = " ".join(matched_texts)
            print(f"第 {ppt['page']} 页处理完成，匹配到 {len(matched_texts)} 段字幕。")

        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ppt_data, f, ensure_ascii=False, indent=4)
        
        return ppt_data
    
    @staticmethod
    async def process_single_page_subtitle(item, image_dir):
        # 这里的 SEM 依然起作用，控制最大并发
        async with SEM:
            img_path = os.path.join(image_dir, item['file'])
            raw_subscribe = item.get('subscribe', '')
            
            prompt = f"""
作为一名学术助手，请分析这张 PPT 图片及对应的原始识别字幕，并对原始字幕进行修正。

【原始字幕】：
{raw_subscribe}

【任务要求】：
1. 学术价值判定：判断画面是否有实质学术内容（如公式、图表、逻辑点、详细描述）。如果只是纯标题页、致谢页或过渡页，判定为 false。
2. 内容对齐：由于原始字幕可能包含对其他页PPT内容的讲解，因此从原始字幕中提取仅与本页 PPT 内容相关的讲解。
3. 术语修正：根据 PPT 上的文字，修正字幕中识别错误的专业名词、缩写等。
4. 去口语化：去除“然后”、“这个”、“嗯”、“啊”等语气词，将口语转化为书面学术语言。

【注意】：在修正原始字幕的过程中，请尽量避免信息丢失，不要过度压缩内容。

请严格按照以下 JSON 格式输出，不要包含任何额外解释：
{{
    "is_academic": boolean,
    "refined_content": "修正后的学术文本"
}}
"""

            max_retries = 10  # 最大重试次数
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    img = PIL.Image.open(img_path)
                    
                    # 建议增加一个 request_options 的 timeout，防止死等
                    response = await client.aio.models.generate_content(
                        model=os.getenv("GEMINI_MODEL"),
                        contents=[prompt, img],
                        config={
                            'response_mime_type': 'application/json',
                            'response_schema': SubtitleResult,
                        }
                    )

                    res_json = response.parsed 
                    if res_json.is_academic:
                        item["subtitle"] = res_json.refined_content
                        return item
                    return None

                except Exception as e:
                    err_msg = str(e)
                    # 如果遇到 503 (繁忙) 或 429 (限流)
                    if "503" in err_msg or "429" in err_msg:
                        retry_count += 1
                        # 指数退避：第一次等 4s, 第二次等 8s, 依此类推，加点随机数防止同步冲撞
                        wait_time = (2 ** retry_count) + random.uniform(1, 3)
                        print(f"\n[页码 {item['page']}] 遇到 503/429，第 {retry_count} 次重试，等待 {wait_time:.1f}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        # 其他不可恢复的错误（如图片损坏、认证失败）直接退出
                        print(f"\n[页码 {item['page']}] 发生不可恢复错误: {e}")
                        return None
            
            print(f"\n[页码 {item['page']}] 达到最大重试次数，放弃该页。")
            return None

    @staticmethod
    async def subtitle_refine(group_json_path, image_dir, output_json_path):
        with open(group_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 创建任务列表
        tasks = [VideoProcessor.process_single_page_subtitle(item, image_dir) for item in data]
        
        # 使用 tqdm 观察异步进度
        results = []
        completed_tasks = await tqdm.gather(*tasks, total=len(tasks))
        for res in completed_tasks:
            if res is not None:
                results.append(res)

        # 排序：因为异步返回是乱序的，需要按页码排回来
        results.sort(key=lambda x: x['page'])

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
    @staticmethod
    async def process_single_ppt(item, image_dir):
        async with SEM:
            img_path = os.path.join(image_dir, item['file'])
            prompt = f"""
作为学术视觉识别专家，请精确提取 PPT 中的信息，遵循以下准则：

1. 语言原则：
   - 提取 PPT 上的文字时，请保持其【原始语言】（如果是英文则保留英文，不要翻译）。
   - 仅在第 4 项“视觉逻辑描述”中使用【中文】进行总结和解释。

2. 任务要求：
   - 层级文字：提取 PPT 的版面标题、目录、正文段落和列表点。使用 #, ## 维护层级。
     【注意】：请跳过图片内部、流程图方框内、图表坐标轴上的零散文字，不要将它们放入此项。PPT装饰性内容，如Logo、背景、底部嵌入字幕中的文字，也不要放入此项。
   - LaTeX 公式：所有数学公式、算式、物理符号必须转换为标准 $LaTeX$ 格式（行内使用 $...$, 独立行使用 $$...$$）。
   - Markdown 表格：将图片中的数据表完整转换为标准 Markdown 表格。
   - 视觉逻辑（中文）：用自然语言描述PPT中**有实质学术意义的**插图的内容（流程图、架构图、趋势图或统计图）。
     【严禁描述】：禁止描述 PPT 的背景底纹、装饰性线条、学校/实验室 Logo、主讲人头像、PPT页码或视频进度条等非学术干扰项。
     【要求】：请将图片内部看到的关键文字（如流程图的步骤名、坐标轴含义、架构组件名）整合进这段中文描述中，使其成为逻辑连贯的句子。
      例如：不要只提取 "Input"，而应写成 "输入层（Input）接收原始信号并传递给..."。

请严格按照以下 JSON 格式输出，不要包含任何额外解释：
{{
    "text_hierarchy": "Markdown格式的版面层级文字，不可为null",
    "latex_formulas": ["$公式1$", "$公式2$"] 或 null,
    "table_markdown": "Markdown表格字符串" 或 null,
    "visual_description": "中文描述图片/流程图逻辑" 或 null
}}
【注意】：如果图中不存在公式、表格或可描述的视觉图形，请将对应字段设为 null。
"""

            max_retries = 4
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    with PIL.Image.open(img_path) as img:
                        # 第一次调用：提取
                        response = await client.aio.models.generate_content(
                            model=os.getenv("GEMINI_MODEL"),
                            contents=[prompt, img],
                            config={
                                'response_mime_type': 'application/json',
                                'response_schema': PPTContent,
                            }
                        )

                    res_json = response.parsed 
                    if not res_json:
                        return None # 或抛出异常进入重试

                    # --- 核心修改：在这里处理数据组装 ---
                    knowledge_blocks = [f"#### [Structure]\n{res_json.text_hierarchy}"]
                    blocks_wo_md = [f"#### [Structure]\n{res_json.text_hierarchy}"]
                    
                    if res_json.latex_formulas:
                        formulas = "\n".join([f"- {f}" for f in res_json.latex_formulas])
                        knowledge_blocks.append(f"#### [Formulas]\n{formulas}")
                        blocks_wo_md.append(f"#### [Formulas]\n{formulas}")
                    
                    if res_json.table_markdown:
                        # --- 第二次调用：表格总结，带独立的小重试逻辑 ---
                        table_des = "表格总结失败" # 默认值
                        sub_retry = 0
                        while sub_retry < 4:
                            try:
                                md_prompt = f"""
### 角色
你是一个科研数据索引专家，擅长将复杂的学术表格压缩成单句语义摘要。

### 任务
请阅读提供的 Markdown 表格，并生成1-3句话摘要。

### 编写准则
1. **三要素原则**：摘要必须包含“实验对象”、“核心指标/维度”和“关键结论”。
2. **术语优先**：保留模型名（如 DeepSeekMath-Base）、数据集名（如 GSM8K）等硬核关键词。
3. **Latex 兼容**：摘要中如涉及变量名请保持 $LaTeX$ 格式。

### 示例参考
- 输入：[关于 ResNet 和 VGG 在 ImageNet 上的 Top-1 错误率对比表]
- 输出：对比了 ResNet 与 VGG 在 ImageNet 上的 Top-1 错误率，结果显示 ResNet-50 的推理精度显著优于 VGG-19。

### 待处理表格
{res_json.table_markdown}

### 输出格式
直接输出摘要的自然语言段落，不要包含表格原文、Markdown 代码块标签或任何解释性废话。
"""
                                sub_response = await client.aio.models.generate_content(
                                    model=os.getenv("GEMINI_MODEL"),
                                    contents=[md_prompt]
                                )
                                table_des = sub_response.text
                                break # 成功则跳出内层循环
                            except Exception as sub_e:
                                print("内层失败！")
                                sub_retry += 1
                                await asyncio.sleep(2**sub_retry) # 短暂等待
                        
                        blocks_wo_md.append(f"#### [Table]\n{table_des}")
                        knowledge_blocks.append(f"#### [Table]\n{res_json.table_markdown}")
                        
                    if res_json.visual_description:
                        knowledge_blocks.append(f"#### [Visual Logic]\n{res_json.visual_description}")
                        blocks_wo_md.append(f"#### [Visual Logic]\n{res_json.visual_description}")

                    item["PPT"] = "\n\n".join(knowledge_blocks)
                    item["PPT_wo_md"] = "\n\n".join(blocks_wo_md)
                    return item

                except Exception as e:
                    # 外部重试逻辑只管第一个大 API
                    err_msg = str(e)
                    if "503" in err_msg or "429" in err_msg:
                        retry_count += 1
                        wait_time = (2 ** retry_count) + random.uniform(1, 3)
                        print(f"\n[页码 {item['page']}] 遇到 503/429，第 {retry_count} 次重试...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"\n[页码 {item['page']}] 不可恢复错误: {e}")
                        return None
            return None

    @staticmethod
    async def PPT_describe(group_json_path, image_dir, output_json_path):
        with open(group_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 创建任务列表
        tasks = [VideoProcessor.process_single_ppt(item, image_dir) for item in data]
        
        # 使用 tqdm 观察异步进度
        results = []
        completed_tasks = await tqdm.gather(*tasks, total=len(tasks))
        for res in completed_tasks:
            if res is not None:
                results.append(res)

        # 排序：因为异步返回是乱序的，需要按页码排回来
        results.sort(key=lambda x: x['page'])

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
# VideoProcessor.keyframe_extract("./data/videos/GRPO.mp4", "./data/video_tmp/GRPO")
# VideoProcessor.transcribe_extract("./data/videos/GRPO.mp4", "./data/video_tmp/GRPO")
# VideoProcessor.merge_keyframe_and_subtitles("./data/video_tmp/GRPO/metadata.json", "./data/video_tmp/GRPO/subscribe.json", "./data/video_tmp/GRPO/group.json")
# asyncio.run(VideoProcessor.subtitle_refine("./data/video_tmp/GRPO/group.json", "./data/video_tmp/GRPO", "./data/video_tmp/GRPO/group_1.json"))
if __name__ == "__main__":
    asyncio.run(VideoProcessor.PPT_describe("./data/video_tmp/GRPO/group_1.json", "./data/video_tmp/GRPO", "./data/video_tmp/GRPO/group_2.json"))