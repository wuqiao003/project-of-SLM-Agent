#!/usr/bin/env python
"""
模型评估脚本 - 测试微调后的工具调用能力（统一 name/arguments JSON 格式）
"""

import json
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from edge_slm.pipeline.prompts import build_system_prompt
from edge_slm.pipeline.acceptance import parse_tool_call_from_text


def load_model(model_path: str):
    """加载微调后的模型"""
    try:
        from unsloth import FastLanguageModel
        print(f"使用 Unsloth 加载模型: {model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        print(f"加载模型: {model_path}")
        
        # 检查是否是 LoRA adapter
        adapter_config = Path(model_path) / "adapter_config.json"
        if adapter_config.exists():
            print("检测到 LoRA adapter，加载基础模型...")
            base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-3B-Instruct",
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer


def generate_response(model, tokenizer, messages: list, max_new_tokens: int = 512):
    """生成模型响应"""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def evaluate_tool_calling(model, tokenizer):
    """评估工具调用能力"""
    
    system_prompt = build_system_prompt()

    # 测试用例 - 20个，从简单到复杂递增
    # expected_params: 期望提取的参数及其值（用于验证参数提取能力）
    test_cases = [
        # ============ 简单级别 (1-5) - 单工具、直接指令 ============
        {
            "query": "解析视频 https://example.com/video.mp4",
            "expected_tools": ["parse_video"],
            "expected_params": {"video_url": "https://example.com/video.mp4"},
            "difficulty": "简单",
            "description": "最基础的单工具调用",
        },
        {
            "query": "导出项目 proj_001，mp4格式",
            "expected_tools": ["export_project"],
            "expected_params": {"project_id": "proj_001", "output_format": "mp4"},
            "difficulty": "简单",
            "description": "导出项目基础调用",
        },
        {
            "query": "给视频加字幕 https://test.com/movie.mp4 语言是英语",
            "expected_tools": ["generate_subtitles"],
            "expected_params": {"video_url": "https://test.com/movie.mp4", "source_language": "英语"},
            "difficulty": "简单",
            "description": "生成字幕基础调用",
        },
        {
            "query": "翻译字幕 sub_123 到日语",
            "expected_tools": ["translate_subtitles"],
            "expected_params": {"subtitle_file": "sub_123", "source_language": "zh", "target_language": "ja"},
            "difficulty": "简单",
            "description": "字幕翻译基础调用",
        },
        {
            "query": "给 https://example.com/clip.mp4 配音，目标语言中文",
            "expected_tools": ["generate_dubbing"],
            "expected_params": {
                "video_url": "https://example.com/clip.mp4",
                "target_language": "zh",
            },
            "difficulty": "简单",
            "description": "配音基础调用（仅校验可提取参数）",
        },
        
        # ============ 中等级别 (6-10) - 单工具、口语化表达 ============
        {
            "query": "我想看看这个视频的时长和分辨率，链接是 https://youtube.com/watch?v=abc123",
            "expected_tools": ["parse_video"],
            "expected_params": {"video_url": "https://youtube.com/watch?v=abc123"},
            "difficulty": "中等",
            "description": "口语化的视频信息解析需求",
        },
        {
            "query": "能帮我把这个日本动漫的字幕翻译成中文吗？字幕文件编号是 sub_anime_456",
            "expected_tools": ["translate_subtitles"],
            "expected_params": {"subtitle_file": "sub_anime_456", "source_language": "ja", "target_language": "zh"},
            "difficulty": "中等",
            "description": "带场景描述的翻译请求",
        },
        {
            "query": "我下载了一个韩剧视频 https://drama.com/ep01.mp4，想给它自动生成韩语字幕",
            "expected_tools": ["generate_subtitles"],
            "expected_params": {"video_url": "https://drama.com/ep01.mp4", "source_language": "韩语"},
            "difficulty": "中等",
            "description": "带背景说明的字幕生成",
        },
        {
            "query": "处理完的视频我想导出来，文件ID是 processed_789，要高清mp4",
            "expected_tools": ["export_project"],
            "expected_params": {"project_id": "processed_789", "output_format": "mp4"},
            "difficulty": "中等",
            "description": "口语化下载请求",
        },
        {
            "query": "这个英语教学视频需要配上英语旁白 https://edu.com/lesson1.mp4",
            "expected_tools": ["generate_dubbing"],
            "expected_params": {
                "video_url": "https://edu.com/lesson1.mp4",
                "target_language": "en",
            },
            "difficulty": "中等",
            "description": "带具体要求的配音请求（仅校验可提取参数）",
        },
        
        # ============ 较难级别 (11-15) - 多步骤暗示、复杂场景 ============
        {
            "query": "我是个UP主，刚录了个游戏解说视频 https://bilibili.com/video/BV123，想先看看视频时长和分辨率信息",
            "expected_tools": ["parse_video"],
            "expected_params": {"video_url": "https://bilibili.com/video/BV123"},
            "difficulty": "较难",
            "description": "带身份和场景的解析请求",
        },
        {
            "query": "公司要做一个产品宣传片的多语言版本，原片是 https://company.com/promo.mp4，先帮我识别出中文字幕",
            "expected_tools": ["generate_subtitles"],
            "expected_params": {"video_url": "https://company.com/promo.mp4", "source_language": "中文"},
            "difficulty": "较难",
            "description": "企业场景的字幕生成",
        },
        {
            "query": "我们团队翻译好了一份西班牙语字幕 sub_spanish_doc，现在需要转成葡萄牙语给巴西分公司用",
            "expected_tools": ["translate_subtitles"],
            "expected_params": {"subtitle_file": "sub_spanish_doc", "source_language": "es", "target_language": "pt"},
            "difficulty": "较难",
            "description": "跨国业务场景的翻译",
        },
        {
            "query": "我在做一个面向东南亚市场的APP介绍视频 https://app.com/intro.mp4，需要泰语配音",
            "expected_tools": ["generate_dubbing"],
            "expected_params": {
                "video_url": "https://app.com/intro.mp4",
                "target_language": "th",
            },
            "difficulty": "较难",
            "description": "详细要求的配音场景（仅校验可提取参数）",
        },
        {
            "query": "客户催着要最终版视频了，文件编号 final_cut_2024，导出成mov格式方便他们在Mac上编辑",
            "expected_tools": ["export_project"],
            "expected_params": {"project_id": "final_cut_2024", "output_format": "mov"},
            "difficulty": "较难",
            "description": "紧急业务场景的下载",
        },
        
        # ============ 复杂级别 (16-18) - 多工具串联暗示 ============
        {
            "query": "我有一个英文的TED演讲视频 https://ted.com/talk123.mp4，想做成中文版发到B站。首先帮我解析一下这个视频的时长和分辨率信息",
            "expected_tools": ["parse_video"],
            "expected_params": {"video_url": "https://ted.com/talk123.mp4"},
            "difficulty": "复杂",
            "description": "多步骤任务的第一步（解析元信息）",
        },
        {
            "query": "继续上个任务，视频分析完了，现在需要先提取出英文字幕，视频地址还是 https://ted.com/talk123.mp4",
            "expected_tools": ["generate_subtitles"],
            "expected_params": {"video_url": "https://ted.com/talk123.mp4", "source_language": "英文"},
            "difficulty": "复杂",
            "description": "多步骤任务的中间步骤",
        },
        {
            "query": "字幕提取好了，编号是 sub_ted_en_001，请把它翻译成简体中文，我要用来做双语字幕",
            "expected_tools": ["translate_subtitles"],
            "expected_params": {"subtitle_file": "sub_ted_en_001", "source_language": "en", "target_language": "zh"},
            "difficulty": "复杂",
            "description": "多步骤任务的后续步骤",
        },
        
        # ============ 非常复杂级别 (19-20) - 多工具、复杂场景、多条件 ============
        {
            "query": "我是一个自媒体博主，最近接了个跨境电商的推广单。客户给了个英文产品介绍视频 https://amazon.com/product_demo.mp4，我需要：1）先获取视频的时长和分辨率信息；2）然后提取英文字幕。视频大概3分钟，产品是智能手表。先帮我做第一步，解析视频信息",
            "expected_tools": ["parse_video"],
            "expected_params": {"video_url": "https://amazon.com/product_demo.mp4"},
            "difficulty": "非常复杂",
            "description": "完整业务场景+多步骤规划+具体产品描述",
        },
        {
            "query": "我们是一家MCN机构，正在帮一个日本美妆博主做中国市场本地化。她的最新视频 https://youtube.com/beauty_tips.mp4 需要完整处理：目前视频是日语的，我们已经人工翻译好了中文字幕文件 sub_beauty_cn_final，现在需要找一个甜美风格的中文女声来配音，让中国观众听起来更亲切自然。请帮我添加配音",
            "expected_tools": ["generate_dubbing"],
            "expected_params": {
                "video_url": "https://youtube.com/beauty_tips.mp4",
                "subtitle_file": "sub_beauty_cn_final",
                "target_language": "zh",
            },
            "difficulty": "非常复杂",
            "description": "MCN业务场景+跨国本地化+详细声音要求+完整背景说明",
        },
    ]
    
    print("\n" + "=" * 70)
    print("工具调用能力评估 - 20个测试用例（简单→非常复杂）")
    print("=" * 70)
    
    results = {
        "简单": {"tool_correct": 0, "param_correct": 0, "total": 0},
        "中等": {"tool_correct": 0, "param_correct": 0, "total": 0},
        "较难": {"tool_correct": 0, "param_correct": 0, "total": 0},
        "复杂": {"tool_correct": 0, "param_correct": 0, "total": 0},
        "非常复杂": {"tool_correct": 0, "param_correct": 0, "total": 0},
    }
    
    total = len(test_cases)
    total_tool_score = 0
    total_param_score = 0
    
    for i, case in enumerate(test_cases, 1):
        difficulty = case["difficulty"]
        results[difficulty]["total"] += 1
        
        print(f"\n{'='*70}")
        print(f"[测试 {i}/{total}] 难度: {difficulty}")
        print(f"场景: {case['description']}")
        print(f"{'='*70}")
        print(f"输入: {case['query'][:100]}..." if len(case['query']) > 100 else f"输入: {case['query']}")
        print(f"期望工具: {case['expected_tools']}")
        print(f"期望参数: {case['expected_params']}")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": case["query"]},
        ]
        
        response = generate_response(model, tokenizer, messages)
        print(f"\n输出: {response[:400]}..." if len(response) > 400 else f"\n输出: {response}")
        
        # 评估结果
        expected_tools = case["expected_tools"]
        expected_params = case["expected_params"]
        
        tool_score = 0
        param_score = 0
        found_tool = None
        found_params = {}
        
        # 解析 JSON（name/arguments 或兼容旧 tool/params）
        try:
            parsed = parse_tool_call_from_text(response)
            if parsed:
                found_tool = parsed.get("name")
                found_params = parsed.get("arguments", {})
                
                # 1. 检查工具是否正确
                if found_tool in expected_tools:
                    tool_score = 1
                    print(f"\n✅ 工具正确: {found_tool}")
                else:
                    print(f"\n❌ 工具错误: 期望 {expected_tools}, 得到 {found_tool}")
                
                # 2. 检查参数提取
                if found_params:
                    print(f"\n参数提取检查:")
                    matched_params = 0
                    total_expected = len(expected_params)
                    
                    for param_name, expected_value in expected_params.items():
                        actual_value = found_params.get(param_name)
                        
                        if actual_value is None:
                            print(f"  ❌ {param_name}: 未提取 (期望: {expected_value})")
                        else:
                            # 检查值是否匹配（模糊匹配）
                            if check_param_match(expected_value, actual_value):
                                print(f"  ✅ {param_name}: {actual_value}")
                                matched_params += 1
                            else:
                                print(f"  ⚠️ {param_name}: {actual_value} (期望: {expected_value})")
                                matched_params += 0.5  # 部分匹配
                    
                    # 检查是否有多余参数
                    extra_params = set(found_params.keys()) - set(expected_params.keys())
                    if extra_params:
                        print(f"  ℹ️ 额外参数: {extra_params}")
                    
                    param_score = matched_params / total_expected if total_expected > 0 else 0
                    print(f"\n  参数得分: {matched_params}/{total_expected} ({param_score*100:.0f}%)")
                else:
                    print(f"\n❌ 未提取到任何参数")
            else:
                print(f"\n❌ 未找到有效的工具调用 JSON")
                    
        except (json.JSONDecodeError, ValueError) as e:
            # JSON 解析失败，检查是否包含工具名
            for tool in expected_tools:
                if tool in response:
                    print(f"\n⚠️ 包含工具名 '{tool}' 但JSON格式不标准")
                    tool_score = 0.5
                    break
            else:
                print(f"\n❌ 未找到有效的工具调用")
        
        # 汇总本条得分
        total_tool_score += tool_score
        total_param_score += param_score
        results[difficulty]["tool_correct"] += tool_score
        results[difficulty]["param_correct"] += param_score
        
        # 综合评价
        if tool_score == 1 and param_score >= 0.8:
            print(f"\n🎯 综合评价: 优秀")
        elif tool_score >= 0.5 and param_score >= 0.5:
            print(f"\n👍 综合评价: 良好")
        elif tool_score >= 0.5 or param_score >= 0.3:
            print(f"\n⚠️ 综合评价: 一般")
        else:
            print(f"\n❌ 综合评价: 较差")
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("评估结果汇总")
    print("=" * 70)
    
    # 按难度显示结果
    print("\n按难度分类 (工具识别 / 参数提取):")
    print("-" * 60)
    for difficulty in ["简单", "中等", "较难", "复杂", "非常复杂"]:
        r = results[difficulty]
        if r["total"] > 0:
            tool_acc = r["tool_correct"] / r["total"] * 100
            param_acc = r["param_correct"] / r["total"] * 100
            tool_bar = "█" * int(tool_acc / 10) + "░" * (10 - int(tool_acc / 10))
            param_bar = "█" * int(param_acc / 10) + "░" * (10 - int(param_acc / 10))
            print(f"  {difficulty:8s}:")
            print(f"    工具: {r['tool_correct']:4.1f}/{r['total']} ({tool_acc:5.1f}%) {tool_bar}")
            print(f"    参数: {r['param_correct']:4.1f}/{r['total']} ({param_acc:5.1f}%) {param_bar}")
    
    # 总体结果
    tool_accuracy = total_tool_score / total * 100
    param_accuracy = total_param_score / total * 100
    overall_accuracy = (tool_accuracy + param_accuracy) / 2
    
    print("-" * 60)
    print(f"  {'总计':8s}:")
    print(f"    工具识别: {total_tool_score:4.1f}/{total} ({tool_accuracy:5.1f}%)")
    print(f"    参数提取: {total_param_score:4.1f}/{total} ({param_accuracy:5.1f}%)")
    print(f"    综合得分: {overall_accuracy:.1f}%")
    print()
    
    # 评价
    if overall_accuracy >= 85:
        print("🎉 模型表现优秀! 工具调用和参数提取能力都很强")
    elif overall_accuracy >= 70:
        print("👍 模型表现良好，可继续优化参数提取准确度")
    elif overall_accuracy >= 50:
        print("⚠️ 模型表现一般，建议增加训练数据")
    else:
        print("❌ 模型表现较差，需要更多训练数据和优化")
    
    # 给出改进建议
    print("\n改进建议:")
    
    # 分析工具识别弱项
    weak_tool_areas = []
    weak_param_areas = []
    for difficulty in ["简单", "中等", "较难", "复杂", "非常复杂"]:
        r = results[difficulty]
        if r["total"] > 0:
            if r["tool_correct"] / r["total"] < 0.6:
                weak_tool_areas.append(difficulty)
            if r["param_correct"] / r["total"] < 0.6:
                weak_param_areas.append(difficulty)
    
    if weak_tool_areas:
        print(f"  - 工具识别在 {', '.join(weak_tool_areas)} 级别较弱")
    if weak_param_areas:
        print(f"  - 参数提取在 {', '.join(weak_param_areas)} 级别较弱")
    
    if param_accuracy < tool_accuracy - 10:
        print("  - 参数提取能力明显弱于工具识别，建议增加参数提取的训练样本")
    
    if results["非常复杂"]["total"] > 0:
        complex_score = (results["非常复杂"]["tool_correct"] + results["非常复杂"]["param_correct"]) / (results["非常复杂"]["total"] * 2)
        if complex_score < 0.5:
            print("  - 复杂场景理解能力不足，建议增加带详细场景描述的训练数据")
    
    return overall_accuracy


def check_param_match(expected: str, actual: str) -> bool:
    """
    检查参数值是否匹配（支持模糊匹配）
    """
    if expected is None or actual is None:
        return False
    
    expected_str = str(expected).lower().strip()
    actual_str = str(actual).lower().strip()
    
    # 完全匹配
    if expected_str == actual_str:
        return True
    
    # URL 匹配（忽略协议差异）
    if expected_str.startswith("http"):
        expected_clean = expected_str.replace("https://", "").replace("http://", "")
        actual_clean = actual_str.replace("https://", "").replace("http://", "")
        if expected_clean == actual_clean:
            return True
    
    # 语言名称模糊匹配
    language_aliases = {
        "中文": ["中文", "chinese", "zh", "cn", "简体中文", "中国语"],
        "英语": ["英语", "english", "en", "英文"],
        "日语": ["日语", "japanese", "ja", "jp", "日文"],
        "韩语": ["韩语", "korean", "ko", "kr", "韩文"],
        "泰语": ["泰语", "thai", "th"],
        "葡萄牙语": ["葡萄牙语", "portuguese", "pt", "葡语"],
        "西班牙语": ["西班牙语", "spanish", "es"],
    }
    
    for lang, aliases in language_aliases.items():
        if expected_str in [a.lower() for a in aliases]:
            if actual_str in [a.lower() for a in aliases]:
                return True
    
    # 包含匹配（实际值包含期望值的关键部分）
    if expected_str in actual_str or actual_str in expected_str:
        return True
    
    return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="评估微调模型")
    parser.add_argument("--model-path", type=str, default="outputs/tool_model", help="模型路径")
    args = parser.parse_args()
    
    # 检查模型路径
    model_path = Path(args.model_path)
    if not model_path.exists():
        # 尝试合并模型路径
        merged_path = Path(f"{args.model_path}_merged")
        if merged_path.exists():
            model_path = merged_path
            print(f"使用合并模型: {model_path}")
        else:
            print(f"错误: 模型路径不存在: {args.model_path}")
            print("请先运行训练: python generate_and_train.py")
            return
    
    print(f"模型路径: {model_path}")
    
    # 加载模型
    model, tokenizer = load_model(str(model_path))
    
    # 评估
    accuracy = evaluate_tool_calling(model, tokenizer)
    
    print(f"\n最终准确率: {accuracy:.1f}%")


if __name__ == "__main__":
    main()
