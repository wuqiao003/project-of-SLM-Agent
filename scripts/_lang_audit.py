import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
path = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "data/prepared/_clean_droptest.jsonl")

LANG = {
    "zh": ["中文", "中字", "汉语", "简体", "繁体", "普通话", "chinese", "mandarin"],
    "en": ["英文", "英语", "english"],
    "ja": ["日文", "日语", "日本", "japanese"],
    "ko": ["韩文", "韩语", "韩国", "korean"],
    "es": ["西班牙", "spanish", "español"],
    "fr": ["法文", "法语", "french"],
    "de": ["德文", "德语", "german"],
    "pt": ["葡萄牙", "portuguese"],
    "th": ["泰文", "泰语", "thai"],
    "ru": ["俄文", "俄语", "russian"],
    "it": ["意大利", "italian"],
    "ar": ["阿拉伯", "arabic"],
}

def langs_in(text):
    t = text.lower()
    found = []
    for code, kws in LANG.items():
        for kw in kws:
            i = t.find(kw.lower())
            if i >= 0:
                found.append((i, code))
                break
    found.sort()
    return [c for _, c in found]

tools = {}
contradict = 0
total_lang_args = 0
samples = []
for line in open(path, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    m = r["messages"]
    q = next((x["content"] for x in m if x["role"] == "user"), "")
    call = json.loads(m[-1]["content"])
    name = call["name"]; args = call.get("arguments", {})
    qlangs = set(langs_in(q))
    for f in ("source_language", "target_language", "language"):
        v = args.get(f)
        if isinstance(v, str) and v:
            total_lang_args += 1
            # normalize labeled value to a code if it's a keyword
            code = v.lower()
            mapped = None
            for c, kws in LANG.items():
                if code == c or code in [k.lower() for k in kws]:
                    mapped = c; break
            if mapped and qlangs and mapped not in qlangs:
                contradict += 1
                tools[name] = tools.get(name, 0) + 1
                if len(samples) < 10:
                    samples.append((name, f, v, sorted(qlangs), q[:55]))

print("lang args total:", total_lang_args, "contradicting query:", contradict)
print("by tool:", tools)
for s in samples:
    print(" ", s)
