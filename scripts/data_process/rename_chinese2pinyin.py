import os

try:
    import pypinyin
except ImportError:
    print("try 'pip install -U pypinyin'")
    os.system("pip install -U pypinyin")


def chinese_to_pinyin(text: str) -> str:
    """将中文文本转换为拼音，非中文部分保留"""
    result = []
    for char in text:
        if "一" <= char <= "鿿":
            # 取第一个读音，去掉声调
            pys = pypinyin.pinyin(char, style=pypinyin.NORMAL)
            if pys:
                result.append(pys[0][0])
        else:
            result.append(char)
    return "".join(result)


def has_chinese(text: str) -> bool:
    """检查是否包含中文字符"""
    return any("一" <= c <= "鿿" for c in text)


def main(dir_path: str):
    renamed = 0
    for filename in os.listdir(dir_path):
        if not has_chinese(filename):
            continue

        name, ext = os.path.splitext(filename)
        new_name = chinese_to_pinyin(name)
        # 替换可能的空格为无空格（拼音通常不加空格）
        new_name = new_name.replace(" ", "")
        new_filename = new_name + ext

        if new_filename == filename:
            continue

        src = os.path.join(dir_path, filename)
        dst = os.path.join(dir_path, new_filename)

        # 避免覆盖已有文件
        if os.path.exists(dst):
            print(f"⚠ 跳过 {filename} → {new_filename} (目标已存在)")
            continue

        os.rename(src, dst)
        print(f"[OK] {filename} -> {new_filename}")
        renamed += 1

    print(f"\n共重命名 {renamed} 个文件")


if __name__ == "__main__":
    dir_path = r"D:\code\pytorch\railway_tools_detection\data\datasets\20250528\train"
    main(dir_path)
