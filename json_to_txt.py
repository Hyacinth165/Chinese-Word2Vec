# 开发者：洪昕 Steven Hong
# 开发时间：2025/4/13 13:01
import json

def json_lines_to_pretty_txt(json_file_path, txt_file_path):
    try:
        count = 0
        with open(json_file_path, 'r', encoding='utf-8') as json_file, \
                open(txt_file_path, 'w', encoding='utf-8') as txt_file:

            for line in json_file:
                line = line.strip()
                if not line:
                    continue  

                try:
                    data = json.loads(line)
                    txt_file.write(f"QID: {data.get('qid', '')}\n")
                    txt_file.write(f"分类: {data.get('category', '')}\n")
                    txt_file.write(f"标题: {data.get('title', '')}\n")
                    txt_file.write(f"描述: {data.get('desc', '')}\n")
                    txt_file.write(f"回答: {data.get('answer', '')}\n")
                    txt_file.write("-" * 40 + "\n")
                    count += 1

                    if count % 1000 == 0:
                        print(f"已处理 {count} 行...")

                except json.JSONDecodeError as e:
                    print(f"第 {count + 1} 行解析失败：{e}")
                    continue

        print(f"转换完成，总共处理了 {count} 行。")

    except Exception as e:
        print(f"转换失败：{e}")

json_lines_to_pretty_txt("baike_qa_train.json", "corpus.txt")
