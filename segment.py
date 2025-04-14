# 开发者：洪昕 Steven Hong
# 开发时间：2025/4/13 12:41
import jieba
import logging


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    jieba.set_dictionary('jieba_dict/dict.txt.big')

    stopword_set = set()
    with open('jieba_dict/stop_words.txt', 'r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip())

    with open('corpus.txt', 'r', encoding='utf-8') as content, \
            open('corpus_cut.txt', 'w', encoding='utf-8') as output:

        for texts_num, line in enumerate(content):
            line = line.strip()
            words = jieba.cut(line, cut_all=False)
            filtered = [word for word in words if word not in stopword_set]
            output.write(' '.join(filtered) + '\n')

            if (texts_num + 1) % 10000 == 0:
                logging.info("已完成前 %d 行" % (texts_num + 1))

    logging.info("全部完成！")


if __name__ == '__main__':
    main()