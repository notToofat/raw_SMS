#以下代码块用于支持所有操作的环境搭建
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from paddlenlp import Taskflow
import os, re, time
import pandas as pd
import numpy as np
from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec

##############必要的全局变量##################
tag = Taskflow('pos_tagging', user_dict ='./dict/user_dict.txt', mode='accurate', batch_size=32)
filter_list = ['v','n','an']
#############################################

#环境部署
def creat_environment(start_year=int(input('研究开始范围：')), end_year=int(input('研究结束范围：')),segfile_number=int(input('请输入阶段划分数量：'))):

    try:

        os.makedirs('./dict', exist_ok=True)
        os.makedirs('./fr', exist_ok=True)
        os.makedirs('./model', exist_ok=True)
        os.makedirs('./tfidf', exist_ok=True)
        os.makedirs('./按照阶段合并', exist_ok=True)
        os.makedirs('./按照检索词合并', exist_ok=True)
        os.makedirs('./各阶段共词网络', exist_ok=True)

        for i in range(segfile_number):
            os.makedirs(os.path.join('./按照阶段合并','第{}阶段'.format(str(i+1))), exist_ok=True)
            os.makedirs(os.path.join('./经过分词的文本', 'S第{}阶段seg_str'.format(str(i+1))), exist_ok=True)
        os.makedirs('./按照年份合并')
        for i in range(start_year, end_year+1):
            os.makedirs(os.path.join('./按照年份合并', i), exist_ok=True)
        print('- -' * 10)
        print('环境创建完毕，请您完善分析样本之后再次运行本程序')
        print('正在关闭程序...')
        time.sleep(3)
    except(FileNotFoundError, IndexError):
        print('- -' * 10)
        print('环境部署失败，缺少管理员权限!')
        time.sleep(5)
    pass

#Azc信息捕捉
class Azc():
    def __init__(self, compare_path='./按照检索词合并', merge_path='./按照年份合并'):
        self.compare_path = compare_path
        self.merge_path = merge_path

    def get_common_info(self, compare_path, merge_path, compare_file_id_list = [], out_file_id_list = []):

        for compare_search_words_file in os.listdir(compare_path):
            for compare_file_number in os.listdir(os.path.join(compare_path, compare_search_words_file)):
                for compare_file_id in os.listdir(os.path.join(compare_path, compare_search_words_file, compare_file_number, compare_file_id)):
                    compare_file_id_list.append(compare_file_id)
        
        for out_file_number in os.listdir(merge_path):
            for out_file_id in os.listdir(os.path.join(merge_path, out_file_number)):
                out_file_id_list.append(out_file_id)

        common_list = list(set(compare_file_id_list) & set(out_file_id_list))
        return common_list, compare_file_id_list, out_file_id_list

    def merge(self):
        common_list, compare_file_id_list = self.get_common_info(self.compare_path, self.merge_path)

        for common in compare_file_id_list:#检查相同词汇是否在对比路径
            if common in common_list:#如果在
                for compare_search_words_file in os.listdir(self.compare_path):#遍历对比路径所有关键词文件夹
                    for compare_file_number in os.listdir(os.path.join(self.compare_path, compare_search_words_file)):#遍历所有文件夹下面的年份文件名称
                        for compare_file_id in os.listdir(os.path.join(self.compare_path, compare_search_words_file, compare_file_number)):#得到具体政策文件名称
                            if compare_file_id == common:#如果政策文件名称等于共同名称
                                compare_cor = open(os.path.join(self.compare_path, compare_search_words_file, compare_file_number, compare_file_id), 'r', encoding='utf-8').read()
                                with open(os.path.join(self.merge_path, compare_file_number, compare_file_id), 'a', encoding='utf-8') as f:
                                    f.write(compare_cor)
            else:
                for unique in os.listdir(self.compare_path):
                    for compare_file_number in os.listdir(os.path.join(self.compare_path, unique)):#遍历所有文件夹下面的年份文件名称
                        for compare_file_id in os.listdir(os.path.join(self.compare_path, unique, compare_file_number)):
                            if compare_file_id == common:
                                compare_cor = open(os.path.join(self.compare_path, compare_search_words_file, compare_file_number, compare_file_id), 'r', encoding='utf-8').read()
                                with open(os.path.join(self.merge_path, compare_file_number, compare_file_id), 'a', encoding='utf-8') as ff:
                                    ff.write(compare_cor)

#辅助：字典创建
def creat_dict(originfile_name='./dict/user_dict.txt', comparefile_name='./dict/hit_dict.txt'):

    #按行遍历origin字典
    words = [i for i in open(os.path.join('./dict', originfile_name), 'r', encoding='utf-8').readlines()]
    #按行遍历compare字典
    for new_word in open(os.path.join('./dict', comparefile_name), 'r', encoding='utf-8').readlines():
        if new_word not in words:#如果compare中存在origin中没有的词语
            print(new_word, end='\n',file = open(os.path.join('./dict', originfile_name), 'a', encoding='utf-8'))#将其追加在origin末尾


#读取停用词词典
def LL_stop(): 
    f_stop = open('./dict/hit_stopwords.txt', 'r', encoding='utf-8') 
    sw = [line.strip() for line in f_stop]
    f_stop.close()
    return sw

#核心分词
def seg():
    global filter_list

    for file_id in os.listdir('./按照阶段合并'):
        for id in os.listdir(os.path.join('./按照阶段合并', file_id)):

            filter_result = []
            pre_cor = open(os.path.join('按照阶段合并', file_id, id), 'r', encoding='utf-8').read()
            re_tag = [item for item in tag(pre_cor)]

            for word,flag in re_tag:
                if flag in filter_list:
                    filter_result.append(word)

            write_LL_fenci = [term for term in filter_result if term not in LL_stop() and len(term) > 1]

            ee = open(os.path.join('./经过分词的文本', 's'+file_id+'seg_str', id), 'w', encoding='utf-8')#输出结果
            ee.write(' '.join(write_LL_fenci))
            ee.close()

#回收模块
def re_final():

    re_final = []
    for seger_file in os.listdir('./经过分词的文本'):
        pre_re_final = []
        for seger in os.listdir(os.path.join('./经过分词的文本', seger_file)):
            pre_cor = open(os.path.join('./经过分词的文本', seger_file, seger), 'r', encoding='utf-8').read()
            pre_cor = re.split(' |\n',pre_cor)
            for cor in pre_cor:
                if len(cor) > 1:
                    pre_re_final.append(cor)
        re_final.append(pre_re_final)

    return re_final#返回一个包含n个维度的矩阵


#计算全部文档级别的TF-IDF
def tf_idf():
    
    vectorizer = CountVectorizer()#将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()#统计每个词语的tf-idf权值

    for re_final_ele in re_final():
        counter = 1
        for re_final_txt in re_final_ele:
            tfidf = transformer.fit_transform(vectorizer.fit_transform(re_final_txt))
            word = vectorizer.get_feature_names_out()
            weight = tfidf.toarray()
            data = {'word': word,'tfidf': weight.sum(axis=0).tolist()}

            df = pd.DataFrame(data)
            sorted_df = df.sort_values(by="tfidf", ascending = False).iloc[0:100]
            sorted_df.to_excel(os.path.join('./tfidf', '第{}阶段的tfidf.txt'.format(counter)), index=False)#将sorted_df结果写入到xlsx
            counter = counter + 1

#共现词矩阵模块（1）-辅助dataarray
class Matrix():

    def __init__(self, data_array_path='./按照阶段合并'):
        self.data_array_path = data_array_path
    
    #从原始数据（按阶段划分）形成可供遍历的二维矩阵
    def data_array(self, data_array_path):

        data_array_all = []
        for file_id in os.listdir(data_array_path):
            data_array = []
            for id in os.listdir(os.path.join('./按照阶段合并', file_id)):
                res_list = []
                raw_contend = open(os.path.join('./按照阶段合并', file_id, id), 'r', encoding = 'utf-8').read()
                raw_contend = str(re.split('；|。|：|\n',raw_contend))
                seg_result = [item for item in tag(raw_contend)]

                for word,flag in seg_result:
                    if flag in filter_list:
                        res_list.append(word)
                data_array.append([item for item in res_list if item not in LL_stop() and len(item) > 1])
            data_array_all.append(data_array)
        return data_array_all#[[第一阶段array]，[第二阶段array],[第三阶段array]...[第n阶段array]]

    #共现词矩阵模块（2）读取边
    def get_edge(self, edge_path = './tfidf'):
        edge_all = []
        for tf_idf_file in os.listdir(edge_path):
            raw_edge = pd.read_excel(os.path.join(edge_path, tf_idf_file), sheet_name=0,usecols=[0])
            raw_edge = raw_edge.values.tolist()
            edge = []
            for key_word in raw_edge:
                edge.append(key_word[0])
        edge_all.append(edge)
        return edge_all

    #填充具体的共现频次
    def count_matrix(self, matrix, data_array):
        edge_all = self.get_edge(edge_path = './tfidf')

        counter_all = 0
        for edge in edge_all:
            #构建基于edge的矩阵实体
            matrix = [['' for j in range(len(edge) + 1)] for i in range(len(edge) + 1)]
            matrix[0][1:] = np.array(edge)
            for q in range(1, len(edge)+1):
                matrix[q][0] = edge[q-1]
                #为矩阵创建空白对角线
            for row in range(1, len(matrix)):
                for col in range(1, len(matrix)):
                    if matrix[0][row] == matrix[col][0]:
                        matrix[col][row] = str(0)
                        #计算空白对角线之外的数值
                    else:
                        counter = 0
                        for ech in data_array:
                            if matrix[0][row] in ech and matrix[col][0] in ech:
                                counter += 1
                            else:
                                continue
                        matrix[col][row] = str(counter)
            counter_all += 1

            matrix_filled = pd.DataFrame(matrix)
            matrix_filled.to_csv(os.path.join('./各阶段共词网络', '第' + str(counter_all) + '阶段共词矩阵.csv'), index=0, columns=None, encoding='utf_8_sig', header=False)
    
#训练语言模型Word2Vec
class cluster():
    def __init__(self, model_path = './model'):
        self.model_path = model_path
    #语言模型本地训练模块
    def train_Word2vec(self):

        seg_str = word2vec.Text8Corpus(self.model_path + '/' + 'train_data.txt')
        self.model = Word2Vec(seg_str, sg=1, vector_size=100,window=5,min_count=3, negative=5, hs=1)
        self.model.save(os.path.join(self.model_path, 'word2vec_model1.bin'))
        self.model.wv.save_word2vec_format(os.path.join(self.model_path, 'word2vec_model2.txt'), binary = "False")
    #装载本地语言模型模块
    def load_model(self):
        model = Word2Vec.load(self.model_path + '/' + 'word2vec_model1.bin')
        self.keys = model.wv.key_to_index.keys()
    #读取各阶段共词网络的第一列作为关键词
    def key_words_all(LL_corhence_path = './各阶段共词网络'):
        key_words_all = []
        for file_id in os.listdir(LL_corhence_path):
            key_words = pd.read_excel(os.path.join(LL_corhence_path, file_id), skiprows=1, header=None).iloc[:, 0]
            key_words = key_words.to_list()
            for item in key_words:
                key_words_all.append(item)
        # 使用 with 语句打开文件进行写入操作
        with open(os.path.join(LL_corhence_path, '关键词汇总.txt'), 'w', encoding='utf-8') as file:
            for item in key_words_all:
                file.write(item + '\n')
            file.close()
    #输出向量化之后词汇模块
    def vector_str(self):
        edge = list(set([line.strip() for line in open('./各阶段共词网络/关键词汇总.txt','r',encoding='utf-8').readlines()]))

        word_vector = [self.model.wv[key] for key in self.keys if key in edge]
        edge_wordvector = [word for word in self.keys if word in edge]

        col_num = []#声明一个列表
        for num in range(1, 101):
            col_num.append(str(num)+'维')

        df_wordvector = pd.DataFrame(word_vector,index=list(edge_wordvector), columns=list(col_num))
        df_wordvector.to_csv('关键词向量化.csv',encoding='utf_8_sig')