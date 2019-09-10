from collections import Counter
import numpy as np

# 数据预处理&模型快速验证
# 查看dataset
def pretty_print_review_and_label(i):
    print(labels[i] + "\t:\t" + reviews[i][:80] + "...")
g = open('reviews.txt','r')
reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()
g = open('labels.txt','r')
labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close()
‘’‘
len(reviews)
reviews[0]
labels[0]
print("labels.txt \t : \t reviews.txt\n")
pretty_print_review_and_label(2137)
pretty_print_review_and_label(12816)
pretty_print_review_and_label(6267)
pretty_print_review_and_label(21934)
pretty_print_review_and_label(5297)
pretty_print_review_and_label(4998)
’‘’
# 创建3个Counter对象存储 positive, negative 和 total counts
positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()
# 遍历评论所有词增加对应counter对象计数
for i in range(len(reviews)):
    if(labels[i] == 'POSITIVE'):
        for word in reviews[i].split(" "):
            positive_counts[word] += 1
            total_counts[word] += 1
    else:
        for word in reviews[i].split(" "):
            negative_counts[word] += 1
            total_counts[word] += 1
 # check积极评论和消极评论中的常见词           
 positive_counts.most_common()
 negative_counts.most_common()
 # 计算出现100此以上常见词“积极vs消极“比率
 pos_neg_ratios = Counter()
 for term,cnt in list(total_counts.most_common()):
    if(cnt > 100):
        pos_neg_ratio = positive_counts[term] / float(negative_counts[term]+1)
        pos_neg_ratios[term] = pos_neg_ratio
‘’‘
print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))
’‘’
#对数化
for word,ratio in pos_neg_ratios.most_common():
    pos_neg_ratios[word] = np.log(ratio)
‘’‘
print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))
pos_neg_ratios.most_common()
list(reversed(pos_neg_ratios.most_common()))[0:30]
’‘’

# 创建输入/输出
vocab = set(total_counts.keys())
vocab_size = len(vocab)
#print(vocab_size)
layer_0 = np.zeros((1,vocab_size))
#layer_0.shape
# 创建映射到索引的词典
word2index = {}
for i,word in enumerate(vocab):
    word2index[word] = i
#word2index
# 修改layer_0将评论以向量形式表示，每个位置元素标示了给定词在评论中的出现次数
def update_input_layer(review):
    global layer_0
    layer_0 *= 0
    for word in review.split(" "):
        layer_0[0][word2index[word]] += 1
 # 查看layer_0
update_input_layer(reviews[0])
layer_0
# 转换label为binary
def get_target_for_label(label):
    if(label == 'POSITIVE'):
        return 1
    else:
        return 0
#labels[0]
#labels[1]

# 构建NN
import time
import sys
import numpy as np

class SentimentNetwork:
    def __init__(self, reviews,labels,hidden_nodes = 10, learning_rate = 0.1):
        """
        Args:
            reviews(list) - 评论训练数据
            labels(list) - 标签训练数据
            hidden_nodes(int) - 隐藏层节点数
            learning_rate(float) - 学习率
        
        """
        # 随机数种子以确保结果复现
        np.random.seed(1)

        # 预处理评论及标签数据
        self.pre_process_data(reviews, labels)
        
        # 初始化网络
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels):
        
        # 评论所有词放到review_vocab集合
        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                review_vocab.add(word)

        # 转换为列表
        self.review_vocab = list(review_vocab)
        
        # 标签所有项放到label_vocab集合
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
        
        # 转换为列表
        self.label_vocab = list(label_vocab)
        
        # 分别记录列表长度
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        # hash词表索引
        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i
        
        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i
        
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # 设置输入、隐藏、输出层节点数
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # 设置学习率
        self.learning_rate = learning_rate

        # 初始化权重
        # 输入和隐藏层之间的权重
        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))
    
        # 隐藏和输出层之间的权重
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))
        
        # 输入层shape： 1 x input_nodes
        self.layer_0 = np.zeros((1,input_nodes))
    
    def update_input_layer(self,review):

        # 重置layer_0状态
        self.layer_0 *= 0
        
        for word in review.split(" "):
            if(word in self.word2index.keys()):
                self.layer_0[0][self.word2index[word]] += 1
                
    def get_target_for_label(self,label):
        if(label == 'POSITIVE'):
            return 1
        else:
            return 0
        
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_output_2_derivative(self,output):
        return output * (1 - output)
    
    def train(self, training_reviews, training_labels):
        
        # 确保评论数匹配标签数
        assert(len(training_reviews) == len(training_labels))
        
        # 记录正确预测数
        correct_so_far = 0

        start = time.time()
        
        # Training
        for i in range(len(training_reviews)):
            
            # 取得下一项评论及标签
            review = training_reviews[i]
            label = training_labels[i]
            
            ### 前向传播 ###
            
            # 输入层
            self.update_input_layer(review)

            # 隐藏层
            layer_1 = self.layer_0.dot(self.weights_0_1)

            # 输出层
            layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))
            
            ### 反向传播 ###

            # 输出层error term
            layer_2_error = layer_2 - self.get_target_for_label(label) 
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)

            # 隐藏层error term
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T) 
            layer_1_delta = layer_1_error 

            # 更新权重
            self.weights_1_2 -= layer_1.T.dot(layer_2_delta) * self.learning_rate 
            self.weights_0_1 -= self.layer_0.T.dot(layer_1_delta) * self.learning_rate 

            # 记录正确预测数
            if(layer_2 >= 0.5 and label == 'POSITIVE'):
                correct_so_far += 1
            elif(layer_2 < 0.5 and label == 'NEGATIVE'):
                correct_so_far += 1
            
            # 打印训练速度和Training Accuracy
            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
    
    def test(self, testing_reviews, testing_labels):
        """
        预测测试集评论并使用测试集标签计算准确率
        """
        
        # 记录正确预测数
        correct = 0
        
        start = time.time()

        # Testing
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            
            # 打印训练速度和Testing Accuracy
            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0        
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def run(self, review):
        """
        执行给定评论的感情倾向预测任务
        """
        
        # 输入层
        self.update_input_layer(review.lower())

        # 隐藏层
        layer_1 = self.layer_0.dot(self.weights_0_1)

        # 输出层
        layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))
        
        # 大于等于0.5的输出返回"POSITIVE"
        # 其他值返回"NEGATIVE"
        if(layer_2[0] >= 0.5):
            return "POSITIVE"
        else:
            return "NEGATIVE"
        
