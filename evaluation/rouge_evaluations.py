from rouge import Rouge
import numpy as np

# 初始化ROUGE评测工具
rouge = Rouge()

# 计算ROUGE-L分数
# hyp_ids: 预测的token id序列
# ref_ids: 参考答案的token id序列
# tokenizer: 分词器，用于将id序列解码为文本

def rouge_score(hyp_ids, ref_ids, tokenizer):
    # 将预测id解码为字符串
    hyps = [tokenizer.decode(hyp_ids, skip_special_tokens=True)]
    # 如果预测为空，直接返回0分
    if len(hyps[0]) == 0:
        return 0.0
    # 将参考答案id解码为字符串
    refs = [tokenizer.decode(ref_ids, skip_special_tokens=True)]
    try:
        # 计算ROUGE-L的F1分数
        rouge_score = rouge.get_scores(hyps, refs)[0]['rouge-l']['f']
    except ValueError:
        # 若解码或评测出错，返回0分
        return 0.0
    return rouge_score

# 计算分类准确率
# preds: 预测标签列表
# labels: 真实标签列表
def acc_score(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    # 统计预测正确的比例
    return np.sum(preds == labels) / float(len(labels))