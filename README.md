## 入门机器学习

- KNN算法

  添加相亲网站预测类型的分类案例
  
  添加0-9数字识别
  
  总结：
  
  k-近邻算法是分类数据最简单的最有效的算法；
  
  但是，执行效率不高，每个测试向量都得做距离运算，每个距离计算需要包含矩阵N维度的浮点运算，而且还要为测试数据准备额外的存储空间；
  这样的算法存储空间大，计算时间开销大;
  
  另一个缺陷是它无法给出任何数据的基础结构信息，因此无法也无法知晓平均实例样本和典型实例样本具有什么特征

- 决策树
  
  使用熵确定最优方案划分数据集，即用增益熵判断，用递归的方式构建树
  
  缺陷是在样本中需要明确知道所属**分类**
  

- 朴素贝叶斯算法
  
  添加文本分类算法案例
  
  添加识别垃圾邮件
  
  总结：
  
  朴素贝叶斯概率模型是通过特征之间的条件独立性的假设,尽管这个建设并不正确，但是任然是有效的分类器；
  可能，存在下溢或者上溢的风险，需要有做些优化.
  