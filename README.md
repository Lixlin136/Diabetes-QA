# Diabetes-QA
集成langchian和Django的糖尿病问答系统
# 糖尿病专家问答系统

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Django](https://img.shields.io/badge/django-4.0+-green.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)
![LangChain](https://img.shields.io/badge/langchain-0.1+-yellow.svg)

基于微调ChatGLM3-6B的糖尿病领域智能问答系统，集成FAISS向量数据库实现专业知识检索。

## 核心功能

**智能问答：**
- 基于微调后的ChatGLM3-6B-32K模型提供专业回答
-支持参数动态调整（top_p, temperature等）
- 自动判断问题相关性，非糖尿病问题简短回答

**2.知识库管理：**
- 支持PDF文档上传和解析
- 自动分块处理（默认800字符块大小，400字符重叠）
- 使用all-roberta-large-v1模型生成嵌入
- FAISS向量数据库存储和检索

**3.混合问答模式：**
- 优先检索知识库内容回答
- 无相关知识时依靠模型自身知识
- 可动态加载/卸载不同知识库

# 快速开始
## 环境要求
* Python 3.8+
* 推荐CUDA GPU（显存≥16GB）
* PyTorch 2.0+
## 安装步骤
```text
git clone https://github.com/你的用户名/糖尿病问答系统.git
cd 糖尿病问答系统

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```
## 模型准备
1. 下载ChatGLM3-6B模型
2. 下载all-roberta-large-v1
## 使用指南
```text
python manage.py runserver

python manage.py makemigrations

python manage.py migrate
```
访问 http://localhost:8000/chat 进入问答界面

**项目架构**
```
graph LR
    A[用户提问] --> B[Django后端]
    B --> C{是否关联知识库?}
    C -->|是| D[FAISS向量检索]
    C -->|否| E[直接生成回答]
    D --> F[答案合成]
    E --> F
    F --> G[返回专业回答]
```
**致谢**
- 清华大学KEG实验室（ChatGLM3模型）

- LangChain开发团队

- Hugging Face开源社区
