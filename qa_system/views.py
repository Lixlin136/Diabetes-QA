import os
import re

from django.shortcuts import render
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import torch
from transformers import AutoTokenizer, AutoModel
from langchain.llms.base import LLM
from typing import Optional, List, Dict, Any
from langchain.pydantic_v1 import BaseModel, Field
from qa_system.models import KnowledgeBase
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import datetime
import os

from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# 全局设置
kb_setting = {
    "max_length": 100,
    "top_p": 0.7,
    "temperature": 0.2,
}


class ChatGLM3(LLM):
    tokenizer: Any = Field(exclude=True)
    model: Any = Field(exclude=True)
    max_new_tokens: int = Field(exclude=True)
    temperature: float = Field(exclude=True)
    top_k: int = Field(exclude=True)

    def __init__(self, tokenizer, model, max_new_tokens, temperature, top_k):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens

    @property
    def _llm_type(self) -> str:
        return "chatglm3-6b-32k"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
            print('模型调用时当前参数设置:', self.max_new_tokens, self.temperature, self.top_k)
            outputs = self.model.generate(**inputs, do_sample=True, max_new_tokens=self.max_new_tokens,
                                          temperature=self.temperature,
                                          top_p=self.top_k)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if not response.strip():
                return "抱歉，我无法生成回答。请尝试重新提问或提供更多上下文。"
            return response
        except Exception as e:
            print(f"模型生成错误: {e}")
            return "处理请求时出错，请稍后再试。"


# Create your views here.
def chat(request):
    return render(request, 'chat.html')


def person_setting(request):
    return render(request, 'django_set_person.html')


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb', ) as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text


@csrf_exempt
def create_knowledgebase(request):
    print('request.method', request.method)
    if request.method == 'POST':
        try:
            # 创建必要的目录结构
            temp_dir = '/root/autodl-tmp/pycharm-project/kb_tmp'
            os.makedirs(temp_dir, exist_ok=True)
            # 获取知识库名称
            kb_name = request.POST.get('kb_name')
            # 获取上传的文件
            files = request.FILES.getlist('documents')
            try:
                KnowledgeBase.objects.create(name=kb_name)
                print('KnowledgeBase created!', kb_name)
            except Exception as e:
                print('知识库导入数据库出错', e)
            print(kb_name)
            print(files)
            all_text = []
            # 在保存文件前添加：
            # 临时保存文件并提取文本
            for file in files:
                # 安全处理文件名
                safe_name = os.path.basename(file.name).replace('/', '_')
                temp_file_path = os.path.join(temp_dir, safe_name)

                # 保存文件
                with open(temp_file_path, 'wb+') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)
                text = extract_text_from_pdf(temp_file_path)
                all_text.append(text)
            documents = [Document(
                page_content=text,
            ) for text in all_text]
            for i, document in enumerate(documents):
                print('files[i].name', files[i].name)
                document.metadata = {"source": files[i].name}
            # 替换为实际模型路径（需包含config.json和模型权重）
            model_path = "/root/autodl-tmp/models/sentence-transformers/all-roberta-large-v1"
            embeddings = HuggingFaceEmbeddings(model_name=model_path)
            # 文档分块
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=400)
            chunks = text_splitter.split_documents(documents)

            # 创建FAISS向量库（仅使用本地模型）
            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings  # 使用本地模型生成嵌入
            )
            # 持久化存储
            persist_directory = f'/root/autodl-tmp/pycharm-project/kb_{kb_name}'
            vectorstore.save_local(persist_directory)
            print(f'向量数据库已保存到 {persist_directory}')
            return JsonResponse({
                'status': 'success',
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    else:
        return JsonResponse({'status': 'error', 'message': '仅支持POST请求'}, status=405)


def get_knowledgebases(request):
    # 获取所有知识库并按创建时间倒序排列
    knowledgebases = KnowledgeBase.objects.all().order_by('-create_time')
    kb_list = [{'id': kb.id, 'name': kb.name} for kb in knowledgebases]
    return JsonResponse({'knowledgebases': kb_list})


def load_models(max_new_tokens=kb_setting['max_length'], temperature=kb_setting['temperature'],
                top_k=kb_setting['top_p']):
    # 初始化ChatGLM3模型
    # model_path = "/root/autodl-tmp/models/ZhipuAI/chatglm3-6b-32k"
    model_path = "/root/autodl-tmp/models/new_model"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"当前使用的设备: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    model = model.eval()
    chatglm3_llm = ChatGLM3(tokenizer=tokenizer, model=model, max_new_tokens=max_new_tokens, temperature=temperature,
                            top_k=top_k)
    return chatglm3_llm


def load_kb(kb_base='糖尿病资料库1'):
    try:
        # 加载嵌入模型和向量数据库
        model_path = "/root/autodl-tmp/models/sentence-transformers/all-roberta-large-v1"
        embeddings = HuggingFaceEmbeddings(model_name=model_path)

        persist_directory = f'/root/autodl-tmp/pycharm-project/kb_{kb_base}'
        vectorstore = FAISS.load_local(
            persist_directory,
            embeddings,
            allow_dangerous_deserialization=True
        )
        kb_setting['kb_base'] = kb_base
        global retriever
        retriever =  vectorstore.as_retriever(search_kwargs={"k": 1})
        return retriever
    except Exception as e:
        print('加载知识库时发生错误',e)



# 初始化模型和向量数据库（全局变量，避免重复加载）
# 全局初始化（Django启动时加载）
chatglm3_llm, retriever = load_models(), load_kb()


@csrf_exempt
def smart_query(request):
    if request.method == 'POST':
        try:
            # 解析JSON请求体
            data = json.loads(request.body)
            query = data.get('query', '').strip()
            print(query)

            if not query:
                return JsonResponse({'error': '查询内容不能为空'}, status=400)

            # 设置系统提示
            system_prompt = """你是一名糖尿病研究领域的专家，擅长回答用户关于糖尿病的各种问题。
            要求：
            1. 精确、专业的回答用户的问题
            2. 优先参考以下知识库内容进行回答（如果有）：
            {context}
            3. 若无相关内容，请直接基于你的专业知识回答用户问题。
            4. 若用户问题与糖尿病无关，则直接进行简短的回答。
            """

            # 创建问答链
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")
            ])
            if kb_setting.get('kb_base'):
                question_answer_chain = create_stuff_documents_chain(chatglm3_llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            else:
                rag_chain = create_stuff_documents_chain(chatglm3_llm, prompt)

            # 执行查询
            try:
                response = rag_chain.invoke({"input": query})
                # print(" response['answer']", response['answer'])
            except Exception as e:
                print(e)
            answer_text = response['answer']
            prefix = f"Human: {query}"
            answer = answer_text.split(prefix, 1)[1].strip()
            if 'Expert:' or '专家:' in answer:
                answer = answer.replace('Expert:', '')
            # 准备返回数据
            result = {
                'answer': answer,
                'context': [doc.page_content for doc in response['context']]
            }

            return JsonResponse(result)

        except json.JSONDecodeError:
            return JsonResponse({'error': '无效的JSON格式'}, status=400)
        except Exception as e:
            print(f"处理请求时出错: {e}")
            return JsonResponse({'error': '处理请求时出错'}, status=500)

    return JsonResponse({'error': '只接受POST请求'}, status=405)


@csrf_exempt
def update_params(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            # 只更新三个参数
            kb_setting['max_length'] = int(data.get('max_length', kb_setting['max_length']))
            kb_setting['top_p'] = float(data.get('top_p', kb_setting['top_p']))
            kb_setting['temperature'] = float(data.get('temperature', kb_setting['temperature']))

            print('更新kb_setting', kb_setting)
            print('正在重新加载模型....')
            global chatglm3_llm
            if chatglm3_llm is not None:
                # 将模型移到 CPU 上
                chatglm3_llm.model.to('cpu')
                # 释放缓存
                del chatglm3_llm
                torch.cuda.empty_cache()
            chatglm3_llm = load_models(kb_setting['max_length'], kb_setting['top_p'], kb_setting['temperature'])
            print('成功重新加载模型....')
            return JsonResponse({
                'success': True,
                'message': '参数更新成功',
                'settings': kb_setting
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': f'参数更新失败: {str(e)}'
            }, status=400)
    return JsonResponse({
        'success': False,
        'message': '无效的请求方法'
    }, status=405)


@csrf_exempt
def load_knowledgebase(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            kb_name = data.get('kb_base', '')
            print('当前装载的知识库名称',kb_name)
            if kb_name == '无':
                kb_setting['kb_base'] = None
                return JsonResponse({
                    'success': True,
                    'message': f'设置成功！当前未引用任何知识库！',
                    'kb_base': kb_setting['kb_base']
                })
            # 加载嵌入模型和向量数据库
            model_path = "/root/autodl-tmp/models/sentence-transformers/all-roberta-large-v1"
            embeddings = HuggingFaceEmbeddings(model_name=model_path)

            persist_directory = f'/root/autodl-tmp/pycharm-project/kb_{kb_name}'
            print('当前知识库路径',persist_directory)
            vectorstore = FAISS.load_local(
                persist_directory,
                embeddings,
                allow_dangerous_deserialization=True
            )
            global retriever
            retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
            kb_setting['kb_base'] = kb_name
            print(kb_setting)
            return JsonResponse({
                'success': True,
                'message': f'知识库 {kb_name} 装载成功',
                'kb_base': kb_setting['kb_base']
            })
        except Exception as e:
            print('知识库装载失败', str(e))
            return JsonResponse({
                'success': False,
                'message': f'知识库装载失败: {str(e)}'
            }, status=400)
    return JsonResponse({
        'success': False,
        'message': '无效的请求方法'
    }, status=405)


@csrf_exempt
def unload_knowledgebase(request):
    if request.method == 'POST':
        try:
            # 直接将kb_base设为空字符串
            kb_setting['kb_base'] = ""

            return JsonResponse({
                'success': True,
                'message': '知识库卸载成功',
                'kb_base': kb_setting['kb_base']
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': f'知识库卸载失败: {str(e)}'
            }, status=400)
    return JsonResponse({
        'success': False,
        'message': '无效的请求方法'
    }, status=405)
