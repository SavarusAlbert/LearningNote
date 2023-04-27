# Chapter01 ChatGPT基础操作
## 1.1 账号及API接口注册流程
- [中国区注册OpenAI账号教程参考](https://readdevdocs.com/blog/makemoney/%E4%B8%AD%E5%9B%BD%E5%8C%BA%E6%B3%A8%E5%86%8COpenAI%E8%B4%A6%E5%8F%B7%E8%AF%95%E7%94%A8ChatGPT%E6%8C%87%E5%8D%97.html#%E5%89%8D%E6%9C%9F%E5%87%86%E5%A4%87)
- 遇到Access denied、不支持地区等连接问题，尝试更换代理。
- [OpenAI API接口获取方法](https://zhuanlan.zhihu.com/p/613782276)
- [ProxyError错误解决方法](https://zhuanlan.zhihu.com/p/350015032)

## 1.2 API基本使用方法
- 安装 `openai` 工具包
```powershell
pip install openai
```
- 导入工具包
```python
import openai
OPENAI_API_KEY = "填入专属的API key"
openai.api_key = OPENAI_API_KEY
```
- `Completion` 接口(可以理解为续写)：
    - model：指定的模型，[可调用模型参考资料](https://zhuanlan.zhihu.com/p/608309509)。
    - prompt：提示，默认为 `<|endoftext|>`，它是模型在训练期间看到的文档分隔符，因此如果未指定Prompt，模型将像从新文档的开始一样。简单来说，就是给模型的提示语。
    - max_tokens：生成的最大Token数，默认为16。
    - temperature：采样温度，默认为1，介于0和2之间。通常建议调整这个参数或下面的top_p，但不能同时更改两者。
    - top_p：采样topN分布，默认为1。0.1意味着Next Token只选择前10%概率的。
    - stop：停止的Token或序列，默认为null，最多4个，如果遇到该Token或序列就停止继续生成。注意生成的结果中不包含stop。
    - presence_penalty：存在惩罚，默认为0，介于-2.0和2.0之间的数字。正值会根据新Token到目前为止是否出现在文本中来惩罚它们，从而增加模型讨论新主题的可能性。
    - frequency_penalty：频率惩罚，默认为0，介于-2.0和2.0之间的数字。正值会根据新Token到目前为止在文本中的现有频率来惩罚新Token，降低模型重复生成同一行的可能性。
```python
def complete(prompt):
    response = openai.Completion.create(
      model="text-davinci-003",                     # 模型引擎
      prompt=prompt,                                # 提的问题
      temperature=0,                                # 生成文本时的随机性、多样性
      max_tokens=64,                                # 生成文本的最长长度
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )
    ans = response.choices[0].text
    return ans
```
- `ChatGPT Style` (可以理解为对话，也就是ChatGPT，有关提示词可以参考[Prompt 工程](https://yam.gift/2023/01/25/NLP/2023-01-25-ChatGPT-Prompt-Engineering/))：
    - model：指定的模型，gpt-3.5-turbo就是ChatGPT。
    - messages：会话消息，支持多轮，多轮就是多条。每一条消息为一个字典，包含「role」和「content」两个字段。如：[{"role": "user", "content": "Hello!"}]
    - 其他参数与 `Completion` 接口中类似。
```python
def ask(content):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": content}]
    )
    ans = response.get("choices")[0].get("message").get("content")
    return ans
```
- 多轮对话函数(注意以下方法会带来额外的tokens消耗，因为每次传入的tokens包含了之前的所有tokens)
```python
class Chat:
    def __init__(self,conversation_list=[]):
        self.conversation_list = []

    def show_conversation(self,msg_list):
        for msg in msg_list:
            if msg['role'] == 'user':
                print(f"问: {msg['content']}\n")
            else:
                print(f"答: {msg['content']}\n")

    def ask(self,prompt):
        self.conversation_list.append({"role":"user","content":prompt})
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=self.conversation_list)
        answer = response.choices[0].message['content']
        self.conversation_list.append({"role":"assistant","content":answer})
        self.show_conversation(self.conversation_list)
```