# tests/test1.py
from swarm import Swarm, Agent
import gradio as gr

# 假设你的 API 密钥是 "your_api_key"
api_key = "25695d738fa54b0d28d36a3fcaa49ee7.rZ7GIwYQXz1qYFIo"

# 创建 Swarm 实例时传入 api_key
client = Swarm(api_key=api_key)

def transfer_to_agent_b():
    return agent_b

agent_a_model = "glm-4-0520"  # 定义代理 A 的模型名称
agent_b_model = "glm-4-flash"  # 定义代理 B 的模型名称

agent_a = Agent(
    name="代理 A",
    instructions="你是一个乐于助人的代理。",
    functions=[transfer_to_agent_b],
    model=agent_a_model  # 添加模型参数
)

agent_b = Agent(
    name="代理 B",
    instructions="只用俳句交流。",
    model=agent_b_model  # 添加模型参数
)

def chat_with_agents(user_input):
    conversation_history = []  # 用于存储对话历史

    try:
        # 获取代理 A 和代理 B 的回复，并进行循环纠错
        while True:
            # 获取代理 A 的回复
            response_a = client.run(
                agent=agent_a,
                messages=[{"role": "user", "content": user_input}],
            )

            # 获取代理 B 的回复
            response_b = client.run(
                agent=agent_b,
                messages=[{"role": "user", "content": user_input}],
            )

            # 打印当前的回复
            conversation_history.append(f"代理 A ({agent_a_model}) 的回复: {response_a.messages[-1]['content']}")
            conversation_history.append(f"代理 B ({agent_b_model}) 的回复: {response_b.messages[-1]['content']}")

            # 更新输出
            yield "\n".join(conversation_history)

            # 检查两个模型的回复是否一致
            if response_a.messages[-1]["content"] == response_b.messages[-1]["content"]:
                break  # 如果一致，退出循环

            # 让代理 A 和代理 B 进行对话以纠正错误
            correction_input = f"代理 A 说: {response_a.messages[-1]['content']}，代理 B 说: {response_b.messages[-1]['content']}。请纠正错误。"
            response_a = client.run(
                agent=agent_a,
                messages=[{"role": "user", "content": correction_input}],
            )
            response_b = client.run(
                agent=agent_b,
                messages=[{"role": "user", "content": correction_input}],
            )

            # 打印纠正后的回复
            conversation_history.append(f"纠正后的代理 A ({agent_a_model}) 的回复: {response_a.messages[-1]['content']}")
            conversation_history.append(f"纠正后的代理 B ({agent_b_model}) 的回复: {response_b.messages[-1]['content']}")

            # 更新输出
            yield "\n".join(conversation_history)

    except Exception as e:
        # 捕获异常并打印错误信息
        error_message = f"发生错误: {str(e)}"
        yield "\n".join(conversation_history) + "\n" + error_message  # 返回对话历史和错误信息

# 创建 Gradio 界面
iface = gr.Interface(fn=chat_with_agents, inputs="text", outputs="text", title="代理对话纠错系统", description="输入一句话，代理 A 和代理 B 将进行对话并纠正错误。", live=False)

# 启动 Gradio 应用
iface.launch()