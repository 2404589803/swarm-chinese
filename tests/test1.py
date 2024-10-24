import gradio as gr
from swarm import create_swarm, get_model_config, Agent, CLIENT_CONFIG
import time
import matplotlib.pyplot as plt
import random
from Levenshtein import distance
import os
import json

# 存储不同厂家的客户端
clients = {}
model_configs = {}


def initialize_model_configs():
    """初始化所有厂家的模型配置"""
    global model_configs
    for vendor in CLIENT_CONFIG.keys():
        model_configs[vendor] = get_model_config(vendor)
    return model_configs


# 初始化模型配置
model_configs = initialize_model_configs()


def calculate_loss(prediction: str, perfect_answer: str) -> float:
    """Calculate loss based on difference from perfect answer"""
    if prediction == perfect_answer:
        return 0.0

    base_loss = distance(prediction, perfect_answer) / max(len(prediction), len(perfect_answer))
    random_fluctuation = random.uniform(-0.1, 0.1)
    final_loss = base_loss + random_fluctuation

    return max(0.0, min(1.0, final_loss))


def create_agent(name: str, instructions: str, vendor: str, model: str) -> Agent:
    """Create an agent with specified parameters"""
    return Agent(
        name=name,
        instructions=instructions,
        model=model
    )


def plot_loss(losses: list, agent_name: str):
    """Create a loss plot for an agent"""
    plt.figure(figsize=(8, 4))
    plt.plot(losses, 'o-', label=f'{agent_name} Loss')
    plt.title(f'{agent_name} Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (Distance from Perfect Answer)')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    fig = plt.gcf()
    plt.close()
    return fig


def save_conversation_history(conversation_history, vendor_a, model_a, vendor_b, model_b, user_input, perfect_answer):
    """Save conversation history to a file"""
    if not os.path.exists("conversation_history"):
        os.makedirs("conversation_history")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"conversation_history/{timestamp}_{vendor_a}_{model_a}_{vendor_b}_{model_b}.json"

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(conversation_history, f, ensure_ascii=False, indent=4)


def get_vendor_models(vendor: str):
    """获取指定厂家的模型列表"""
    return model_configs[vendor]["options"]


def update_model_choices(vendor_a: str, vendor_b: str):
    """更新模型选择下拉框的选项"""
    models_a = get_vendor_models(vendor_a)
    models_b = get_vendor_models(vendor_b)
    return (
        gr.update(choices=models_a, value=models_a[0]),
        gr.update(choices=models_b, value=models_b[0])
    )


def update_api_keys(vendor_a: str, api_key_a: str, vendor_b: str, api_key_b: str):
    """更新API密钥并创建客户端"""
    global clients
    status_messages_a = []
    status_messages_b = []

    try:
        if api_key_a:
            clients[vendor_a] = create_swarm(vendor_a, api_key_a)
            status_messages_a.append(f"{vendor_a} API Key设置成功")
        else:
            clients.pop(vendor_a, None)
            status_messages_a.append(f"{vendor_a} API Key已清除")
    except Exception as e:
        status_messages_a.append(f"{vendor_a} API Key设置失败: {str(e)}")

    try:
        if api_key_b:
            clients[vendor_b] = create_swarm(vendor_b, api_key_b)
            status_messages_b.append(f"{vendor_b} API Key设置成功")
        else:
            clients.pop(vendor_b, None)
            status_messages_b.append(f"{vendor_b} API Key已清除")
    except Exception as e:
        status_messages_b.append(f"{vendor_b} API Key设置失败: {str(e)}")

    return "\n".join(status_messages_a), "\n".join(status_messages_b)


def chat_between_agents(vendor_a, model_a, vendor_b, model_b, user_input, perfect_answer, stop_event):
    """Main chat function between agents"""
    if not (clients.get(vendor_a) and clients.get(vendor_b)):
        return (
            [{"role": "assistant", "content": "请先设置两个厂家的API Key", "color": "red"}],
            plot_loss([], "Agent A"),
            plot_loss([], "Agent B")
        )

    if not perfect_answer:
        return (
            [{"role": "assistant", "content": "请输入完美答案用于评估", "color": "red"}],
            plot_loss([], "Agent A"),
            plot_loss([], "Agent B")
        )

    conversation_history = []
    losses_a = []
    losses_b = []

    try:
        # Create agents
        agent_a = create_agent(f"代理 A ({vendor_a})", "追求完美表达的代理", vendor_a, model_a)
        agent_b = create_agent(f"代理 B ({vendor_b})", "追求完美表达的代理", vendor_b, model_b)

        # Record user input
        conversation_history.append({
            "role": "user",
            "content": f"用户输入: {user_input}",
            "color": "purple"
        })

        # Get initial responses
        response_a = clients[vendor_a].run(agent=agent_a, messages=[{"role": "user", "content": user_input}],
                                           selected_model=model_a)
        response_b = clients[vendor_b].run(agent=agent_b, messages=[{"role": "user", "content": user_input}],
                                           selected_model=model_b)

        # Calculate initial losses
        loss_a = calculate_loss(response_a.messages[-1]['content'], perfect_answer)
        loss_b = calculate_loss(response_b.messages[-1]['content'], perfect_answer)

        losses_a.append(loss_a)
        losses_b.append(loss_b)

        # Add responses to history
        conversation_history.extend([
            {
                "role": "assistant",
                "content": f"代理 A ({vendor_a}-{model_a}) [Loss: {loss_a:.4f}]: {response_a.messages[-1]['content']}"
            },
            {
                "role": "assistant",
                "content": f"代理 B ({vendor_b}-{model_b}) [Loss: {loss_b:.4f}]: {response_b.messages[-1]['content']}"
            }
        ])

        # Update plots
        fig_a = plot_loss(losses_a, f"Agent A ({vendor_a})")
        fig_b = plot_loss(losses_b, f"Agent B ({vendor_b})")
        yield conversation_history, fig_a, fig_b

        # Iteration loop
        for iteration in range(5):
            if stop_event:
                break

            # Agent A improvement
            prompt_a = f"""
            请尝试改进你的表达，使其更准确、清晰和完整。考虑：
            1. 是否有遗漏的重要信息？
            2. 表达是否足够准确？
            3. 逻辑是否连贯？

            你的当前回答: {response_a.messages[-1]['content']}
            """
            correction_a = clients[vendor_a].run(agent=agent_a, messages=[{"role": "user", "content": prompt_a}],
                                                 selected_model=model_a)
            loss_a = calculate_loss(correction_a.messages[-1]['content'], perfect_answer)
            losses_a.append(loss_a)

            # Agent B improvement
            prompt_b = f"""
            请尝试改进你的表达，使其更准确、清晰和完整。考虑：
            1. 是否有遗漏的重要信息？
            2. 表达是否足够准确？
            3. 逻辑是否连贯？

            你的当前回答: {response_b.messages[-1]['content']}
            """
            correction_b = clients[vendor_b].run(agent=agent_b, messages=[{"role": "user", "content": prompt_b}],
                                                 selected_model=model_b)
            loss_b = calculate_loss(correction_b.messages[-1]['content'], perfect_answer)
            losses_b.append(loss_b)

            # Update history and responses
            conversation_history.extend([
                {
                    "role": "assistant",
                    "content": f"代理 A ({vendor_a}-{model_a}) [Loss: {loss_a:.4f}]: {correction_a.messages[-1]['content']}",
                    "color": "yellow"
                },
                {
                    "role": "assistant",
                    "content": f"代理 B ({vendor_b}-{model_b}) [Loss: {loss_b:.4f}]: {correction_b.messages[-1]['content']}",
                    "color": "yellow"
                }
            ])

            response_a = correction_a
            response_b = correction_b

            # Update plots
            fig_a = plot_loss(losses_a, f"Agent A ({vendor_a})")
            fig_b = plot_loss(losses_b, f"Agent B ({vendor_b})")
            yield conversation_history, fig_a, fig_b

        # Summary
        if not stop_event:
            best_loss_a = min(losses_a)
            best_loss_b = min(losses_b)

            conversation_history.append({
                "role": "assistant",
                "content": f"对话结束:\n"
                           f"代理 A ({vendor_a}-{model_a}) 最佳 Loss: {best_loss_a:.4f}\n"
                           f"代理 B ({vendor_b}-{model_b}) 最佳 Loss: {best_loss_b:.4f}\n"
                           f"表现更好的代理: {'代理 A' if best_loss_a < best_loss_b else '代理 B'}"
            })

            save_conversation_history(conversation_history, vendor_a, model_a, vendor_b, model_b, user_input,
                                      perfect_answer)

            fig_a = plot_loss(losses_a, f"Agent A ({vendor_a})")
            fig_b = plot_loss(losses_b, f"Agent B ({vendor_b})")
            yield conversation_history, fig_a, fig_b

    except Exception as e:
        conversation_history.append({
            "role": "assistant",
            "content": f"发生错误: {str(e)}"
        })
        fig_a = plot_loss(losses_a, f"Agent A ({vendor_a})")
        fig_b = plot_loss(losses_b, f"Agent B ({vendor_b})")
        yield conversation_history, fig_a, fig_b


def clear_chat():
    """Clear chat history and plots"""
    return [], plot_loss([], "Agent A"), plot_loss([], "Agent B")


def stop_chat(stop_event):
    """Stop chat interaction"""
    return True


# Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# Swarm跨厂商代理对话优化系统")

    with gr.Row():
        conversation_history = gr.Chatbot(type="messages", label="对话历史")
        with gr.Column():
            gr.Markdown("### Loss曲线 (与完美答案的差距)")
            plot_a = gr.Plot(label="代理 A Loss")
            plot_b = gr.Plot(label="代理 B Loss")

    stop_event = gr.State(value=False)

    with gr.Row():
        with gr.Column():
            with gr.Row():
                vendor_a = gr.Dropdown(
                    label="代理A厂家",
                    choices=list(CLIENT_CONFIG.keys()),
                    value=list(CLIENT_CONFIG.keys())[0]
                )
                vendor_b = gr.Dropdown(
                    label="代理B厂家",
                    choices=list(CLIENT_CONFIG.keys()),
                    value=list(CLIENT_CONFIG.keys())[0]
                )

            with gr.Row():
                api_key_a = gr.Textbox(label="代理A API Key", placeholder="输入API Key", type="password")
                api_key_b = gr.Textbox(label="代理B API Key", placeholder="输入API Key", type="password")

            with gr.Row():
                api_status_a = gr.Textbox(label="代理A API状态", value="未设置API Key", interactive=False)
                api_status_b = gr.Textbox(label="代理B API状态", value="未设置API Key", interactive=False)

            with gr.Row():
                model_a = gr.Dropdown(
                    label="代理A模型",
                    choices=get_vendor_models(list(CLIENT_CONFIG.keys())[0]),
                    value=get_vendor_models(list(CLIENT_CONFIG.keys())[0])[0]
                )
                model_b = gr.Dropdown(
                    label="代理B模型",
                    choices=get_vendor_models(list(CLIENT_CONFIG.keys())[0]),
                    value=get_vendor_models(list(CLIENT_CONFIG.keys())[0])[0]
                )

            user_input = gr.Textbox(label="输入对话", placeholder="输入对话内容")
            perfect_answer = gr.Textbox(label="完美答案（仅用于评估）", placeholder="输入完美答案")

            with gr.Row():
                submit_btn = gr.Button("开始对话")
                stop_btn = gr.Button("停止对话")
                clear_btn = gr.Button("清空对话")

    # Event handlers
    vendor_a.change(
        fn=update_model_choices,
        inputs=[vendor_a, vendor_b],
        outputs=[model_a, model_b]
    ).then(
        fn=lambda: gr.update(value=""),
        outputs=[api_key_a]
    )

    vendor_b.change(
        fn=update_model_choices,
        inputs=[vendor_a, vendor_b],
        outputs=[model_a, model_b]
    ).then(
        fn=lambda: gr.update(value=""),
        outputs=[api_key_b]
    )

    api_key_a.submit(
        fn=update_api_keys,
        inputs=[vendor_a, api_key_a, vendor_b, api_key_b],
        outputs=[api_status_a, api_status_b]
    )

    api_key_b.submit(
        fn=update_api_keys,
        inputs=[vendor_a, api_key_a, vendor_b, api_key_b],
        outputs=[api_status_a, api_status_b]
    )

    submit_btn.click(
        fn=chat_between_agents,
        inputs=[vendor_a, model_a, vendor_b, model_b, user_input, perfect_answer, stop_event],
        outputs=[conversation_history, plot_a, plot_b]
    )

    stop_btn.click(
        fn=stop_chat,
        inputs=[stop_event],
        outputs=[stop_event]
    )

    clear_btn.click(
        fn=clear_chat,
        outputs=[conversation_history, plot_a, plot_b]
    )

# Launch the interface
if __name__ == "__main__":
    iface.launch()