import gradio as gr
from swarm import create_swarm, get_model_config, Agent, CLIENT_CONFIG  # 导入 CLIENT_CONFIG
import time
import matplotlib.pyplot as plt
import random
from Levenshtein import distance
import os
import json

# API key configuration
API_KEY = None  # Remove hardcoded API key
client_type = "Default"  # Default client type
client = None  # Initialize client as None

# Get initial model configuration
model_config = get_model_config(client_type)
model_options = model_config["options"]
default_model = model_config["default"]


def calculate_loss(prediction: str, perfect_answer: str) -> float:
    """Calculate loss based on difference from perfect answer"""
    if prediction == perfect_answer:
        return 0.0

    base_loss = distance(prediction, perfect_answer) / max(len(prediction), len(perfect_answer))
    random_fluctuation = random.uniform(-0.1, 0.1)
    final_loss = base_loss + random_fluctuation

    return max(0.0, min(1.0, final_loss))


def create_agent(name: str, instructions: str, model: str) -> Agent:
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


def save_conversation_history(conversation_history, model_a, model_b, user_input, perfect_answer):
    """Save conversation history to a file"""
    if not os.path.exists("conversation_history"):
        os.makedirs("conversation_history")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"conversation_history/{timestamp}_modelA_{model_a}_modelB_{model_b}.json"

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(conversation_history, f, ensure_ascii=False, indent=4)


def update_client_config(new_client_type: str, api_key: str):
    """Update client and model configurations"""
    global client, model_options, default_model

    # Create new client instance
    client = create_swarm(new_client_type, api_key)

    # Get new model configuration
    model_config = get_model_config(new_client_type)
    model_options = model_config["options"]
    default_model = model_config["default"]

    return client, model_options, default_model, default_model, default_model


def chat_between_agents(model_a, model_b, user_input, perfect_answer, stop_event):
    """Main chat function between agents"""
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
        agent_a = create_agent("代理 A", "追求完美表达的代理", model_a)
        agent_b = create_agent("代理 B", "追求完美表达的代理", model_b)

        # Record user input
        conversation_history.append({
            "role": "user",
            "content": f"用户输入: {user_input}",
            "color": "purple"
        })

        # Get initial responses
        response_a = client.run(agent=agent_a, messages=[{"role": "user", "content": user_input}])
        response_b = client.run(agent=agent_b, messages=[{"role": "user", "content": user_input}])

        # Calculate initial losses
        loss_a = calculate_loss(response_a.messages[-1]['content'], perfect_answer)
        loss_b = calculate_loss(response_b.messages[-1]['content'], perfect_answer)

        losses_a.append(loss_a)
        losses_b.append(loss_b)

        # Add responses to history
        conversation_history.extend([
            {
                "role": "assistant",
                "content": f"代理 A ({model_a}) [Loss: {loss_a:.4f}]: {response_a.messages[-1]['content']}"
            },
            {
                "role": "assistant",
                "content": f"代理 B ({model_b}) [Loss: {loss_b:.4f}]: {response_b.messages[-1]['content']}"
            }
        ])

        # Update plots
        fig_a = plot_loss(losses_a, "Agent A")
        fig_b = plot_loss(losses_b, "Agent B")
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
            correction_a = client.run(agent=agent_a, messages=[{"role": "user", "content": prompt_a}])
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
            correction_b = client.run(agent=agent_b, messages=[{"role": "user", "content": prompt_b}])
            loss_b = calculate_loss(correction_b.messages[-1]['content'], perfect_answer)
            losses_b.append(loss_b)

            # Update history and responses
            conversation_history.extend([
                {
                    "role": "assistant",
                    "content": f"代理 A ({model_a}) [Loss: {loss_a:.4f}]: {correction_a.messages[-1]['content']}",
                    "color": "yellow"
                },
                {
                    "role": "assistant",
                    "content": f"代理 B ({model_b}) [Loss: {loss_b:.4f}]: {correction_b.messages[-1]['content']}",
                    "color": "yellow"
                }
            ])

            response_a = correction_a
            response_b = correction_b

            # Update plots
            fig_a = plot_loss(losses_a, "Agent A")
            fig_b = plot_loss(losses_b, "Agent B")
            yield conversation_history, fig_a, fig_b

        # Summary
        if not stop_event:
            best_loss_a = min(losses_a)
            best_loss_b = min(losses_b)

            conversation_history.append({
                "role": "assistant",
                "content": f"对话结束:\n"
                           f"代理 A 最佳 Loss: {best_loss_a:.4f}\n"
                           f"代理 B 最佳 Loss: {best_loss_b:.4f}\n"
                           f"表现更好的代理: {'代理 A' if best_loss_a < best_loss_b else '代理 B'}"
            })

            save_conversation_history(conversation_history, model_a, model_b, user_input, perfect_answer)

            fig_a = plot_loss(losses_a, "Agent A")
            fig_b = plot_loss(losses_b, "Agent B")
            yield conversation_history, fig_a, fig_b

    except Exception as e:
        conversation_history.append({
            "role": "assistant",
            "content": f"发生错误: {str(e)}"
        })
        fig_a = plot_loss(losses_a, "Agent A")
        fig_b = plot_loss(losses_b, "Agent B")
        yield conversation_history, fig_a, fig_b


def clear_chat():
    """Clear chat history and plots"""
    return [], plot_loss([], "Agent A"), plot_loss([], "Agent B")


def stop_chat(stop_event):
    """Stop chat interaction"""
    return True


# Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# Swarm代理对话优化系统")

    with gr.Row():
        conversation_history = gr.Chatbot(type="messages", label="对话历史")
        with gr.Column():
            gr.Markdown("### Loss曲线 (与完美答案的差距)")
            plot_a = gr.Plot(label="代理 A Loss")
            plot_b = gr.Plot(label="代理 B Loss")

    stop_event = gr.State(value=False)

    with gr.Row():
        with gr.Column():
            client_type_dropdown = gr.Dropdown(
                label="选择厂家客户端",
                choices=list(CLIENT_CONFIG.keys()),
                value="Default"
            )
            api_key_input = gr.Textbox(label="输入API Key", placeholder="输入API Key", visible=False)
            model_a = gr.Dropdown(
                label="模型 A",
                choices=model_options,
                value=default_model
            )
            model_b = gr.Dropdown(
                label="模型 B",
                choices=model_options,
                value=default_model
            )
            user_input = gr.Textbox(label="输入对话", placeholder="输入对话内容")
            perfect_answer = gr.Textbox(label="完美答案（仅用于评估）", placeholder="输入完美答案")

            with gr.Row():
                submit_btn = gr.Button("开始对话")
                stop_btn = gr.Button("停止对话")
                clear_btn = gr.Button("清空对话")

    # Event handlers
    submit_btn.click(
        fn=chat_between_agents,
        inputs=[model_a, model_b, user_input, perfect_answer, stop_event],
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

    # Client type change handler
    client_type_dropdown.change(
        fn=lambda client_type: (
            gr.update(visible=True),
            gr.update(choices=get_model_config(client_type)["options"]),
            gr.update(choices=get_model_config(client_type)["options"])
        ),
        inputs=[client_type_dropdown],
        outputs=[api_key_input, model_a, model_b]
    )

    # API key input handler
    api_key_input.submit(
        fn=update_client_config,
        inputs=[client_type_dropdown, api_key_input],
        outputs=[client, model_a, model_b, model_a, model_b]
    )

# Launch the interface
if __name__ == "__main__":
    iface.launch()