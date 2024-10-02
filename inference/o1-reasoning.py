# app_qwen.py
import os
import json
import time
import streamlit as st
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

def make_api_call(messages, max_tokens, is_final_answer=False):
    for attempt in range(3):
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_tokens
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            content = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"Raw API response: {content}")  # 添加此行来打印原始响应
            try:
                return json.loads(content)
            except json.JSONDecodeError as json_error:
                print(f"JSON解析错误: {json_error}")
                # 如果JSON解析失败，返回一个包含原始内容的字典
                return {
                    "title": "API Response",
                    "content": content,
                    "next_action": "final_answer" if is_final_answer else "continue"
                }
        except Exception as e:
            if attempt == 2:
                return {
                    "title": "Error",
                    "content": f"Failed after 3 attempts. Error: {str(e)}",
                    "next_action": "final_answer"
                }
            time.sleep(1)  # 重试前等待1秒

def generate_response(prompt):
    messages = [
        {"role": "system", "content": """
        你是一位具有高级推理能力的专家。你的任务是提供详细的、逐步的思维过程解释。对于每一步:
        1. 提供一个清晰、简洁的标题,描述当前的推理阶段。
        2. 在内容部分详细阐述你的思维过程。
        3. 决定是继续推理还是提供最终答案。

        输出格式说明:
        输出请严格遵循JSON格式, 包含以下键: 'title', 'content', 'next_action'(值只能为'continue' 或 'final_answer'二者之一)

        关键指示:
        - 至少使用5个不同的推理步骤。
        - 承认你作为AI的局限性,明确说明你能做什么和不能做什么。
        - 主动探索和评估替代答案或方法。
        - 批判性地评估你自己的推理;识别潜在的缺陷或偏见。
        - 当重新审视时,采用根本不同的方法或视角。
        - 至少使用3种不同的方法来得出或验证你的答案。
        - 在你的推理中融入相关的领域知识和最佳实践。
        - 在适用的情况下,量化每个步骤和最终结论的确定性水平。
        - 考虑你推理中可能存在的边缘情况或例外。
        - 为排除替代假设提供清晰的理由。

        示例JSON输出:
        {
            "title": "初步问题分析",
            "content": "为了有效地解决这个问题,我首先会将给定的信息分解为关键组成部分。这涉及到识别...[详细解释]...通过这样构建问题,我们可以系统地解决每个方面。",
            "next_action": "continue"
        }

        记住: 全面性和清晰度至关重要。每一步都应该为最终解决方案提供有意义的进展。
        再次提醒: 输出请务必严格遵循JSON格式, 包含以下键: 'title', 'content', 'next_action'(值只能为'continue' 或 'final_answer'二者之一)
        """},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "现在我将一步步思考，从分析问题开始并将问题分解。"}
    ]
    
    steps = []
    step_count = 1
    total_thinking_time = 0
    
    while True:
        start_time = time.time()
        step_data = make_api_call(messages, 1024)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time
        
        title = step_data.get('title', f'Step {step_count}')
        content = step_data.get('content', 'No content provided')
        next_action = step_data.get('next_action', 'continue')
        
        steps.append((f"Step {step_count}: {title}", content, thinking_time))
        
        messages.append({"role": "assistant", "content": json.dumps(step_data)})
        
        if next_action == 'final_answer' or step_count > 25: # 最多25步，以防止无限的思考。可以适当调整。
            break
        
        step_count += 1

        yield steps, None  # 在结束时生成总时间

    # 生成最终答案
    messages.append({"role": "user", "content": "请根据你上面的推理提供最终答案。"})
    
    start_time = time.time()
    final_data = make_api_call(messages, 1000, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time
    
    final_content = final_data.get('content', '没有推理出最终结果')
    steps.append(("最终推理结果", final_content, thinking_time))

    yield steps, total_thinking_time

def main():
    st.set_page_config(page_title="Qwen2.5 o1-like Reasoning Chain", page_icon="💬", layout="wide")
    
    st.title("Qwen2.5实现类似o1 model的推理链")
    
    st.markdown("""
    通过vLLM部署调用Qwen2.5-7B-Instruct并实现类似OpenAI o1 model的长推理链效果以提高对复杂问题的推理准确性。
    """)
    
    # 用户输入查询
    user_query = st.text_input("输入问题:", placeholder="示例：strawberry中有多少个字母r？")
    
    if user_query:
        st.write("正在生成推理链中...")
        
        # 创建空元素以保存生成的文本和总时间
        response_container = st.empty()
        time_container = st.empty()
        
        # 生成并显示响应
        for steps, total_thinking_time in generate_response(user_query):
            with response_container.container():
                for i, (title, content, thinking_time) in enumerate(steps):
                    if title.startswith("最终推理结果"):
                        st.markdown(f"### {title}")
                        st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
                    else:
                        with st.expander(title, expanded=True):
                            st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
            
            # 仅在结束时显示总时间
            if total_thinking_time is not None:
                time_container.markdown(f"**总推理时间: {total_thinking_time:.2f} 秒**")

if __name__ == "__main__":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2___5-7B-Instruct",
        torch_dtype="auto",
        device_map="auto",
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2___5-7B-Instruct")
    
    main()
    # 启动
    # streamlit run o1-reasoning.py --server.address 127.0.0.1 --server.port 6006