from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
# 从llm_template导入模板
from .llm_template import get_template_for_model
import torch.nn.functional as F
from .tokenfilter import TokenFilter
import time
import tqdm

def entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-6), dim=-1)

def miner(num_iterations, model, tokenizer, all_token_embeddings, no_need_tokens, batch_size=16, k=100, start_id=0, chat_template=None, if_print=True, print_language="CN"):
    device = model.device
    all_token_embeddings = all_token_embeddings.to(device)

    # 创建掩码，排除不需要的token
    mask = torch.ones(all_token_embeddings.shape[0], dtype=torch.bool, device=device)
    mask[no_need_tokens] = False

    # 获取模板格式
    system_format = chat_template.system_format
    user_format = chat_template.user_format
    assistant_prefill = ' Sure, the string is: "«'
    system_message = ''

    #valid_tokens = torch.where(mask)[0]
    current_token_id = start_id#valid_tokens[torch.randint(0, len(valid_tokens), (1,))].item() #start_id #

    total_tokens_checked = 0
    glitch_tokens_count = 0
    glitch_tokens = []
    glitch_token_ids = []

    for iteration in range(num_iterations):
        if if_print:
            print(f"迭代次数: {iteration}" if print_language == "CN" else f"Iteration: {iteration}")

        # 步骤1：计算当前token的熵值和梯度
        token = tokenizer.decode([current_token_id])
        user_prompt = f'Please repeat the string: "«{token}»"'
        formatted_user = user_format.format(content=user_prompt)
        formatted_input = (system_format.format(content=system_message) if system_format else "") + formatted_user + assistant_prefill
        quote_id = tokenizer.tokenizer.encode('"«',add_special_tokens=False)[-1]
        
        input_ids = tokenizer.encode(formatted_input)
        current_token_position = input_ids.index(quote_id)+1
        input_ids = torch.tensor([input_ids], device=device)
        inputs_embeds = model.get_input_embeddings()(input_ids).clone().detach()
        inputs_embeds.requires_grad_(True)
        

        outputs = model(inputs_embeds=inputs_embeds, use_cache=False)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_probs = F.softmax(next_token_logits, dim=-1)
        entropy_value = entropy(next_token_probs)

        grads = torch.autograd.grad(
            outputs=entropy_value,
            inputs=inputs_embeds,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )[0]

        # 步骤2：计算最近的k个token的近似熵
        current_token_embedding = model.lm_head.weight[current_token_id].detach().to(device)
        grad = grads[0, current_token_position, :].detach().to(device)

        valid_mask = mask.clone()
        valid_mask[current_token_id] = False
        # 首先获取模型权重所在的设备
        weight_device = model.lm_head.weight.device
        
        # 确保所有张量在同一设备上
        current_token_embedding = model.lm_head.weight[current_token_id].detach().to(weight_device)
        grad = grads[0, current_token_position, :].detach().to(weight_device)

        valid_mask = mask.clone()
        valid_mask[current_token_id] = False
        valid_token_ids = torch.where(valid_mask)[0].to(weight_device)
        valid_embeddings = model.lm_head.weight[valid_token_ids]

        # 标准化操作
        normalized_current = F.normalize(current_token_embedding.unsqueeze(0), p=2, dim=1)
        normalized_valid = F.normalize(valid_embeddings, p=2, dim=1)
        
        # 计算距离
        normalized_l2_distances = torch.norm(normalized_valid - normalized_current, p=2, dim=1)
        nearest_indices = torch.topk(normalized_l2_distances, k=min(k, len(valid_token_ids)), largest=False).indices

        candidate_token_ids = valid_token_ids[nearest_indices]

        delta_embeddings = valid_embeddings[nearest_indices] - current_token_embedding
        entropy_approximations = entropy_value.item() + (delta_embeddings @ grad).detach()

        # 步骤3：选取前batch_size个近似熵最大的token
        top_batch_indices = torch.topk(entropy_approximations, k=min(batch_size, len(candidate_token_ids))).indices
        batch_token_ids = candidate_token_ids[top_batch_indices]
        
        # 步骤4：评估是否为glitch token
        model.eval()
        with torch.no_grad():
            batch_entropies = []
            for xxx, token_id in enumerate(batch_token_ids):
                token = tokenizer.decode([token_id.item()])
                user_prompt = f'Please repeat the string: "«{token}»"'
                formatted_input = (system_format.format(content=system_message) if system_format else "") + user_format.format(content=user_prompt) + assistant_prefill
                input_ids = tokenizer.tokenizer.encode(formatted_input, return_tensors="pt").to(device)
                
                outputs = model(input_ids=input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                entropy_value = entropy(next_token_probs)
                batch_entropies.append(entropy_value.item())

                max_prob_token_id = next_token_probs.argmax().item()
                is_glitch = max_prob_token_id != token_id.item()
                total_tokens_checked += 1
                
                if is_glitch:
                    glitch_tokens_count += 1
                    glitch_tokens.append(token)
                    glitch_token_ids.append(token_id.item())

                if if_print:
                    print_str = f"  当前token: '{token}', token id: {token_id.item()}, 是否为glitch token: {'是' if is_glitch else '否'}, 熵值: {entropy_value.item():.4f}" if print_language == "CN" else f"  Current token: '{token}', token id: {token_id.item()}, is glitch token: {'Yes' if is_glitch else 'No'}, entropy: {entropy_value.item():.4f}"
                    print(print_str)

        # 步骤5：选择熵值最大的token进行下一次迭代
        max_entropy_index = torch.argmax(torch.tensor(batch_entropies))
        current_token_id = batch_token_ids[max_entropy_index].item()

        # 更新mask
        mask[batch_token_ids] = False

        # 清理内存
        del outputs, entropy_value, grads
        torch.cuda.empty_cache()

    glitch_frequency = glitch_tokens_count / total_tokens_checked if total_tokens_checked > 0 else 0
    if if_print:
        print(f"检测的总token数: {total_tokens_checked}" if print_language == "CN" else f"Total tokens checked: {total_tokens_checked}")
        print(f"glitch token数: {glitch_tokens_count}" if print_language == "CN" else f"Number of glitch tokens: {glitch_tokens_count}")
        print(f"glitch token出现的频率: {glitch_frequency:.4f}" if print_language == "CN" else f"Frequency of glitch tokens: {glitch_frequency:.4f}")

    return glitch_tokens, glitch_token_ids



def initialize_model_and_tokenizer(model_path, device="auto", quant_type="bfloat16"):
    # 设置设备映射
    if device == "auto":
        device_map = "auto"  # 使用所有可用的GPU
    elif device.startswith("cuda"):
        device_map = {"": device}  # 使用指定的单个GPU
    else:
        device_map = {"": device}  # 使用CPU或其他指定设备
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if quant_type == 'bfloat16':
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
    elif quant_type == 'float16':
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_threshold=4.0,
                llm_int8_skip_modules=["lm_head"]
            ),
        )
    
    model.requires_grad_(False)
    return model, tokenizer

def GlitchMiner(model, tokenizer, num_iterations, batch_size, k, if_print=True, print_language="CN", skip_tokens=None):
    # 获取嵌入
    embedding_device = next(model.get_input_embeddings().parameters()).device
    all_token_embeddings = model.get_input_embeddings().weight.detach().to(embedding_device)
    vocab_size = all_token_embeddings.shape[0]
    # 过滤token
    token_filter = TokenFilter(model, tokenizer)
    if skip_tokens is None:
        skip_tokens = token_filter.filter_token()
    else:
        skip_tokens = skip_tokens + token_filter.filter_token()
    
    # 获取聊天模板
    chat_template = get_template_for_model(model.config._name_or_path)
    norms = []
    all_token_embeddings_cpu = all_token_embeddings.cpu().clone()
    for token_id in range(vocab_size):
        if token_id in skip_tokens:
                continue
        l2_norm = torch.norm(all_token_embeddings_cpu[token_id], p=2)
        norms.append((token_id, l2_norm))

    norms.sort(key=lambda x: x[1])
    start_id = norms[0][0]

    glitch_tokens, glitch_token_ids = miner(
            num_iterations, model, token_filter, all_token_embeddings, skip_tokens, 
            batch_size=batch_size, k=k, start_id=start_id, chat_template=chat_template, if_print=if_print, print_language=print_language
        )
    return glitch_tokens, glitch_token_ids

def chat_token(model_path, token_id, max_size=10, device="auto", quant_type="bfloat16"):
    # 设置设备映射
    if device == "auto":
        device_map = "auto"  # 使用所有可用的GPU
    elif device.startswith("cuda"):
        device_map = {"": device}  # 使用指定的单个GPU
    else:
        device_map = {"": device}  # 使用CPU或其他指定设备
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if quant_type == 'bfloat16':
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_threshold=4.0,
                llm_int8_skip_modules=["lm_head"]
            ),
        )
    
    model.eval()
    
    tf = TokenFilter(model, tokenizer)
    chat_template = get_template_for_model(model_path)

    # 获取模板格式
    system_format = chat_template.system_format
    user_format = chat_template.user_format
    assistant_format = chat_template.assistant_format
    stop_word = chat_template.stop_word
    system_message = ''  # chat_template.system

    with torch.no_grad():
        token = tf.decode([token_id])
        user_prompt = f'''Please repeat the string: "«{token}»"'''
        assistant_prefill = ' Sure, the string is: "«'

        # 构建输入
        if system_format is not None:
            formatted_system = system_format.format(content=system_message)
        else:
            formatted_system = ""
        formatted_user = user_format.format(content=user_prompt)
        if assistant_prefill != '':
            formatted_input = formatted_system + formatted_user + assistant_prefill
        else:
            formatted_input = formatted_system + formatted_user
        
        # 对输入进行分词
        inputs = tf.tokenizer(formatted_input, return_tensors="pt").to(model.device)

        # 生成模型响应
        outputs = model.generate(**inputs, max_new_tokens=max_size, do_sample=False, output_scores=True, return_dict_in_generate=True)

        # 获取模型输出的第一个token的概率分布
        first_token_probs = torch.softmax(outputs.scores[0][0], dim=-1)

        # 计算检测token的概率
        good_prob = first_token_probs[token_id].item()
        
        # 获取概率最大的5个下标
        top_5_indices = torch.topk(first_token_probs, k=5).indices
        
        # 获取概率最大的token及其概率
        max_prob_index = torch.argmax(first_token_probs)
        max_prob_token = tokenizer.decode([max_prob_index])
        max_prob = first_token_probs[max_prob_index].item()

        # 解码生成的文本
        generated_text = tf.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)

        result = {
            "检测token id": token_id,
            "检测token": token,
            "模型输出的第一个token为检测token的概率": good_prob,
            "模型输出的第一个token位置上概率最大的5个下标": top_5_indices.tolist(),
            "概率最大的token": max_prob_token,
            "概率最大的token概率": max_prob,
            "概率最大的token下标": max_prob_index.item(),
            "大模型对话的结果": generated_text
        }

        return result


def glitch_verify_message1(chat_template,token):
    system_format = chat_template.system_format
    user_format = chat_template.user_format
    assistant_prefill = ' Sure, the string is: "«'
    system_message = ''
    user_prompt = f'Please repeat the string: "«{token}»"'
    
    formatted_system = system_format.format(content=system_message) if system_format else ""
    formatted_user = user_format.format(content=user_prompt)
    
    formatted_input = formatted_system + formatted_user + assistant_prefill
    
    return formatted_input

def glitch_verify_message2(chat_template,token):
    system_format = chat_template.system_format
    user_format = chat_template.user_format
    assistant_format = chat_template.assistant_format
    assistant_prefill="'"
    system_message = 'This helpful assistant explains what a phrase means, replying with identical responses to identical questions for reliability.'
    
    user_prompt1 = "What does 'edslfkj' mean?"
    assistant_answer1 = "'edslfkj' has no specific meaning I am aware of."
    user_prompt2 = f"What does '{token}' mean?"
    
    formatted_system = system_format.format(content=system_message) if system_format else ""
    formatted_user1 = user_format.format(content=user_prompt1)
    formatted_assitant1 = assistant_format.format(content=assistant_answer1)
    formatted_user2 = user_format.format(content=user_prompt2)
    
    formatted_input = formatted_system + formatted_user1 + formatted_assitant1 +formatted_user2 + assistant_prefill
    return formatted_input

def glitch_verify_message3(chat_template,token):
    system_format = chat_template.system_format
    user_format = chat_template.user_format
    assistant_prefill = " Of course! Here is the repeated string:'"
    system_message = ''
    user_prompt = f"Question: Please can you repeat the string '{token}'?"
    
    formatted_system = system_format.format(content=system_message) if system_format else ""
    formatted_user = user_format.format(content=user_prompt)
    
    formatted_input = formatted_system + formatted_user + assistant_prefill
    return formatted_input

def strictly_glitch_verify(model, tokenizer, token_id, chat_template=None):
    # 获取模型设备
    device = model.device
    
    # 解码 token_id 获取对应的文本
    token = tokenizer.decode([token_id])
    
    # 如果未提供 chat_template，则根据模型名称获取
    if chat_template is None:
        chat_template = get_template_for_model(model.config._name_or_path)
    
    formatted_input1 = glitch_verify_message1(chat_template,token)
    formatted_input2 = glitch_verify_message2(chat_template,token)
    formatted_input3 = glitch_verify_message3(chat_template,token)

    
    # 对输入进行分词
    input_ids1 = tokenizer.tokenizer.encode(formatted_input1, return_tensors='pt').to(device)
    input_ids2 = tokenizer.tokenizer.encode(formatted_input2, return_tensors='pt').to(device)
    input_ids3 = tokenizer.tokenizer.encode(formatted_input3, return_tensors='pt').to(device)

    inputs = [input_ids1,input_ids2,input_ids3]
    # 获取模型的输出
    is_glitch = True
    for input_ids in inputs:
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            # 获取下一个 token 的 logits
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            # 获取概率最大的 token 的 id
            predicted_token_id = next_token_probs.argmax(dim=-1).item()

            # 判断预测的 token 是否与输入的 token 一致
            is_glitch = predicted_token_id != token_id
        if not is_glitch:
            break
        
    return is_glitch

def strictly_glitch_verification(model, tokenizer, id_list):
    """检测整个 id_list 中的 glitch token 数量"""
    # 初始化 TokenFilter 过滤器
    token_filter = TokenFilter(model, tokenizer)
    skip_tokens = token_filter.filter_token()  # 获取需要跳过的 token
    
    glitch_count = 0  # 初始化 glitch token 计数
    verified_glitch_ids=[]
    # 遍历 id_list 并检测 glitch token
    for token_id in tqdm.tqdm(id_list, desc="检测 Glitch Token"):
        if token_id not in skip_tokens:
            if strictly_glitch_verify(model, token_filter, token_id):
                glitch_count += 1  # 计数器加 1
                verified_glitch_ids.append(token_id)
    return glitch_count, verified_glitch_ids


if __name__ == "__main__":
    model_path = "/home/hui/workspace/hf-hub/Qwen2.5-7B-Instruct/"
    model, tokenizer = initialize_model_and_tokenizer(model_path, device="auto", quant_type="float16")
    
    start_time = time.time()
    
    glitch_tokens, glitch_token_ids = GlitchMiner(
        model,
        tokenizer,
        num_iterations=50,
        batch_size=8,
        k=32,
        if_print=True,
        print_language="ENG",
    )
    
    end_time = time.time()
    runtime = end_time - start_time
    print(f"GlitchMiner 运行时间: {runtime:.2f} 秒")
    #result = chat_token(model_path="/data/wzh777/MLLMs/llama-2-7b-chat-hf", token_id=15879, max_size=1)
    #print(result)
