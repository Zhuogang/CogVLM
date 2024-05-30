# -*- encoding: utf-8 -*-
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from sat.model.mixins import CachedAutoregressiveMixin
from sat.quantization.kernels import quantize
from sat.model import AutoModel

from utils.utils import chat, llama2_tokenizer, llama2_text_processor_inference, get_image_processor
from utils.models import CogAgentModel, CogVLMModel
import cv2
import re
from tqdm import tqdm
import pandas as pd


def process_image(image_path, model, text_processor_infer, image_processor, cross_image_processor, query, args):
    with torch.no_grad():
        history = None
        cache_image = None

        try:
            response, history, cache_image = chat(
                image_path,
                model,
                text_processor_infer,
                image_processor,
                query,
                history=history,
                cross_img_processor=cross_image_processor,
                image=cache_image,
                max_length=args.max_length,
                top_p=args.top_p,
                temperature=args.temperature,
                top_k=args.top_k,
                invalid_slices=text_processor_infer.invalid_slices,
                args=args
            )
            return response
        except Exception as e:
            print(e)
            return None

def main(input_path, query):
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=0.01, help='temperature for sampling')
    parser.add_argument("--version", type=str, default="chat_old", choices=['chat', 'vqa', 'chat_old', 'base'], help='version of language process. if there is \"text_processor_version\" in model_config.json, this option will be overwritten')
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None, help='quantization bits')
    parser.add_argument("--from_pretrained", type=str, default="cogagent-chat", help='pretrained ckpt')
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", default=True)
    parser.add_argument("--stream_chat", default=True)
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    args = parser.parse_args()

    # Load model
    model, model_args = AutoModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
            deepspeed=None,
            local_rank=rank,
            rank=rank,
            world_size=world_size,
            model_parallel_size=world_size,
            mode='inference',
            skip_init=True,
            use_gpu_initialization=True if (torch.cuda.is_available() and args.quant is None) else False,
            device='cpu' if args.quant else 'cuda',
            **vars(args)
        ), overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {}
    )

    model = model.eval()

    from sat.mpu import get_model_parallel_world_size

    assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

    language_processor_version = model_args.text_processor_version if 'text_processor_version' in model_args else args.version

    print("[Language processor version]:", language_processor_version)

    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=language_processor_version)

    image_processor = get_image_processor(model_args.eva_args["image_size"][0])

    cross_image_processor = get_image_processor(model_args.cross_image_pix) if "cross_image_pix" in model_args else None

    if args.quant:
        quantize(model, args.quant)
        if torch.cuda.is_available():
            model = model.cuda()

    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

    text_processor_infer = llama2_text_processor_inference(tokenizer, args.max_length, model.image_length)

    # Process input path
    responses = {}

    if os.path.isfile(input_path):
        response = process_image(input_path, model, text_processor_infer, image_processor, cross_image_processor, query, args)
        if response:
            responses.append(response)
    elif os.path.isdir(input_path):
        valid_extensions = {".jpg", ".jpeg", ".png"}
        for image_name in tqdm(os.listdir(input_path)):
            image_path = os.path.join(input_path, image_name)
            if os.path.isfile(image_path) and os.path.splitext(image_path)[1].lower() in valid_extensions:
                response = process_image(image_path, model, text_processor_infer, image_processor, cross_image_processor, query, args)
                if response:
                    responses[image_name] = response

    # Convert dictionary to DataFrame
    df = pd.DataFrame.from_dict(responses, orient='index').reset_index()

    # Rename columns
    df.columns = ['img_name', 'description']
    df.to_csv("results.csv", index=False)

    return responses



def write_response_on_image(image_path, response, output_folder="outputs"):
    # Read the image
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    
    # Define the font, size, color, and thickness
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2  # Adjust this value to make the font bigger
    color = (255, 0, 0)  # Blue color in BGR
    thickness = 10  # Adjust thickness if needed
    position = (10, int(h / 2))  # Adjust position if needed
    
    # Put the response text on the image
    cv2.putText(image, response, position, font, font_scale, color, thickness, cv2.LINE_AA)
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Define the output path
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    
    # Save the image
    cv2.imwrite(output_path, image)


def extract_keywords(output_text):
    # Define the keywords to look for
    keywords = ["good coverage", "bad coverage", "clear", "blurry"]
    
    # Use regular expressions to find the keywords in the output text
    found_keywords = []
    for keyword in keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', output_text):
            found_keywords.append(keyword)
    
    return ",".join(found_keywords)


def analyze_image_folder(input_folder, prompt, output_folder="outputs"):
    valid_extensions = {".jpg", ".jpeg", ".png"}

    image_paths = [os.path.join(input_folder, image_name) for image_name in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, image_name)) and os.path.splitext(image_name)[1].lower() in valid_extensions]

    responses = main(input_folder, prompt)
    
    for key, value in tqdm(responses.items()):
        info = extract_keywords(value)
        image_path = os.path.join(input_folder, key)
        write_response_on_image(image_path, info, output_folder)

    # # Convert dictionary to DataFrame
    # df = pd.DataFrame.from_dict(responses, orient='index').reset_index()

    # # Rename columns
    # df.columns = ['img_name', 'description']
    # df.to_csv(os.path.join(output_folder, "results.csv"), index=False)
    print("Processing complete!")


if __name__ == "__main__":
    prompt = """
        describe the following image in details about shelf organizations, if it provide a good view to the shelf 
    """

    input_folder = "/home/ubuntu/CogVLM/demo_images"
    analyze_image_folder(input_folder, prompt, output_folder="outputs")