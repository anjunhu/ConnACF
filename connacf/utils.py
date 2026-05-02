import importlib
import asyncio
from recbole.utils import get_model as recbole_get_model
from collections import defaultdict
import os
import torch
import random
import numpy as np
import pandas as pd
import json
import pickle
import gzip
from tqdm import tqdm
true=True
false=False

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
       

def get_model(model_name):
    # "ConnaCF" is the config name; maps to our MultiChannelRecAgent implementation
    lookup = 'connacf' if model_name == 'ConnaCF' else model_name.lower()
    if importlib.util.find_spec(f'model.{lookup}', __name__):
        model_module = importlib.import_module(f'model.{lookup}', __name__)
        # Try the new name first, fall back to legacy name
        cls = getattr(model_module, 'MultiChannelRecAgent', None) or getattr(model_module, model_name)
        return cls
    else:
        return recbole_get_model(model_name)


async def dispatch_bedrock_requests(
    messages_list,
    llm_instance,
    temperature: float = 0.2
):
    """Dispatches requests to Bedrock API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to Bedrock API.
        llm_instance: Bedrock LLM instance to use.
        temperature: Temperature to use for the model.
    Returns:
        List of responses from Bedrock API.
    """
    # Use the LLM instance's async method
    return await llm_instance.agenerate_response(messages_list)


def dispatch_single_bedrock_request(
    message,
    llm_instance,
    temperature: float = 0.2
):
    """Dispatches single request to Bedrock API.
    
    Args:
        message: Message to be sent to Bedrock API.
        llm_instance: Bedrock LLM instance to use.
        temperature: Temperature to use for the model.
    Returns:
        Response from Bedrock API.
    """
    # Use the LLM instance's sync method
    result = llm_instance.generate_response(message)
    return result.content if hasattr(result, 'content') else result


amazon_dataset2fullname = {
    'Beauty': 'All_Beauty',
    'Fashion': 'AMAZON_FASHION',
    'Appliances': 'Appliances',
    'Arts': 'Arts_Crafts_and_Sewing',
    'Automotive': 'Automotive',
    'Books': 'Books',
    'CDs': 'CDs_and_Vinyl',
    'Cell': 'Cell_Phones_and_Accessories',
    'Clothing': 'Clothing_Shoes_and_Jewelry',
    'Music': 'Digital_Music',
    'Electronics': 'Electronics',
    'Gift': 'Gift_Cards',
    'Food': 'Grocery_and_Gourmet_Food',
    'Home': 'Home_and_Kitchen',
    'Scientific': 'Industrial_and_Scientific',
    'Kindle': 'Kindle_Store',
    'Luxury': 'Luxury_Beauty',
    'Magazine': 'Magazine_Subscriptions',
    'Movies': 'Movies_and_TV',
    'Instruments': 'Musical_Instruments',
    'Office': 'Office_Products',
    'Garden': 'Patio_Lawn_and_Garden',
    'Pantry': 'Prime_Pantry',
    'Pet': 'Pet_Supplies',
    'Software': 'Software',
    'Sports': 'Sports_and_Outdoors',
    'Tools': 'Tools_and_Home_Improvement',
    'Toys': 'Toys_and_Games',
    'Games': 'Video_Games'
}

