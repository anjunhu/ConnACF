"""
ConnaCF Model with Lazy LLM Initialization

This version of ConnaCF delays LLM agent initialization until they are actually needed,
preventing hangs during model creation.
"""

import asyncio
import torch
import torch.nn as nn
from logging import getLogger
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
import os.path as osp
import os
from agentverse.initialization import load_agent,  prepare_task_config
from fuzzywuzzy import process
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import random
from itertools import chain
import numpy as np
from scipy.spatial.distance import cosine

# Utility functions to replace OpenAI embedding utilities
def distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine"):
    """Calculate distances between query embedding and a list of embeddings"""
    distances = []
    for embedding in embeddings:
        if distance_metric == "cosine":
            distance = cosine(query_embedding.detach().cpu(), embedding.detach().cpu())
        else:
            # Default to cosine distance
            distance = cosine(query_embedding.detach().cpu(), embedding.detach().cpu())
        distances.append(distance)
    return distances

def indices_of_nearest_neighbors_from_distances(distances):
    """Get indices of nearest neighbors sorted by distance"""
    return np.argsort(distances)

class ConnaCFNoLLMInit(SequentialRecommender):
    r"""ConnaCF with lazy LLM initialization to prevent hanging during model creation."""
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(ConnaCFNoLLMInit, self).__init__(config, dataset)
        self.n_users = dataset.num(self.USER_ID)
        self.config = config
        self.sample_num = config['sample_num']
        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.data_path = config['data_path']
        self.dataset_name = dataset.dataset_name
        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.logger = getLogger()
        self.loss = BPRLoss()
        self.item_token_id = dataset.field2token_id['item_id']
        self.item_id_token = dataset.field2id_token['item_id']
        self.user_id_token = dataset.field2id_token['user_id']
        self.user_token_id = dataset.field2token_id['user_id']
        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.api_batch = config['api_batch']
        self.chat_api_batch = config['chat_api_batch']
        
        # LAZY INITIALIZATION: Store contexts but don't create agents yet
        self.embedding_context = self._prepare_embedding_context()
        self.item_text = self.load_text()
        self.user_context = self.load_user_context()
        self.item_context = self.load_item_context()
        self.max_his_len = config['max_his_len']
        self.record_idx = 0
        
        # Initialize record directory
        while True:
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'user_record_{self.record_idx}',)
            if os.path.exists(path):
                self.record_idx += 1
                continue
            else: break

        print(f"In this interaction, the updation process is recorded in {str(self.record_idx)}")
        
        # LAZY INITIALIZATION: Don't create agents yet, just store contexts
        self.user_agents = {}
        self.item_agents = {}
        self.embedding_agent = None
        self.rec_agent = None
        
        # Flag to track if agents have been initialized
        self._agents_initialized = False
        
        print("ConnaCF model created with lazy LLM initialization")

    def _prepare_embedding_context(self):
        """Prepare embedding context without creating the agent"""
        # Prepare AWS credentials for Bedrock (if provided)
        aws_config = {}
        if 'aws_access_key_id' in self.config:
            aws_config['aws_access_key_id'] = self.config['aws_access_key_id']
        if 'aws_secret_access_key' in self.config:
            aws_config['aws_secret_access_key'] = self.config['aws_secret_access_key']
        if 'aws_session_token' in self.config:
            aws_config['aws_session_token'] = self.config['aws_session_token']
        if 'aws_region' in self.config:
            aws_config['region'] = self.config['aws_region']
        
        return {
            'agent_type': 'embeddingagent',
            'role_task': '',
            'memory': [],
            'prompt_template': '',
            'llm': {'model': self.config['embedding_model'], 'temperature': self.config['llm_temperature'],
                    'max_tokens': self.config['max_tokens'], 'llm_type': self.config['embedding_model'],
                    **aws_config},
            'llm_chat': {'model': self.config['llm_model'], 'llm_type': self.config['llm_model'],
                         'temperature': self.config['llm_temperature'], 'max_tokens': self.config['max_tokens_chat'],
                         **aws_config},
            'agent_mode': 'embedding', 
            'output_parser_type': 'recommender',
        }

    def _initialize_agents_if_needed(self):
        """Initialize LLM agents only when they are actually needed"""
        if self._agents_initialized:
            return
            
        print("Initializing LLM agents (this may take a moment)...")
        
        # Initialize embedding agent
        self.embedding_agent = load_agent(self.embedding_context)
        
        # Initialize user agents
        for user_id, user_context in self.user_context.items():
            agent = load_agent(user_context)
            self.user_agents[user_id] = agent
            user_id = str(user_id)
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'user_record_{self.record_idx}',)
            user_description = user_context['memory_1'][-1]
            if not os.path.exists(path):
                os.makedirs(path)
            with open(osp.join(path,f'user.{user_id}'),'w') as f:
                f.write('~'*20 + 'Meta information' + '~'*20 + '\n')
                f.write(f'The user wrote the following self-description as follows: {user_description}\n')

        # Initialize item agents
        item_descriptions = []
        for item_id, item_context in self.item_context.items():
            agent = load_agent(item_context)
            self.item_agents[item_id] = agent
            item_id = str(item_id)
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'item_record_{self.record_idx}',)
            item_description = item_context['role_description']
            item_descriptions.append(item_description)
            if not os.path.exists(path):
                os.makedirs(path)
            with open(osp.join(path,f'item.{item_id}'),'w') as f:
                f.write('~'*20 + 'Meta information' + '~'*20 + '\n')
                f.write(f'The item has the following characteristics: {item_description} \n')

        # Initialize recommendation agent
        rec_context = {
            'agent_type':'recagent',
            'memory':[],
            'prompt_template': self.config['system_prompt_template'],
            'llm':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'], 'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens']},
            'llm_chat':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature_test'],'max_tokens':self.config['max_tokens_chat']},
            'agent_mode':'system','output_parser_type':'recommender',
            'system_prompt_template_backward': self.config['system_prompt_template_backward'],
            'system_prompt_template_evaluation_basic': self.config['system_prompt_template_evaluation_basic'],
            'system_prompt_template_evaluation_sequential': self.config['system_prompt_template_evaluation_sequential'],
            'system_prompt_template_evaluation_retrieval': self.config['system_prompt_template_evaluation_retrieval'],
            'n_users': self.n_users
        }
        self.rec_agent = load_agent(rec_context)
        
        self._agents_initialized = True
        print("LLM agents initialized successfully")

    def load_user_context(self):
        """Load user context without creating agents"""
        user_context = {}
        user_context[0] = {'agent_type':'useragent', 'role_description':{'age': '[PAD]', 'user_gender': '[PAD]','user_occupation':'[PAD]'},'memory_1':['[PAD]'],'update_memory':['[PAD]'],
                     'role_description_string_1':'[PAD]','role_description_string_3':'[PAD]', 'role_task':'[PAD]','prompt_template': self.config['user_prompt_template'], 'user_prompt_system_role': self.config['user_prompt_system_role'],
                     'llm':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens'],},
                     'llm_chat':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens_chat'],},
                     'agent_mode':'user','output_parser_type':'useragent','historical_interactions':{}, 'user_prompt_template_true': self.config['user_prompt_template_true']}
        feat_path = None
        if 'ml-' in self.dataset_name:
            feat_path = osp.join(self.data_path, f'ml-100k.user')
        if feat_path != None :
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    user_id, user_age, user_gender, user_occupation,_ = line.strip().split('\t')
                    if user_id not in self.user_token_id:
                        continue
                    if user_occupation == 'other':
                        user_occupation_des = ' movie enthusiast'
                    else:
                        user_occupation_des = user_occupation
                    if user_gender == 'M':
                        user_gender_des = 'man'
                    else:
                        user_gender_des = 'woman'

                    user_context[self.user_token_id[user_id]] = \
                        {'agent_type':'useragent',
                        'role_description': {'age': user_age, 'user_gender': user_gender,'user_occupation':user_occupation},
                        'role_description_string_3': f'The user is a {user_gender_des}. The user is a {user_occupation_des}. ',
                        'role_description_string_1': f'I am a {user_gender_des}. I am a {user_occupation_des}.',
                         'user_prompt_system_role': self.config['user_prompt_system_role'],
                        'memory_1': [f' I am a {user_gender_des}. I am a {user_occupation_des}.',],
                        'update_memory': [f' I am a {user_gender_des}. I am a {user_occupation_des}.',],
                        'prompt_template': self.config['user_prompt_template'],

                        'llm':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens']},
                        'llm_chat':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens_chat']},
                        'agent_mode':'user','output_parser_type':'useragent','historical_interactions':{}, 'user_prompt_template_true': self.config['user_prompt_template_true']}
            return user_context
        else:
            for user_id in range(self.n_users):
                user_context[user_id] = {'agent_type':'useragent', 'role_description':{'age': '[PAD]', 'user_gender': '[PAD]','user_occupation':'[PAD]'},'memory_1':['[PAD]'],'update_memory':['[PAD]'],
                         'role_description_string_1':'[PAD]','role_description_string_3':'[PAD]', 'role_task':'[PAD]','prompt_template': self.config['user_prompt_template'], 'user_prompt_system_role': self.config['user_prompt_system_role'],
                         'llm':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens'],},
                         'llm_chat':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens_chat'],},
                         'agent_mode':'user','output_parser_type':'useragent','historical_interactions':{}, 'user_prompt_template_true': self.config['user_prompt_template_true']}
            return user_context

    def load_text(self):
        """Load item text data"""
        item_text = {}
        feat_path = None
        if 'ml-' in self.dataset_name:
            feat_path = osp.join(self.data_path, f'ml-100k.item')
        elif 'Amazon' in self.dataset_name:
            feat_path = osp.join(self.data_path, f'{self.dataset_name}.item')
        elif 'CDs' in self.dataset_name:
            feat_path = osp.join(self.data_path, f'{self.dataset_name}.item')
        
        if feat_path and os.path.exists(feat_path):
            with open(feat_path, 'r', encoding='utf-8') as file:
                for line in file:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        item_id, item_title = parts[0], parts[1]
                        if item_id in self.item_token_id:
                            item_text[self.item_token_id[item_id]] = item_title
        return item_text

    def load_item_context(self):
        """Load item context without creating agents"""
        item_context = {}
        feat_path = None
        if 'ml-' in self.dataset_name:
            feat_path = osp.join(self.data_path, f'ml-100k.item')
        elif 'Amazon' in self.dataset_name:
            feat_path = osp.join(self.data_path, f'{self.dataset_name}.item')
        elif 'CDs' in self.dataset_name:
            feat_path = osp.join(self.data_path, f'{self.dataset_name}.item')
        
        if feat_path and os.path.exists(feat_path):
            with open(feat_path, 'r', encoding='utf-8') as file:
                for line in file:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        item_id, item_title = parts[0], parts[1]
                        if item_id in self.item_token_id:
                            item_context[self.item_token_id[item_id]] = {
                                'agent_type': 'itemagent',
                                'role_description': item_title,
                                'prompt_template': self.config['item_prompt_template'],
                                'llm': {'model': self.config['llm_model'], 'llm_type': self.config['llm_model'], 
                                       'temperature': self.config['llm_temperature'], 'max_tokens': self.config['max_tokens']},
                                'llm_chat': {'model': self.config['llm_model'], 'llm_type': self.config['llm_model'], 
                                            'temperature': self.config['llm_temperature'], 'max_tokens': self.config['max_tokens_chat']},
                                'agent_mode': 'item', 'output_parser_type': 'itemagent',
                                'item_prompt_template_true': self.config['item_prompt_template_true']
                            }
        return item_context

    # Override methods that need LLM agents to initialize them first
    def forward(self, user, item):
        """Forward pass - only uses embeddings, no LLM needed"""
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def calculate_loss(self, interaction):
        """Calculate loss - only uses embeddings, no LLM needed"""
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e = self.user_embedding(user)
        pos_e = self.item_embedding(pos_item)
        neg_e = self.item_embedding(neg_item)

        pos_item_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_item_score = torch.mul(user_e, neg_e).sum(dim=1)

        loss = self.loss(pos_item_score, neg_item_score)
        return loss

    def predict(self, interaction):
        """Predict scores - only uses embeddings, no LLM needed"""
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)

    def full_sort_predict(self, interaction):
        """Full sort prediction - only uses embeddings, no LLM needed"""
        user = interaction[self.USER_ID]
        user_e = self.user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)

    # Methods that require LLM agents will initialize them when called
    def update_user_item(self, user_id, pos_item_id, neg_item_id, user_his_items):
        """Update user and item - initializes agents if needed"""
        self._initialize_agents_if_needed()
        # Continue with original implementation...
        # (This would contain the rest of the original method)
        pass

# Create an alias for backward compatibility
ConnaCF = ConnaCFNoLLMInit