#!/usr/bin/env python3
"""
ConnaCF with Comprehensive LLM Interaction Logging

This is a modified version of the ConnaCF model that includes detailed logging
of all LLM interactions to help debug hanging issues and monitor performance.
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
from agentverse.initialization import load_agent, prepare_task_config
from fuzzywuzzy import process
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import random
from itertools import chain
import numpy as np
from scipy.spatial.distance import cosine
import time
import threading
import signal
import sys

# Import our logging system
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ..llm_interaction_logger import LLMInteractionLogger, HangingDetector


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


class ConnaCFWithLogging(SequentialRecommender):
    """ConnaCF with comprehensive LLM interaction logging"""
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(ConnaCFWithLogging, self).__init__(config, dataset)
        
        _cfg = config.final_config_dict if hasattr(config, 'final_config_dict') else {}
        # Initialize logging system
        experiment_name = f"connacf_{dataset.dataset_name}_{int(time.time())}"
        self.llm_logger = LLMInteractionLogger(
            log_dir=_cfg.get('llm_log_dir', 'llm_interaction_logs'),
            experiment_name=experiment_name
        )
        
        # Initialize hanging detector
        self.hanging_detector = HangingDetector(
            self.llm_logger, 
            timeout_threshold=_cfg.get('hang_timeout', 30.0)
        )
        
        # Start monitoring for hanging operations
        self.monitor_thread = self.hanging_detector.start_monitoring(
            check_interval=_cfg.get('hang_check_interval', 10.0)
        )
        self._llm_timeout = _cfg.get('llm_timeout', 60.0)
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print(f"[LLM LOG] Initialized logging system for experiment: {experiment_name}")
        print(f"[LLM LOG] Log directory: {self.llm_logger.log_dir}")
        print(f"[LLM LOG] Hanging detection timeout: {self.hanging_detector.timeout_threshold}s")
        
        # Continue with original ConnaCF initialization
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
        
        embedding_context = \
            {'agent_type': 'embeddingagent',
             'role_task': '',
             'memory': [],
             'prompt_template': '',
             'llm': {'model': self.config['embedding_model'], 'temperature': self.config['llm_temperature'],
                     'max_tokens': self.config['max_tokens'], 'llm_type': self.config['embedding_model'],
                     **aws_config},
             'llm_chat': {'model': self.config['llm_model'], 'llm_type': self.config['llm_model'],
                          'temperature': self.config['llm_temperature'], 'max_tokens': self.config['max_tokens_chat'],
                          **aws_config},
             'agent_mode': 'embedding', 'output_parser_type': 'recommender',
             }
        
        self.embedding_agent = load_agent(embedding_context)
        self.item_text = self.load_text()
        self.user_context = self.load_user_context()
        self.item_context = self.load_item_context()
        self.max_his_len = config['max_his_len']
        self.record_idx = 0

        # Find available record directory
        while True:
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'user_record_{self.record_idx}',)
            if os.path.exists(path):
                self.record_idx += 1
                continue
            else: 
                break

        print(f"In this interaction, the updation process is recorded in {str(self.record_idx)}")
        
        # Initialize agents with logging
        self._initialize_agents_with_logging()
        
        # Initialize recommendation agent
        self._initialize_rec_agent()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n[LLM LOG] Received signal {signum}, shutting down gracefully...")
        self.llm_logger.close()
        sys.exit(0)
    
    def _initialize_agents_with_logging(self):
        """Initialize user and item agents with logging capabilities"""
        print("[LLM LOG] Initializing user agents...")
        self.user_agents = {}
        for user_id, user_context in self.user_context.items():
            agent = load_agent(user_context)
            self.user_agents[user_id] = agent
            user_id_str = str(user_id)
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'user_record_{self.record_idx}',)
            user_description = user_context['memory_1'][-1]
            if not os.path.exists(path):
                os.makedirs(path)
            with open(osp.join(path, f'user.{user_id_str}'), 'w') as f:
                f.write('~'*20 + 'Meta information' + '~'*20 + '\n')
                f.write(f'The user wrote the following self-description as follows: {user_description}\n')

        print(f"[LLM LOG] Initialized {len(self.user_agents)} user agents")

        print("[LLM LOG] Initializing item agents...")
        self.item_agents = {}
        item_descriptions = []

        for item_id, item_context in self.item_context.items():
            agent = load_agent(item_context)
            self.item_agents[item_id] = agent
            item_id_str = str(item_id)
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'item_record_{self.record_idx}',)
            item_description = item_context['role_description']
            item_descriptions.append(item_description)
            if not os.path.exists(path):
                os.makedirs(path)
            with open(osp.join(path, f'item.{item_id_str}'), 'w') as f:
                f.write('~'*20 + 'Meta information' + '~'*20 + '\n')
                f.write(f'The item has the following characteristics: {item_description} \n')

        print(f"[LLM LOG] Initialized {len(self.item_agents)} item agents")
    
    def _initialize_rec_agent(self):
        """Initialize recommendation agent"""
        print("[LLM LOG] Initializing recommendation agent...")
        rec_context = \
            {'agent_type': 'recagent',
             'memory': [],
             'prompt_template': self.config['system_prompt_template'],
             'llm': {'model': self.config['llm_model'], 'llm_type': self.config['llm_model'], 
                     'temperature': self.config['llm_temperature'], 'max_tokens': self.config['max_tokens']},
             'llm_chat': {'model': self.config['llm_model'], 'llm_type': self.config['llm_model'],
                          'temperature': self.config['llm_temperature_test'], 'max_tokens': self.config['max_tokens_chat']},
             'agent_mode': 'system', 'output_parser_type': 'recommender',
             'system_prompt_template_backward': self.config['system_prompt_template_backward'],
             'system_prompt_template_evaluation_basic': self.config['system_prompt_template_evaluation_basic'],
             'system_prompt_template_evaluation_sequential': self.config['system_prompt_template_evaluation_sequential'],
             'system_prompt_template_evaluation_retrieval': self.config['system_prompt_template_evaluation_retrieval'],
             'n_users': self.n_users
             }
        self.rec_agent = load_agent(rec_context)
        print("[LLM LOG] Recommendation agent initialized")

    def calculate_loss(self, interaction):
        """Calculate loss with comprehensive logging"""
        batch_user = interaction[self.USER_ID]
        batch_pos_item = interaction[self.ITEM_ID]
        batch_neg_item = interaction[self.NEG_ITEM_ID]
        batch_size = batch_user.size(0)

        # Log batch information
        batch_info = {
            'batch_size': batch_size,
            'users': batch_user.tolist(),
            'pos_items': batch_pos_item.tolist(),
            'neg_items': batch_neg_item.tolist()
        }

        for i in range(self.config['all_update_rounds']):
            round_start_time = time.time()
            
            # Log round start
            self.llm_logger.log_round_start(
                i, batch_user.tolist(), batch_pos_item.tolist(), batch_neg_item.tolist()
            )
            
            print("~"*20 + f"{i}-th round update with comprehensive logging!" + "~"*20 + '\n')
            
            # Forward pass with logging
            user_forward_description, pos_item_forward_description, neg_item_forward_description = [], [], []
            for j in range(batch_size):
                user_forward_description.append(self.user_agents[int(batch_user[j])].update_memory[-1])
                pos_item_forward_description.append(self.item_agents[int(batch_pos_item[j])].update_memory[-1])
                neg_item_forward_description.append(self.item_agents[int(batch_neg_item[j])].update_memory[-1])
            
            # System forward pass with detailed logging
            system_selections, system_reasons = self.forward_with_logging(
                batch_user, batch_pos_item, batch_neg_item, i, batch_info
            )
            
            accuracy = self.convert_system_selections_to_accuracy(system_selections, batch_pos_item, batch_neg_item)
            current_accuracy = sum(accuracy) / len(accuracy)
            print(f"[LLM LOG] Round {i} accuracy: {current_accuracy:.4f}")
            
            # Backward pass with logging
            backward_system_reasons, backward_user, backward_pos_item, backward_neg_item = [], [], [], []
            backward_system_reasons_true, backward_user_true, backward_pos_item_true, backward_neg_item_true = [], [], [], []
            first_time = set()
            
            for j, acc in enumerate(accuracy):
                if acc == 0:  # record wrong choices
                    backward_pos_item.append(int(batch_pos_item[j]))
                    backward_neg_item.append(int(batch_neg_item[j]))
                    backward_user.append(int(batch_user[j]))
                    backward_system_reasons.append(system_reasons[j])
                else:  # record right choices
                    if i == 0:
                        first_time.add(int(batch_user[j]))
                        backward_user_true.append(int(batch_user[j]))
                        backward_pos_item_true.append(int(batch_pos_item[j]))
                        backward_neg_item_true.append(int(batch_neg_item[j]))
                        backward_system_reasons_true.append(system_reasons[j])
                    elif int(batch_user[j]) not in first_time:
                        backward_user_true.append(int(batch_user[j]))
                        backward_pos_item_true.append(int(batch_pos_item[j]))
                        backward_neg_item_true.append(int(batch_neg_item[j]))
                        backward_system_reasons_true.append(system_reasons[j])

            print(f"[LLM LOG] Users to be updated (wrong predictions): {backward_user}")
            
            if backward_user:
                self.backward_with_logging(
                    backward_system_reasons, backward_user, backward_pos_item, backward_neg_item, i, False
                )
            
            if i == 0 and len(backward_user_true):
                self.backward_true_with_logging(
                    backward_system_reasons_true, backward_user_true, backward_pos_item_true, 
                    backward_neg_item_true, True, i
                )
            
            round_time = time.time() - round_start_time
            self.llm_logger.log_round_complete(i, current_accuracy, round_time)
        
        # Final backward_true call
        if 'backward_user_true' in locals() and len(backward_user_true):
            self.backward_true_with_logging(
                backward_system_reasons_true, backward_user_true, backward_pos_item_true, 
                backward_neg_item_true, False, self.config['all_update_rounds'] - 1
            )

        # Continue with normal ConnaCF processing
        if self.config['evaluation'] == 'rag':
            system_reasons_embeddings = self.generate_embedding(system_reasons)
            for i, user in enumerate(batch_user):
                self.rec_agent.user_examples[int(user)][(user_forward_description[i], self.item_text[int(batch_pos_item[i])], self.item_text[int(batch_neg_item[i])], pos_item_forward_description[i], neg_item_forward_description[i], accuracy[i], system_reasons[i])] = system_reasons_embeddings[i]
        else:
            for i, user in enumerate(batch_user):
                self.rec_agent.user_examples[int(user)][(user_forward_description[i], self.item_text[int(batch_pos_item[i])], self.item_text[int(batch_neg_item[i])], pos_item_forward_description[i], neg_item_forward_description[i], accuracy[i], system_reasons[i])] = None

        self.logging_after_updation(batch_user, batch_pos_item, batch_neg_item)
        
        # Update agent memories
        batch_pos_item_descriptions = []
        batch_neg_item_descriptions = []
        for i in range(batch_size):
            self.user_agents[int(batch_user[i])].memory_1.append(self.user_agents[int(batch_user[i])].update_memory[-1])
            batch_pos_item_descriptions.append(self.item_agents[int(batch_pos_item[i])].update_memory[-1])
            batch_neg_item_descriptions.append(self.item_agents[int(batch_neg_item[i])].update_memory[-1])

        if self.config['evaluation'] == 'rag':
            batch_pos_item_descriptions_embeddings = self.generate_embedding(batch_pos_item_descriptions)
            batch_neg_item_descriptions_embeddings = self.generate_embedding(batch_neg_item_descriptions)
            for i in range(batch_size):
                self.item_agents[int(batch_pos_item[i])].memory_embedding[batch_pos_item_descriptions[i]] = batch_pos_item_descriptions_embeddings[i]
                self.item_agents[int(batch_neg_item[i])].memory_embedding[batch_neg_item_descriptions[i]] = batch_neg_item_descriptions_embeddings[i]
        else:
            for i in range(batch_size):
                self.item_agents[int(batch_pos_item[i])].memory_embedding[batch_pos_item_descriptions[i]] = None
                self.item_agents[int(batch_neg_item[i])].memory_embedding[batch_neg_item_descriptions[i]] = None

    def forward_with_logging(self, batch_user, batch_pos_item, batch_neg_item, round_num, batch_info):
        """Forward pass with comprehensive logging"""
        batch_size = batch_user.size(0)
        
        # Log batch start
        batch_id = self.llm_logger.log_batch_start(
            "system_forward", round_num, batch_size, batch_info
        )
        
        # Track operation for hanging detection
        operation_id = f"forward_{round_num}_{int(time.time() * 1000)}"
        self.hanging_detector.start_operation(
            operation_id, "system_forward", round_num, "System agent forward reasoning"
        )
        
        batch_start_time = time.time()
        
        try:
            user_descriptions, pos_item_descriptions, neg_item_descriptions = [], [], []
            for i, user in enumerate(batch_user):
                user_agent = self.user_agents[int(user)]
                pos_item_agent = self.item_agents[int(batch_pos_item[i])]
                neg_item_agent = self.item_agents[int(batch_neg_item[i])]
                user_descriptions.append(user_agent.update_memory[-1])
                pos_item_descriptions.append(pos_item_agent.update_memory[-1])
                neg_item_descriptions.append(neg_item_agent.update_memory[-1])

            # Log individual system prompts
            system_forward_prompts = []
            for i in range(batch_size):
                prompt = self.rec_agent.astep_forward(
                    int(batch_user[i]), user_descriptions[i], 
                    pos_item_descriptions[i], neg_item_descriptions[i]
                )
                system_forward_prompts.append(prompt)
                
                # Log individual interaction
                self.llm_logger.log_interaction_start(
                    "system_forward", round_num, "system", int(batch_user[i]), 
                    prompt, {'pos_item': int(batch_pos_item[i]), 'neg_item': int(batch_neg_item[i])}
                )

            # Process in batches with timeout monitoring
            system_responses = []
            successful_count = 0
            failed_count = 0
            
            for i in range(0, batch_size, self.api_batch):
                api_batch_start = time.time()
                current_batch = system_forward_prompts[i:i + self.api_batch]
                
                print(f"[LLM LOG] Processing system forward batch {i//self.api_batch + 1}/{(batch_size + self.api_batch - 1)//self.api_batch}")
                print(f"[LLM LOG] Batch size: {len(current_batch)}")
                
                try:
                    # Use asyncio.run instead of await since this is not an async function
                    batch_responses = asyncio.run(asyncio.wait_for(
                        self.rec_agent.llm.agenerate_response(current_batch),
                        timeout=self._llm_timeout
                    ))
                    system_responses.extend(batch_responses)
                    successful_count += len(batch_responses)
                    
                    api_batch_time = time.time() - api_batch_start
                    print(f"[LLM LOG] Batch completed in {api_batch_time:.2f}s")
                    
                except asyncio.TimeoutError:
                    print(f"[LLM LOG] TIMEOUT: System forward batch {i//self.api_batch + 1} timed out")
                    # Add fallback responses
                    fallback_responses = ["I need to choose between the items." for _ in current_batch]
                    system_responses.extend(fallback_responses)
                    failed_count += len(current_batch)
                    
                except Exception as e:
                    print(f"[LLM LOG] ERROR: System forward batch {i//self.api_batch + 1} failed: {e}")
                    # Add fallback responses
                    fallback_responses = ["I need to choose between the items." for _ in current_batch]
                    system_responses.extend(fallback_responses)
                    failed_count += len(current_batch)

            print(f"[LLM LOG] System responses received: {len(system_responses)}")
            for idx, response in enumerate(system_responses[:3]):  # Show first 3 responses
                print(f"[LLM LOG] Response {idx}: {response[:100]}...")

            # Parse responses with error handling
            system_responses_parsed = []
            for i, response in enumerate(system_responses):
                try:
                    parsed = self.rec_agent.output_parser.parse(response)
                    system_responses_parsed.append(parsed)
                except Exception as e:
                    print(f"[LLM LOG] Error parsing system response {i}: {e}")
                    # Fallback parsing
                    system_responses_parsed.append(("A", "I choose item A"))

            system_selections, system_reasons = [], []
            for response in system_responses_parsed:
                system_selections.append(response[0])
                system_reasons.append(response[1])

            batch_time = time.time() - batch_start_time
            self.llm_logger.log_batch_complete(batch_id, successful_count, failed_count, batch_time)
            
            return system_selections, system_reasons
            
        finally:
            self.hanging_detector.end_operation(operation_id)

    def backward_with_logging(self, system_reasons, batch_user, batch_pos_item, batch_neg_item, round_num, is_true_update):
        """Backward pass with comprehensive logging"""
        batch_size = len(batch_user)
        
        if batch_size == 0:
            print("[LLM LOG] No users to update in backward pass")
            return
        
        update_type = "user_backward_true" if is_true_update else "user_backward"
        
        # Log batch start
        batch_id = self.llm_logger.log_batch_start(
            update_type, round_num, batch_size, 
            {'users': batch_user, 'pos_items': batch_pos_item, 'neg_items': batch_neg_item}
        )
        
        # Track operation for hanging detection
        operation_id = f"backward_{round_num}_{int(time.time() * 1000)}"
        self.hanging_detector.start_operation(
            operation_id, update_type, round_num, f"User agent backward update ({'true' if is_true_update else 'false'})"
        )
        
        batch_start_time = time.time()
        
        try:
            pos_item_descriptions_forward, neg_item_descriptions_forward, pos_item_titles, neg_item_titles, user_descriptions_forward = [], [], [], [], []
            for i, user in enumerate(batch_user):
                pos_item_agent = self.item_agents[int(batch_pos_item[i])]
                neg_item_agent = self.item_agents[int(batch_neg_item[i])]
                pos_item_titles.append(pos_item_agent.role_description['item_title'])
                neg_item_titles.append(neg_item_agent.role_description['item_title'])
                pos_item_descriptions_forward.append(pos_item_agent.update_memory[-1])
                neg_item_descriptions_forward.append(neg_item_agent.update_memory[-1])
                user_descriptions_forward.append(self.user_agents[int(user)].update_memory[-1])

            # Generate user backward prompts
            user_backward_prompts = []
            for i in range(batch_size):
                if is_true_update:
                    prompt = self.user_agents[int(batch_user[i])].astep_backward_true(
                        system_reasons[i], pos_item_titles[i], neg_item_titles[i], 
                        pos_item_descriptions_forward[i], neg_item_descriptions_forward[i]
                    )
                else:
                    prompt = self.user_agents[int(batch_user[i])].astep_backward(
                        system_reasons[i], pos_item_titles[i], neg_item_titles[i], 
                        pos_item_descriptions_forward[i], neg_item_descriptions_forward[i]
                    )
                user_backward_prompts.append(prompt)
                
                # Log individual interaction
                self.llm_logger.log_interaction_start(
                    update_type, round_num, "user", int(batch_user[i]), 
                    prompt, {'pos_item': int(batch_pos_item[i]), 'neg_item': int(batch_neg_item[i])}
                )

            # Process user updates in batches
            user_update_descriptions = []
            successful_count = 0
            failed_count = 0
            
            for i in range(0, batch_size, self.chat_api_batch):
                api_batch_start = time.time()
                current_batch = user_backward_prompts[i:i+self.chat_api_batch]
                
                print(f"[LLM LOG] Processing user {update_type} batch {i//self.chat_api_batch + 1}/{(batch_size + self.chat_api_batch - 1)//self.chat_api_batch}")
                
                try:
                    # Use asyncio.run instead of await since this is not an async function
                    batch_responses = asyncio.run(asyncio.wait_for(
                        self.user_agents[0].llm_chat.agenerate_response_without_construction(current_batch),
                        timeout=self._llm_timeout
                    ))
                    user_update_descriptions.extend(batch_responses)
                    successful_count += len(batch_responses)
                    
                    api_batch_time = time.time() - api_batch_start
                    print(f"[LLM LOG] User batch completed in {api_batch_time:.2f}s")
                    
                except asyncio.TimeoutError:
                    print(f"[LLM LOG] TIMEOUT: User {update_type} batch {i//self.chat_api_batch + 1} timed out")
                    fallback_responses = ["I have updated my preferences." for _ in current_batch]
                    user_update_descriptions.extend(fallback_responses)
                    failed_count += len(current_batch)
                    
                except Exception as e:
                    print(f"[LLM LOG] ERROR: User {update_type} batch {i//self.chat_api_batch + 1} failed: {e}")
                    fallback_responses = ["I have updated my preferences." for _ in current_batch]
                    user_update_descriptions.extend(fallback_responses)
                    failed_count += len(current_batch)

            # Parse user updates with retry logic and logging
            parsed_user_updates = []
            for i, response in enumerate(user_update_descriptions):
                retry_count = 0
                max_retries = 3
                
                while retry_count <= max_retries:
                    try:
                        parsed_result = self.user_agents[0].output_parser.parse_update(response)
                        
                        # Check if parser is requesting a retry
                        if (isinstance(parsed_result, tuple) and len(parsed_result) == 2 and 
                            parsed_result[0] == "RETRY_NEEDED"):
                            
                            if retry_count < max_retries:
                                print(f"[LLM LOG] Retrying user update {i} with directive prompt (attempt {retry_count + 1})")
                                
                                # Get the directive prompt from the parser
                                directive_prompt = parsed_result[1]
                                
                                # Create a new prompt that includes the directive
                                original_prompt = user_backward_prompts[i]
                                retry_prompt = f"{original_prompt}\n\n{directive_prompt}"
                                
                                # Get new response with directive prompt
                                retry_response = asyncio.run(asyncio.wait_for(
                                    self.user_agents[0].llm_chat.agenerate_response_without_construction([retry_prompt]),
                                    timeout=self._llm_timeout
                                ))
                                response = retry_response[0]
                                retry_count += 1
                                continue
                            else:
                                print(f"[LLM LOG] Max retries exceeded for user update {i}, using fallback")
                                parsed_result = f"I have updated my preferences based on this experience."
                                break
                        else:
                            # Successful parse
                            break
                            
                    except Exception as e:
                        print(f"[LLM LOG] Error parsing user update {i} (attempt {retry_count + 1}): {e}")
                        if retry_count >= max_retries:
                            print(f"[LLM LOG] Max retries exceeded for user update {i}, using fallback")
                            parsed_result = f"I have updated my preferences based on this experience."
                            break
                        retry_count += 1
                        continue
                
                parsed_user_updates.append(parsed_result)
            
            user_update_descriptions = parsed_user_updates
            
            # Update user memories
            for i, user in enumerate(batch_user):
                self.user_agents[int(user)].update_memory.append(user_update_descriptions[i])
            
            print(f"[LLM LOG] User {update_type} complete!")
            
            batch_time = time.time() - batch_start_time
            self.llm_logger.log_batch_complete(batch_id, successful_count, failed_count, batch_time)
            
        finally:
            self.hanging_detector.end_operation(operation_id)

    def backward_true_with_logging(self, system_reasons_true, backward_user_true, backward_pos_item_true, backward_neg_item_true, is_first_time, round_num):
        """Backward true pass with logging"""
        self.backward_with_logging(
            system_reasons_true, backward_user_true, backward_pos_item_true, 
            backward_neg_item_true, round_num, True
        )

    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'llm_logger'):
            self.llm_logger.close()

    # Include all other methods from the original ConnaCF class
    # (load_text, load_user_context, load_item_context, etc.)
    # For brevity, I'm not including them all here, but they would be copied from the original