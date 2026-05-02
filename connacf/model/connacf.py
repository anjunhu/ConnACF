
"""
Multi-Channel Recommendation Agent (MultiChannelRecAgent)

This module extends the base ConnaCF sequential recommendation model
(Zhang et al., "Collaborative Learning with Autonomous Language Agents
for Recommender Systems", WWW 2024, https://arxiv.org/abs/2310.09233)
with the following original contributions:

  - User-User (UU) social interaction channel with async batching
  - User-Item (UI) dialogue channel (pitch / concern / address rounds)
  - Private PII system: per-user identity, contact, location, temporal,
    regulated data fields; public vs. private profile separation
  - Attack hook interface for injecting adversarial agents into any channel
  - Conversation logging and per-turn metric collection
  - Configurable interaction topology (UU/UI independently togglable)
  - Multi-candidate ranking support (binary / ranking / N-cand)

The base forward/backward recommendation loop and agent initialisation
follow the ConnaCF design; all channel extensions and the PII subsystem
are original work.
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

class MultiChannelRecAgent(SequentialRecommender):
    r"""
    Multi-channel LLM-based recommendation agent.

    Extends the ConnaCF base (pairwise BPR + LLM profile update) with
    optional UU/UI interaction channels, a private PII subsystem, and
    an attack-hook interface for adversarial benchmarking.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(ConnaCF, self).__init__(config, dataset)
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
             #  'role_description': dict(),
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



        while True:
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'user_record_{self.record_idx}',)
            if os.path.exists(path):
                self.record_idx += 1
                continue
            else: break

        print(f"In this interaction, the updation process is recorded in {str(self.record_idx)}")
        self.user_agents = {}
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


        self.item_agents = {}
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


        rec_context = \
            {'agent_type':'recagent',
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
        
        # Initialize InteractionConfig for U-U and U-I interactions
        try:
            from connacf.model.interaction_config import InteractionConfig
        except ImportError:
            from model.interaction_config import InteractionConfig
        # Convert RecBole Config to dict using final_config_dict
        config_dict = config.final_config_dict if hasattr(config, 'final_config_dict') else {}
        
        # Debug: Print UU/UI config values
        print(f"[DEBUG] Config enable_uu_interaction: {config_dict.get('enable_uu_interaction', 'NOT SET')}")
        print(f"[DEBUG] Config enable_ui_interaction: {config_dict.get('enable_ui_interaction', 'NOT SET')}")
        
        self.interaction_config = InteractionConfig.from_config(config_dict)
        self.logger.info(f"Interaction config: {self.interaction_config}")
        print(f"[DEBUG] Final interaction config: {self.interaction_config}")
        
        # Initialize ConversationLogger for logging NL interactions
        from attack.visualization.conversation_logger import ConversationLogger as NLConversationLogger
        
        self.conversation_logger = NLConversationLogger(
            output_dir=self.config['record_path'],
            task_id_or_dataset=self.dataset_name,
            record_idx=self.record_idx,
            async_logging=True
        )
        
        # Attack hooks placeholder (set externally when running attacks)
        self.attack_hooks = None





    def load_user_context(self):
        user_context = {}
        user_context[0] = {'agent_type':'useragent', 'role_description':{'age': '[PAD]', 'user_gender': '[PAD]','user_occupation':'[PAD]'},'memory_1':['[PAD]'],'update_memory':['[PAD]'],
                     'role_description_string_1':'[PAD]','role_description_string_3':'[PAD]', 'role_task':'[PAD]','prompt_template': self.config['user_prompt_template'], 'user_prompt_system_role': self.config['user_prompt_system_role'],
                     'llm':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens'],},
                     'llm_chat':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens_chat'],},
                     'agent_mode':'user','output_parser_type':'useragent','historical_interactions':{}, 'user_prompt_template_true': self.config['user_prompt_template_true'],
                     'private_pii': {}}  # MAMA: Private PII storage
        feat_path = None
        if 'ml-' in self.dataset_name:
            feat_path = osp.join(self.data_path, f'ml-100k.user')
        if feat_path != None :
            with open(feat_path, 'r', encoding='utf-8') as file:
                header = file.readline().strip().split('\t')
                
                # Check if we have the enhanced PII file
                has_pii = len(header) > 5 and 'pseudonym' in header
                
                for line in file:
                    parts = line.strip().split('\t')
                    
                    # Parse based on file format
                    if has_pii and len(parts) >= 14:
                        # Enhanced format with PII
                        user_id, user_age, user_gender, user_occupation, zipcode, \
                        pseudonym, email, phone, address, city, state, dob, unique_id, ssn = parts[:14]
                    else:
                        # Original format
                        if len(parts) < 5:
                            continue
                        user_id, user_age, user_gender, user_occupation, zipcode = parts[:5]
                        pseudonym = email = phone = address = city = state = dob = unique_id = ssn = None
                    
                    if user_id not in self.user_token_id:
                        continue
                    
                    # Better occupation phrasing
                    if user_occupation == 'other':
                        user_occupation_des = 'a movie enthusiast'
                    elif user_occupation in ['administrator', 'educator', 'engineer', 'entertainment', 
                                           'executive', 'healthcare', 'homemaker', 'lawyer', 'librarian',
                                           'marketing', 'programmer', 'scientist', 'technician', 'writer']:
                        user_occupation_des = f'working in {user_occupation}'
                    elif user_occupation == 'student':
                        user_occupation_des = 'a student'
                    elif user_occupation == 'retired':
                        user_occupation_des = 'retired'
                    else:
                        user_occupation_des = f'a {user_occupation}'
                    
                    if user_gender == 'M':
                        user_gender_des = 'man'
                    else:
                        user_gender_des = 'woman'

                    # PUBLIC profile (visible to all agents - NO PII)
                    public_profile = f' I am a {user_gender_des}. I am {user_occupation_des}.'
                    
                    # PRIVATE PII (following MAMA's 5 categories)
                    # Only stored in role_description and private_pii, NOT in memory_1
                    private_pii = {
                        # 1. Identity (High-Sensitivity Anchors)
                        'pseudonym': pseudonym if pseudonym else None,
                        
                        # 2. Contact (Structured Identifiers)
                        'email': email if email else None,
                        'phone': phone if phone else None,
                        
                        # 3. Location (High-Context Attributes)
                        'address': address if address else None,
                        'city': city if city else None,
                        'state': state if state else None,
                        'zipcode': zipcode,
                        
                        # 4. Temporal (High-Context Attributes)
                        'age': user_age,
                        'dob': dob if dob else None,
                        
                        # 5. Regulated Identifiers (High-Sensitivity Anchors)
                        'unique_id': unique_id if unique_id else None,
                        'ssn': ssn if ssn else None,
                    }

                    user_context[self.user_token_id[user_id]] = \
                        {'agent_type':'useragent',
                        
                        # Basic demographics (some overlap with PII)
                        'role_description': {
                            'age': user_age, 
                            'user_gender': user_gender,
                            'user_occupation': user_occupation,
                            # Extended with PII
                            'pseudonym': pseudonym,
                            'zipcode': zipcode,
                        },
                        
                        # MAMA: Private PII storage (5 categories)
                        'private_pii': private_pii,
                        
                        # Public descriptions (NO PII)
                        'role_description_string_3': f'The user is a {user_gender_des}. The user is {user_occupation_des}. ',
                        'role_description_string_1': public_profile,
                        
                        # Public memory (NO PII - this is what other agents see)
                        'memory_1': [public_profile],
                        'update_memory': [public_profile],
                        
                        # Prompts and LLM config
                        'user_prompt_system_role': self.config['user_prompt_system_role'],
                        'prompt_template': self.config['user_prompt_template'],
                        'llm':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens']},
                        'llm_chat':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens_chat']},
                        
                        # Agent metadata
                        'agent_mode':'user',
                        'output_parser_type':'useragent',
                        'historical_interactions':{}, 
                        'user_prompt_template_true': self.config['user_prompt_template_true']}
            return user_context
        else:
            for user_id in range(self.n_users):
                user_context[user_id] = \
                        {'agent_type':'useragent',
                        'role_description': {},
                        'private_pii': {},  # MAMA: Empty for non-MovieLens datasets
                        'role_description_string_3': f'This user enjoys listening CDs very much.',
                        'role_description_string_1': f'I enjoy listening to CDs very much.',
                         'user_prompt_system_role': self.config['user_prompt_system_role'],
                        'memory_1': [f' I enjoy listening to CDs very much.',],
                        'update_memory': [f' I enjoy listening to CDs very much.', ],
                        'prompt_template': self.config['user_prompt_template'],

                        'llm':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens'], },
                        'llm_chat':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens_chat'], },
                        'agent_mode':'user','output_parser_type':'useragent','historical_interactions':{}, 'user_prompt_template_true': self.config['user_prompt_template_true']}
            return user_context



    def load_item_context(self):
        item_context = {}
        item_context[0] = {'agent_type':'itemagent', 'role_description':{'item_title': '[PAD]', 'item_release_year': '[PAD]','item_class':'[PAD]'},'memory':['[PAD]'],'memory_embedding':{},'update_memory':['[PAD]'], 'item_prompt_template_true': self.config['item_prompt_template_true'],
                     'role_description_string':'[PAD]', 'role_task':'[PAD]', 'prompt_template': self.config['user_prompt_template'],
                     'llm':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens'], },
                     'llm_chat':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens_chat'], },
                     'agent_mode':'user','output_parser_type':'itemagent'}
        feat_path = None
        init_item_descriptions = []
        if 'ml-' in self.dataset_name:
            feat_path = osp.join(self.data_path, f'ml-100k.item')
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    # ml-100k.item has 3 fields: item_id, title (with year), category
                    item_id, item_title, item_class = line.strip().split('\t')
                    if item_id not in self.item_token_id:
                        continue
                    role_description_string = f'The movie is called {self.item_text[self.item_token_id[item_id]]}. The theme of this movie is about {item_class}.'
                    item_context[self.item_token_id[item_id]] = \
                        {'agent_type':'itemagent',
                         'update_memory':[role_description_string],
                        'role_description':{'item_title': self.item_text[self.item_token_id[item_id]], 'item_class':item_class},
                        'role_description_string': role_description_string,
                        'prompt_template': self.config['item_prompt_template'],
                        'llm':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens']},
                        'llm_chat':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens_chat']},
                        'agent_mode':'item',
                         'item_prompt_template_true': self.config['item_prompt_template_true'],
                        'output_parser_type':'itemagent'}
                    init_item_descriptions.append(role_description_string)
            # return item_context
        else:
            feat_path = osp.join(self.data_path, f'CDs.item')
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    try:
                        item_id, item_title, item_class = line.strip().split('\t')
                    except ValueError:
                        item_id, item_title = line.strip().split('\t')
                        item_class = 'CDs'
                    if item_id not in self.item_token_id:
                        continue
                    role_description_string = f"The CD is called '{self.item_text[self.item_token_id[item_id]]}'. The category of this CD is: '{item_class}'."
                    # role_description_string = f"The CD is called '{self.item_text[self.item_token_id[item_id]]}'."
                    item_context[self.item_token_id[item_id]] = \
                        {'agent_type':'itemagent',
                         'update_memory': [role_description_string],
                        'role_description':{'item_title': self.item_text[self.item_token_id[item_id]], 'item_class':item_class},
                        'role_description_string': role_description_string,
                        'prompt_template': self.config['item_prompt_template'],
                         'item_prompt_template_true': self.config['item_prompt_template_true'],
                        'llm':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens']},
                        'llm_chat':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens_chat']},
                        'agent_mode':'item',
                        'output_parser_type':'itemagent'}
                    init_item_descriptions.append(role_description_string)

            # return item_context
        if self.config['evaluation'] == 'rag':
            init_item_description_embeddings = self.generate_embedding(init_item_descriptions)
            for i, item in enumerate(item_context.keys()):
                if item == 0: continue
                item_context[item]['memory_embedding'] = {init_item_descriptions[i-1]: init_item_description_embeddings[i-1]}
        else:
            for i, item in enumerate(item_context.keys()):
                if item == 0: continue
                item_context[item]['memory_embedding'] = {init_item_descriptions[i - 1]: None}

        return item_context





    def load_text(self):
        token_text = {}
        item_text = ['[PAD]']
        if 'ml-' in self.dataset_name:
            feat_path = osp.join(self.data_path, f'ml-100k.item')
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    parts = line.strip().split('\t')
                    if len(parts) == 4:
                        # Old format: item_id, movie_title, release_year, genre
                        item_id, movie_title, release_year, genre = parts
                    elif len(parts) == 3:
                        # New format: item_id, title (with year), category
                        item_id, movie_title, genre = parts
                    else:
                        continue
                    token_text[item_id] = movie_title
            for i, token in enumerate(self.item_id_token):
                if token == '[PAD]': continue
                raw_text = token_text[token]
                if raw_text.endswith(', The'):
                    raw_text = 'The ' + raw_text[:-5]
                elif raw_text.endswith(', A'):
                    raw_text = 'A ' + raw_text[:-3]
                item_text.append(raw_text)
            return item_text
        else:
            feat_path = osp.join(self.data_path, f'CDs.item')
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    try:
                        item_id, movie_title, genre = line.strip().split('\t')
                    except ValueError:
                        print(line)
                        item_id, movie_title = line.strip().split('\t')
                    token_text[item_id] = movie_title
            for i, token in enumerate(self.item_id_token):
                if token == '[PAD]': continue
                raw_text = token_text[token]
                item_text.append(raw_text)
            return item_text




    def generate_embedding(self, embedding_contents):
        batch_size = len(embedding_contents)
        embeddings = []
        
        # Bedrock Titan Embed v2 hard limit: 8192 tokens.
        # Agent memories tokenize at ~3.5-3.7 chars/token (structured text with
        # emails, IDs, interaction logs tokenizes more densely than plain prose).
        # 8326 tokens observed from 30000 chars → ~3.6 chars/token.
        # Use 7500 token target × 3.5 chars/token = 26250 chars for safe headroom.
        MAX_EMBEDDING_CHARS = 26000
        
        # Filter out empty strings, track indices, and truncate long content
        valid_contents = []
        valid_indices = []
        for i, content in enumerate(embedding_contents):
            if content and content.strip():  # Only include non-empty strings
                # Truncate if too long — use tail-preserving truncation so that
                # attack payloads appended at the end (TOMA canaries, MAMA PII blocks)
                # are retained.  We keep the first 40% and last 60% of the budget
                # so both the base preference profile and the appended payload survive.
                if len(content) > MAX_EMBEDDING_CHARS:
                    self.logger.warning(
                        f"Truncating embedding content from {len(content)} to "
                        f"{MAX_EMBEDDING_CHARS} chars (index {i})"
                    )
                    head = int(MAX_EMBEDDING_CHARS * 0.4)
                    tail = MAX_EMBEDDING_CHARS - head
                    content = content[:head] + " [...] " + content[-tail:]
                valid_contents.append(content)
                valid_indices.append(i)
            else:
                self.logger.warning(f"Skipping empty embedding content at index {i}")
        
        if not valid_contents:
            self.logger.error("All embedding contents are empty!")
            # Return zero embeddings for all
            embedding_dim = 1536  # Default dimension for embeddings
            return torch.zeros(batch_size, embedding_dim).to(self.device)
        
        # Generate embeddings only for valid contents
        for i in range(0, len(valid_contents), self.api_batch):
            embeddings += asyncio.run(self.embedding_agent.llm.agenerate_response(valid_contents[i:i+self.api_batch]))

        embeddings = [_["data"][0]["embedding"] for _ in embeddings]
        embeddings_tensor = torch.Tensor(embeddings).to(self.device) # valid_size, embedding_size
        embeddings_tensor = embeddings_tensor / embeddings_tensor.norm(p=2,dim=-1, keepdim=True)
        
        # Create full embedding tensor with zeros for empty contents
        embedding_dim = embeddings_tensor.shape[1]
        full_embeddings = torch.zeros(batch_size, embedding_dim).to(self.device)
        for i, valid_idx in enumerate(valid_indices):
            full_embeddings[valid_idx] = embeddings_tensor[i]
        
        return full_embeddings

    # ── Base recommendation loop (ConnaCF, WWW 2024) ─────────────────────────
    def forward(self, batch_user, batch_pos_item, batch_neg_item, round_idx=0, batch_idx=0):
        batch_size = batch_user.size(0)
        user_descriptions, pos_item_descriptions, neg_item_descriptions = [], [], []
        
        # Store enriched contexts from U-U interactions
        enriched_contexts = {}
        
        # Log UU/UI status
        uu_enabled = self.interaction_config.enable_uu_interaction
        ui_enabled = self.interaction_config.enable_ui_interaction
        if uu_enabled or ui_enabled:
            print(f"[Forward] Batch {batch_idx}: Processing {batch_size} users (UU={uu_enabled}, UI={ui_enabled})")
        
        # Collect base descriptions first
        base_user_descs = []
        base_pos_descs = []
        base_neg_descs = []
        user_ids = []
        candidate_items_list = []
        
        for i, user in enumerate(batch_user):
            user_id = int(user)
            user_ids.append(user_id)
            user_agent = self.user_agents[user_id]
            pos_item_agent = self.item_agents[int(batch_pos_item[i])]
            neg_item_agent = self.item_agents[int(batch_neg_item[i])]
            
            base_user_descs.append(user_agent.update_memory[-1])
            base_pos_descs.append(pos_item_agent.update_memory[-1])
            base_neg_descs.append(neg_item_agent.update_memory[-1])
            candidate_items_list.append([int(batch_pos_item[i]), int(batch_neg_item[i])])
        
        # === BATCHED U-U INTERACTIONS ===
        if uu_enabled:
            print(f"  [UU] Running batched U-U interactions for {batch_size} users...")
            try:
                uu_results = asyncio.run(self._batched_user_user_interactions(
                    user_ids=user_ids,
                    candidate_items_list=candidate_items_list,
                    attack_hooks=self.attack_hooks,
                    round_idx=round_idx,
                    batch_idx=batch_idx
                ))
                for i, user_id in enumerate(user_ids):
                    enriched_contexts[user_id] = uu_results[i].get('enriched_context', '')
                print(f"  [UU] Completed batched U-U interactions")
            except Exception as e:
                self.logger.warning(f"Batched U-U interaction failed: {e}")
                import traceback
                traceback.print_exc()
                for user_id in user_ids:
                    enriched_contexts[user_id] = ''
        
        # === BATCHED U-I DIALOGUES ===
        ui_results = {}
        if ui_enabled:
            print(f"  [UI] Running batched U-I dialogues for {batch_size} users...")
            try:
                ui_results_list = asyncio.run(self._batched_user_item_dialogues(
                    user_ids=user_ids,
                    candidate_items_list=candidate_items_list,
                    enriched_contexts=enriched_contexts,
                    attack_hooks=self.attack_hooks,
                    round_idx=round_idx,
                    batch_idx=batch_idx
                ))
                for i, user_id in enumerate(user_ids):
                    ui_results[user_id] = ui_results_list[i]
                print(f"  [UI] Completed batched U-I dialogues")
            except Exception as e:
                self.logger.warning(f"Batched U-I dialogue failed: {e}")
                import traceback
                traceback.print_exc()
        
        # === BUILD FINAL DESCRIPTIONS ===
        for i, user_id in enumerate(user_ids):
            user_desc = base_user_descs[i]
            pos_desc = base_pos_descs[i]
            neg_desc = base_neg_descs[i]
            
            # Enhance with UU results
            if uu_enabled and user_id in enriched_contexts:
                uu_result = uu_results[i] if uu_enabled else {}
                if uu_result.get('synthesis'):
                    user_desc = f"{user_desc}\n\nFriend consultation insights: {uu_result['synthesis']}"
            
            # Enhance with UI results
            if ui_enabled and user_id in ui_results:
                dialogue_summaries = ui_results[user_id].get('dialogue_summaries', {})
                pos_item_id = candidate_items_list[i][0]
                neg_item_id = candidate_items_list[i][1]
                
                if pos_item_id in dialogue_summaries:
                    summary = dialogue_summaries[pos_item_id]
                    pos_desc = f"{pos_desc}\n\nDialogue sentiment: {summary.overall_sentiment}"
                if neg_item_id in dialogue_summaries:
                    summary = dialogue_summaries[neg_item_id]
                    neg_desc = f"{neg_desc}\n\nDialogue sentiment: {summary.overall_sentiment}"
            
            user_descriptions.append(user_desc)
            pos_item_descriptions.append(pos_desc)
            neg_item_descriptions.append(neg_desc)

        print(f"[Forward] Building system prompts for {batch_size} users...")
        system_forward_prompts = [self.rec_agent.astep_forward(int(batch_user[i]), user_descriptions[i], pos_item_descriptions[i], neg_item_descriptions[i]) for i in range(batch_size)]
        print(f"[Forward] Calling LLM for system responses (api_batch={self.api_batch})...")
        system_responses = []
        for i in range(0, batch_size, self.api_batch):
            batch_end = min(i + self.api_batch, batch_size)
            print(f"[Forward] LLM batch {i//self.api_batch + 1}: prompts {i}-{batch_end}...")
            system_responses += asyncio.run(self.rec_agent.llm.agenerate_response(system_forward_prompts[i:i + self.api_batch]))
        print(f"[Forward] Parsing {len(system_responses)} responses...")
        # print(system_responses)
        system_responses = [self.rec_agent.output_parser.parse(response) for response in system_responses]
        system_selections, system_reasons = [], []
        for response in system_responses:
            system_selections.append(response[0])
            system_reasons.append(response[1])
        return system_selections, system_reasons


    def backward(self, system_reasons, batch_user, batch_pos_item, batch_neg_item):
        batch_size = len(batch_user)
        pos_item_descriptions_forward, neg_item_descriptions_forward, pos_item_titles, neg_item_titles, user_descriptions_forward = [], [], [], [], []
        for i, user in enumerate(batch_user):
            pos_item_agent = self.item_agents[int(batch_pos_item[i])]
            neg_item_agent = self.item_agents[int(batch_neg_item[i])]
            pos_item_titles.append(pos_item_agent.role_description['item_title'])
            neg_item_titles.append(neg_item_agent.role_description['item_title'])
            pos_item_descriptions_forward.append(pos_item_agent.update_memory[-1])
            neg_item_descriptions_forward.append(neg_item_agent.update_memory[-1])
            user_descriptions_forward.append(self.user_agents[int(user)].update_memory[-1])

        user_backward_prompts = [self.user_agents[int(batch_user[i])].astep_backward(system_reasons[i], pos_item_titles[i], neg_item_titles[i], pos_item_descriptions_forward[i], neg_item_descriptions_forward[i]) for i in range(batch_size)]
        user_update_descriptions = []
        for i in range(0, batch_size, self.chat_api_batch):
            user_update_descriptions += asyncio.run(self.user_agents[0].llm_chat.agenerate_response_without_construction(user_backward_prompts[i:i+self.chat_api_batch]))

        # Parse user updates with retry logic
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
                            print(f"Retrying user update {i} with directive prompt (attempt {retry_count + 1})")
                            
                            # Get the directive prompt from the parser
                            directive_prompt = parsed_result[1]
                            
                            # Create a new prompt that includes the directive
                            original_prompt = user_backward_prompts[i]
                            retry_prompt = f"{original_prompt}\n\n{directive_prompt}"
                            
                            # Get new response with directive prompt
                            retry_response = asyncio.run(
                                self.user_agents[0].llm_chat.agenerate_response_without_construction([retry_prompt])
                            )[0]
                            
                            response = retry_response
                            retry_count += 1
                            continue
                        else:
                            print(f"Max retries exceeded for user update {i}, using fallback")
                            parsed_result = f"I have updated my preferences based on this music experience."
                            break
                    else:
                        # Successful parse
                        break
                        
                except Exception as e:
                    print(f"Error parsing user update {i} (attempt {retry_count + 1}): {e}")
                    if retry_count >= max_retries:
                        print(f"Max retries exceeded for user update {i}, using fallback")
                        parsed_result = f"I have updated my preferences based on this music experience."
                        break
                    retry_count += 1
                    continue
            
            parsed_user_updates.append(parsed_result)
        
        user_update_descriptions = parsed_user_updates
        _cfg = self.config.final_config_dict if hasattr(self.config, 'final_config_dict') else {}
        if _cfg.get('disable_user_backward', False):
            print("*"*10 + "User Backward Skipped (disable_user_backward=True)" + "*"*10 + '\n')
        else:
            for i, user in enumerate(batch_user):
                user_id = int(user)
                self.user_agents[user_id].update_memory.append(user_update_descriptions[i])
                # Truncate memory to prevent token limit issues (keep last 10 entries)
                if len(self.user_agents[user_id].update_memory) > 10:
                    self.user_agents[user_id].update_memory = self.user_agents[user_id].update_memory[-10:]
        print("*"*10 + "User Update Is Over!" + "*"*10 + '\n')

        if _cfg.get('disable_item_backward', False):
            print("*"*10 + "Item Backward Skipped (disable_item_backward=True)" + "*"*10 + '\n')
            self.logging_during_updation(batch_user, system_reasons, user_backward_prompts, pos_item_descriptions_forward, neg_item_descriptions_forward, user_update_descriptions, [])
        else:
            item_backward_prompts = [
                self.item_agents[int(batch_pos_item[i])].astep_backward(
                    system_reasons[i], pos_item_titles[i], neg_item_titles[i], pos_item_descriptions_forward[i],
                    neg_item_descriptions_forward[i], user_update_descriptions[i])
                for i in range(batch_size)]

            item_update_memories = []
            for i in range(0, batch_size, self.chat_api_batch):
                item_update_memories += asyncio.run(
                    self.item_agents[0].llm_chat.agenerate_response(item_backward_prompts[i:i + self.chat_api_batch]))

            # Parse responses with error handling and retry logic
            parsed_memories = []
            for i, response in enumerate(item_update_memories):
                retry_count = 0
                max_retries = 3

                while retry_count <= max_retries:
                    try:
                        parsed_result = self.item_agents[0].output_parser.parse(response)

                        if (isinstance(parsed_result, tuple) and len(parsed_result) == 2 and
                                parsed_result[0] == "RETRY_NEEDED"):
                            if retry_count < max_retries:
                                print(f"Retrying item update {i} with directive prompt (attempt {retry_count + 1})")
                                directive_prompt = parsed_result[1]
                                original_prompt = item_backward_prompts[i]
                                retry_prompt = f"{original_prompt}\n\n{directive_prompt}"
                                retry_response = asyncio.run(
                                    self.item_agents[0].llm_chat.agenerate_response([retry_prompt])
                                )[0]
                                response = retry_response
                                retry_count += 1
                                continue
                            else:
                                print(f"Max retries exceeded for item update {i}, using empty strings")
                                parsed_result = ("", "")
                                break
                        else:
                            break
                    except Exception as e:
                        print(f"Error parsing response {i} (attempt {retry_count + 1}): {e}")
                        if retry_count >= max_retries:
                            print(f"Max retries exceeded for item update {i}, using empty strings")
                            parsed_result = ("", "")
                            break
                        retry_count += 1
                        continue

                parsed_memories.append(parsed_result)

            item_update_memories = parsed_memories

            for i in range(batch_size):
                if len(item_update_memories[i]) != 2:
                    print("*" * 10 + "item update 出现 bug" + "*" * 10 + '\n')
                elif item_update_memories[i][0] == "" and item_update_memories[i][1] == "":
                    print(f"*" * 10 + "item update returned empty strings for batch {i}" + "*" * 10 + '\n')
                else:
                    pos_item_id = int(batch_pos_item[i])
                    self.item_agents[pos_item_id].update_memory.append(item_update_memories[i][1])
                    if len(self.item_agents[pos_item_id].update_memory) > 10:
                        self.item_agents[pos_item_id].update_memory = self.item_agents[pos_item_id].update_memory[-10:]
                    print(f"[ITEM_UPDATE] Item {pos_item_id} updated using 1 user profile and 1 user decision")
                    if self.config['update_neg_item']:
                        neg_item_id = int(batch_neg_item[i])
                        self.item_agents[neg_item_id].update_memory.append(item_update_memories[i][0])
                        if len(self.item_agents[neg_item_id].update_memory) > 10:
                            self.item_agents[neg_item_id].update_memory = self.item_agents[neg_item_id].update_memory[-10:]
                        print(f"[ITEM_UPDATE] Item {neg_item_id} (neg) updated using 1 user profile and 1 user decision")
            print("*" * 10 + "Item Update Is Over!" + "*" * 10 + '\n')
            self.logging_during_updation(batch_user, system_reasons, user_backward_prompts, pos_item_descriptions_forward, neg_item_descriptions_forward, user_update_descriptions, item_update_memories)



    def backward_true(self, system_reasons, batch_user, batch_pos_item, batch_neg_item, round_1):
        batch_size = len(batch_user)
        pos_item_descriptions_forward, neg_item_descriptions_forward, pos_item_titles, neg_item_titles, user_descriptions_forward = [], [], [], [], []
        for i, user in enumerate(batch_user):
            pos_item_agent = self.item_agents[int(batch_pos_item[i])]
            neg_item_agent = self.item_agents[int(batch_neg_item[i])]
            pos_item_titles.append(pos_item_agent.role_description['item_title'])
            neg_item_titles.append(neg_item_agent.role_description['item_title'])
            pos_item_descriptions_forward.append(pos_item_agent.update_memory[-1])
            neg_item_descriptions_forward.append(neg_item_agent.update_memory[-1])
            user_descriptions_forward.append(self.user_agents[int(user)].update_memory[-1])

        if round_1:
            user_backward_prompts = [self.user_agents[int(batch_user[i])].astep_backward_true(system_reasons[i], pos_item_titles[i], neg_item_titles[i], pos_item_descriptions_forward[i], neg_item_descriptions_forward[i]) for i in range(batch_size)]
            user_update_descriptions = []
            for i in range(0, batch_size, self.chat_api_batch):
                user_update_descriptions += asyncio.run(self.user_agents[0].llm_chat.agenerate_response_without_construction(user_backward_prompts[i:i+self.chat_api_batch]))

            # Parse user updates with retry logic (round_1)
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
                                print(f"Retrying user update {i} (round_1) with directive prompt (attempt {retry_count + 1})")
                                
                                # Get the directive prompt from the parser
                                directive_prompt = parsed_result[1]
                                
                                # Create a new prompt that includes the directive
                                original_prompt = user_backward_prompts[i]
                                retry_prompt = f"{original_prompt}\n\n{directive_prompt}"
                                
                                # Get new response with directive prompt
                                retry_response = asyncio.run(
                                    self.user_agents[0].llm_chat.agenerate_response_without_construction([retry_prompt])
                                )[0]
                                
                                response = retry_response
                                retry_count += 1
                                continue
                            else:
                                print(f"Max retries exceeded for user update {i} (round_1), using fallback")
                                parsed_result = f"I have updated my preferences based on this music experience."
                                break
                        else:
                            # Successful parse
                            break
                            
                    except Exception as e:
                        print(f"Error parsing user update {i} (round_1) (attempt {retry_count + 1}): {e}")
                        if retry_count >= max_retries:
                            print(f"Max retries exceeded for user update {i} (round_1), using fallback")
                            parsed_result = f"I have updated my preferences based on this music experience."
                            break
                        retry_count += 1
                        continue
                
                parsed_user_updates.append(parsed_result)
            
            user_update_descriptions = parsed_user_updates
            for i, user in enumerate(batch_user):
                user_id = int(user)
                self.user_agents[user_id].update_memory.append(user_update_descriptions[i])
                # Truncate memory to prevent token limit issues (keep last 10 entries)
                if len(self.user_agents[user_id].update_memory) > 10:
                    self.user_agents[user_id].update_memory = self.user_agents[user_id].update_memory[-10:]

            item_backward_prompts = [
                self.item_agents[int(batch_pos_item[i])].astep_backward_true(
                    system_reasons[i], pos_item_titles[i], neg_item_titles[i], pos_item_descriptions_forward[i],
                    neg_item_descriptions_forward[i], user_update_descriptions[i])
                for i in range(batch_size)]
        else:
            item_backward_prompts = [
                self.item_agents[int(batch_pos_item[i])].astep_backward_true(
                    system_reasons[i], pos_item_titles[i], neg_item_titles[i], pos_item_descriptions_forward[i],
                    neg_item_descriptions_forward[i], user_descriptions_forward[i])
                for i in range(batch_size)]

        item_update_memories = []
        for i in range(0, batch_size, self.chat_api_batch):
            item_update_memories += asyncio.run(
                self.item_agents[0].llm_chat.agenerate_response(item_backward_prompts[i:i + self.chat_api_batch]))

        item_update_memories = [self.item_agents[0].output_parser.parse(response)
                                for response in item_update_memories]

        # for descriptions in item_update_memories:
        #     print(descriptions)
        #     print('\n\n')
        #     input()

        for i in range(batch_size):
            if len(item_update_memories[i]) != 2:
                print("*" * 10 + "item update 出现 bug" + "*" * 10 + '\n')
            else:
                pos_item_id = int(batch_pos_item[i])
                self.item_agents[pos_item_id].update_memory.append(item_update_memories[i][1])
                # Truncate memory to prevent token limit issues (keep last 10 entries)
                if len(self.item_agents[pos_item_id].update_memory) > 10:
                    self.item_agents[pos_item_id].update_memory = self.item_agents[pos_item_id].update_memory[-10:]
                # Log user profile and decision counts for item update (backward_true)
                print(f"[ITEM_UPDATE_TRUE] Item {pos_item_id} updated using 1 user profile and 1 user decision")
                # TODO: whether to update negative item?
                if self.config['update_neg_item']:
                    neg_item_id = int(batch_neg_item[i])
                    self.item_agents[neg_item_id].update_memory.append(item_update_memories[i][0])
                    # Truncate memory to prevent token limit issues (keep last 10 entries)
                    if len(self.item_agents[neg_item_id].update_memory) > 10:
                        self.item_agents[neg_item_id].update_memory = self.item_agents[neg_item_id].update_memory[-10:]
                    print(f"[ITEM_UPDATE_TRUE] Item {neg_item_id} (neg) updated using 1 user profile and 1 user decision")


    def convert_system_selections_to_accuracy(self, system_selections, pos_items, neg_items):
        """
        判断系统的推荐结果是否正确
        """
        accuracy = []
        for i, selection in enumerate(system_selections):
            pos_item_title = self.item_text[int(pos_items[i])]
            neg_item_title = self.item_text[int(neg_items[i])]
            matched_name, _ = process.extractOne(selection, [pos_item_title, neg_item_title])
            if matched_name == pos_item_title:
                accuracy.append(1)
            else:
                accuracy.append(0)
        return accuracy


    def calculate_loss(self, interaction):
        print(f"[calculate_loss] Starting...")
        print(f"User ID is : {interaction[self.USER_ID]}")
        print(f"BPR pos item (pre-candidate override): {interaction[self.ITEM_ID]}")
        batch_user = interaction[self.USER_ID]
        batch_pos_item = interaction[self.ITEM_ID]
        batch_neg_item = interaction[self.NEG_ITEM_ID]
        batch_size = batch_user.size(0)
        print(f"[calculate_loss] Batch size: {batch_size}, all_update_rounds: {self.config['all_update_rounds']}")

        # have_recorded_idx = set()

        for i in range(self.config['all_update_rounds']):
            print("~"*20 + f"{i}-th round update!" + "~"*20 + '\n')
            first_time = set()
            # TODO: forward part i.e. candidate item selection
            user_forward_description, pos_item_forward_description, neg_item_forward_description = [], [], []
            for j in range(batch_size):
                user_forward_description.append(self.user_agents[int(batch_user[j])].update_memory[-1])
                pos_item_forward_description.append(self.item_agents[int(batch_pos_item[j])].update_memory[-1])
                neg_item_forward_description.append(self.item_agents[int(batch_neg_item[j])].update_memory[-1])
            system_selections, system_reasons = self.forward(batch_user, batch_pos_item, batch_neg_item, round_idx=i, batch_idx=0)
            accuracy = self.convert_system_selections_to_accuracy(system_selections, batch_pos_item, batch_neg_item)
            print(f"Current accuracy is {sum(accuracy) / len(accuracy)}")

            backward_system_reasons, backward_user, backward_pos_item, backward_neg_item, backward_system_reasons_true, backward_user_true, backward_pos_item_true, backward_neg_item_true = [], [], [], [], [], [], [], []
            for j, acc in enumerate(accuracy):
                if acc == 0: # record wrong choices
                    backward_pos_item.append(int(batch_pos_item[j]))
                    backward_neg_item.append(int(batch_neg_item[j]))
                    backward_user.append(int(batch_user[j]))
                    backward_system_reasons.append(system_reasons[j])
                else: # record right choices
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

            # if sum(accuracy) / len(accuracy) > 0.9: break
            print(f"the user who are about to be updated: {backward_user}")
            self.backward(backward_system_reasons, backward_user, backward_pos_item, backward_neg_item)
            if i == 0 and len(backward_user_true):
                self.backward_true(backward_system_reasons_true, backward_user_true, backward_pos_item_true, backward_neg_item_true, True)
        self.backward_true(backward_system_reasons_true, backward_user_true, backward_pos_item_true,
                               backward_neg_item_true, False)


        if self.config['evaluation'] == 'rag':
            system_reasons_embeddings = self.generate_embedding(system_reasons)
            for i, user in enumerate(batch_user):
                self.rec_agent.user_examples[int(user)][(user_forward_description[i], self.item_text[int(batch_pos_item[i])], self.item_text[int(batch_neg_item[i])], pos_item_forward_description[i], neg_item_forward_description[i], accuracy[i], system_reasons[i])] = system_reasons_embeddings[i]
        else:
            for i, user in enumerate(batch_user):
                self.rec_agent.user_examples[int(user)][(user_forward_description[i], self.item_text[int(batch_pos_item[i])], self.item_text[int(batch_neg_item[i])], pos_item_forward_description[i], neg_item_forward_description[i], accuracy[i], system_reasons[i])] = None

        self.logging_after_updation(batch_user, batch_pos_item, batch_neg_item)
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

    def logging_during_updation(self, batch_user, system_explanations, user_backward_prompts, pos_item_descriptions_forward, neg_item_descriptions_forward, user_update_descriptions, item_update_memories):
        batch_size = len(batch_user)
        for i in range(batch_size):
            user_id = int(batch_user[i])
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'user_record_{self.record_idx}')
            if not os.path.exists(path):
                os.makedirs(path)
            with open(osp.join(path, f'user.{str(user_id)}'), 'a') as f:
                f.write('~' * 20 + 'Updation during reflection' + '~' * 20 + '\n')
                f.write(
                    f'There are two candidate CDs. \n The positive CD has the following information: {pos_item_descriptions_forward[i]}. \n The negative CD has the following information: {neg_item_descriptions_forward[i]}\n\n')
                f.write(
                    f'The recommender system made unsuitable recommendation. \n Its reasons are as follows: {system_explanations[i]}\n\n'
                )
                f.write(
                    f"The user's previous self-description is as follows: {self.user_agents[user_id].memory_1[-1]}\n\n")
                f.write(
                    f"The prompts to update the user's descriptions is as follows: {user_backward_prompts[i]}\n\n"
                )
                f.write(
                    f'The user updates his self-description as follows: {user_update_descriptions[i]}\n\n')
                # Log user profile and decision counts for item updates
                f.write(
                    f'[Item Update Stats] User profiles used: 1, User decisions used: 1\n')
                if self.config['update_neg_item']:
                    f.write(
                        f'The two candidate CDs update their description. \n The first CD has the following updated information: {item_update_memories[i][1]}\n The second CD has the following updated information {item_update_memories[i][0]} \n\n')
                else:
                    f.write(
                        f'The positive CD has the following updated information: {item_update_memories[i][1]}\n\n')

    def logging_after_updation(self, batch_user, batch_pos_item, batch_neg_item):
        print("~" * 20 + f"loging in record_{self.record_idx}" + "~" * 20)
        batch_size = batch_user.size(0)
        for i, user in enumerate(batch_user):
            user_id = int(user)
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'user_record_{self.record_idx}')
            if not os.path.exists(path):
                os.makedirs(path)
            with open(osp.join(path, f'user.{str(user_id)}'), 'a') as f:
                f.write('~' * 20 + 'New interaction' + '~' * 20 + '\n')
                f.write(
                    f'There are two candidate CDs. \n The first CD has the following information: {list(self.item_agents[int(batch_pos_item[i])].memory_embedding.keys())[-1]}. \n The second CD has the following information: {list(self.item_agents[int(batch_neg_item[i])].memory_embedding.keys())[-1]}\n\n')
                f.write(
                    f"The user's previous self-description is as follows: {self.user_agents[user_id].memory_1[-1]}\n\n")
                f.write(
                    f'The user updates his self-description as follows: {self.user_agents[user_id].update_memory[-1]} \n\n')
                if self.config['update_neg_item']:
                    f.write(
                        f'The two candidate CDs update their description. \n The first CD has the following updated information: {self.item_agents[int(batch_pos_item[i])].update_memory[-1]}\n The second CD has the following updated information {self.item_agents[int(batch_neg_item[i])].update_memory[-1]} \n\n')
                else:
                    f.write(
                        f'The positive CD has the following updated information: {self.item_agents[int(batch_pos_item[i])].update_memory[-1]}\n\n')

        for i in range(batch_size):
            pos_item_id = int(batch_pos_item[i])
            user_id = int(batch_user[i])
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'item_record_{self.record_idx}')
            if not os.path.exists(path):
                os.makedirs(path)
            with open(osp.join(path, f'item.{str(pos_item_id)}'), 'a') as f:
                f.write('~' * 20 + 'New interaction' + '~' * 20 + '\n')
                # Log user profile and decision counts for item updates
                f.write(f'[Item Update Stats] User profiles used: 1, User decisions used: 1\n\n')
                f.write(
                    f"You: {self.item_agents[pos_item_id].role_description['item_title']} and the other movie: {self.item_agents[int(batch_neg_item[i])].role_description['item_title']} are recommended to a user.\n\n")
                f.write(
                    f'You have the following description: {list(self.item_agents[int(batch_pos_item[i])].memory_embedding.keys())[-1]}\n\n')
                f.write(
                    f'The other movie has the following description: {list(self.item_agents[int(batch_neg_item[i])].memory_embedding.keys())[-1]}\n\n')
                f.write(
                    f"The user's previous self-description is as follows: {self.user_agents[user_id].memory_1[-1]}\n\n")
                f.write(
                    f'The user updates his self-description as follows: {self.user_agents[user_id].update_memory[-1]}\n\n')
                f.write(
                    f'You update your description as follows: {self.item_agents[int(batch_pos_item[i])].update_memory[-1]}\n\n')
                if self.config['update_neg_item']:
                    f.write(
                        f'The other item updates the  following description: {self.item_agents[int(batch_neg_item[i])].update_memory[-1]}\n\n')
            if self.config['update_neg_item']:
                neg_item_id = int(batch_neg_item[i])
                path = osp.join(self.config['record_path'], self.dataset_name, 'record',
                                f'item_record_{self.record_idx}')
                if not os.path.exists(path):
                    os.makedirs(path)
                with open(osp.join(path, f'item.{str(neg_item_id)}'), 'a') as f:
                    f.write('~' * 20 + 'New interaction' + '~' * 20 + '\n')
                    # Log user profile and decision counts for item updates
                    f.write(f'[Item Update Stats] User profiles used: 1, User decisions used: 1\n\n')
                    f.write(
                        f"You: {self.item_agents[neg_item_id].role_description['item_title']} and the other movie: {self.item_agents[pos_item_id].role_description['item_title']} are recommended to a user.\n\n")
                    f.write(
                        f'The other movie has the following description: {list(self.item_agents[int(batch_pos_item[i])].memory_embedding.keys())[-1]}\n\n')
                    f.write(
                        f"The user's previous self-description is as follows: {self.user_agents[user_id].memory_1[-1]}\n\n")
                    f.write(
                        f'The user updates his self-description as follows: {self.user_agents[user_id].update_memory[-1]}\n\n')
                    f.write(
                        f'You update your description as follows: {self.item_agents[int(batch_neg_item[i])].update_memory[-1]}\n\n')

    
    def full_sort_predict(self, interaction, idxs):
        """
        Main function to rank with LLMs

        :param interaction:
        :param idxs: item id retrieved by candidate generation models [batch_size, candidate_size]
        :return: score
        """

        batch_size = idxs.shape[0]
        batch_pos_item = interaction[self.ITEM_ID]
        # TODO: load previously saved user and item agents' memories
        if self.config['loaded']:
            self.record_idx = self.config['saved_idx']
            path = osp.join(self.config['data_path'], 'saved', f'{self.record_idx}',)
            with open(f'{path}/user','r') as f:
                f.readline()
                for line in f:
                    user, user_description = line.strip().split('\t')
                    user_id = self.user_token_id[user]
                    self.user_agents[user_id].memory_1.append(user_description)
            batch_user = interaction[self.USER_ID]
            for i, user in enumerate(batch_user):
                self.user_agents[int(user)].historical_interactions = np.load(f'{path}/user_embeddings_{self.user_id_token[int(user)]}.npy',allow_pickle=True).item()
                self.rec_agent.user_examples[int(user)] = np.load(f'{path}/user_examples_{self.user_id_token[int(user)]}.npy', allow_pickle=True).item()
            for i, item in enumerate(range(self.n_items)):
                if os.path.exists(f'{path}/item_embeddings_{self.item_id_token[int(item)]}.npy'):
                    self.item_agents[int(item)].memory_embedding = np.load(f'{path}/item_embeddings_{self.item_id_token[int(item)]}.npy',allow_pickle=True).item()

        if self.config['saved'] and not self.config['loaded']:
            path = osp.join(self.config['data_path'], 'saved', f'{self.record_idx}',)
            if not os.path.exists(path):
                os.makedirs(path)
            for item_id, item_context in self.item_agents.items():
                np.save(f'{path}/item_embeddings_{self.item_id_token[item_id]}.npy',
                        item_context.memory_embedding)
            with open(f'{path}/user','w') as f:
                f.write('user_id:token\tuser_description:token_seq\n')
                for user_id, user_context in self.user_agents.items():
                    user_description = user_context.memory_1[-1]
                    f.write(str(self.user_id_token[user_id]) + '\t' + user_description.replace('\n',' ') + '\n')
                    np.save(f'{path}/user_embeddings_{self.user_id_token[user_id]}.npy', user_context.historical_interactions)
                    np.save(f'{path}/user_examples_{self.user_id_token[user_id]}.npy',
                            self.rec_agent.user_examples[int(user_id)])

        all_candidate_idxs = set(idxs.view(-1).tolist())
        untrained_candidates = []
        for item in range(1, self.n_items):
            if list(self.item_agents[item].memory_embedding.keys())[-1].startswith('The CD is called'):
                if item in all_candidate_idxs:
                    untrained_candidates.append(item)
        print(f"In the reranking stage, there are {len(set(all_candidate_idxs))} candidates in total. \n There are {len(untrained_candidates)} have not been trained.")
        print("!!!")


        batch_user = interaction['user_id']
        batch_user_descriptions = []
        for i in range(batch_size):
            batch_user_descriptions.append(self.user_agents[int(batch_user[i])].memory_1[-1])
        if self.config['evaluation'] == 'rag' and self.config['item_representation'] == 'rag':
            batch_user_embedding_description = self.generate_embedding(batch_user_descriptions)
        else:
            batch_user_embedding_description = None

        scores = torch.full((batch_user.shape[0], self.n_items), -10000.)
        user_descriptions, list_of_item_descriptions, candidate_texts, user_his_texts, batch_user_embedding_explanations, batch_user_his, batch_select_examples = [], [], [], [], [], [], None
        for i in range(batch_size):
            user_id = int(batch_user[i])
            user_his_text, candidate_text, candidate_text_order, candidate_idx, candidate_text_order_description = self.get_batch_inputs(interaction, idxs, i, batch_user_embedding_description)
            user_descriptions.append(self.user_agents[user_id].memory_1[-1])
            user_his_texts.append(user_his_text)
            list_of_item_descriptions.append('\n\n'.join(candidate_text_order_description))
            candidate_texts.append(candidate_text)
            batch_user_his.append(list(self.rec_agent.user_examples[user_id].keys()))


        if self.config['evaluation'] == 'rag':
            batch_select_examples = []
            query_embeddings = self.generate_embedding(list_of_item_descriptions)
            for i in tqdm(range(batch_size)):
                user_memory = self.user_agents[int(batch_user[i])].memory_1
                # Check if there are enough elements to slice [1:-1]
                if len(user_memory) <= 2:
                    # If memory has 2 or fewer elements, use the last element as fallback
                    user_his_descriptions = [user_memory[-1]] if user_memory else [""]
                else:
                    user_his_descriptions = user_memory[1:-1]
                
                # Only proceed if we have descriptions to work with
                if user_his_descriptions and user_his_descriptions[0]:
                    user_his_description_embeddings = self.generate_embedding(user_his_descriptions)
                    distances = distances_from_embeddings(query_embeddings[i], user_his_description_embeddings)
                    indices = indices_of_nearest_neighbors_from_distances(distances)
                    if len(indices) > 0:
                        index = indices[0]
                        batch_select_examples.append(user_his_descriptions[index])
                    else:
                        # Fallback to the last memory entry
                        batch_select_examples.append(user_memory[-1] if user_memory else "")
                else:
                    # Fallback to the last memory entry
                    batch_select_examples.append(user_memory[-1] if user_memory else "")
            np.save(os.path.join(path,'batch_select_examples.npy'), np.array(batch_select_examples))

        if self.config['evaluation'] != 'sequential':
            user_his_texts = None
        evaluation_prompts, messages = self.evaluation(batch_user, user_descriptions, user_his_texts, list_of_item_descriptions, batch_select_examples)



        # # TODO: logging
        # for i in range(batch_size):
        #     user_id = int(batch_user[i])
        #     pos_item = int(batch_pos_item[i])
        #     path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'user_record_{self.record_idx}')
        #     with open(osp.join(path, f'user.{user_id}'), 'a') as f:
        #         f.write('~' * 20 + 'New interaction' + '~' * 20 + '\n')
        #         f.write(
        #             f'The evaluation prompts are as follows: {evaluation_prompts[i]}\n\n')
        #
        #         f.write(f'The system results are as follows: {messages[i]} \n\n')
        #         f.write(
        #             f"The pos item is: {self.item_agents[pos_item].role_description['item_title']}. Its related descriptions is as follows: {list(self.item_agents[pos_item].memory_embedding.keys())[-1]} \n\n")

        batch_pos_item = interaction[self.ITEM_ID]
    
        self.parsing_output_text(scores, messages, idxs, candidate_texts,batch_pos_item)
        return scores




    def evaluation(self, batch_user, user_descriptions, user_his_texts, list_of_item_descriptions, batch_select_examples=None):
        batch_size = len(user_descriptions)
        if batch_select_examples != None:
            # retrieval mode:
            evaluation_prompts = [self.rec_agent.astep_evaluation(int(batch_user[i]), user_descriptions[i], [], list_of_item_descriptions[i], batch_select_examples[i]) for i in range(batch_size)]
        else:
            if self.config['evaluation'] == 'sequential':
                evaluation_prompts = [self.rec_agent.astep_evaluation(int(batch_user[i]), user_descriptions[i], user_his_texts[i],
                                                list_of_item_descriptions[i]) for i in range(batch_size)]
            else:
                evaluation_prompts = [
                    self.rec_agent.astep_evaluation(int(batch_user[i]), user_descriptions[i], [],
                                                    list_of_item_descriptions[i]) for i in range(batch_size)]

        messages = []
        for i in tqdm(range(0, batch_size, self.chat_api_batch)):
            messages += asyncio.run(self.user_agents[0].llm_chat.agenerate_response_without_construction(evaluation_prompts[i:i+self.chat_api_batch]))

        messages = [self.rec_agent.output_parser.parse_evaluation(response) for response in messages]
        return evaluation_prompts, messages

    def get_batch_inputs(self, interaction, idxs, i, user_embedding):
        user_his = interaction[self.ITEM_SEQ]
        user_his_len = interaction[self.ITEM_SEQ_LEN]
        real_his_len = min(self.max_his_len, user_his_len[i].item())
        user_his_text = [str(j+1) + '. ' + self.item_text[user_his[i, user_his_len[i].item() - real_his_len + j].item()] \
                for j in range(real_his_len)]

        candidate_text = [self.item_text[idxs[i, j]]
                          for j in range(idxs.shape[1])]
        candidate_text_order = [str(j + 1) + '. ' + self.item_text[idxs[i, j].item()]
                                for j in range(idxs.shape[1])]

        if self.config['item_representation'] == 'direct':
            candidate_text_order_description = [str(j + 1) + '. ' + self.item_text[idxs[i, j].item()] + ': ' +
                                                list(self.item_agents[idxs[i, j].item()].memory_embedding.keys())[-1]
                                                for j in range(idxs.shape[1])]
        elif self.config['item_representation'] == 'rag' and self.config['evaluation'] == 'rag':
            item_descriptions = []
            for item in idxs[i]:
                item = int(item)
                item_embeddings = list(self.item_agents[item].memory_embedding.values())
                distances = distances_from_embeddings(user_embedding, item_embeddings)
                indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)[0]
                item_descriptions.append(list(self.item_agents[item].memory_embedding.keys())[indices_of_nearest_neighbors])
            candidate_text_order_description = [str(j+1) + '. ' + self.item_text[idxs[i, j].item()] + ': ' +
                                                item_descriptions[j] for j in range(idxs.shape[1])]





        candidate_idx = idxs[i].tolist()

        return user_his_text, candidate_text, candidate_text_order, candidate_idx, candidate_text_order_description
    

    def parsing_output_text(self, scores, messages, idxs, candidate_texts,batch_pos_item):
        all_recommendation_ranking_results = []
        for i, message in enumerate(messages):
            ranking_result = []
            candidate_text = candidate_texts[i]
            matched_names = []
            for j, item_detail in enumerate(message):
                if len(item_detail) < 1:
                    continue
                if item_detail.endswith('candidate movies:'):
                    continue
                pr = item_detail.find('. ')
                if item_detail[:pr].isdigit():
                    item_name = item_detail[pr + 2:].strip()
                else:
                    item_name = item_detail.strip()

                if self.config['match_rule'] == 'exact':
                    for id, candidate_text_single in enumerate(candidate_text):
                        if candidate_text_single in item_name:
                            item_id = idxs[i,id]
                            if scores[i, item_id] > -5000.: break # has been recommended
                            scores[i, item_id] = self.config['recall_budget'] - j
                            break
                elif self.config['match_rule'] == 'fuzzy':
                    matched_name, sim_score = process.extractOne(item_name, candidate_text)
                    matched_names.append(matched_name)
                    matched_idx = candidate_text.index(matched_name)
                    item_id = idxs[i,matched_idx]
                    if scores[i, item_id] > -5000.: continue # has been recommended
                    ranking_result.append(self.item_id_token[item_id])
                    scores[i, item_id] = self.config['recall_budget'] - j
            all_recommendation_ranking_results.append(ranking_result)

    # ── Natural language interaction channels (original contribution) ─────────

    def _find_similar_users(self, user_id: int, k: int = 3) -> list:
        """Find k most similar users via embedding similarity.
        
        Uses user embeddings to find the k nearest neighbors.
        
        Args:
            user_id: ID of the user to find similar users for
            k: Number of similar users to return
            
        Returns:
            List of user IDs sorted by ascending embedding distance
        """
        # Get user embedding
        user_embedding = self.user_embedding.weight[user_id]
        
        # Get all other user embeddings
        other_user_ids = [uid for uid in range(self.n_users) if uid != user_id and uid != 0]
        if not other_user_ids:
            return []
        
        other_embeddings = [self.user_embedding.weight[uid] for uid in other_user_ids]
        
        # Calculate distances
        distances = distances_from_embeddings(user_embedding, other_embeddings)
        
        # Get indices of nearest neighbors
        sorted_indices = indices_of_nearest_neighbors_from_distances(distances)
        
        # Return k nearest user IDs
        k = min(k, len(other_user_ids))
        return [other_user_ids[idx] for idx in sorted_indices[:k]]
    
    async def user_user_interaction(
        self,
        user_id: int,
        candidate_items: list,
        attack_hooks=None,
        round_idx: int = 0,
        batch_idx: int = 0
    ) -> dict:
        """Execute U-U interaction phase.
        
        Finds similar users (friends) and collects their opinions on
        candidate items. Opinions are passed through attack hooks for
        potential interception.
        
        Args:
            user_id: ID of the user consulting friends
            candidate_items: List of candidate item IDs
            attack_hooks: Optional AttackHooks for message interception
            round_idx: Current training round index
            batch_idx: Current batch index
            
        Returns:
            Dict with 'enriched_context', 'friend_opinions', and 'synthesis'
        """
        from model.audience_context import AudienceContext
        from model.conversation_models import FriendOpinion
        
        # Get interaction config
        friends_count = int(self.config['uu_friends_count']) if 'uu_friends_count' in self.config else 3
        max_tokens = int(self.config['uu_opinion_max_tokens']) if 'uu_opinion_max_tokens' in self.config else 150
        
        # Find similar users
        friend_ids = self._find_similar_users(user_id, k=friends_count)
        
        # Include adversarial agents if configured
        adversarial_friend_ids = []
        if attack_hooks:
            adversarial_agents = attack_hooks.get_adversarial_agents()
            for agent in adversarial_agents:
                if hasattr(agent, 'agent_id') and 'user' in str(agent.agent_id).lower():
                    # Extract user ID from agent_id like "user_123"
                    try:
                        adv_id = int(str(agent.agent_id).split('_')[-1])
                        adversarial_friend_ids.append(adv_id)
                    except (ValueError, IndexError):
                        pass
        
        all_friend_ids = friend_ids + adversarial_friend_ids
        
        if not all_friend_ids:
            # No friends found - return empty context
            return {
                'enriched_context': '',
                'friend_opinions': [],
                'synthesis': ''
            }
        
        # Get item descriptions for opinion generation
        item_descriptions = []
        for item_id in candidate_items[:5]:  # Limit to 5 items for opinion
            if item_id in self.item_agents:
                item_agent = self.item_agents[item_id]
                desc = item_agent.update_memory[-1] if item_agent.update_memory else item_agent.role_description_string
                item_descriptions.append(desc)
        
        # Collect opinions from friends
        friend_opinions = []
        conversation_history = []
        
        for friend_idx, friend_id in enumerate(all_friend_ids):
            if friend_id not in self.user_agents:
                continue
            
            print(f"    [UU] User {user_id}: Getting opinion from friend {friend_idx+1}/{len(all_friend_ids)} (friend_id={friend_id})")
            
            friend_agent = self.user_agents[friend_id]
            is_adversarial = friend_id in adversarial_friend_ids
            
            # Calculate similarity score
            user_emb = self.user_embedding.weight[user_id]
            friend_emb = self.user_embedding.weight[friend_id]
            similarity = 1.0 - cosine(user_emb.detach().cpu(), friend_emb.detach().cpu())
            
            # Generate opinion
            opinion = await friend_agent.give_opinion(
                items=item_descriptions,
                requester_id=user_id,
                max_tokens=max_tokens
            )
            
            original_opinion = opinion
            
            # Pass through attack hooks with audience context
            if attack_hooks:
                audience_context = AudienceContext.for_uu_opinion(
                    target_user_id=user_id,
                    friend_id=friend_id,
                    other_friend_ids=[fid for fid in all_friend_ids if fid != friend_id],
                    user_preferences=self.user_agents[user_id].update_memory[-1] if self.user_agents[user_id].update_memory else None,
                    conversation_history=conversation_history,
                    round_idx=round_idx,
                    batch_idx=batch_idx
                )
                
                opinion = attack_hooks.intercept_message(
                    f"user_{friend_id}",
                    opinion,
                    audience_context
                )
                
                # Log interaction
                attack_hooks.log_interaction({
                    'type': 'uu_opinion',
                    'user_id': user_id,
                    'friend_id': friend_id,
                    'opinion': opinion,
                    'original_opinion': original_opinion if opinion != original_opinion else None,
                    'is_adversarial': is_adversarial,
                    'round_idx': round_idx,
                    'batch_idx': batch_idx
                })
            
            # Store opinion
            friend_opinions.append(FriendOpinion(
                friend_id=friend_id,
                opinion=opinion,
                similarity_score=similarity,
                is_adversarial=is_adversarial
            ))
            
            conversation_history.append({
                'speaker_id': f'user_{friend_id}',
                'message': opinion
            })
            
            # Log to conversation logger if available
            if hasattr(self, 'conversation_logger') and self.conversation_logger:
                self.conversation_logger.log_uu_interaction(
                    user_id=user_id,
                    friend_id=friend_id,
                    opinion=opinion,
                    round_idx=round_idx,
                    batch_idx=batch_idx,
                    was_intercepted=(opinion != original_opinion),
                    original_opinion=original_opinion if opinion != original_opinion else None,
                    similarity_score=similarity,
                    is_adversarial=is_adversarial
                )
            
            # MAMA: Track PII leakage in U-U interactions
            if hasattr(self, 'attacker') and hasattr(self.attacker, 'on_uu_interaction'):
                # Call MAMA hook for each friend opinion
                self.attacker.on_uu_interaction(
                    user_id=user_id,
                    friend_ids=[friend_id],
                    messages=[opinion],
                    round_idx=round_idx
                )
        
        # Synthesize opinions
        user_agent = self.user_agents[user_id]
        own_preferences = user_agent.update_memory[-1] if user_agent.update_memory else user_agent.role_description_string_1
        
        synthesis = await user_agent.synthesize_opinions(
            friend_opinions=[{'friend_id': op.friend_id, 'opinion': op.opinion} for op in friend_opinions],
            items=item_descriptions,
            own_preferences=own_preferences
        )
        
        return {
            'enriched_context': synthesis,
            'friend_opinions': friend_opinions,
            'synthesis': synthesis
        }
    
    async def _batched_user_user_interactions(
        self,
        user_ids: list,
        candidate_items_list: list,
        attack_hooks=None,
        round_idx: int = 0,
        batch_idx: int = 0
    ) -> list:
        """Execute U-U interactions for multiple users concurrently.
        
        Batches all friend opinion requests and synthesis calls for efficiency.
        
        Args:
            user_ids: List of user IDs
            candidate_items_list: List of candidate item lists (one per user)
            attack_hooks: Optional AttackHooks for message interception
            round_idx: Current training round index
            batch_idx: Current batch index
            
        Returns:
            List of result dicts (one per user)
        """
        from model.audience_context import AudienceContext
        from model.conversation_models import FriendOpinion
        
        friends_count = int(self.config['uu_friends_count']) if 'uu_friends_count' in self.config else 3
        max_tokens = int(self.config['uu_opinion_max_tokens']) if 'uu_opinion_max_tokens' in self.config else 150
        
        # Phase 1: Collect all friend info and build prompts
        print(f"    [UU-Batch] Phase 1: Collecting friend info for {len(user_ids)} users...")
        all_opinion_tasks = []  # (user_idx, friend_id, friend_agent, items_desc, is_adversarial, similarity)
        user_friend_map = {}  # user_idx -> list of friend indices in all_opinion_tasks
        
        for user_idx, user_id in enumerate(user_ids):
            friend_ids = self._find_similar_users(user_id, k=friends_count)
            
            # Include adversarial agents if configured
            adversarial_friend_ids = []
            if attack_hooks:
                adversarial_agents = attack_hooks.get_adversarial_agents()
                for agent in adversarial_agents:
                    if hasattr(agent, 'agent_id') and 'user' in str(agent.agent_id).lower():
                        try:
                            adv_id = int(str(agent.agent_id).split('_')[-1])
                            adversarial_friend_ids.append(adv_id)
                        except (ValueError, IndexError):
                            pass
            
            all_friend_ids = friend_ids + adversarial_friend_ids
            user_friend_map[user_idx] = []
            
            # Get item descriptions
            item_descriptions = []
            for item_id in candidate_items_list[user_idx][:5]:
                if item_id in self.item_agents:
                    item_agent = self.item_agents[item_id]
                    desc = item_agent.update_memory[-1] if item_agent.update_memory else item_agent.role_description_string
                    item_descriptions.append(desc)
            items_desc = '\n'.join([f"{i+1}. {item}" for i, item in enumerate(item_descriptions)])
            
            for friend_id in all_friend_ids:
                if friend_id not in self.user_agents:
                    continue
                
                friend_agent = self.user_agents[friend_id]
                is_adversarial = friend_id in adversarial_friend_ids
                
                # Calculate similarity
                user_emb = self.user_embedding.weight[user_id]
                friend_emb = self.user_embedding.weight[friend_id]
                similarity = 1.0 - cosine(user_emb.detach().cpu(), friend_emb.detach().cpu())
                
                task_idx = len(all_opinion_tasks)
                all_opinion_tasks.append((user_idx, user_id, friend_id, friend_agent, items_desc, is_adversarial, similarity))
                user_friend_map[user_idx].append(task_idx)
        
        # Phase 2: Generate all opinions concurrently
        print(f"    [UU-Batch] Phase 2: Generating {len(all_opinion_tasks)} friend opinions concurrently...")
        
        async def get_opinion(task):
            user_idx, user_id, friend_id, friend_agent, items_desc, is_adversarial, similarity = task
            try:
                opinion = await friend_agent.give_opinion(
                    items=items_desc.split('\n'),
                    requester_id=user_id,
                    max_tokens=max_tokens
                )
                return opinion
            except Exception as e:
                return f"I think these items look interesting."
        
        # Run all opinion generations concurrently
        opinion_results = await asyncio.gather(*[get_opinion(task) for task in all_opinion_tasks])
        
        # Phase 3: Process opinions through attack hooks and build friend_opinions
        print(f"    [UU-Batch] Phase 3: Processing opinions and building results...")
        user_friend_opinions = {user_idx: [] for user_idx in range(len(user_ids))}
        
        for task_idx, (user_idx, user_id, friend_id, friend_agent, items_desc, is_adversarial, similarity) in enumerate(all_opinion_tasks):
            opinion = opinion_results[task_idx]
            original_opinion = opinion
            
            # Pass through attack hooks
            if attack_hooks:
                all_friend_ids_for_user = [all_opinion_tasks[i][2] for i in user_friend_map[user_idx]]
                audience_context = AudienceContext.for_uu_opinion(
                    target_user_id=user_id,
                    friend_id=friend_id,
                    other_friend_ids=[fid for fid in all_friend_ids_for_user if fid != friend_id],
                    user_preferences=self.user_agents[user_id].update_memory[-1] if self.user_agents[user_id].update_memory else None,
                    conversation_history=[],
                    round_idx=round_idx,
                    batch_idx=batch_idx
                )
                
                opinion = attack_hooks.intercept_message(f"user_{friend_id}", opinion, audience_context)
                
                attack_hooks.log_interaction({
                    'type': 'uu_opinion',
                    'user_id': user_id,
                    'friend_id': friend_id,
                    'opinion': opinion,
                    'original_opinion': original_opinion if opinion != original_opinion else None,
                    'is_adversarial': is_adversarial,
                    'round_idx': round_idx,
                    'batch_idx': batch_idx
                })
            
            # Log to conversation logger if available
            if hasattr(self, 'conversation_logger') and self.conversation_logger:
                self.conversation_logger.log_uu_interaction(
                    user_id=user_id,
                    friend_id=friend_id,
                    opinion=opinion,
                    round_idx=round_idx,
                    batch_idx=batch_idx,
                    was_intercepted=(opinion != original_opinion),
                    original_opinion=original_opinion if opinion != original_opinion else None,
                    similarity_score=similarity,
                    is_adversarial=is_adversarial
                )
            
            # MAMA: Track PII leakage in U-U interactions (visible messages)
            if hasattr(self, 'attacker') and hasattr(self.attacker, 'on_uu_interaction'):
                self.attacker.on_uu_interaction(
                    user_id=user_id,
                    friend_ids=[friend_id],
                    messages=[opinion],
                    round_idx=round_idx
                )
            
            user_friend_opinions[user_idx].append(FriendOpinion(
                friend_id=friend_id,
                opinion=opinion,
                similarity_score=similarity,
                is_adversarial=is_adversarial
            ))
        
        # Phase 4: Synthesize opinions for all users concurrently
        print(f"    [UU-Batch] Phase 4: Synthesizing opinions for {len(user_ids)} users concurrently...")
        
        async def synthesize_for_user(user_idx, user_id):
            friend_opinions = user_friend_opinions[user_idx]
            if not friend_opinions:
                return ''
            
            user_agent = self.user_agents[user_id]
            own_preferences = user_agent.update_memory[-1] if user_agent.update_memory else user_agent.role_description_string_1
            
            # Get item descriptions again
            item_descriptions = []
            for item_id in candidate_items_list[user_idx][:5]:
                if item_id in self.item_agents:
                    item_agent = self.item_agents[item_id]
                    desc = item_agent.update_memory[-1] if item_agent.update_memory else item_agent.role_description_string
                    item_descriptions.append(desc)
            
            try:
                synthesis = await user_agent.synthesize_opinions(
                    friend_opinions=[{'friend_id': op.friend_id, 'opinion': op.opinion} for op in friend_opinions],
                    items=item_descriptions,
                    own_preferences=own_preferences
                )
                return synthesis
            except Exception as e:
                return ''
        
        synthesis_results = await asyncio.gather(*[
            synthesize_for_user(user_idx, user_id) 
            for user_idx, user_id in enumerate(user_ids)
        ])
        
        # Build final results
        results = []
        for user_idx, user_id in enumerate(user_ids):
            results.append({
                'enriched_context': synthesis_results[user_idx],
                'friend_opinions': user_friend_opinions[user_idx],
                'synthesis': synthesis_results[user_idx]
            })
        
        print(f"    [UU-Batch] Completed batched U-U interactions")
        return results
    
    async def _batched_user_item_dialogues(
        self,
        user_ids: list,
        candidate_items_list: list,
        enriched_contexts: dict,
        attack_hooks=None,
        round_idx: int = 0,
        batch_idx: int = 0
    ) -> list:
        """Execute U-I dialogues for multiple users concurrently.
        
        Batches all dialogue turns for efficiency.
        
        Args:
            user_ids: List of user IDs
            candidate_items_list: List of candidate item lists (one per user)
            enriched_contexts: Dict of user_id -> enriched context from UU
            attack_hooks: Optional AttackHooks for message interception
            round_idx: Current training round index
            batch_idx: Current batch index
            
        Returns:
            List of result dicts (one per user)
        """
        from model.audience_context import AudienceContext
        from model.conversation_models import DialogueSummary, ConversationTurn
        
        dialogue_rounds = int(self.config['ui_dialogue_rounds']) if 'ui_dialogue_rounds' in self.config else 2
        pitch_max_tokens = int(self.config['ui_pitch_max_tokens']) if 'ui_pitch_max_tokens' in self.config else 200
        response_max_tokens = int(self.config['ui_response_max_tokens']) if 'ui_response_max_tokens' in self.config else 150
        
        # For simplicity, we'll batch by dialogue round
        # Round 1: All item pitches -> All user responses -> All item addresses
        # Round 2: Repeat
        
        # Initialize dialogue state for each (user, item) pair
        dialogue_states = {}  # (user_idx, item_id) -> DialogueSummary
        
        for user_idx, user_id in enumerate(user_ids):
            for item_id in candidate_items_list[user_idx][:2]:  # Limit to 2 items for speed
                if item_id not in self.item_agents:
                    continue
                item_agent = self.item_agents[item_id]
                item_title = item_agent.role_description.get('item_title', f'Item {item_id}')
                dialogue_states[(user_idx, item_id)] = DialogueSummary(
                    item_id=item_id,
                    item_title=item_title,
                    pitch=""
                )
        
        print(f"    [UI-Batch] Processing {len(dialogue_states)} user-item dialogues over {dialogue_rounds} rounds...")
        
        for round_num in range(dialogue_rounds):
            print(f"    [UI-Batch] Round {round_num + 1}/{dialogue_rounds}: Generating pitches...")
            
            # Phase A: All item pitches
            pitch_tasks = []
            for (user_idx, item_id), summary in dialogue_states.items():
                user_id = user_ids[user_idx]
                item_agent = self.item_agents[item_id]
                user_agent = self.user_agents[user_id]
                user_preferences = user_agent.update_memory[-1] if user_agent.update_memory else user_agent.role_description_string_1
                
                if enriched_contexts.get(user_id):
                    user_preferences = f"{user_preferences}\n\nBased on friend consultations: {enriched_contexts[user_id]}"
                
                pitch_tasks.append((user_idx, user_id, item_id, item_agent, user_preferences))
            
            async def generate_pitch(task):
                user_idx, user_id, item_id, item_agent, user_preferences = task
                try:
                    pitch = await item_agent.pitch_to_user(
                        user_preferences=user_preferences,
                        user_id=user_id,
                        max_tokens=pitch_max_tokens
                    )
                    return pitch
                except Exception as e:
                    return "This is a great item that I think you'll enjoy!"
            
            pitch_results = await asyncio.gather(*[generate_pitch(t) for t in pitch_tasks])
            
            # Process pitches through attack hooks
            for i, (user_idx, user_id, item_id, item_agent, user_preferences) in enumerate(pitch_tasks):
                pitch = pitch_results[i]
                original_pitch = pitch
                
                if attack_hooks:
                    audience_context = AudienceContext.for_ui_pitch(
                        target_user_id=user_id,
                        item_id=item_id,
                        user_preferences=user_preferences,
                        conversation_history=[],
                        round_idx=round_idx,
                        batch_idx=batch_idx
                    )
                    pitch = attack_hooks.intercept_message(f"item_{item_id}", pitch, audience_context)
                
                summary = dialogue_states[(user_idx, item_id)]
                if round_num == 0:
                    summary.pitch = pitch
                
                turn = ConversationTurn.from_item(item_id, pitch)
                if pitch != original_pitch:
                    turn.mark_intercepted(original_pitch)
                summary.add_turn(turn)
                pitch_results[i] = pitch  # Update with potentially modified pitch
            
            print(f"    [UI-Batch] Round {round_num + 1}/{dialogue_rounds}: Generating user responses...")
            
            # Phase B: All user responses
            response_tasks = []
            for i, (user_idx, user_id, item_id, item_agent, user_preferences) in enumerate(pitch_tasks):
                user_agent = self.user_agents[user_id]
                pitch = pitch_results[i]
                response_tasks.append((user_idx, user_id, item_id, user_agent, pitch))
            
            async def generate_response(task):
                user_idx, user_id, item_id, user_agent, pitch = task
                try:
                    response = await user_agent.respond_to_item(
                        item_pitch=pitch,
                        item_id=item_id,
                        max_tokens=response_max_tokens
                    )
                    return response
                except Exception as e:
                    return "I'm interested but have some concerns."
            
            response_results = await asyncio.gather(*[generate_response(t) for t in response_tasks])
            
            # Add responses to summaries
            for i, (user_idx, user_id, item_id, user_agent, pitch) in enumerate(response_tasks):
                response = response_results[i]
                summary = dialogue_states[(user_idx, item_id)]
                turn = ConversationTurn.from_user(user_id, response)
                summary.add_turn(turn)
            
            print(f"    [UI-Batch] Round {round_num + 1}/{dialogue_rounds}: Generating item addresses...")
            
            # Phase C: All item addresses (only if not last round or if we want final address)
            address_tasks = []
            for i, (user_idx, user_id, item_id, item_agent, user_preferences) in enumerate(pitch_tasks):
                user_response = response_results[i]
                address_tasks.append((user_idx, user_id, item_id, item_agent, user_response, user_preferences))
            
            async def generate_address(task):
                user_idx, user_id, item_id, item_agent, user_response, user_preferences = task
                try:
                    address = await item_agent.address_concerns(
                        concerns=user_response,
                        user_preferences=user_preferences,
                        max_tokens=response_max_tokens
                    )
                    return address
                except Exception as e:
                    return "I understand your concerns and I think you'll find this item meets your needs."
            
            address_results = await asyncio.gather(*[generate_address(t) for t in address_tasks])
            
            # Add addresses to summaries
            for i, (user_idx, user_id, item_id, item_agent, user_response, user_preferences) in enumerate(address_tasks):
                address = address_results[i]
                original_address = address
                
                if attack_hooks:
                    audience_context = AudienceContext.for_ui_pitch(
                        target_user_id=user_id,
                        item_id=item_id,
                        user_preferences="",
                        conversation_history=[],
                        round_idx=round_idx,
                        batch_idx=batch_idx
                    )
                    address = attack_hooks.intercept_message(f"item_{item_id}", address, audience_context)
                
                summary = dialogue_states[(user_idx, item_id)]
                turn = ConversationTurn.from_item(item_id, address)
                if address != original_address:
                    turn.mark_intercepted(original_address)
                summary.add_turn(turn)
        
        # Build final results and fire MAMA / logging hooks
        results = []
        for user_idx, user_id in enumerate(user_ids):
            dialogue_summaries = {}
            for item_id in candidate_items_list[user_idx][:2]:
                if (user_idx, item_id) in dialogue_states:
                    summary = dialogue_states[(user_idx, item_id)]
                    dialogue_summaries[item_id] = summary
                    
                    # Log to conversation logger if available
                    if hasattr(self, 'conversation_logger') and self.conversation_logger:
                        self.conversation_logger.log_ui_dialogue(
                            user_id=user_id,
                            item_id=item_id,
                            item_title=summary.item_title,
                            dialogue_turns=[t.to_dict() for t in summary.dialogue_turns],
                            round_idx=round_idx,
                            batch_idx=batch_idx,
                            overall_sentiment=summary.overall_sentiment
                        )
                    
                    # MAMA: Track PII leakage in U-I interactions (visible messages)
                    if hasattr(self, 'attacker') and hasattr(self.attacker, 'on_ui_interaction'):
                        for turn in summary.dialogue_turns:
                            turn_dict = turn.to_dict()
                            if turn_dict['speaker_type'] == 'user':
                                user_message = turn_dict['message']
                                # Find corresponding item response (next turn)
                                item_response = ""
                                turn_idx = summary.dialogue_turns.index(turn)
                                if turn_idx + 1 < len(summary.dialogue_turns):
                                    next_turn = summary.dialogue_turns[turn_idx + 1]
                                    if next_turn.to_dict()['speaker_type'] == 'item':
                                        item_response = next_turn.to_dict()['message']
                                
                                self.attacker.on_ui_interaction(
                                    user_id=user_id,
                                    item_id=item_id,
                                    user_message=user_message,
                                    item_response=item_response,
                                    round_idx=round_idx
                                )
                    
            results.append({'dialogue_summaries': dialogue_summaries})
        
        print(f"    [UI-Batch] Completed batched U-I dialogues")
        return results

    async def user_item_dialogue(
        self,
        user_id: int,
        candidate_items: list,
        enriched_context: str = None,
        attack_hooks=None,
        round_idx: int = 0,
        batch_idx: int = 0
    ) -> dict:
        """Execute U-I dialogue phase.
        
        Initiates dialogues between the user and candidate items.
        Each item pitches to the user, user responds with concerns,
        and item addresses those concerns.
        
        Args:
            user_id: ID of the user engaging in dialogues
            candidate_items: List of candidate item IDs
            enriched_context: Optional context from U-U interaction
            attack_hooks: Optional AttackHooks for message interception
            round_idx: Current training round index
            batch_idx: Current batch index
            
        Returns:
            Dict with 'dialogue_summaries' for each item
        """
        from model.audience_context import AudienceContext
        from model.conversation_models import DialogueSummary, ConversationTurn
        
        # Get interaction config
        dialogue_rounds = int(self.config['ui_dialogue_rounds']) if 'ui_dialogue_rounds' in self.config else 2
        pitch_max_tokens = int(self.config['ui_pitch_max_tokens']) if 'ui_pitch_max_tokens' in self.config else 200
        response_max_tokens = int(self.config['ui_response_max_tokens']) if 'ui_response_max_tokens' in self.config else 150
        
        user_agent = self.user_agents[user_id]
        user_preferences = user_agent.update_memory[-1] if user_agent.update_memory else user_agent.role_description_string_1
        
        # Add enriched context to preferences if available
        if enriched_context:
            user_preferences = f"{user_preferences}\n\nBased on friend consultations: {enriched_context}"
        
        dialogue_summaries = {}
        
        for item_idx, item_id in enumerate(candidate_items[:5]):  # Limit to 5 items for dialogue
            if item_id not in self.item_agents:
                continue
            
            print(f"    [UI] User {user_id}: Dialogue with item {item_idx+1}/{min(len(candidate_items), 5)} (item_id={item_id})")
            
            item_agent = self.item_agents[item_id]
            item_title = item_agent.role_description.get('item_title', f'Item {item_id}')
            
            summary = DialogueSummary(
                item_id=item_id,
                item_title=item_title,
                pitch=""
            )
            
            conversation_history = []
            
            for round_num in range(dialogue_rounds):
                print(f"      [UI] Item {item_id}: Dialogue round {round_num+1}/{dialogue_rounds}")
                # Item pitches to user
                pitch = await item_agent.pitch_to_user(
                    user_preferences=user_preferences,
                    user_id=user_id,
                    max_tokens=pitch_max_tokens
                )
                
                original_pitch = pitch
                
                # Pass through attack hooks
                if attack_hooks:
                    audience_context = AudienceContext.for_ui_pitch(
                        target_user_id=user_id,
                        item_id=item_id,
                        user_preferences=user_preferences,
                        conversation_history=conversation_history,
                        round_idx=round_idx,
                        batch_idx=batch_idx
                    )
                    
                    pitch = attack_hooks.intercept_message(
                        f"item_{item_id}",
                        pitch,
                        audience_context
                    )
                
                if round_num == 0:
                    summary.pitch = pitch
                
                # Add to conversation
                pitch_turn = ConversationTurn.from_item(item_id, pitch)
                if pitch != original_pitch:
                    pitch_turn.mark_intercepted(original_pitch)
                summary.add_turn(pitch_turn)
                conversation_history.append({'speaker': f'item_{item_id}', 'message': pitch})
                
                # User responds with concerns
                concerns = await user_agent.respond_to_item(
                    item_pitch=pitch,
                    item_info=item_agent.role_description,
                    max_tokens=response_max_tokens
                )
                
                summary.add_concern(concerns)
                concern_turn = ConversationTurn.from_user(user_id, concerns)
                summary.add_turn(concern_turn)
                conversation_history.append({'speaker': f'user_{user_id}', 'message': concerns})
                
                # Item addresses concerns
                response = await item_agent.address_concerns(
                    concerns=concerns,
                    user_preferences=user_preferences,
                    max_tokens=response_max_tokens
                )
                
                original_response = response
                
                # Pass through attack hooks
                if attack_hooks:
                    audience_context = AudienceContext.for_ui_concern_response(
                        target_user_id=user_id,
                        item_id=item_id,
                        user_preferences=user_preferences,
                        concerns_text=concerns,
                        conversation_history=conversation_history,
                        round_idx=round_idx,
                        batch_idx=batch_idx
                    )
                    
                    response = attack_hooks.intercept_message(
                        f"item_{item_id}",
                        response,
                        audience_context
                    )
                
                summary.add_response(response)
                response_turn = ConversationTurn.from_item(item_id, response)
                if response != original_response:
                    response_turn.mark_intercepted(original_response)
                summary.add_turn(response_turn)
                conversation_history.append({'speaker': f'item_{item_id}', 'message': response})
            
            # Compute sentiment
            summary.compute_sentiment()
            
            # Log to conversation logger if available
            if hasattr(self, 'conversation_logger') and self.conversation_logger:
                self.conversation_logger.log_ui_dialogue(
                    user_id=user_id,
                    item_id=item_id,
                    item_title=item_title,
                    dialogue_turns=[t.to_dict() for t in summary.dialogue_turns],
                    round_idx=round_idx,
                    batch_idx=batch_idx,
                    overall_sentiment=summary.overall_sentiment
                )
            
            # MAMA: Track PII leakage in U-I interactions
            if hasattr(self, 'attacker') and hasattr(self.attacker, 'on_ui_interaction'):
                # Extract user and item messages from dialogue
                for turn in summary.dialogue_turns:
                    turn_dict = turn.to_dict()
                    if turn_dict['speaker_type'] == 'user':
                        user_message = turn_dict['message']
                        # Find corresponding item response (next turn)
                        item_response = ""
                        turn_idx = summary.dialogue_turns.index(turn)
                        if turn_idx + 1 < len(summary.dialogue_turns):
                            next_turn = summary.dialogue_turns[turn_idx + 1]
                            if next_turn.to_dict()['speaker_type'] == 'item':
                                item_response = next_turn.to_dict()['message']
                        
                        # Call MAMA hook
                        self.attacker.on_ui_interaction(
                            user_id=user_id,
                            item_id=item_id,
                            user_message=user_message,
                            item_response=item_response,
                            round_idx=round_idx
                        )
            
            # Log to attack hooks if available
            if attack_hooks:
                attack_hooks.log_interaction({
                    'type': 'ui_dialogue',
                    'user_id': user_id,
                    'item_id': item_id,
                    'item_title': item_title,
                    'dialogue_turns': [t.to_dict() for t in summary.dialogue_turns],
                    'sentiment': summary.overall_sentiment,
                    'round_idx': round_idx,
                    'batch_idx': batch_idx
                })
            
            dialogue_summaries[item_id] = summary
        
        return {
            'dialogue_summaries': dialogue_summaries
        }


                        


    

# Backward-compatibility aliases
ConnaCF = MultiChannelRecAgent   # new canonical name
AgentCF = MultiChannelRecAgent   # RecBole config files reference "AgentCF"
